"""
Minimal NumPy implementation of a two-layer LSTM regressor with
variational (per-sequence) dropout, trained with Adam.  Dropout is kept
active at inference time to obtain Monte-Carlo (MC) dropout samples in
the sense of Gal & Ghahramani (2016).

Everything here is deliberately dependency-free (numpy only) so that the
experiments in the RAI-Edge paper are reproducible without a deep
learning framework.
"""
from __future__ import annotations
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTMLayer:
    def __init__(self, d_in: int, d_h: int, rng: np.random.Generator):
        k = 1.0 / np.sqrt(d_h)
        self.Wx = rng.uniform(-k, k, size=(d_in, 4 * d_h))
        self.Wh = rng.uniform(-k, k, size=(d_h, 4 * d_h))
        self.b = np.zeros(4 * d_h)
        self.b[d_h:2 * d_h] = 1.0          # forget-gate bias trick
        self.d_in, self.d_h = d_in, d_h

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def forward(self, X):
        """X: (B, T, d_in) -> H: (B, T, d_h). Caches for backward."""
        B, T, _ = X.shape
        H = self.d_h
        h = np.zeros((B, H)); c = np.zeros((B, H))
        self.cache = []
        Hs = np.empty((B, T, H))
        for t in range(T):
            A = X[:, t] @ self.Wx + h @ self.Wh + self.b
            i = _sigmoid(A[:, :H]); f = _sigmoid(A[:, H:2 * H])
            o = _sigmoid(A[:, 2 * H:3 * H]); g = np.tanh(A[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            self.cache.append((X[:, t], h, c, i, f, o, g, c_new, tc))
            h, c = h_new, c_new
            Hs[:, t] = h
        return Hs

    def backward(self, dHs):
        """dHs: (B, T, d_h) upstream grads -> dX (B, T, d_in); accumulates grads."""
        B, T, H = dHs.shape
        dWx = np.zeros_like(self.Wx); dWh = np.zeros_like(self.Wh); db = np.zeros_like(self.b)
        dX = np.empty((B, T, self.d_in))
        dh_next = np.zeros((B, H)); dc_next = np.zeros((B, H))
        for t in reversed(range(T)):
            x, h_prev, c_prev, i, f, o, g, c_new, tc = self.cache[t]
            dh = dHs[:, t] + dh_next
            do = dh * tc
            dc = dh * o * (1 - tc ** 2) + dc_next
            df = dc * c_prev; di = dc * g; dg = dc * i
            dc_next = dc * f
            dA = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
            dWx += x.T @ dA; dWh += h_prev.T @ dA; db += dA.sum(0)
            dX[:, t] = dA @ self.Wx.T
            dh_next = dA @ self.Wh.T
        self.grads = [dWx, dWh, db]
        return dX


class MCDropoutLSTM:
    """Two-layer LSTM -> linear head.  Variational dropout masks are
    sampled once per sequence and shared across time steps."""

    def __init__(self, d_in=3, d_h=32, p_drop=0.2, seed=0):
        self.rng = np.random.default_rng(seed)
        self.l1 = LSTMLayer(d_in, d_h, self.rng)
        self.l2 = LSTMLayer(d_h, d_h, self.rng)
        k = 1.0 / np.sqrt(d_h)
        self.Wo = self.rng.uniform(-k, k, size=(d_h, 1)); self.bo = np.zeros(1)
        self.p = p_drop; self.d_h = d_h
        self.params = self.l1.params() + self.l2.params() + [self.Wo, self.bo]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0
        self.sigma_a_bins = None  # heteroscedastic aleatoric noise (fitted)

    # ---------------- forward / backward ----------------
    def forward(self, X, dropout=True, rng=None):
        rng = rng or self.rng
        B = X.shape[0]
        H1 = self.l1.forward(X)
        if dropout and self.p > 0:
            m1 = (rng.random((B, 1, self.d_h)) > self.p) / (1 - self.p)
            m2 = (rng.random((B, self.d_h)) > self.p) / (1 - self.p)
        else:
            m1 = np.ones((B, 1, self.d_h)); m2 = np.ones((B, self.d_h))
        self._m1, self._m2 = m1, m2
        H2 = self.l2.forward(H1 * m1)
        self._hlast = H2[:, -1] * m2
        return self._hlast @ self.Wo + self.bo   # (B,1)

    def backward(self, dY):
        dWo = self._hlast.T @ dY; dbo = dY.sum(0)
        dh = (dY @ self.Wo.T) * self._m2
        B, H = dh.shape
        dH2 = np.zeros((B, self.l2.cache.__len__(), H)); dH2[:, -1] = dh
        dH1 = self.l2.backward(dH2) * self._m1
        self.l1.backward(dH1)
        self.grads = self.l1.grads + self.l2.grads + [dWo, dbo]

    def adam_step(self, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.step += 1
        gn = np.sqrt(sum((g ** 2).sum() for g in self.grads))
        scale = min(1.0, clip / (gn + 1e-12))
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v):
            g = g * scale
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** self.step); vh = v / (1 - b2 ** self.step)
            p -= lr * mh / (np.sqrt(vh) + eps)

    # ---------------- training ----------------
    def fit(self, X, y, Xv=None, yv=None, epochs=20, batch=128, lr=2e-3, verbose=True):
        n = X.shape[0]
        hist = []
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                bi = idx[s:s + batch]
                yh = self.forward(X[bi], dropout=True)
                err = yh - y[bi][:, None]
                loss = (err ** 2).mean()
                self.backward(2 * err / len(bi))
                self.adam_step(lr=lr)
                tot += loss * len(bi)
            msg = f"epoch {ep+1:2d} train_mse={tot/n:.5f}"
            if Xv is not None:
                yv_hat = self.forward(Xv, dropout=False)[:, 0]
                msg += f" val_mse={((yv_hat-yv)**2).mean():.5f}"
            hist.append(msg)
            if verbose:
                print(msg)
        return hist

    # ---------------- inference ----------------
    def predict_point(self, X):
        """Deterministic forecast (dropout off)."""
        return self.forward(X, dropout=False)[:, 0]

    def mc_samples(self, X, K=50, rng=None, aleatoric=True):
        """K stochastic forward passes -> (K, B) samples.  If an aleatoric
        noise model was fitted, i.i.d. Gaussian observation noise with the
        fitted (heteroscedastic) scale is added to each sample."""
        rng = rng or self.rng
        outs = []
        chunk = max(1, 20000 // K)                # bound memory of the BPTT caches
        for s in range(0, X.shape[0], chunk):
            Xr = np.repeat(X[s:s + chunk], K, axis=0)      # (b*K, T, d)
            outs.append(self.forward(Xr, dropout=True, rng=rng)[:, 0].reshape(-1, K).T)
        yh = np.concatenate(outs, axis=1)
        if aleatoric and self.sigma_a_bins is not None:
            mu = yh.mean(0)
            if self.resid_pool is not None and not self.gaussian_aleatoric:
                # residual bootstrap: draw held-out residuals from the bin of mu
                bins = self.aleatoric_bin(mu)
                eps = np.empty_like(yh)
                for b in np.unique(bins):
                    cols = np.nonzero(bins == b)[0]; pool = self.resid_pool[b]
                    eps[:, cols] = pool[rng.integers(0, len(pool), size=(yh.shape[0], len(cols)))]
                yh = yh + eps
            else:
                s = self.aleatoric_sigma(mu)
                yh = yh + rng.standard_normal(yh.shape) * s[None, :]
        return yh

    def fit_aleatoric(self, X, y, K=50, nbins=6, max_n=4000):
        """Non-parametric heteroscedastic noise model fitted on held-out
        windows.  For each quantile bin of the MC-mean forecast mu we store
        (i) the empirical residuals y - mu (used by residual bootstrap) and
        (ii) their standard deviation (used by the Gaussian comparator)."""
        if X.shape[0] > max_n:
            sel = self.rng.choice(X.shape[0], max_n, replace=False); X, y = X[sel], y[sel]
        mc = self.mc_samples(X, K=K, aleatoric=False)
        mu = mc.mean(0)
        resid = y - mu
        edges = np.quantile(mu, np.linspace(0, 1, nbins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        sig = np.zeros(nbins); pools = []
        for b in range(nbins):
            sel = (mu > edges[b]) & (mu <= edges[b + 1])
            r = resid[sel] if sel.sum() >= 20 else resid
            pools.append(r - r.mean())            # centred residuals
            sig[b] = r.std()
        self.sigma_a_bins = (edges, sig)
        self.resid_pool = pools
        return edges, sig

    resid_pool = None
    gaussian_aleatoric = False

    def aleatoric_bin(self, mu):
        edges, sig = self.sigma_a_bins
        return np.clip(np.searchsorted(edges, mu, side="left") - 1, 0, len(sig) - 1)

    def aleatoric_sigma(self, mu):
        return self.sigma_a_bins[1][self.aleatoric_bin(mu)]

    # ---------------- persistence ----------------
    def save(self, path):
        d = {f"p{i}": p for i, p in enumerate(self.params)}
        d["meta"] = np.array([self.p, self.d_h])
        if self.sigma_a_bins is not None:
            d["edges"], d["sig"] = self.sigma_a_bins
            for b, pool in enumerate(self.resid_pool or []):
                d[f"pool{b}"] = pool
        np.savez(path, **d)

    @classmethod
    def load(cls, path, seed=0):
        z = np.load(path)
        p, dh = z["meta"]
        d_in = z["p0"].shape[0]
        m = cls(d_in=d_in, d_h=int(dh), p_drop=float(p), seed=seed)
        for i, q in enumerate(m.params):
            q[:] = z[f"p{i}"]
        if "edges" in z:
            m.sigma_a_bins = (z["edges"], z["sig"])
            pools = [z[f"pool{b}"] for b in range(len(z["sig"])) if f"pool{b}" in z]
            m.resid_pool = pools if pools else None
        return m


def gradient_check(seed=0):
    """Finite-difference check of the hand-written backward pass."""
    rng = np.random.default_rng(seed)
    m = MCDropoutLSTM(d_in=3, d_h=5, p_drop=0.0, seed=seed)
    X = rng.standard_normal((4, 6, 3)); y = rng.standard_normal(4)
    def loss():
        yh = m.forward(X, dropout=False)[:, 0]
        return ((yh - y) ** 2).mean()
    L0 = loss()
    yh = m.forward(X, dropout=False)[:, 0]
    m.backward((2 * (yh - y) / 4)[:, None])
    worst = 0.0
    for p, g in zip(m.params, m.grads):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]; h = 1e-5
            p[idx] = old + h; lp = loss(); p[idx] = old - h; lm = loss(); p[idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx]) + 1e-8))
    return worst


if __name__ == "__main__":
    print("max relative gradient error:", gradient_check())
