"""
RAI-Edge discrete-time simulator (slot length Delta = 1 s).

Models (all documented in the paper):
  * K fog nodes (micro / macro) + 1 cloud node, SDN-controlled backhaul graph.
  * Mobile IoT devices (Random Waypoint or Gauss-Markov, or a CSV trace),
    each attached to the nearest fog access point (AP).
  * ON/OFF-modulated Poisson task arrivals, uniform per-task workload (GHz.s)
    and uplink size (bytes).
  * Wireless uplink: 3GPP-style path loss + log-normal shadowing, Shannon rate.
  * Backhaul: fluid per-slot link model (utilisation-based slowdown),
    1 ms per hop processing/propagation.  Controller routes task flows and
    migration flows on the residual-capacity graph (widest-shortest path)
    or on static min-hop paths.
  * Compute: M/G/1-PS per node (response time w / (C (1-rho))) plus a
    backlog term when the offered load exceeds capacity.
  * Energy: device radio + backhaul switching + node compute (J per GHz.s,
    heterogeneous) + migration transfer.
  * Stop-and-copy service migration with a per-slot migration budget.

The orchestration policies (RAI-Edge and all baselines) live in the
Controller class at the bottom of the file.
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.stats import norm as _norm

# --------------------------------------------------------------------------
#  Configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    # --- time
    delta: float = 1.0          # slot length [s]
    T_warm: int = 10            # warm-up slots (no decisions, no metrics)
    T_eval: int = 120           # evaluated slots
    # --- area / topology
    area: float = 1000.0        # [m]
    K: int = 10                 # fog nodes
    n_macro: int = 4
    C_micro: float = 16.0       # [GHz]
    C_macro: float = 48.0
    mem_micro: float = 4096.0   # [MB]
    mem_macro: float = 16384.0
    eps_micro: float = 0.6      # [J per GHz.s]
    eps_macro: float = 1.4
    C_cloud: float = 1.0e4
    eps_cloud: float = 2.0
    link_bw: float = 100e6      # [bit/s] backhaul link
    cloud_bw: float = 1e9
    tau_hop: float = 1e-3       # [s] per-hop processing + propagation
    tau_cloud: float = 40e-3    # [s] one-way latency on the cloud link
    k_nearest: int = 3
    # --- wireless
    B_sub: float = 5e6          # [Hz] per-device uplink bandwidth
    P_tx: float = 0.2           # [W] (23 dBm)
    noise_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    shadow_db: float = 4.0
    # --- devices / services
    v_min: float = 1.0; v_max: float = 10.0; t_pause: float = 5.0
    mobility: str = "rwp"       # rwp | gm | trace
    gm_alpha: float = 0.85
    lam_on: float = 3.0; lam_off: float = 0.3      # tasks per second
    p_on_off: float = 0.05; p_off_on: float = 0.03 # ON/OFF switching probs per slot
    wbar_min: float = 0.15; wbar_max: float = 0.35 # mean workload [GHz.s]
    w_spread: float = 0.4                           # per-task w ~ U[(1-s)wbar,(1+s)wbar]
    d_min: float = 20e3; d_max: float = 100e3       # uplink bytes per task
    r_min: float = 0.5e6; r_max: float = 2.0e6      # service state [bytes]
    m_min: float = 64.0; m_max: float = 256.0       # memory footprint [MB]
    max_tasks: int = 14
    tau_sla: float = 0.100      # [s]
    eps_bh: float = 1e-8        # [J/bit/hop]
    rho_max: float = 0.95       # cap for the PS formula
    backlog_cap: float = 2.0    # max backlog in units of C*delta (excess work is shed)
    # --- predictor
    window: int = 10
    K_mc: int = 50
    # --- orchestration defaults
    alpha: float = 0.5
    gamma: float = 0.1
    beta: float = 0.15
    delta_margin: float = 0.10
    T_lh: int = 5
    T_bud: int = 10
    Gamma_robust: float = 2.0
    E_ref: float = 0.15         # [J] normalisation
    r_ref: float = 1e6          # [bytes]
    ewma: float = 0.3
    init_placement: str = "nearest"   # nearest | random

# --------------------------------------------------------------------------
#  Topology
# --------------------------------------------------------------------------

class Topology:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        K = cfg.K
        self.K = K; self.cloud = K; self.n = K + 1
        # jittered grid positions
        side = int(math.ceil(math.sqrt(K)))
        pts = []
        for i in range(K):
            gx, gy = i % side, i // side
            pts.append([(gx + 0.5) / side * cfg.area + rng.uniform(-0.12, 0.12) * cfg.area / side,
                        (gy + 0.5) / side * cfg.area + rng.uniform(-0.12, 0.12) * cfg.area / side])
        self.pos = np.clip(np.array(pts), 0, cfg.area)
        kind = np.zeros(K, dtype=int)
        kind[rng.choice(K, cfg.n_macro, replace=False)] = 1
        self.kind = np.append(kind, 2)                      # 2 = cloud
        self.C = np.where(kind == 1, cfg.C_macro, cfg.C_micro).astype(float)
        self.C = np.append(self.C, cfg.C_cloud)
        self.mem = np.append(np.where(kind == 1, cfg.mem_macro, cfg.mem_micro).astype(float), 1e12)
        self.eps = np.append(np.where(kind == 1, cfg.eps_macro, cfg.eps_micro).astype(float), cfg.eps_cloud)
        # links: k-nearest + MST for connectivity
        D = np.linalg.norm(self.pos[:, None] - self.pos[None], axis=-1)
        edges = set()
        for i in range(K):
            for j in np.argsort(D[i])[1:cfg.k_nearest + 1]:
                edges.add((min(i, j), max(i, j)))
        # MST (Prim)
        inT = [0];
        while len(inT) < K:
            best = None
            for i in inT:
                for j in range(K):
                    if j in inT: continue
                    if best is None or D[i, j] < best[0]: best = (D[i, j], i, j)
            edges.add((min(best[1], best[2]), max(best[1], best[2]))); inT.append(best[2])
        edges.add((0, K))                                   # gateway -> cloud
        self.edges = sorted(edges)
        self.E = len(self.edges)
        self.eid = {e: k for k, e in enumerate(self.edges)}
        self.B = np.array([cfg.cloud_bw if K in e else cfg.link_bw for e in self.edges])
        self.prop = np.array([cfg.tau_cloud if K in e else cfg.tau_hop for e in self.edges])
        self.adj = [[] for _ in range(self.n)]
        for k, (u, v) in enumerate(self.edges):
            self.adj[u].append((v, k)); self.adj[v].append((u, k))
        # static min-hop paths (all pairs, BFS)
        # static min-hop paths with NOMINAL link capacities (used by non-joint schemes for cost estimates)
        self.minhop = self.all_pairs_paths(self.B.copy())

    def all_pairs_paths(self, residual):
        """Widest-shortest paths on the residual graph. Returns dict with
        'hops', 'bneck' (n x n arrays) and 'links' (list of lists of edge ids)."""
        n = self.n
        hops = np.zeros((n, n), dtype=int); bneck = np.zeros((n, n)); links = [[None] * n for _ in range(n)]
        prop = np.zeros((n, n))
        for s in range(n):
            best_b = np.full(n, -1.0); best_h = np.full(n, 10 ** 9); prev = [None] * n
            best_b[s] = 1e18; best_h[s] = 0; done = np.zeros(n, bool)
            for _ in range(n):
                cand = [(-best_b[v], best_h[v], v) for v in range(n) if not done[v] and best_b[v] >= 0]
                if not cand: break
                _, _, u = min(cand); done[u] = True
                for v, k in self.adj[u]:
                    b = min(best_b[u], residual[k]); h = best_h[u] + 1
                    if b > best_b[v] + 1e-9 or (abs(b - best_b[v]) <= 1e-9 and h < best_h[v]):
                        best_b[v] = b; best_h[v] = h; prev[v] = (u, k)
            for v in range(n):
                path = []; x = v
                while x != s and prev[x] is not None:
                    path.append(prev[x][1]); x = prev[x][0]
                links[s][v] = path[::-1]
                hops[s, v] = len(path); bneck[s, v] = best_b[v] if v != s else 1e18
                prop[s, v] = self.prop[path].sum() if path else 0.0
        return dict(hops=hops, bneck=bneck, links=links, prop=prop)

# --------------------------------------------------------------------------
#  Mobility
# --------------------------------------------------------------------------

class Mobility:
    def __init__(self, cfg: Config, N: int, rng: np.random.Generator, trace=None):
        self.cfg, self.N, self.rng = cfg, N, rng
        self.pos = rng.uniform(0, cfg.area, size=(N, 2))
        self.kind = cfg.mobility
        if self.kind == "rwp":
            self.target = rng.uniform(0, cfg.area, size=(N, 2))
            self.speed = rng.uniform(cfg.v_min, cfg.v_max, N)
            self.pause = np.zeros(N)
        elif self.kind == "gm":
            self.vbar = rng.uniform(cfg.v_min, cfg.v_max, N)
            self.v = self.vbar.copy()
            self.theta = rng.uniform(0, 2 * np.pi, N)
            self.thbar = self.theta.copy()
        elif self.kind == "trace":
            self.trace = trace                               # (N, T, 2)
            self.pos = trace[:, 0].copy()
        self.t = 0

    def step(self):
        cfg, rng = self.cfg, self.rng
        self.t += 1
        if self.kind == "rwp":
            d = self.target - self.pos; dist = np.linalg.norm(d, axis=1)
            moving = self.pause <= 0
            stepv = np.minimum(self.speed * cfg.delta, dist)
            self.pos[moving] += d[moving] / np.maximum(dist[moving, None], 1e-9) * stepv[moving, None]
            self.pause[~moving] -= cfg.delta
            arrived = moving & (dist <= self.speed * cfg.delta + 1e-9)
            n = arrived.sum()
            if n:
                self.target[arrived] = rng.uniform(0, cfg.area, size=(n, 2))
                self.speed[arrived] = rng.uniform(cfg.v_min, cfg.v_max, n)
                self.pause[arrived] = cfg.t_pause
        elif self.kind == "gm":
            a = cfg.gm_alpha
            self.v = a * self.v + (1 - a) * self.vbar + math.sqrt(1 - a * a) * 1.0 * rng.standard_normal(self.N)
            self.v = np.clip(self.v, 0.0, cfg.v_max * 1.5)
            self.theta = a * self.theta + (1 - a) * self.thbar + math.sqrt(1 - a * a) * 0.5 * rng.standard_normal(self.N)
            self.pos += np.stack([np.cos(self.theta), np.sin(self.theta)], 1) * (self.v * cfg.delta)[:, None]
            # reflect at the borders
            for k in range(2):
                lo = self.pos[:, k] < 0; hi = self.pos[:, k] > cfg.area
                self.pos[lo, k] = -self.pos[lo, k]; self.pos[hi, k] = 2 * cfg.area - self.pos[hi, k]
                if lo.any() or hi.any():
                    self.thbar[lo | hi] = (self.thbar[lo | hi] + np.pi) % (2 * np.pi)
                    self.theta[lo | hi] = self.thbar[lo | hi]
        else:
            self.pos = self.trace[:, min(self.t, self.trace.shape[1] - 1)].copy()
        return self.pos

# --------------------------------------------------------------------------
#  World (devices, services, nodes, links)
# --------------------------------------------------------------------------

class World:
    def __init__(self, cfg: Config, N: int, seed: int, trace=None):
        self.cfg, self.N = cfg, N
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.topo = Topology(cfg, rng)
        self.mob = Mobility(cfg, N, np.random.default_rng(seed + 1), trace)
        T = cfg.T_warm + cfg.T_eval + 2
        self.T = T
        # per-device static parameters
        self.wbar = rng.uniform(cfg.wbar_min, cfg.wbar_max, N)
        self.dbar = rng.uniform(cfg.d_min, cfg.d_max, N)
        self.r = rng.uniform(cfg.r_min, cfg.r_max, N)
        self.m = rng.uniform(cfg.m_min, cfg.m_max, N)
        # ON/OFF modulated Poisson arrivals, pre-generated for the whole episode
        state = (rng.random(N) < cfg.p_off_on / (cfg.p_off_on + cfg.p_on_off)).astype(int)
        self.ntask = np.zeros((N, T), dtype=int)
        self.wtask = np.zeros((N, T, cfg.max_tasks))
        self.dtask = np.zeros((N, T, cfg.max_tasks))
        for t in range(T):
            lam = np.where(state == 1, cfg.lam_on, cfg.lam_off)
            n = np.minimum(rng.poisson(lam), cfg.max_tasks)
            self.ntask[:, t] = n
            u = rng.uniform(1 - cfg.w_spread, 1 + cfg.w_spread, (N, cfg.max_tasks))
            self.wtask[:, t] = u * self.wbar[:, None]
            self.dtask[:, t] = rng.uniform(0.6, 1.4, (N, cfg.max_tasks)) * self.dbar[:, None]
            flip = rng.random(N) < np.where(state == 1, cfg.p_on_off, cfg.p_off_on)
            state = np.where(flip, 1 - state, state)
        self.Wtot = np.array([[self.wtask[i, t, :self.ntask[i, t]].sum() for t in range(T)] for i in range(N)])
        self.Dtot = np.array([[self.dtask[i, t, :self.ntask[i, t]].sum() for t in range(T)] for i in range(N)])
        # noise power
        self.noise_w = 10 ** ((cfg.noise_dbm_hz + 10 * math.log10(cfg.B_sub) + cfg.noise_figure_db - 30) / 10)
        # dynamic state
        self.t = 0
        self.attach = self.nearest_ap(self.mob.pos)
        self.x = np.zeros(N, dtype=int)                # placement
        self.mem_used = np.zeros(self.topo.n)
        order = rng.permutation(N)
        for i in order:                                 # initial: nearest fog node with free memory
            d = np.linalg.norm(self.topo.pos - self.mob.pos[i], axis=1)
            if cfg.init_placement == "random": d = rng.random(self.topo.K)
            for f in np.argsort(d):
                if self.mem_used[f] + self.m[i] <= self.topo.mem[f]:
                    self.x[i] = f; self.mem_used[f] += self.m[i]; break
        self.backlog = np.zeros(self.topo.n)
        self.h = np.zeros(self.topo.n)                 # offered load [GHz] of last slot
        self.util = np.zeros(self.topo.E)              # link utilisation of last slot
        self.mig_remaining = np.zeros(N)               # remaining migration time [s]
        self.mig_links = [None] * N; self.mig_bytes_left = np.zeros(N)
        self.last_Ltx = np.zeros(N)
        self.nserv = np.bincount(self.x, minlength=self.topo.n)

    # ---- helpers
    def nearest_ap(self, pos):
        d = np.linalg.norm(pos[:, None] - self.topo.pos[None], axis=-1)
        return d.argmin(1)

    def uplink_rate(self, pos, ap, rng):
        cfg = self.cfg
        d = np.maximum(np.linalg.norm(pos - self.topo.pos[ap], axis=1), 10.0)
        pl = 128.1 + 37.6 * np.log10(d / 1000.0) + cfg.shadow_db * rng.standard_normal(len(d))
        prx = cfg.P_tx * 10 ** (-pl / 10)
        return cfg.B_sub * np.log2(1 + prx / self.noise_w)

    def node_features(self):
        """Features used by the demand predictor (per node, fog only)."""
        K = self.topo.K
        return np.stack([self.h[:K] / self.topo.C[:K], self.backlog[:K] / self.topo.C[:K],
                         self.nserv[:K] / 50.0, self.topo.kind[:K].astype(float)], 1)

    def oracle_next_demand(self):
        """True offered load of the coming slot under the current placement."""
        W = np.bincount(self.x, weights=self.Wtot[:, self.t], minlength=self.topo.n)
        return (self.backlog + W) / self.cfg.delta

    # ---- one slot
    def step(self, migrations, routing, mig_paths):
        """migrations: list of (s, f_from, f_to); routing: dict from all_pairs_paths
        used for task flows this slot; mig_paths: list of link-lists for migrations."""
        cfg, topo, t = self.cfg, self.topo, self.t
        N = self.N; rng = self.rng
        # demand that would have materialised without this slot's migrations
        W_pre = np.bincount(self.x, weights=self.Wtot[:, t], minlength=topo.n)
        h_nomig = (self.backlog + W_pre) / cfg.delta
        # apply migrations (stop-and-copy: service state moves now, tasks of this slot wait)
        for (s, f0, f1), lk in zip(migrations, mig_paths):
            self.x[s] = f1; self.mem_used[f0] -= self.m[s]; self.mem_used[f1] += self.m[s]
            self.mig_links[s] = lk; self.mig_bytes_left[s] = self.r[s]
        self.nserv = np.bincount(self.x, minlength=topo.n)
        # tasks of the slot
        n = self.ntask[:, t]
        has = n > 0
        ap = self.attach
        # link loads (bytes) from task flows and active migrations
        load = np.zeros(topo.E)
        hops = np.zeros(N, dtype=int); flow_links = [None] * N
        for i in np.nonzero(has)[0]:
            f = self.x[i]
            if f != ap[i]:
                lk = routing["links"][ap[i]][f]
                flow_links[i] = lk; hops[i] = len(lk)
                load[lk] += self.Dtot[i, t]
        active_mig = np.nonzero(self.mig_bytes_left > 0)[0]
        for s in active_mig:
            load[self.mig_links[s]] += self.mig_bytes_left[s]
        util = load / (topo.B * cfg.delta)
        rate = topo.B / np.maximum(1.0, util)          # fluid fair share [bit/s]
        # migration transfer times / progress
        mig_time = np.zeros(N); mig_done = []; E_mig = 0.0
        for s in active_mig:
            lk = self.mig_links[s]
            r_path = rate[lk].min() if len(lk) else 1e18
            tfull = self.mig_bytes_left[s] * 8 / r_path + topo.prop[lk].sum()
            mig_time[s] = tfull
            sent = min(self.mig_bytes_left[s], r_path * cfg.delta / 8)
            E_mig += cfg.eps_bh * 8 * sent * len(lk)
            if tfull <= cfg.delta:
                self.mig_bytes_left[s] = 0.0; mig_done.append((s, tfull))
            else:
                self.mig_bytes_left[s] -= sent                 # continues next slot
        # uplink
        rate_ul = self.uplink_rate(self.mob.pos, ap, rng)
        self.last_Ltx = self.dbar / rate_ul                  # mean-size estimate kept for the controller
        # backhaul bottleneck rate / propagation per device flow
        rmin_dev = np.full(N, 1e18); prop_dev = np.zeros(N)
        for i in np.nonzero(has)[0]:
            if hops[i]:
                lk = flow_links[i]
                rmin_dev[i] = rate[lk].min(); prop_dev[i] = topo.prop[lk].sum()
        # compute (M/G/1-PS + backlog)
        W = np.bincount(self.x, weights=self.Wtot[:, t], minlength=topo.n)
        h = (self.backlog + W) / cfg.delta
        rho = np.minimum(h / topo.C, cfg.rho_max)
        inv = 1.0 / (topo.C * (1 - rho))
        Lq = self.backlog / topo.C
        # per-task latency
        idx = np.nonzero(has)[0]
        rep = np.repeat(idx, n[idx])
        wt = np.concatenate([self.wtask[i, t, :n[i]] for i in idx]) if len(idx) else np.zeros(0)
        dt = np.concatenate([self.dtask[i, t, :n[i]] for i in idx]) if len(idx) else np.zeros(0)
        f_of = self.x[rep]
        L_tx = dt * 8 / rate_ul[rep]
        L_bh = np.where(hops[rep] > 0, dt * 8 / rmin_dev[rep] + prop_dev[rep], 0.0)
        L_comp = wt * inv[f_of] + Lq[f_of]
        L_mig = mig_time[rep]
        L = L_tx + L_bh + L_comp + L_mig
        E_task = cfg.P_tx * L_tx + cfg.eps_bh * 8 * dt * hops[rep] + topo.eps[f_of] * wt
        # state update: bounded backlog (work beyond b_max is shed and counted as violated tasks)
        b_max = cfg.backlog_cap * topo.C * cfg.delta
        carry = np.maximum(0.0, self.backlog + W - topo.C * cfg.delta)
        dropped_work = np.maximum(0.0, carry - b_max)
        self.backlog = np.minimum(carry, b_max)
        wmean_node = np.bincount(f_of, weights=wt, minlength=topo.n) / np.maximum(1, np.bincount(f_of, minlength=topo.n))
        n_drop = int(np.round(dropped_work / np.maximum(wmean_node, 1e-3)).sum())
        self.h = h; self.util = util
        self.t += 1
        self.mob.step(); self.attach = self.nearest_ap(self.mob.pos)
        ntask = len(L)
        return dict(
            ntask=ntask, n_drop=n_drop,
            lat_sum=L.sum(), lat_mean=L.mean() if ntask else 0.0,
            viol=(L > cfg.tau_sla).sum() + n_drop,
            E_sum=E_task.sum() + E_mig,
            overload=(h[:topo.K] > topo.C[:topo.K]).astype(float),
            h=h.copy(), h_nomig=h_nomig, rho=h[:topo.K] / topo.C[:topo.K],
            n_mig=len(migrations), mig_times=[tt for _, tt in mig_done],
            Ltx=L_tx.sum(), Lbh=L_bh.sum(), Lcomp=L_comp.sum(), Lmig=L_mig.sum(),
            W_dev=self.Wtot[:, t], ntask_dev=n)

# --------------------------------------------------------------------------
#  Risk statistics
# --------------------------------------------------------------------------

def empirical_cvar(samples, beta):
    """samples: (K, F).  CVaR at upper-tail probability beta = mean of the
    ceil(beta K) largest samples."""
    Kn = samples.shape[0]; k = max(1, int(math.ceil(beta * Kn)))
    return np.sort(samples, axis=0)[-k:].mean(0)

def empirical_var(samples, beta):
    return np.quantile(samples, 1 - beta, axis=0)

def gaussian_cvar(mu, sigma, beta):
    z = _norm.ppf(1 - beta)
    return mu + sigma * _norm.pdf(z) / beta

# --------------------------------------------------------------------------
#  Controller (RAI-Edge + baselines)
# --------------------------------------------------------------------------

@dataclass
class Scheme:
    name: str
    risk: str = "cvar"          # cvar | gauss | mean | robust | var | oracle | measured | ensemble
    rule: str = "framework"     # framework | ctmc | static
    routing: str = "joint"      # joint | minhop
    migrate: bool = True
    beta: float | None = None
    T_lh: int | None = None
    T_bud: int | None = None
    Gamma: float | None = None

class Controller:
    def __init__(self, cfg: Config, world: World, scheme: Scheme, model=None, ensemble=None, seed=0):
        self.cfg, self.w, self.sc = cfg, world, scheme
        self.model, self.ensemble = model, ensemble
        self.rng = np.random.default_rng(seed + 7)
        n = world.topo.n; K = world.topo.K
        self.hist = []                                       # list of node feature arrays
        self.what = np.zeros(world.N); self.lam = np.zeros(world.N)
        self.ctmc_counts = np.ones((K, K)) * 0.1
        self.last_attach = world.attach.copy()
        self.beta = scheme.beta if scheme.beta is not None else cfg.beta
        self.T_lh = scheme.T_lh if scheme.T_lh is not None else cfg.T_lh
        self.T_bud = scheme.T_bud if scheme.T_bud is not None else cfg.T_bud
        self.Gamma = scheme.Gamma if scheme.Gamma is not None else cfg.Gamma_robust
        self.diag = dict(pit=[], mu=[], sig=[], real=[], real_post=[], cvar_e=[], cvar_g=[], safe_flag=[], samples_keep=[])
        self.keep_samples = False
        self.runtime = []; self.rt_parts = []
        self.pending_real = None

    # ---- telemetry update (called after every slot)
    def observe(self, res):
        w, cfg = self.w, self.cfg
        feats = w.node_features()
        self.hist.append(feats)
        if len(self.hist) > cfg.window: self.hist.pop(0)
        self.what = (1 - cfg.ewma) * self.what + cfg.ewma * res["W_dev"] / cfg.delta
        self.lam = (1 - cfg.ewma) * self.lam + cfg.ewma * res["ntask_dev"]
        # CTMC attachment statistics
        a0, a1 = self.last_attach, w.attach
        np.add.at(self.ctmc_counts, (a0, a1), 1.0)
        self.last_attach = a1.copy()
        # calibration diagnostics: compare last forecast with realised demand
        if self.pending_real is not None:
            samples, mu, sig, cv_e, cv_g, safe = self.pending_real
            C = w.topo.C[:w.topo.K]
            real_nomig = res["h_nomig"][:w.topo.K] / C      # same placement the forecast assumed
            real = res["h"][:w.topo.K] / C                  # what actually happened after decisions
            self.diag["pit"].append((samples < real_nomig[None]).mean(0))
            self.diag["mu"].append(mu); self.diag["sig"].append(sig)
            self.diag["real"].append(real_nomig); self.diag["real_post"].append(real)
            self.diag["cvar_e"].append(cv_e); self.diag["cvar_g"].append(cv_g); self.diag["safe_flag"].append(safe)
            if self.keep_samples: self.diag["samples_keep"].append(samples)
            self.pending_real = None

    # ---- forecast
    def forecast(self):
        """Returns (samples (K_mc, K_fog) normalised by C, mu, sigma)."""
        cfg, w = self.cfg, self.w
        H = np.stack(self.hist, 1)                               # (K_fog, window, feat)
        if H.shape[1] < cfg.window:
            H = np.concatenate([np.repeat(H[:, :1], cfg.window - H.shape[1], 1), H], 1)
        if self.sc.risk == "ensemble":
            per = max(2, cfg.K_mc // len(self.ensemble))
            outs = []
            for m in self.ensemble:
                mu = m.predict_point(H)
                s = m.aleatoric_sigma(mu) if m.sigma_a_bins is not None else 0.0
                outs.append(mu[None] + self.rng.standard_normal((per, len(mu))) * s)
            samples = np.concatenate(outs, 0)
        else:
            samples = self.model.mc_samples(H, K=cfg.K_mc, rng=self.rng)
        samples = np.maximum(samples, 0.0)
        mu = self.model.predict_point(H) if self.sc.risk != "ensemble" else samples.mean(0)
        return samples, mu, samples.std(0)

    def risk_statistic(self, samples, mu, sig):
        """Risk-adjusted demand (GHz) per fog node according to the scheme."""
        w = self.w; C = w.topo.C[:w.topo.K]; r = self.sc.risk; b = self.beta
        if r in ("cvar", "ensemble"): v = empirical_cvar(samples, b)
        elif r == "gauss": v = gaussian_cvar(samples.mean(0), samples.std(0), b)
        elif r == "mean": v = mu
        elif r == "robust": v = samples.mean(0) + self.Gamma * samples.std(0)
        elif r == "var": v = empirical_var(samples, b)
        elif r == "oracle": v = w.oracle_next_demand()[:w.topo.K] / C
        elif r == "measured": v = w.h[:w.topo.K] / C
        else: raise ValueError(r)
        return np.maximum(v, 0.0) * C

    # ---- cost model used by the greedy algorithm
    def cost(self, s, hhat, paths, targets):
        """Per-slot expected normalised cost of service s hosted at each node in `targets`."""
        cfg, w = self.cfg, self.w; topo = w.topo
        ai = w.attach[s]
        add = np.where(targets == w.x[s], 0.0, self.what[s])
        rho = np.minimum((hhat[targets] + add) / topo.C[targets], cfg.rho_max)
        Lcomp = w.wbar[s] / (topo.C[targets] * (1 - rho)) + w.backlog[targets] / topo.C[targets]
        hp = paths["hops"][ai, targets]
        bn = np.maximum(paths["bneck"][ai, targets], 1e3)
        Lbh = np.where(hp > 0, w.dbar[s] * 8 / bn + paths["prop"][ai, targets], 0.0)
        Ltx = w.last_Ltx[s]
        L = Ltx + Lbh + Lcomp
        E = cfg.P_tx * Ltx + cfg.eps_bh * 8 * w.dbar[s] * hp + topo.eps[targets] * w.wbar[s]
        return self.lam[s] * (cfg.alpha * E / cfg.E_ref + (1 - cfg.alpha) * L / cfg.tau_sla)

    def mig_cost(self, s, f0, f1, paths):
        cfg, w = self.cfg, self.w
        hp = paths["hops"][f0, f1]; bn = max(paths["bneck"][f0, f1], 1e3)
        Lmig = w.r[s] * 8 / bn + paths["prop"][f0, f1]
        return ((1 - cfg.alpha) * self.lam[s] * Lmig / cfg.tau_sla
                + cfg.alpha * cfg.eps_bh * 8 * w.r[s] * hp / cfg.E_ref
                + cfg.gamma * w.r[s] / cfg.r_ref)

    # ---- main decision step
    def decide(self):
        t0 = time.perf_counter()
        cfg, w, sc = self.cfg, self.w, self.sc; topo = w.topo; K = topo.K; n = topo.n
        residual = topo.B * np.maximum(0.0, 1.0 - w.util)
        paths = topo.all_pairs_paths(residual) if sc.routing == "joint" else topo.minhop
        migrations, mig_paths = [], []
        hhat = np.zeros(n)
        if sc.rule == "static" or not sc.migrate:
            self.runtime.append(time.perf_counter() - t0)
            return migrations, paths, mig_paths
        if sc.rule == "ctmc":
            # Mobility-driven baseline [wang2019mobility]: the candidate target is the
            # most likely attachment node T_lh slots ahead under an estimated Markov
            # mobility chain; the decision uses the SAME cost/gain test as RAI-Edge but
            # with the last *measured* demand (no forecast, no uncertainty).
            hhat[:K] = w.h[:K]
            cap = (1 - cfg.delta_margin) * topo.C
            P = self.ctmc_counts / self.ctmc_counts.sum(1, keepdims=True)
            Pn = np.linalg.matrix_power(P, self.T_lh)
            pi = Pn[w.attach]                                   # (N, K)
            target = pi.argmax(1); conf = pi.max(1)
            mem_used = w.mem_used.copy()
            order = np.argsort(-conf)
            for s in order:
                if len(migrations) >= self.T_bud: break
                f0, f1 = int(w.x[s]), int(target[s])
                if f1 == f0 or w.mig_bytes_left[s] > 0 or self.lam[s] <= 1e-6: continue
                if hhat[f1] + self.what[s] > cap[f1] or mem_used[f1] + w.m[s] > topo.mem[f1]: continue
                J = self.cost(s, hhat, paths, np.array([f0, f1]))
                if self.T_lh * (J[0] - J[1]) - self.mig_cost(s, f0, f1, paths) <= 0: continue
                migrations.append((s, f0, f1)); mig_paths.append(paths["links"][f0][f1])
                hhat[f0] -= self.what[s]; hhat[f1] += self.what[s]
                mem_used[f0] -= w.m[s]; mem_used[f1] += w.m[s]
            self.runtime.append(time.perf_counter() - t0)
            return migrations, paths, mig_paths
        # ----- framework (RAI-Edge and its ablations) -- Algorithm 1 of the paper
        t_route = time.perf_counter() - t0                                   # initial all-pairs routing
        t1 = time.perf_counter()
        samples, mu, sig = self.forecast()                                   # lines 1-3
        hhat[:K] = self.risk_statistic(samples, mu, sig)                    # line 4
        t_pred = time.perf_counter() - t1
        cap = (1 - cfg.delta_margin) * topo.C
        cv_e = empirical_cvar(samples, self.beta)
        cv_g = gaussian_cvar(samples.mean(0), samples.std(0), self.beta)
        mem_used = w.mem_used.copy()
        excess = (hhat[w.x] - cap[w.x]) / topo.C[w.x]                       # line 6: ordering key
        order = np.lexsort((-self.lam, -excess))
        all_nodes = np.arange(n)
        for s in order:                                                      # line 7
            if len(migrations) >= self.T_bud: break                          # budget
            if w.mig_bytes_left[s] > 0 or self.lam[s] <= 1e-6: continue      # in-flight / idle
            f0 = w.x[s]
            feas = (all_nodes != f0) & (hhat + self.what[s] <= cap) & (mem_used + w.m[s] <= topo.mem)
            feas &= paths["bneck"][f0] > 0                                   # line 9: F_safe
            cand = all_nodes[feas]
            if len(cand) == 0: continue
            tg = np.concatenate([[f0], cand])
            J = self.cost(s, hhat, paths, tg)                                # line 10
            j = int(J[1:].argmin()); f1 = int(cand[j])
            gain = self.T_lh * (J[0] - J[1 + j]) - self.mig_cost(s, f0, f1, paths)   # line 11
            if gain > 0:                                                     # line 12
                migrations.append((s, f0, f1)); lk = paths["links"][f0][f1]; mig_paths.append(lk)
                hhat[f0] -= self.what[s]; hhat[f1] += self.what[s]           # line 13: state update
                mem_used[f0] -= w.m[s]; mem_used[f1] += w.m[s]
                if sc.routing == "joint" and len(lk):                        # line 14: reserve path bw
                    residual[lk] = np.maximum(0.0, residual[lk] - w.r[s] * 8 / cfg.delta)
                    tr = time.perf_counter(); paths = topo.all_pairs_paths(residual); t_route += time.perf_counter() - tr
        safe = (hhat[:K] <= cap[:K]).astype(float)
        self.pending_real = (samples, mu, sig, cv_e, cv_g, safe)
        tot = time.perf_counter() - t0
        self.runtime.append(tot); self.rt_parts.append((t_pred, tot - t_pred - t_route, t_route))
        return migrations, paths, mig_paths

    # ---- one-slot (linearised) instance for the exact-MIP comparison
    def oneslot_instance(self):
        """Returns the data of the per-slot problem P_t evaluated at the current
        risk-adjusted load (the same quantities Algorithm 1 uses)."""
        cfg, w, sc = self.cfg, self.w, self.sc; topo = w.topo; K = topo.K; n = topo.n
        residual = topo.B * np.maximum(0.0, 1.0 - w.util)
        paths = topo.all_pairs_paths(residual) if sc.routing == "joint" else topo.minhop
        samples, mu, sig = self.forecast()
        hhat = np.zeros(n); hhat[:K] = self.risk_statistic(samples, mu, sig)
        cap = (1 - cfg.delta_margin) * topo.C
        N = w.N; all_nodes = np.arange(n)
        J = np.zeros((N, n)); Cm = np.zeros((N, n)); feas = np.zeros((N, n), bool)
        for s in range(N):
            J[s] = self.cost(s, hhat, paths, all_nodes)
            for f in range(n):
                Cm[s, f] = 0.0 if f == w.x[s] else self.mig_cost(s, int(w.x[s]), f, paths)
            feas[s] = (paths["bneck"][w.x[s]] > 0) | (all_nodes == w.x[s])
            if w.mig_bytes_left[s] > 0 or self.lam[s] <= 1e-6: feas[s] = all_nodes == w.x[s]
        return dict(J=J, Cm=Cm, feas=feas, what=self.what.copy(), m=w.m.copy(), hhat=hhat, cap=cap,
                    mem=topo.mem.copy(), mem_used=w.mem_used.copy(), x=w.x.copy(), T_lh=self.T_lh,
                    T_bud=self.T_bud, lam=self.lam.copy(), order=np.lexsort((-self.lam, -((hhat[w.x] - cap[w.x]) / topo.C[w.x]))))

# --------------------------------------------------------------------------
#  Episode runner
# --------------------------------------------------------------------------

def run_episode(cfg: Config, N: int, seed: int, scheme: Scheme, model=None, ensemble=None,
                trace=None, collect_series=False):
    w = World(cfg, N, seed, trace)
    ctrl = Controller(cfg, w, scheme, model, ensemble, seed)
    ctrl.keep_samples = collect_series
    T = cfg.T_warm + cfg.T_eval
    acc = dict(ntask=0, n_drop=0, lat_sum=0.0, viol=0, E_sum=0.0, overload=0.0, n_mig=0, mig_times=[],
               Ltx=0.0, Lbh=0.0, Lcomp=0.0, Lmig=0.0, node_slots=0)
    series = dict(n_mig=[], viol_rate=[], lat=[]) if collect_series else None
    # routing for the warm-up: min-hop
    routing = w.topo.minhop
    for t in range(T):
        if t >= cfg.T_warm:
            migrations, routing, mig_paths = ctrl.decide()
        else:
            migrations, mig_paths = [], []
        res = w.step(migrations, routing, mig_paths)
        ctrl.observe(res)
        if t >= cfg.T_warm:
            for k in ("ntask", "n_drop", "lat_sum", "viol", "E_sum", "n_mig", "Ltx", "Lbh", "Lcomp", "Lmig"):
                acc[k] += res[k]
            acc["overload"] += res["overload"].sum(); acc["node_slots"] += w.topo.K
            acc["mig_times"] += res["mig_times"]
            if collect_series:
                series["n_mig"].append(res["n_mig"]); series["lat"].append(res["lat_mean"])
                series["viol_rate"].append(res["viol"] / max(1, res["ntask"]))
    nt = max(1, acc["ntask"])
    out = dict(
        scheme=scheme.name, N=N, seed=seed,
        latency_ms=1000 * acc["lat_sum"] / nt,
        energy_J=acc["E_sum"] / nt,
        sla_viol=acc["viol"] / (nt + acc["n_drop"]),
        drop_rate=acc["n_drop"] / (nt + acc["n_drop"]),
        mig_per_slot=acc["n_mig"] / cfg.T_eval,
        mig_time_ms=1000 * float(np.mean(acc["mig_times"])) if acc["mig_times"] else float("nan"),
        overload_freq=acc["overload"] / acc["node_slots"],
        Ltx_ms=1000 * acc["Ltx"] / nt, Lbh_ms=1000 * acc["Lbh"] / nt,
        Lcomp_ms=1000 * acc["Lcomp"] / nt, Lmig_ms=1000 * acc["Lmig"] / nt,
        runtime_ms=1000 * float(np.mean(ctrl.runtime)) if ctrl.runtime else 0.0,
        runtime_p95_ms=1000 * float(np.percentile(ctrl.runtime, 95)) if ctrl.runtime else 0.0,
    )
    # calibration diagnostics (framework schemes only)
    if ctrl.diag["pit"]:
        pit = np.concatenate(ctrl.diag["pit"]); mu = np.concatenate(ctrl.diag["mu"]); sig = np.concatenate(ctrl.diag["sig"])
        real = np.concatenate(ctrl.diag["real"]); cve = np.concatenate(ctrl.diag["cvar_e"]); cvg = np.concatenate(ctrl.diag["cvar_g"])
        safe = np.concatenate(ctrl.diag["safe_flag"]); real_post = np.concatenate(ctrl.diag["real_post"])
        out["rmse"] = float(np.sqrt(np.mean((mu - real) ** 2))); out["mae"] = float(np.mean(np.abs(mu - real)))
        levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        cov = [float(np.mean((pit >= (1 - l) / 2) & (pit <= 1 - (1 - l) / 2))) for l in levels]
        out["coverage"] = cov
        out["cvar_rel_diff"] = float(np.mean(np.abs(cvg - cve) / np.maximum(cve, 1e-3)))
        out["cvar_e_mean"] = float(cve.mean()); out["cvar_g_mean"] = float(cvg.mean())
        out["p_exceed_cvar"] = float(np.mean(real > cve))            # nominal: <= beta
        out["p_exceed_var"] = float(np.mean(pit > 1 - ctrl.beta))    # nominal: beta
        # realised overload among nodes declared safe after this slot's decisions
        sel = safe > 0.5
        out["p_overload_given_safe"] = float(np.mean(real_post[sel] > 1.0)) if sel.any() else float("nan")
        out["p_overload_given_unsafe"] = float(np.mean(real_post[~sel] > 1.0)) if (~sel).any() else float("nan")
        out["frac_declared_safe"] = float(sel.mean())
        if collect_series:
            out["diag_mu"] = mu.tolist(); out["diag_sig"] = sig.tolist(); out["diag_real"] = real.tolist()
            out["diag_cve"] = cve.tolist(); out["diag_cvg"] = cvg.tolist(); out["diag_pit"] = pit.tolist()
            out["diag_skew"] = [float(np.mean(((s_ - s_.mean()) / (s_.std() + 1e-9)) ** 3)) for s_ in np.concatenate(ctrl.diag["samples_keep"], 1).T] if ctrl.diag.get("samples_keep") else None
    if collect_series:
        out["series"] = series
    return out


# --------------------------------------------------------------------------
#  Telemetry collection for predictor training
# --------------------------------------------------------------------------

def collect_training_data(cfg: Config, N: int, seed: int, T: int = 200, p_random_mig=0.02):
    """Runs the world with a random-migration policy and returns (X, y) windows."""
    cfg2 = Config(**{**asdict(cfg), "T_warm": 0, "T_eval": T})
    w = World(cfg2, N, seed)
    rng = np.random.default_rng(seed + 99)
    feats = []
    routing = w.topo.minhop
    for t in range(T):
        migrations, mig_paths = [], []
        if rng.random() < 0.5:
            for s in rng.choice(N, max(1, int(p_random_mig * N)), replace=False):
                f0 = w.x[s]; f1 = int(rng.integers(0, w.topo.K))
                if f1 != f0 and w.mem_used[f1] + w.m[s] <= w.topo.mem[f1] and w.mig_bytes_left[s] == 0:
                    migrations.append((s, f0, f1)); mig_paths.append(routing["links"][f0][f1])
        w.step(migrations, routing, mig_paths)
        feats.append(w.node_features())
    F = np.stack(feats, 1)                                   # (K, T, feat)
    X, y = [], []
    for f in range(F.shape[0]):
        for t in range(cfg.window, T - 1):
            X.append(F[f, t - cfg.window:t]); y.append(F[f, t, 0])
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------
#  External mobility trace loader (hook for real-world traces)
# --------------------------------------------------------------------------

def load_trace_csv(path, N, T, area, delta=1.0):
    """Reads a CSV with columns  device_id,time_s,x_m,y_m  and returns a
    (N, T, 2) position array sampled every `delta` seconds, rescaled so that
    positions fall inside the [0, area]^2 simulation square.  Devices are the
    first N distinct ids; a device's position is held constant outside the
    time span of its records.  Suitable for e.g. SUMO/LuST FCD exports
    (convert with sumolib) or taxi GPS traces."""
    import csv
    recs = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            try: d, t, x, y = row[0], float(row[1]), float(row[2]), float(row[3])
            except (ValueError, IndexError): continue
            recs.setdefault(d, []).append((t, x, y))
    ids = sorted(recs)[:N]
    if len(ids) < N:
        raise ValueError(f"trace has only {len(ids)} devices, {N} requested")
    allxy = np.array([p[1:] for d in ids for p in recs[d]])
    lo, hi = allxy.min(0), allxy.max(0); scale = area / max(hi - lo)
    out = np.zeros((N, T, 2))
    for k, d in enumerate(ids):
        r = np.array(sorted(recs[d]))
        ts = np.arange(T) * delta + r[0, 0]
        out[k, :, 0] = np.interp(ts, r[:, 0], r[:, 1]); out[k, :, 1] = np.interp(ts, r[:, 0], r[:, 2])
    return np.clip((out - lo) * scale, 0, area)
