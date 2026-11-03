"""Offline training of the MC-dropout LSTM demand predictor and of the
deep-ensemble baseline.  Telemetry is collected from the simulator with a
random-migration exploration policy (training seeds 1..12, evaluation uses
seeds >= 1000, so there is no overlap)."""
import os, json, time, numpy as np
from sim import collect_training_data
from lstm import MCDropoutLSTM
from common import benchmark_config, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)
cfg = benchmark_config()
t0 = time.time()
Xs, ys, Xv, yv = [], [], [], []
for k, N in enumerate([50, 100, 150, 200, 250, 300]):
    for j in range(2):
        X, y = collect_training_data(cfg, N, seed=1 + 2 * k + j, T=200)
        Xs.append(X); ys.append(y)
    X, y = collect_training_data(cfg, N, seed=100 + k, T=120)          # validation episodes
    Xv.append(X); yv.append(y)
X = np.concatenate(Xs); y = np.concatenate(ys); Xv = np.concatenate(Xv); yv = np.concatenate(yv)
print(f"train windows {X.shape}, val windows {Xv.shape}, collect {time.time()-t0:.1f}s")
print("target (h/C): mean %.3f std %.3f p95 %.3f frac>1 %.3f" % (y.mean(), y.std(), np.quantile(y, .95), (y > 1).mean()))

log = {}
m = MCDropoutLSTM(d_in=X.shape[2], d_h=64, p_drop=0.2, seed=0)
t0 = time.time()
hist = m.fit(X, y, Xv, yv, epochs=25, batch=128, lr=2e-3)
m.fit_aleatoric(Xv, yv)                     # residual pools from held-out data
yh = m.predict_point(Xv)
naive = Xv[:, -1, 0]                        # persistence baseline
log["main"] = dict(train_time_s=time.time() - t0, val_rmse=float(np.sqrt(np.mean((yh - yv) ** 2))),
                   persistence_rmse=float(np.sqrt(np.mean((naive - yv) ** 2))),
                   val_mae=float(np.mean(np.abs(yh - yv))), hist=hist,
                   sigma_bins=[list(map(float, m.sigma_a_bins[0])), list(map(float, m.sigma_a_bins[1]))])
m.save(f"{MODEL_DIR}/lstm_mc.npz")
print("main model: val RMSE %.4f (persistence %.4f)" % (log["main"]["val_rmse"], log["main"]["persistence_rmse"]))

# deep ensemble (5 members, deterministic at inference)
ens_rmse = []
for k in range(5):
    e = MCDropoutLSTM(d_in=X.shape[2], d_h=64, p_drop=0.2, seed=10 + k)
    e.fit(X, y, epochs=25, batch=128, lr=2e-3, verbose=False)
    e.fit_aleatoric(Xv, yv)
    e.save(f"{MODEL_DIR}/ens_{k}.npz")
    ens_rmse.append(float(np.sqrt(np.mean((e.predict_point(Xv) - yv) ** 2))))
    print("ensemble member", k, "val RMSE %.4f" % ens_rmse[-1])
log["ensemble_val_rmse"] = ens_rmse
log["data"] = dict(n_train=int(X.shape[0]), n_val=int(Xv.shape[0]), y_mean=float(y.mean()), y_std=float(y.std()))
json.dump(log, open(f"{MODEL_DIR}/train_log.json", "w"), indent=1)
