"""Measured per-slot controller runtime of RAI-Edge vs network size."""
import json, time, os, platform, numpy as np
from sim import World, Controller, Scheme
from lstm import MCDropoutLSTM
from common import benchmark_config, MODEL_DIR, RESULT_DIR

model = MCDropoutLSTM.load(f"{MODEL_DIR}/lstm_mc.npz")
rows = []
for K, N in [(10, 100), (10, 300), (20, 300), (20, 600), (40, 600), (40, 1000), (60, 1000)]:
    cfg = benchmark_config(K=K, n_macro=max(1, K * 4 // 10), area=1000.0 * np.sqrt(K / 10))
    for seed in (3000, 3001):
        w = World(cfg, N, seed); ctrl = Controller(cfg, w, Scheme("RAI-Edge", risk="cvar"), model, None, seed)
        routing = w.topo.minhop
        for t in range(cfg.T_warm + 30):
            mig, routing, mp = ctrl.decide() if t >= cfg.T_warm else ([], routing, [])
            ctrl.observe(w.step(mig, routing, mp))
        rt = np.array(ctrl.runtime) * 1000; parts = np.array(ctrl.rt_parts) * 1000
        rows.append(dict(K=K, N=N, E=w.topo.E, seed=seed, mean_ms=float(rt.mean()), p95_ms=float(np.percentile(rt, 95)),
                         max_ms=float(rt.max()), pred_ms=float(parts[:, 0].mean()), greedy_ms=float(parts[:, 1].mean()),
                         route_ms=float(parts[:, 2].mean()), mig_per_slot=float(np.mean([len(m) for m in []] or [0]))))
        print(rows[-1], flush=True)
cpu = platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
json.dump(dict(rows=rows, cpu=cpu, python=platform.python_version(), numpy=np.__version__), open(f"{RESULT_DIR}/runtime.json", "w"), indent=1)
