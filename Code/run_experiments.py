"""Experiment driver.  Usage:  python run_experiments.py <experiment> [--workers 2]
Results are appended as JSON lines to ../results/<experiment>.jsonl.
Every run is identified by (scheme, N, seed, extra); paired comparisons across
schemes use common random numbers (same seed -> same topology, mobility, arrivals)."""
import os, sys, json, time, argparse, itertools
import numpy as np
from multiprocessing import Pool
from dataclasses import replace, asdict
from sim import Scheme, run_episode
from lstm import MCDropoutLSTM
from common import benchmark_config, SCHEMES, MAIN_SCHEMES, EVAL_SEED0, MODEL_DIR, RESULT_DIR

_MODEL = None; _ENS = None
def _load():
    global _MODEL, _ENS
    if _MODEL is None:
        _MODEL = MCDropoutLSTM.load(f"{MODEL_DIR}/lstm_mc.npz")
        _ENS = [MCDropoutLSTM.load(f"{MODEL_DIR}/ens_{k}.npz") for k in range(5)]
    return _MODEL, _ENS

def one(job):
    cfg_over, scheme_over, N, seed, extra, series = job
    m, ens = _load()
    cfg = benchmark_config(**cfg_over)
    sc = SCHEMES[scheme_over["name"]]
    sc = replace(sc, **{k: v for k, v in scheme_over.items() if k != "name" and k != "label"})
    t0 = time.time()
    r = run_episode(cfg, N, seed, sc, model=m, ensemble=ens, collect_series=series)
    r["label"] = scheme_over.get("label", sc.name); r["extra"] = extra; r["wall_s"] = time.time() - t0
    r["cfg_over"] = cfg_over
    return r

def jobs_for(exp):
    J = []
    S = list(range(EVAL_SEED0, EVAL_SEED0 + 20))
    if exp == "main":
        for N in [50, 100, 150, 200, 250, 300]:
            for name in MAIN_SCHEMES:
                for s in S: J.append(({}, {"name": name}, N, s, {}, False))
    elif exp == "ablation":
        N = 250
        variants = [
            {"name": "RAI-Edge", "label": "RAI-Edge"},
            {"name": "Oracle", "label": "Oracle-point"},
            {"name": "RAI-Edge", "label": "no-joint-routing", "routing": "minhop"},
            {"name": "Static", "label": "routing-only", "routing": "joint"},
            {"name": "LSTM-Det", "label": "no-uncertainty"},
            {"name": "RAI-G", "label": "Gaussian-CVaR"},
            {"name": "RAI-Edge", "label": "T_lh=1", "T_lh": 1},
            {"name": "RAI-Edge", "label": "T_bud=3", "T_bud": 3},
            {"name": "RAI-Edge", "label": "T_bud=inf", "T_bud": 10 ** 6},
        ]
        for v in variants:
            for s in S: J.append(({}, v, N, s, {}, False))
    elif exp == "beta":
        for b in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            for s in S:
                J.append(({}, {"name": "RAI-Edge", "label": f"beta={b}", "beta": b}, 250, s, {"beta": b}, False))
    elif exp == "horizon":
        for tl in [1, 2, 5, 10, 20]:
            for s in S[:10]:
                J.append(({}, {"name": "RAI-Edge", "label": f"T_lh={tl}", "T_lh": tl}, 250, s, {"T_lh": tl}, False))
        for tb in [1, 3, 5, 10, 20, 10 ** 6]:
            for s in S[:10]:
                J.append(({}, {"name": "RAI-Edge", "label": f"T_bud={tb}", "T_bud": tb}, 250, s, {"T_bud": tb}, False))
    elif exp == "alpha":
        for a in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for name in ["RAI-Edge", "LSTM-Det"]:
                for s in S[:10]:
                    J.append(({"alpha": a}, {"name": name}, 250, s, {"alpha": a}, False))
    elif exp == "gm":
        for N in [100, 200, 300]:
            for name in MAIN_SCHEMES:
                for s in S[:10]: J.append(({"mobility": "gm"}, {"name": name}, N, s, {"mobility": "gm"}, False))
    elif exp == "series":
        for name in ["CTMC", "LSTM-Det", "Robust", "VaR", "RAI-Edge"]:
            for s in S[:5]: J.append(({}, {"name": name}, 200, s, {}, True))
    elif exp == "calib":
        for N in [100, 200, 300]:
            for s in S[:4]: J.append(({}, {"name": "RAI-Edge"}, N, s, {}, True))
    elif exp == "burst":
        for sw in [0.5, 1.0, 2.0, 4.0]:
            for name in ["Static", "LSTM-Det", "RAI-Edge"]:
                for s in S[:10]:
                    J.append(({"p_on_off": 0.05 * sw, "p_off_on": 0.03 * sw}, {"name": name}, 250, s, {"switch": sw}, False))
    elif exp == "capacity":
        for sc_ in [0.75, 1.0, 1.5]:
            for name in ["Static", "LSTM-Det", "VaR", "Robust", "RAI-Edge"]:
                for s in S[:10]:
                    J.append(({"C_micro": 16.0 * sc_, "C_macro": 48.0 * sc_}, {"name": name}, 250, s, {"cap_scale": sc_}, False))
    elif exp == "spread":
        for sp in [0.2, 0.4, 0.8]:
            for name in ["LSTM-Det", "VaR", "Robust", "RAI-Edge"]:
                for s in S[:10]:
                    J.append(({"w_spread": sp}, {"name": name}, 250, s, {"w_spread": sp}, False))
    elif exp == "burst2":
        for sw in [0.5, 1.0, 2.0, 4.0]:
            for name in ["VaR", "Robust"]:
                for s in S[:10]:
                    J.append(({"p_on_off": 0.05 * sw, "p_off_on": 0.03 * sw}, {"name": name}, 250, s, {"switch": sw}, False))
    elif exp == "alpha2":
        for a in [0.0, 0.3, 0.5, 0.7]:
            for name in ["VaR", "Robust"]:
                for s in S[:10]:
                    J.append(({"alpha": a}, {"name": name}, 250, s, {"alpha": a}, False))
    elif exp == "backhaul":
        for bw in [20e6, 50e6, 100e6]:
            for v in [{"name": "RAI-Edge", "label": "RAI-Edge"}, {"name": "RAI-Edge", "label": "no-joint-routing", "routing": "minhop"}]:
                for s in S[:10]:
                    J.append(({"link_bw": bw}, v, 250, s, {"link_bw": bw}, False))
    else:
        raise SystemExit("unknown experiment " + exp)
    return J

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)
    J = jobs_for(a.exp)
    out = open(f"{RESULT_DIR}/{a.exp}.jsonl", "w")
    t0 = time.time(); n = 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, J, chunksize=1):
            out.write(json.dumps(r) + "\n"); out.flush(); n += 1
            if n % 20 == 0 or n == len(J):
                print(f"[{a.exp}] {n}/{len(J)} done, {time.time()-t0:.0f}s elapsed", flush=True)
    out.close()
