"""Empirical optimality gap of the greedy orchestration (Algorithm 1)
against the exact optimum of the per-slot linearised problem P_t, solved
with the HiGHS branch-and-bound solver (scipy.optimize.milp).

P_t (data taken from the controller at a given slot):
  min  sum_{s,f} y_{s,f} [ T_lh * J(s,f) + C^mig(s,f) ]
  s.t. sum_f y_{s,f} = 1                                            (single placement)
       sum_{s not on f} y_{s,f} w_s - sum_{s on f} (1-y_{s,f}) w_s <= max(0, cap_f - hhat_f)   (risk capacity)
       sum_{s not on f} y_{s,f} m_s - sum_{s on f} (1-y_{s,f}) m_s <= mem_f - memused_f         (memory)
       sum_s (1 - y_{s,x_s}) <= T_bud                               (migration budget)
       y binary, y_{s,f}=0 where f is unreachable / s in flight.
Greedy objective is evaluated on exactly the same linearised cost."""
import json, time, os, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from sim import World, Controller, Scheme
from lstm import MCDropoutLSTM
from common import benchmark_config, MODEL_DIR, RESULT_DIR

def greedy_on_instance(inst):
    """Re-implements the acceptance loop of Algorithm 1 on the frozen instance,
    with the sequential state updates (hhat, memory) but WITHOUT recomputing the
    cost after each acceptance (that is the linearisation shared with the MIP)."""
    J, Cm, feas = inst["J"], inst["Cm"], inst["feas"]
    what, m, cap, mem = inst["what"], inst["m"], inst["cap"], inst["mem"]
    hhat = inst["hhat"].copy(); mem_used = inst["mem_used"].copy(); x = inst["x"].copy()
    n_mig = 0; obj = 0.0
    y = x.copy()
    for s in inst["order"]:
        f0 = x[s]
        if n_mig >= inst["T_bud"]:
            break
        ok = feas[s].copy(); ok[f0] = False
        ok &= (hhat + what[s] <= cap) & (mem_used + m[s] <= mem)
        if not ok.any(): continue
        tot = inst["T_lh"] * J[s] + Cm[s]
        cand = np.nonzero(ok)[0]; f1 = cand[tot[cand].argmin()]
        if tot[f1] < tot[f0]:
            y[s] = f1; n_mig += 1
            hhat[f0] -= what[s]; hhat[f1] += what[s]; mem_used[f0] -= m[s]; mem_used[f1] += m[s]
    obj = sum(inst["T_lh"] * J[s, y[s]] + Cm[s, y[s]] for s in range(len(y)))
    return obj, y, n_mig

def solve_mip(inst, time_limit=60.0):
    J, Cm, feas = inst["J"], inst["Cm"], inst["feas"]
    N, n = J.shape
    c = (inst["T_lh"] * J + Cm).ravel()
    A = lil_matrix((N + 2 * n + 1, N * n)); lb = []; ub = []
    r = 0
    for s in range(N):                      # single placement
        A[r, s * n:(s + 1) * n] = 1.0; lb.append(1.0); ub.append(1.0); r += 1
    # sum_{s not on f} y w - sum_{s on f}(1-y) w <= R_f  <=>  sum_all y_{s,f} w_s <= R_f + sum_{s on f} w_s
    for f in range(n):                      # risk capacity
        for s in range(N): A[r, s * n + f] = inst["what"][s]
        onf = inst["x"] == f
        lb.append(-np.inf); ub.append(max(0.0, inst["cap"][f] - inst["hhat"][f]) + inst["what"][onf].sum()); r += 1
    for f in range(n):                      # memory
        for s in range(N): A[r, s * n + f] = inst["m"][s]
        onf = inst["x"] == f
        lb.append(-np.inf); ub.append(inst["mem"][f] - inst["mem_used"][f] + inst["m"][onf].sum()); r += 1
    for s in range(N):                      # budget: sum_s (1 - y_{s,x_s}) <= T_bud
        A[r, s * n + inst["x"][s]] = -1.0
    lb.append(-np.inf); ub.append(inst["T_bud"] - N); r += 1
    lo = np.zeros(N * n); hi = feas.astype(float).ravel()
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.ones(N * n),
               bounds=Bounds(lo, hi), options={"time_limit": time_limit})
    dt = time.time() - t0
    if res.x is None:
        return None, dt, res.status
    y = res.x.reshape(N, n).argmax(1)
    return float(res.fun), dt, res.status, y

def randomize_placement(w, rng):
    """Random feasible (memory-respecting) placement: generates hard, independent instances."""
    w.mem_used[:] = 0.0
    for s_ in rng.permutation(w.N):
        for f in rng.permutation(w.topo.K):
            if w.mem_used[f] + w.m[s_] <= w.topo.mem[f]:
                w.x[s_] = f; w.mem_used[f] += w.m[s_]; break
        else:
            w.x[s_] = w.topo.cloud; w.mem_used[w.topo.cloud] += w.m[s_]
    w.nserv = np.bincount(w.x, minlength=w.topo.n)

if __name__ == "__main__":
    model = MCDropoutLSTM.load(f"{MODEL_DIR}/lstm_mc.npz")
    rows = []
    rng = np.random.default_rng(0)
    # small, TIGHT instances: low CPU capacity and memory so that C2/C8 bind, random placement so that many
    # services want to move, and small migration budgets so that C7 binds.
    for (K, n_macro, Cmi, Cma, N) in [(4, 1, 6.0, 16.0, 12), (4, 1, 6.0, 16.0, 24), (6, 2, 6.0, 16.0, 40), (8, 2, 6.0, 16.0, 60)]:
        for seed in range(2000, 2006):
            cfg = benchmark_config(K=K, n_macro=n_macro, C_micro=Cmi, C_macro=Cma, init_placement="random",
                                   mem_micro=640.0, mem_macro=1536.0, T_warm=10, T_eval=10)
            w = World(cfg, N, seed); ctrl = Controller(cfg, w, Scheme("RAI-Edge", risk="cvar"), model, None, seed)
            routing = w.topo.minhop
            for t in range(cfg.T_warm + 8):
                if t >= cfg.T_warm and (t - cfg.T_warm) % 2 == 0:
                    randomize_placement(w, rng)
                    for T_bud in (2, 5, 10):
                        ctrl.T_bud = T_bud
                        inst = ctrl.oneslot_instance()
                        g_obj, g_y, g_n = greedy_on_instance(inst)
                        out = solve_mip(inst)
                        if out[0] is None:
                            rows.append(dict(K=K, N=N, seed=seed, t=t, T_bud=T_bud, status=int(out[2]), solved=False)); continue
                        o_obj, dt, st, o_y = out
                        base = sum(inst["T_lh"] * inst["J"][s, inst["x"][s]] for s in range(N))   # no-migration cost
                        rows.append(dict(K=K, N=N, seed=seed, t=t, T_bud=T_bud, greedy=float(g_obj), opt=float(o_obj), nomig=float(base),
                                         gap=float((g_obj - o_obj) / max(abs(o_obj), 1e-9)), mip_time=dt, status=int(st),
                                         n_mig_greedy=int(g_n), n_mig_opt=int((o_y != inst["x"]).sum()), solved=True,
                                         improv_greedy=float((base - g_obj) / max(abs(base), 1e-9)), improv_opt=float((base - o_obj) / max(abs(base), 1e-9)),
                                         capacity_binding=bool((inst["hhat"][:K] > inst["cap"][:K]).any())))
                    ctrl.T_bud = cfg.T_bud
                migrations, routing, mig_paths = ctrl.decide() if t >= cfg.T_warm else ([], routing, [])
                res = w.step(migrations, routing, mig_paths); ctrl.observe(res)
    os.makedirs(RESULT_DIR, exist_ok=True)
    json.dump(rows, open(f"{RESULT_DIR}/mip_gap.json", "w"), indent=1)
    sol = [r for r in rows if r["solved"]]
    print("instances", len(sol), "mean gap %.4f  max gap %.4f  frac zero %.3f  mean mip time %.3fs  mean n_mig_opt %.2f" % (
        np.mean([r["gap"] for r in sol]), np.max([r["gap"] for r in sol]), np.mean([r["gap"] < 1e-9 for r in sol]),
        np.mean([r["mip_time"] for r in sol]), np.mean([r["n_mig_opt"] for r in sol])))
