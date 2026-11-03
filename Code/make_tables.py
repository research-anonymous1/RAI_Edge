"""Emits LaTeX table bodies from ../results/numbers.json into ../tables/."""
import json, os, math
import numpy as np
R = "../results"; T = "../tables"; os.makedirs(T, exist_ok=True)
n = json.load(open(f"{R}/numbers.json"))

def pm(v, scale=1.0, d=1):
    m, e = v[0] * scale, v[1] * scale
    if m != m: return "--"
    return f"{m:.{d}f}$\\pm${e:.{d}f}"

ORDER = ["Static", "CTMC", "LSTM-Det", "Robust", "VaR", "DE-CVaR", "RAI-G", "RAI-Edge"]
LAB = {"RAI-G": "RAI-G (Gaussian CVaR)", "DE-CVaR": "DE-CVaR (ensemble)", "Robust": "Robust ($\\mu{+}2\\sigma$)", "VaR": "VaR (quantile)"}

# ---- main summary at N=300 (and 250)
with open(f"{T}/main_summary.tex", "w") as f:
    for N in ("300",):
        f.write("\\midrule\n")
        for s in ORDER:
            d = n[f"N{N}"][s]
            f.write(f"{LAB.get(s, s)} & {pm(d['latency_ms'])} & {pm(d['sla_viol'], 100, 1)} & {pm(d['energy_J'], 1, 3)} & {pm(d['mig_per_slot'], 1, 2)} & {pm(d['mig_time_ms'], 1, 0)} \\\\\n")

# ---- ablation
if "ablation" in n:
    rows = [("RAI-Edge", "\\RAI{} (full)"), ("no-uncertainty", "$-$ uncertainty (point forecast)"),
            ("Gaussian-CVaR", "$-$ empirical CVaR (Gaussian)"), ("no-joint-routing", "$-$ rerouting (min-hop paths)"),
            ("routing-only", "$-$ migration (joint routing only)"), ("Oracle-point", "Oracle point forecast, no margin"),
            ("T_lh=1", "$T_{\\mathrm{lh}}=1$ (myopic gain)"), ("T_bud=3", "$T_{\\mathrm{bud}}=3$"), ("T_bud=inf", "$T_{\\mathrm{bud}}=\\infty$")]
    with open(f"{T}/ablation.tex", "w") as f:
        for key, lab in rows:
            if key not in n["ablation"]: continue
            d = n["ablation"][key]
            f.write(f"{lab} & {pm(d['latency_ms'])} & {pm(d['sla_viol'], 100, 1)} & {pm(d['energy_J'], 1, 3)} & {pm(d['mig_per_slot'], 1, 2)} \\\\\n")

# ---- Gauss-Markov mobility
if "gm" in n:
    with open(f"{T}/gm.tex", "w") as f:
        for N in sorted(n["gm"], key=int):
            f.write(f"\\midrule\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{Gauss--Markov, $N={N}$}}}}\\\\\n")
            for s in ORDER:
                if s not in n["gm"][N]: continue
                d = n["gm"][N][s]
                f.write(f"{LAB.get(s, s)} & {pm(d['latency_ms'])} & {pm(d['sla_viol'], 100, 1)} & {pm(d['energy_J'], 1, 3)} & {pm(d['mig_per_slot'], 1, 2)} \\\\\n")

# ---- horizon / budget
if "horizon" in n:
    with open(f"{T}/horizon.tex", "w") as f:
        for key in ["T_lh=1", "T_lh=2", "T_lh=5", "T_lh=10", "T_lh=20", "T_bud=1", "T_bud=3", "T_bud=5", "T_bud=10", "T_bud=20", "T_bud=1000000"]:
            if key not in n["horizon"]: continue
            d = n["horizon"][key]; lab = key.replace("T_lh=", "$T_{\\mathrm{lh}}=").replace("T_bud=", "$T_{\\mathrm{bud}}=").replace("1000000", "\\infty") + "$"
            f.write(f"{lab} & {pm(d['latency_ms'])} & {pm(d['sla_viol'], 100, 1)} & {pm(d['energy_J'], 1, 3)} & {pm(d['mig_per_slot'], 1, 2)} \\\\\n")

# ---- burstiness
if "burst" in n:
    with open(f"{T}/burst.tex", "w") as f:
        for sw in sorted(n["burst"], key=float):
            d = n["burst"][sw]
            f.write(f"{float(sw):g}$\\times$ & " + " & ".join(f"{pm(d[s]['latency_ms'])} / {pm(d[s]['sla_viol'], 100, 1)}" for s in ("Static", "LSTM-Det", "RAI-Edge")) + " \\\\\n")
print("tables written")
