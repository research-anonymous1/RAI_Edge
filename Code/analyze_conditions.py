"""Condition analysis for the research question:
   "Under what workload, uncertainty and resource conditions does risk-aware joint
    orchestration outperform deterministic prediction and simpler risk-aware
    alternatives, and what is the cost of those improvements?"

Reads ../results/{main,burst,burst2,capacity,spread,alpha,alpha2}.jsonl and writes
  ../figures/conditions.pdf      (2 x 4 panels: gain vs LSTM-Det, cost vs LSTM-Det)
  ../figures/main_4panel.pdf     (compact 1 x 4 main-results figure)
  ../results/numbers_conditions.json  (every number quoted in Section VI-C)
All comparisons are paired over seeds (common random numbers) and reported as
relative differences with 95% confidence intervals.
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

R = "../results"; IMG = "../figures"; os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"font.size": 7.0, "font.family": "serif", "axes.labelsize": 7.0, "legend.fontsize": 6.0,
                     "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "lines.markersize": 3.0, "legend.frameon": False, "figure.dpi": 150})
COL = {"Static": "#7f7f7f", "CTMC": "#d62728", "LSTM-Det": "#ff7f0e", "Robust": "#9467bd", "VaR": "#8c564b",
       "DE-CVaR": "#17becf", "RAI-G": "#2ca02c", "RAI-Edge": "#1f77b4"}
MK = {"Static": "x", "CTMC": "v", "LSTM-Det": "s", "Robust": "D", "VaR": "^", "DE-CVaR": "P", "RAI-G": "o", "RAI-Edge": "o"}
LABEL = {"RAI-G": "RAI-G", "DE-CVaR": "DE-CVaR", "Robust": "Robust (μ+2σ)", "VaR": "VaR (quantile)"}

def load(name):
    p = f"{R}/{name}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan, np.nan, 0)
    return (float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0, int(len(v)))

def paired_rel(rows, cond_key, cond, scheme, base, metric):
    """relative reduction (base - scheme)/base, paired over seeds. Positive = scheme better (lower)."""
    b = {r["seed"]: r[metric] for r in rows if r["label"] == base and r[cond_key] == cond}
    s = {r["seed"]: r[metric] for r in rows if r["label"] == scheme and r[cond_key] == cond}
    d = [(b[k] - s[k]) / b[k] for k in s if k in b and b[k] and not math.isnan(b[k]) and not math.isnan(s[k])]
    wins = sum(x > 0 for x in d)
    m, e, n = ci(d)
    return dict(mean=m, ci=e, n=n, wins=wins)

def ratio(rows, cond_key, cond, scheme, base, metric):
    b = ci([r[metric] for r in rows if r["label"] == base and r[cond_key] == cond])[0]
    s = ci([r[metric] for r in rows if r["label"] == scheme and r[cond_key] == cond])[0]
    return s / b if b else float("nan")

def tag(rows, key):
    for r in rows: r[key] = r["extra"].get(key, r.get("N"))
    return rows

# ---------- data sets for the four condition axes
main = load("main")
for r in main: r["N_"] = r["N"]
burst = tag(load("burst") + load("burst2"), "switch")
cap = tag(load("capacity"), "cap_scale")
spread = tag(load("spread"), "w_spread")
alpha = tag(load("alpha") + load("alpha2"), "alpha")

AXES = [  # (rows, condition key, x label, panel title, xscale, xticks)
    (main, "N_", "Devices $N$", "(a) load", "linear", [100, 200, 300]),
    (burst, "switch", "ON/OFF rate $\\times$", "(b) burstiness", "log", [0.5, 1, 2, 4]),
    (cap, "cap_scale", "Capacity $\\times$", "(c) capacity", "linear", [0.75, 1.0, 1.5]),
    (alpha, "alpha", "Energy weight $\\alpha$", "(d) weight $\\alpha$", "linear", [0, 0.3, 0.5, 0.7]),
]
SCHEMES = ["RAI-Edge", "VaR", "Robust"]
numbers = {}

fig, axs = plt.subplots(3, 4, figsize=(3.5, 3.05), sharex="col")
for j, (rows, key, xl, title, xs, xt) in enumerate(AXES):
    conds = sorted(set(r[key] for r in rows if r["label"] == "LSTM-Det"))
    if key == "alpha": conds = [c for c in conds if c <= 0.7]
    numbers[key] = {}
    for sch in SCHEMES:
        cs = [c for c in conds if any(r["label"] == sch and r[key] == c for r in rows)]
        lat = [paired_rel(rows, key, c, sch, "LSTM-Det", "latency_ms") for c in cs]
        sla = [paired_rel(rows, key, c, sch, "LSTM-Det", "sla_viol") for c in cs]
        en = [paired_rel(rows, key, c, sch, "LSTM-Det", "energy_J") for c in cs]
        mig = [ratio(rows, key, c, sch, "LSTM-Det", "mig_per_slot") for c in cs]
        for c, l, s, e, m in zip(cs, lat, sla, en, mig):
            numbers[key].setdefault(str(c), {})[sch] = dict(latency_red=l, sla_red=s, energy_red=e, mig_ratio=m)
        ax = axs[0, j]
        ax.errorbar(cs, [100 * v["mean"] for v in lat], yerr=[100 * v["ci"] for v in lat], color=COL[sch], marker=MK[sch],
                    label=LABEL.get(sch, sch), capsize=1.2, elinewidth=0.6, mfc=COL[sch])
        ax = axs[1, j]
        ax.errorbar(cs, [100 * v["mean"] for v in sla], yerr=[100 * v["ci"] for v in sla], color=COL[sch], marker=MK[sch],
                    capsize=1.2, elinewidth=0.6, mfc=COL[sch])
        ax = axs[2, j]
        ax.errorbar(cs, [-100 * v["mean"] for v in en], yerr=[100 * v["ci"] for v in en], color=COL[sch], marker=MK[sch],
                    capsize=1.2, elinewidth=0.6, mfc=COL[sch])
    # absolute context for the text
    numbers[key]["abs"] = {}
    for c in conds:
        numbers[key]["abs"][str(c)] = {}
        for sch in ["Static", "LSTM-Det"] + SCHEMES:
            rr = [r for r in rows if r["label"] == sch and r[key] == c]
            if rr:
                numbers[key]["abs"][str(c)][sch] = {m: ci([r[m] for r in rr]) for m in ("latency_ms", "sla_viol", "energy_J", "mig_per_slot", "overload_freq")}
    for i in (0, 1, 2):
        ax = axs[i, j]
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_xscale(xs); ax.set_xticks(xt); ax.set_xticklabels([f"{t:g}".replace("0.", ".") if t < 1 and key == "alpha" else f"{t:g}" for t in xt], fontsize=5.8)
        ax.tick_params(axis="y", labelsize=5.8, pad=1.5); ax.tick_params(axis="x", pad=1.5)
        if xs == "log": ax.minorticks_off()
    axs[0, j].set_title(title, fontsize=7.0, pad=2)
    axs[2, j].set_xlabel(xl, fontsize=6.3, labelpad=1)
axs[0, 0].set_ylabel("Latency reduction\nvs LSTM-Det (%)", fontsize=6.3)
axs[1, 0].set_ylabel("SLA-violation red.\nvs LSTM-Det (%)", fontsize=6.3)
axs[2, 0].set_ylabel("Energy increase\nvs LSTM-Det (%)", fontsize=6.3)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.0, fontsize=6.0)
fig.tight_layout(pad=0.15, w_pad=0.5, h_pad=0.3, rect=(0, 0, 1, 0.945))
fig.savefig(f"{IMG}/conditions.pdf"); plt.close(fig)

# ---------- spread sweep (text only)
numbers["w_spread"] = {}
for c in sorted(set(r["w_spread"] for r in spread)):
    numbers["w_spread"][str(c)] = {sch: dict(latency_red=paired_rel(spread, "w_spread", c, sch, "LSTM-Det", "latency_ms"),
                                            sla_red=paired_rel(spread, "w_spread", c, sch, "LSTM-Det", "sla_viol"),
                                            energy_red=paired_rel(spread, "w_spread", c, sch, "LSTM-Det", "energy_J"),
                                            mig_ratio=ratio(spread, "w_spread", c, sch, "LSTM-Det", "mig_per_slot")) for sch in SCHEMES}
    numbers["w_spread"][str(c)]["abs"] = {sch: {m: ci([r[m] for r in spread if r["label"] == sch and r["w_spread"] == c]) for m in ("latency_ms", "sla_viol", "energy_J", "mig_per_slot")} for sch in ["LSTM-Det"] + SCHEMES}

# ---------- compact 1 x 4 main-results figure
ORDER = ["Static", "CTMC", "LSTM-Det", "Robust", "VaR", "DE-CVaR", "RAI-G", "RAI-Edge"]
LS = {"RAI-G": "--"}
Ns = sorted(set(r["N"] for r in main))
def agg(rows, key):
    d = defaultdict(list)
    for r in rows: d[(r["label"], r["N"])].append(r[key])
    return {k: ci(v) for k, v in d.items()}
mets = [("latency_ms", "(a) Latency (ms)", 1), ("energy_J", "(b) Energy (J/task)", 1), ("sla_viol", "(c) SLA viol. (%)", 100), ("mig_per_slot", "(d) Migr./slot", 1)]
fig, axs = plt.subplots(1, 4, figsize=(3.5, 1.42))
for ax, (m, title, sc) in zip(axs, mets):
    A = agg(main, m)
    for sch in ORDER:
        xs_ = [N for N in Ns if (sch, N) in A]
        ys = np.array([A[(sch, N)][0] for N in xs_]) * sc; es = np.array([A[(sch, N)][1] for N in xs_]) * sc
        ax.errorbar(xs_, ys, yerr=es, color=COL[sch], marker=MK[sch], ls=LS.get(sch, "-"), label={"RAI-G": "RAI-G (Gauss.)", "DE-CVaR": "DE-CVaR", "Robust": "Robust", "VaR": "VaR"}.get(sch, sch),
                    capsize=1.0, elinewidth=0.5, mfc="white" if sch == "RAI-G" else COL[sch], markersize=2.4, lw=0.9)
    ax.set_title(title, fontsize=6.3, pad=2); ax.grid(alpha=0.3, lw=0.4)
    ax.set_xticks([100, 200, 300]); ax.tick_params(pad=1.5, labelsize=5.8)
    ax.set_xlabel("Devices $N$", fontsize=6.3, labelpad=1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), handlelength=1.3, columnspacing=0.8, handletextpad=0.3, fontsize=5.5)
fig.tight_layout(pad=0.12, w_pad=0.45, rect=(0, 0, 1, 0.84))
fig.savefig(f"{IMG}/main_4panel.pdf"); plt.close(fig)

json.dump(numbers, open(f"{R}/numbers_conditions.json", "w"), indent=1, default=float)

# ---------- console summary
def fmt(v): return f"{100*v['mean']:+5.1f}±{100*v['ci']:.1f} ({v['wins']}/{v['n']})"
for key in ["N_", "switch", "cap_scale", "alpha", "w_spread"]:
    print(f"== {key}")
    for c, d in numbers[key].items():
        if c == "abs": continue
        for sch in SCHEMES:
            if sch in d:
                v = d[sch]
                print(f"  {c:>5} {sch:8s} lat {fmt(v['latency_red'])}  sla {fmt(v['sla_red'])}  E {fmt(v['energy_red'])}  mig x{v['mig_ratio']:.2f}")

# ---------- combined validation strip: calibration (a,b) + optimality gap (c) + runtime (d)
beta = load("beta")
fig, axs = plt.subplots(1, 4, figsize=(3.5, 1.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ax = axs[0]
ax.plot([0.4, 1], [0.4, 1], "k--", lw=0.6, label="ideal")
for N, c in zip([100, 200, 300], ["#9ecae1", "#4292c6", "#08306b"]):
    rr = [r for r in main if r["label"] == "RAI-Edge" and r["N"] == N and "coverage" in r]
    if rr: ax.plot(levels, np.mean([r["coverage"] for r in rr], 0), marker="o", color=c, label=f"$N$={N}", ms=2.2)
ax.set_xlabel("Nominal coverage", fontsize=6.3, labelpad=1); ax.set_ylabel("Empirical cov.", fontsize=6.0, labelpad=1)
ax.legend(loc="upper left", fontsize=5.0, handlelength=1.2, borderpad=0.2, labelspacing=0.2); ax.grid(alpha=0.3, lw=0.4)
ax.set_title("(a) coverage", fontsize=6.3, pad=2); ax.set_xticks([0.5, 0.7, 0.9]); ax.set_yticks([0.5, 0.7, 0.9])
ax = axs[1]
if beta:
    bs = sorted(set(r["extra"]["beta"] for r in beta))
    pv = [ci([r["p_exceed_var"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
    pc = [ci([r["p_exceed_cvar"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
    ax.plot(bs, bs, "k--", lw=0.6, label="nominal $\\beta$")
    ax.errorbar(bs, [p[0] for p in pv], yerr=[p[1] for p in pv], marker="^", color="#8c564b", label="VaR$_\\beta$", capsize=1.2, ms=2.4)
    ax.errorbar(bs, [p[0] for p in pc], yerr=[p[1] for p in pc], marker="o", color="#1f77b4", label="CVaR$_\\beta$", capsize=1.2, ms=2.4)
    ax.set_xlabel("Tail probability $\\beta$", fontsize=6.3, labelpad=1); ax.set_ylabel("Realised exceed.", fontsize=6.0, labelpad=1)
    ax.legend(loc="upper left", fontsize=5.0, handlelength=1.2, borderpad=0.2, labelspacing=0.2); ax.grid(alpha=0.3, lw=0.4)
    ax.set_title("(b) exceedance", fontsize=6.3, pad=2); ax.set_xticks([0, 0.25, 0.5]); ax.set_yticks([0, 0.25, 0.5])
ax = axs[2]
if os.path.exists(f"{R}/mip_gap.json"):
    mg = [r for r in json.load(open(f"{R}/mip_gap.json")) if r.get("solved")]
    sizes = sorted(set((r["K"], r["N"]) for r in mg))
    data = [[100 * r["gap"] for r in mg if (r["K"], r["N"]) == k] for k in sizes]
    ax.boxplot(data, widths=0.55, showfliers=True, flierprops=dict(marker=".", ms=1.5), medianprops=dict(color="#d62728"), whis=(5, 95))
    ax.set_xticks(range(1, len(sizes) + 1)); ax.set_xticklabels([f"{k[0]+1}/{k[1]}" for k in sizes], fontsize=5.0, rotation=45)
    ax.set_xlabel("$|\\bar{\\mathcal{F}}|$ / $|\\mathcal{S}|$", fontsize=6.3, labelpad=1); ax.set_ylabel("Gap to optimum (%)", fontsize=6.0, labelpad=1)
    ax.grid(axis="y", alpha=0.3, lw=0.4); ax.set_title("(c) gap", fontsize=6.3, pad=2)
ax = axs[3]
if os.path.exists(f"{R}/runtime.json"):
    rt = json.load(open(f"{R}/runtime.json")); rows = rt["rows"]
    keys = sorted(set((r["K"], r["N"]) for r in rows))
    pred = [np.mean([r["pred_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    gre = [np.mean([r["greedy_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    rou = [np.mean([r["route_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    p95 = [np.mean([r["p95_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, pred, color="#fff3d1", edgecolor="k", lw=0.4, label="prediction")
    ax.bar(x, gre, bottom=pred, color="#e3f1e1", edgecolor="k", lw=0.4, label="greedy")
    ax.bar(x, rou, bottom=np.array(pred) + np.array(gre), color="#d9e8ff", edgecolor="k", lw=0.4, label="routing")
    ax.plot(x, p95, "k_", ms=6, mew=1.0, label="p95")
    ax.set_xticks(x); ax.set_xticklabels([f"{k[0]}/{k[1]}" for k in keys], fontsize=4.6, rotation=60)
    ax.set_xlabel("$|\\mathcal{F}|$ / $|\\mathcal{S}|$", fontsize=6.3, labelpad=1); ax.set_ylabel("Time/slot (ms)", fontsize=6.0, labelpad=1)
    ax.legend(loc="upper left", fontsize=4.8, handlelength=1.0, borderpad=0.2, labelspacing=0.2); ax.grid(axis="y", alpha=0.3, lw=0.4)
    ax.set_title("(d) runtime", fontsize=6.3, pad=2)
for ax in axs: ax.tick_params(labelsize=5.6, pad=1.5)
fig.tight_layout(pad=0.12, w_pad=0.7)
fig.savefig(f"{IMG}/validation.pdf"); plt.close(fig)
