"""Builds every figure (PDF) and the numbers used in the manuscript from
the raw per-run results in ../results/*.jsonl. Figures go to ../figures/."""
import json, os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

R = "../results"; IMG = "../figures"; os.makedirs(IMG, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.labelsize": 7.5, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "lines.markersize": 3.2, "legend.frameon": False, "figure.dpi": 150})
COL = {"Static": "#7f7f7f", "CTMC": "#d62728", "LSTM-Det": "#ff7f0e", "Robust": "#9467bd", "VaR": "#8c564b",
       "DE-CVaR": "#17becf", "RAI-G": "#2ca02c", "RAI-Edge": "#1f77b4", "Oracle": "#000000"}
MK = {"Static": "x", "CTMC": "v", "LSTM-Det": "s", "Robust": "D", "VaR": "^", "DE-CVaR": "P", "RAI-G": "o", "RAI-Edge": "o"}
LS = {"RAI-G": "--"}
ORDER = ["Static", "CTMC", "LSTM-Det", "Robust", "VaR", "DE-CVaR", "RAI-G", "RAI-Edge"]
LABEL = {"RAI-G": "RAI-Edge (Gauss. CVaR)", "DE-CVaR": "DE-CVaR (ensemble)", "Robust": "Robust (μ+2σ)", "VaR": "VaR (quantile)"}

def load(name):
    p = f"{R}/{name}.jsonl"
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p)]

def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0: return np.nan, np.nan
    return v.mean(), 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0

def agg(rows, key, by=("label", "N")):
    d = defaultdict(list)
    for r in rows:
        if key in r and r[key] is not None: d[tuple(r[b] for b in by)].append(r[key])
    return {k: ci(v) for k, v in d.items()}

numbers = {}
main = load("main")
Ns = sorted(set(r["N"] for r in main))
metrics = {"latency_ms": "Avg. task latency (ms)", "energy_J": "Energy per task (J)", "sla_viol": "SLA violation rate",
           "mig_per_slot": "Migrations per slot", "overload_freq": "Node-slot overload freq.", "mig_time_ms": "Avg. migration time (ms)"}
A = {m: agg(main, m) for m in metrics}

def curve(ax, metric, pct=False, legend=False, logy=False):
    for sch in ORDER:
        if not any(k[0] == sch for k in A[metric]): continue
        xs = [N for N in Ns if (sch, N) in A[metric]]
        ys = np.array([A[metric][(sch, N)][0] for N in xs]); es = np.array([A[metric][(sch, N)][1] for N in xs])
        if pct: ys, es = 100 * ys, 100 * es
        ax.errorbar(xs, ys, yerr=es, color=COL[sch], marker=MK[sch], ls=LS.get(sch, "-"), label=LABEL.get(sch, sch), capsize=1.5, elinewidth=0.6,
                    mfc="white" if sch == "RAI-G" else COL[sch])
    ax.set_xlabel("Number of IoT devices $N$"); ax.set_ylabel(metrics[metric] + (" (%)" if pct else ""))
    ax.grid(alpha=0.3, lw=0.4)
    if logy: ax.set_yscale("log")
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.figure.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=0.7, fontsize=5.6)

if main:
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7))
    curve(axs[0], "latency_ms", legend=True); curve(axs[1], "energy_J")
    axs[0].set_title("(a)", fontsize=7.5, pad=2); axs[1].set_title("(b)", fontsize=7.5, pad=2)
    fig.tight_layout(pad=0.15, w_pad=0.6, rect=(0, 0, 1, 0.86)); fig.savefig(f"{IMG}/latency_energy.pdf"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7))
    curve(axs[0], "sla_viol", pct=True, legend=True); curve(axs[1], "mig_per_slot")
    axs[0].set_title("(a)", fontsize=7.5, pad=2); axs[1].set_title("(b)", fontsize=7.5, pad=2)
    fig.tight_layout(pad=0.15, w_pad=0.6, rect=(0, 0, 1, 0.86)); fig.savefig(f"{IMG}/sla_migrations.pdf"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5))
    N0 = 250
    schs = [s for s in ORDER if s != "Static" and (s, N0) in A["mig_time_ms"]]
    vals = [A["mig_time_ms"][(s, N0)][0] for s in schs]; errs = [A["mig_time_ms"][(s, N0)][1] for s in schs]
    axs[0].bar(range(len(schs)), vals, yerr=errs, color=[COL[s] for s in schs], capsize=2, width=0.65, error_kw=dict(lw=0.6))
    axs[0].set_xticks(range(len(schs))); axs[0].set_xticklabels(schs, rotation=55, ha="right", fontsize=5.6)
    axs[0].set_ylabel("Avg. migration time (ms)"); axs[0].grid(axis="y", alpha=0.3, lw=0.4); axs[0].set_title("(a) $N=250$", fontsize=7.5, pad=2)
    curve(axs[1], "overload_freq", pct=True); axs[1].set_title("(b)", fontsize=7.5, pad=2)
    fig.tight_layout(pad=0.15, w_pad=0.6); fig.savefig(f"{IMG}/migtime_overload.pdf"); plt.close(fig)

    # headline numbers at N=300 and N=250
    for N in (200, 250, 300):
        numbers[f"N{N}"] = {s: {m: A[m].get((s, N), (np.nan, np.nan)) for m in metrics} for s in ORDER}
    # paired relative improvements (RAI-Edge vs others), by seed
    def paired(metric, a, b, N):
        da = {r["seed"]: r[metric] for r in main if r["label"] == a and r["N"] == N}
        db = {r["seed"]: r[metric] for r in main if r["label"] == b and r["N"] == N}
        common = sorted(set(da) & set(db))
        if not common: return (np.nan, np.nan, 0)
        d = np.array([(db[s] - da[s]) / db[s] for s in common if db[s] > 0])   # relative reduction of a vs b
        return float(d.mean()), float(1.96 * d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0, int((np.array([da[s] < db[s] for s in common])).sum())
    numbers["paired"] = {}
    for N in (150, 200, 250, 300):
        for b in ORDER:
            if b == "RAI-Edge": continue
            for m in ("latency_ms", "sla_viol", "energy_J", "mig_per_slot"):
                numbers["paired"][f"{m}|RAI-Edge|{b}|{N}"] = paired(m, "RAI-Edge", b, N)
    # predictor metrics from RAI-Edge runs
    rr = [r for r in main if r["label"] == "RAI-Edge" and "rmse" in r]
    numbers["pred"] = {N: {k: ci([r[k] for r in rr if r["N"] == N])[0] for k in ("rmse", "mae", "p_exceed_cvar", "p_exceed_var", "p_overload_given_safe", "frac_declared_safe", "cvar_rel_diff", "runtime_ms", "runtime_p95_ms")} for N in Ns}
    numbers["pred"]["coverage"] = {N: np.mean([r["coverage"] for r in rr if r["N"] == N], 0).tolist() for N in Ns}
    numbers["runtime_main"] = {s: ci([r["runtime_ms"] for r in main if r["label"] == s and r["N"] == 300]) for s in ORDER}

# ---------------- calibration figure
beta = load("beta"); calib = load("calib")
if main or beta:
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5))
    levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    axs[0].plot([0.4, 1], [0.4, 1], "k--", lw=0.6, label="ideal")
    for N, c in zip([100, 200, 300], ["#9ecae1", "#4292c6", "#08306b"]):
        rr = [r for r in main if r["label"] == "RAI-Edge" and r["N"] == N and "coverage" in r]
        if rr: axs[0].plot(levels, np.mean([r["coverage"] for r in rr], 0), marker="o", color=c, label=f"$N={N}$")
    rg = [r for r in main if r["label"] == "RAI-G" and "coverage" in r]
    axs[0].set_xlabel("Nominal central coverage"); axs[0].set_ylabel("Empirical coverage"); axs[0].legend(loc="upper left"); axs[0].grid(alpha=0.3, lw=0.4)
    axs[0].set_title("(a)", fontsize=7.5, pad=2)
    if beta:
        bs = sorted(set(r["extra"]["beta"] for r in beta))
        pv = [ci([r["p_exceed_var"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
        pc = [ci([r["p_exceed_cvar"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
        axs[1].plot(bs, bs, "k--", lw=0.6, label="nominal $\\beta$")
        axs[1].errorbar(bs, [p[0] for p in pv], yerr=[p[1] for p in pv], marker="^", color="#8c564b", label="$P(h>\\mathrm{VaR}_\\beta)$", capsize=1.5)
        axs[1].errorbar(bs, [p[0] for p in pc], yerr=[p[1] for p in pc], marker="o", color="#1f77b4", label="$P(h>\\mathrm{CVaR}_\\beta)$", capsize=1.5)
        axs[1].set_xlabel("Upper-tail probability $\\beta$"); axs[1].set_ylabel("Realised exceedance"); axs[1].legend(loc="upper left"); axs[1].grid(alpha=0.3, lw=0.4)
        axs[1].set_title("(b)", fontsize=7.5, pad=2)
        numbers["beta_exceed"] = {b: (pv[i], pc[i]) for i, b in enumerate(bs)}
    fig.tight_layout(pad=0.15, w_pad=0.6); fig.savefig(f"{IMG}/calibration.pdf"); plt.close(fig)

if calib:
    # Gaussian-assumption check: skewness of MC samples, empirical vs Gaussian CVaR
    sk = np.concatenate([r["diag_skew"] for r in calib if r.get("diag_skew")])
    cve = np.concatenate([r["diag_cve"] for r in calib]); cvg = np.concatenate([r["diag_cvg"] for r in calib])
    pit = np.concatenate([r["diag_pit"] for r in calib])
    numbers["gauss_check"] = dict(skew_mean=float(sk.mean()), skew_p90=float(np.quantile(sk, .9)), frac_skew_gt_0p5=float((sk > 0.5).mean()),
                                  cvar_rel_diff_mean=float(np.mean((cvg - cve) / np.maximum(cve, 1e-3))),
                                  cvar_rel_diff_p90=float(np.quantile(np.abs(cvg - cve) / np.maximum(cve, 1e-3), .9)),
                                  frac_gauss_below_emp=float((cvg < cve).mean()), n=int(len(cve)))
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.45))
    axs[0].hist(pit, bins=10, range=(0, 1), color="#1f77b4", alpha=0.85, density=True, edgecolor="white", lw=0.4)
    axs[0].axhline(1, color="k", ls="--", lw=0.6); axs[0].set_xlabel("PIT of realised demand"); axs[0].set_ylabel("Density"); axs[0].set_title("(a)", fontsize=7.5, pad=2)
    axs[1].hist(sk, bins=30, color="#ff7f0e", alpha=0.85, density=True, edgecolor="white", lw=0.4)
    axs[1].axvline(0, color="k", ls="--", lw=0.6); axs[1].set_xlabel("Skewness of predictive samples", fontsize=6.8); axs[1].set_title("(b)", fontsize=7.5, pad=2)
    fig.tight_layout(pad=0.15, w_pad=0.6); fig.savefig(f"{IMG}/pit_skew.pdf"); plt.close(fig)

# ---------------- beta sensitivity
if beta:
    bs = sorted(set(r["extra"]["beta"] for r in beta))
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5))
    lat = [ci([r["latency_ms"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
    sla = [ci([100 * r["sla_viol"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
    mig = [ci([r["mig_per_slot"] for r in beta if r["extra"]["beta"] == b]) for b in bs]
    ax = axs[0]; ax2 = ax.twinx()
    ax.errorbar(bs, [v[0] for v in sla], yerr=[v[1] for v in sla], marker="s", color="#d62728", label="SLA violation (%)", capsize=1.5)
    ax2.errorbar(bs, [v[0] for v in mig], yerr=[v[1] for v in mig], marker="^", color="#2ca02c", ls="--", label="migrations/slot", capsize=1.5)
    ax.set_ylabel("SLA violation (%)", color="#d62728"); ax2.set_ylabel("Migrations per slot", color="#2ca02c")
    axs[1].errorbar(bs, [v[0] for v in lat], yerr=[v[1] for v in lat], marker="o", color="#1f77b4", capsize=1.5)
    axs[1].set_ylabel("Avg. task latency (ms)")
    for a in (ax, axs[1]):
        a.set_xscale("log"); tk = [0.02, 0.05, 0.1, 0.2, 0.5]; a.set_xticks(tk); a.set_xticklabels([f"{b:g}" for b in tk], fontsize=6); a.minorticks_off()
        a.set_xlabel("Upper-tail probability $\\beta$"); a.grid(alpha=0.3, lw=0.4)
    ax.set_title("(a)", fontsize=7.5, pad=2); axs[1].set_title("(b)", fontsize=7.5, pad=2)
    fig.tight_layout(pad=0.15, w_pad=1.2); fig.savefig(f"{IMG}/beta_sensitivity.pdf"); plt.close(fig)
    numbers["beta"] = {b: dict(latency=lat[i], sla=sla[i], mig=mig[i]) for i, b in enumerate(bs)}

# ---------------- alpha Pareto
alpha = load("alpha")
if alpha:
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    for sch, c in [("RAI-Edge", "#1f77b4"), ("LSTM-Det", "#ff7f0e")]:
        als = sorted(set(r["extra"]["alpha"] for r in alpha if r["label"] == sch))
        E = [ci([r["energy_J"] for r in alpha if r["label"] == sch and r["extra"]["alpha"] == a]) for a in als]
        L = [ci([r["latency_ms"] for r in alpha if r["label"] == sch and r["extra"]["alpha"] == a]) for a in als]
        ax.errorbar([l[0] for l in L], [e[0] for e in E], xerr=[l[1] for l in L], yerr=[e[1] for e in E], marker=MK[sch], color=c, label=sch, capsize=1.2, lw=0.8)
        for a, l, e in zip(als, L, E):
            if sch == "RAI-Edge" and a in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0): ax.annotate(f"$\\alpha$={a:g}", (l[0], e[0]), fontsize=5.5, xytext=(4, 3), textcoords="offset points")
        numbers[f"alpha_{sch}"] = {a: dict(latency=L[i], energy=E[i]) for i, a in enumerate(als)}
    st = [r for r in main if r["label"] == "Static" and r["N"] == 250]
    if st: ax.plot(np.mean([r["latency_ms"] for r in st]), np.mean([r["energy_J"] for r in st]), marker="x", color=COL["Static"], ls="none", label="Static")
    ax.set_xlabel("Avg. task latency (ms)"); ax.set_ylabel("Energy per task (J)"); ax.legend(loc="upper right"); ax.grid(alpha=0.3, lw=0.4); ax.set_xlim(28, 215)
    fig.tight_layout(pad=0.15); fig.savefig(f"{IMG}/alpha_pareto.pdf"); plt.close(fig)

# ---------------- migrations over time (series)
series = load("series")
if series:
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.6))
    for sch in ["CTMC", "LSTM-Det", "Robust", "VaR", "RAI-Edge"]:
        rr = [r for r in series if r["label"] == sch]
        if not rr: continue
        m = np.mean([r["series"]["n_mig"] for r in rr], 0); v = np.mean([r["series"]["viol_rate"] for r in rr], 0) * 100
        t = np.arange(1, len(m) + 1)
        axs[0].plot(t, m, color=COL[sch], label=sch, lw=0.9); axs[1].plot(t, v, color=COL[sch], lw=0.9)
    axs[0].set_xlabel("Time slot"); axs[0].set_ylabel("Migrations per slot"); axs[0].grid(alpha=0.3, lw=0.4); axs[0].set_title("(a)", fontsize=7.5, pad=2)
    axs[1].set_xlabel("Time slot"); axs[1].set_ylabel("SLA violation (%)"); axs[1].grid(alpha=0.3, lw=0.4); axs[1].set_title("(b)", fontsize=7.5, pad=2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=5.6, columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(pad=0.15, w_pad=0.6, rect=(0, 0, 1, 0.9)); fig.savefig(f"{IMG}/migrations_time.pdf"); plt.close(fig)

# ---------------- combined gap + runtime figure
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
if os.path.exists(f"{R}/mip_gap.json"):
    mg = [r for r in json.load(open(f"{R}/mip_gap.json")) if r.get("solved")]
    sizes = sorted(set((r["K"], r["N"]) for r in mg))
    data = [[100 * r["gap"] for r in mg if (r["K"], r["N"]) == k] for k in sizes]
    ax = axs[0]
    ax.boxplot(data, widths=0.55, showfliers=True, flierprops=dict(marker=".", ms=2), medianprops=dict(color="#d62728"), whis=(5, 95))
    ax.set_xticks(range(1, len(sizes) + 1)); ax.set_xticklabels([f"{k[0]+1}/{k[1]}" for k in sizes], fontsize=6)
    ax.set_xlabel("$|\\bar{\\mathcal{F}}|$ / $|\\mathcal{S}|$"); ax.set_ylabel("Optimality gap (%)"); ax.grid(axis="y", alpha=0.3, lw=0.4); ax.set_title("(a)", fontsize=7.5, pad=2)
    numbers["mip"] = {f"{k[0]}x{k[1]}": dict(n=len(d), mean_gap=float(np.mean(d)), max_gap=float(np.max(d)), frac_zero=float(np.mean(np.array(d) < 1e-6)),
                                             mip_time=float(np.mean([r["mip_time"] for r in mg if (r["K"], r["N"]) == k])),
                                             improv_greedy=float(np.mean([r["improv_greedy"] for r in mg if (r["K"], r["N"]) == k])),
                                             improv_opt=float(np.mean([r["improv_opt"] for r in mg if (r["K"], r["N"]) == k]))) for k, d in zip(sizes, data)}
    numbers["mip"]["all"] = dict(n=len(mg), mean_gap=float(np.mean([r["gap"] for r in mg])), max_gap=float(np.max([r["gap"] for r in mg])),
                                 frac_zero=float(np.mean([r["gap"] < 1e-6 for r in mg])), p95_gap=float(np.quantile([r["gap"] for r in mg], .95)))
if os.path.exists(f"{R}/runtime.json"):
    rt = json.load(open(f"{R}/runtime.json")); rows = rt["rows"]
    keys = sorted(set((r["K"], r["N"]) for r in rows))
    pred = [np.mean([r["pred_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    gre = [np.mean([r["greedy_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    rou = [np.mean([r["route_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    p95 = [np.mean([r["p95_ms"] for r in rows if (r["K"], r["N"]) == k]) for k in keys]
    x = np.arange(len(keys)); ax = axs[1]
    ax.bar(x, pred, color="#fff3d1", edgecolor="k", lw=0.4, label="prediction")
    ax.bar(x, gre, bottom=pred, color="#e3f1e1", edgecolor="k", lw=0.4, label="greedy")
    ax.bar(x, rou, bottom=np.array(pred) + np.array(gre), color="#d9e8ff", edgecolor="k", lw=0.4, label="routing")
    ax.plot(x, p95, "k_", ms=8, mew=1.0, label="p95")
    ax.set_xticks(x); ax.set_xticklabels([f"{k[0]}/{k[1]}" for k in keys], fontsize=5.6, rotation=45)
    ax.set_xlabel("$|\\mathcal{F}|$ / $|\\mathcal{S}|$"); ax.set_ylabel("Time per slot (ms)"); ax.legend(loc="upper left", ncol=1, fontsize=5.4); ax.grid(axis="y", alpha=0.3, lw=0.4); ax.set_title("(b)", fontsize=7.5, pad=2)
    numbers["runtime"] = {f"{k[0]}x{k[1]}": dict(mean=float(np.mean([r["mean_ms"] for r in rows if (r["K"], r["N"]) == k])), p95=p95[i], max=float(np.max([r["max_ms"] for r in rows if (r["K"], r["N"]) == k]))) for i, k in enumerate(keys)}
    numbers["cpu"] = rt["cpu"]
fig.tight_layout(pad=0.15, w_pad=0.5); fig.savefig(f"{IMG}/gap_runtime.pdf"); plt.close(fig)

# ---------------- tables: ablation, GM mobility, horizon/budget, burst
def table_rows(rows, labels, N=None):
    out = {}
    for lab in labels:
        rr = [r for r in rows if r["label"] == lab and (N is None or r["N"] == N)]
        if not rr: continue
        out[lab] = {m: ci([r[m] for r in rr]) for m in ("latency_ms", "sla_viol", "energy_J", "mig_per_slot", "overload_freq")}
        out[lab]["n"] = len(rr)
    return out
abl = load("ablation")
if abl:
    numbers["ablation"] = table_rows(abl, sorted(set(r["label"] for r in abl)))
gm = load("gm")
if gm:
    numbers["gm"] = {N: table_rows(gm, ORDER, N) for N in sorted(set(r["N"] for r in gm))}
hz = load("horizon")
if hz:
    numbers["horizon"] = table_rows(hz, sorted(set(r["label"] for r in hz)))
bu = load("burst")
if bu:
    numbers["burst"] = {sw: {s: {m: ci([r[m] for r in bu if r["extra"]["switch"] == sw and r["label"] == s]) for m in ("latency_ms", "sla_viol")} for s in ("Static", "LSTM-Det", "RAI-Edge")} for sw in sorted(set(r["extra"]["switch"] for r in bu))}

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o
json.dump(conv(numbers), open(f"{R}/numbers.json", "w"), indent=1)
print("figures written; numbers.json keys:", list(numbers.keys()))
