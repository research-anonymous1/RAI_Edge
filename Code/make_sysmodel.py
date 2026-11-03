"""Draws the three-tier RAI-Edge architecture figure (vector PDF), sized for a
0.62-column-width placement in the 6-page manuscript (2.2 in wide, fonts >= 5.5 pt)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 6, "font.family": "serif", "mathtext.fontset": "dejavuserif"})
fig = plt.figure(figsize=(2.2, 2.25))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, text, fc, ec="k", lw=0.6, fs=5.6, bold=False, z=1, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", zorder=z + 1, linespacing=1.1)

def arrow(p, q, color="k", ls="-", lw=0.7, both=False, z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=5, color=color,
                                 lw=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

# ---- cloud tier -------------------------------------------------------------
box(22, 92, 56, 7, "Cloud $f{=}K{+}1$ (40 ms gateway)", "#e8eef7", fs=5.6)
arrow((50, 92), (50, 88.5), both=True, lw=0.6)

# ---- SDN controller ---------------------------------------------------------
box(1, 50, 98, 38.5, "", "#f3f3f3", lw=0.8, z=0)
ax.text(3, 85.0, "SDN controller (period $\\Delta$)", fontsize=6.2, weight="bold")
box(3, 68.5, 28, 14, "Global view\n$h_{f,t},\\,b_{f,t},\\,U_{\\ell,t}$\n$a_i(t)$", "#ffffff", fs=5.6)
box(34, 68.5, 32, 14, "MC-dropout LSTM\n+ bootstrap\n$\\{\\hat h^{(k)}_{f,t+1}\\}$", "#fff3d1", fs=5.4)
box(69, 68.5, 28, 14, "Risk engine\nemp. $\\mathrm{CVaR}_\\beta$\n$\\Rightarrow \\hat h^{\\beta}_{f}$", "#fde2e1", fs=5.4)
box(3, 52.5, 94, 13, "Greedy joint orchestrator (Alg. 1)\nrisk/memory feasibility, gain vs. migration cost\n$\\Rightarrow$ placement $\\mathbf{x}_t$, migrations $\\mathbf{M}_t$, routes $\\mathbf{R}_t$", "#e3f1e1", fs=5.6)
arrow((31, 75.5), (34, 75.5), lw=0.6); arrow((66, 75.5), (69, 75.5), lw=0.6)
arrow((83, 68.5), (83, 65.5), lw=0.6); arrow((17, 68.5), (17, 65.5), lw=0.6)

# ---- fog tier ---------------------------------------------------------------
box(1, 22, 98, 25, "", "#f9f9f9", lw=0.8, z=0)
ax.text(3, 43.8, "Fog tier: SDN switches + fog nodes", fontsize=6.2, weight="bold")
xs = [11, 30.5, 50, 69.5, 89]; kinds = ["micro", "macro", "micro", "micro", "macro"]
for x, k in zip(xs, kinds):
    fc = "#d9e8ff" if k == "micro" else "#b9cff2"
    box(x - 8.5, 27.5, 17, 11, f"FN {k}\n" + ("16 GHz" if k == "micro" else "48 GHz") + "\n$C_f, \\epsilon_f$", fc, fs=5.4, r=1.0)
for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
    ax.plot([xs[a] + 8.5, xs[b] - 8.5], [33, 33], color="#333", lw=0.7, zorder=0)
ax.plot([xs[0], xs[2]], [27.5, 27.5], color="#333", lw=0.7, zorder=0); ax.plot([xs[2], xs[4]], [27.5, 27.5], color="#333", lw=0.7, zorder=0)
ax.text(50, 23.8, "backhaul $B_\\ell$, $\\tau_\\ell$ (fluid sharing)", fontsize=5.4, ha="center")
arrow((30.5, 38.5), (50, 38.5), color="#b8860b", lw=0.9)
ax.text(40, 40.2, "migration $r_s$", fontsize=5.4, color="#7a5a00", ha="center")

# controller <-> fog
arrow((10, 50), (10, 47), color="#1a5fb4", ls="--", lw=0.7)
ax.text(12, 48.5, "rules $\\mathbf{R}_t$, cmds $\\mathbf{M}_t$", fontsize=5.4, color="#1a5fb4", va="center")
arrow((90, 47), (90, 50), color="#c01c28", ls="--", lw=0.7)
ax.text(88, 48.5, "telemetry", fontsize=5.4, color="#c01c28", va="center", ha="right")

# ---- device tier ------------------------------------------------------------
box(1, 2, 98, 17.5, "", "#fbfbfb", lw=0.8, z=0)
ax.text(3, 16.6, "IoT devices (RWP / GM / trace); uplink to nearest AP", fontsize=5.0, weight="bold")
dev = [(12, 11), (27, 30.5), (42, 50), (57, 50), (72, 69.5), (87, 89)]
for x, apx in dev:
    ax.add_patch(Circle((x, 9), 2.4, fc="#ffe8c7", ec="k", lw=0.5, zorder=2))
    ax.text(x, 9, "$i$", fontsize=5.4, ha="center", va="center", zorder=3)
    arrow((x, 11.4), (apx, 22), color="#2a7f2a", lw=0.55)
ax.text(50, 3.8, "ON/OFF Poisson tasks $(d_\\kappa, w_\\kappa)$; service $s_i$ at host $x_{s_i}$", fontsize=5.4, ha="center")
fig.savefig(f"{OUT}/sysmodel_raiedge.pdf")
print("saved")
