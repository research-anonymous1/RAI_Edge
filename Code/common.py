"""Benchmark configuration shared by every experiment script."""
from sim import Config, Scheme

def benchmark_config(**over):
    base = dict(T_warm=10, T_eval=120,
                lam_on=3.5, lam_off=0.35, wbar_min=0.2, wbar_max=0.4,
                eps_micro=0.8, eps_macro=1.2, eps_cloud=1.6, E_ref=0.25,
                alpha=0.3, gamma=0.1, beta=0.15, delta_margin=0.10, T_lh=5, T_bud=10,
                K_mc=50, window=10)
    base.update(over)
    return Config(**base)

SCHEMES = {
    "Static":   Scheme("Static", rule="static", routing="minhop", migrate=False),
    "CTMC":     Scheme("CTMC", rule="ctmc", risk="measured", routing="minhop"),
    "LSTM-Det": Scheme("LSTM-Det", risk="mean"),
    "Robust":   Scheme("Robust", risk="robust"),
    "VaR":      Scheme("VaR", risk="var"),
    "DE-CVaR":  Scheme("DE-CVaR", risk="ensemble"),
    "RAI-G":    Scheme("RAI-G", risk="gauss"),
    "RAI-Edge": Scheme("RAI-Edge", risk="cvar"),
    "Oracle":   Scheme("Oracle", risk="oracle"),
}
MAIN_SCHEMES = ["Static", "CTMC", "LSTM-Det", "Robust", "VaR", "DE-CVaR", "RAI-G", "RAI-Edge"]
EVAL_SEED0 = 1000
MODEL_DIR = "../models"
RESULT_DIR = "../results"
