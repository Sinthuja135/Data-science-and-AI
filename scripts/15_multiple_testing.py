"""Step 15: Holm-Bonferroni correction per family of paired tests + environment record.

Reads the *_tests.csv files from earlier steps; skips any that are missing.
"""
import sys, platform, importlib.metadata as md, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src import config as C

def holm(p):
    """Holm-adjusted p-values (step-down, kept monotone, capped at 1)."""
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

# ----------------------------------------------------------------------------- correct each test family
families = {"model_comparison": "paired_tests.csv", "effort_aware": "effort_aware_tests.csv", "label_noise": "label_noise_tests.csv",
            "external": "external_tests.csv", "cross_project": "cross_project_tests.csv", "feature_ablation": "ablation_tests.csv"}
frames = []
for fam, f in families.items():
    path = C.RESULTS / f
    if not path.exists(): print("missing", f); continue
    t = pd.read_csv(path); t.insert(0, "family", fam)
    t = t[t.wilcoxon_p.notna()].copy(); t["p_holm"] = holm(t.wilcoxon_p.values)
    t["sig_raw"] = t.wilcoxon_p < 0.05; t["sig_holm"] = t.p_holm < 0.05; frames.append(t)
A = pd.concat(frames, ignore_index=True); A.to_csv(C.RESULTS / "tests_corrected.csv", index=False)
s = A.groupby("family").agg(n_tests=("wilcoxon_p", "size"), sig_raw=("sig_raw", "sum"), sig_holm=("sig_holm", "sum")); s.to_csv(C.RESULTS / "tests_corrected_summary.csv"); print(s)

# ----------------------------------------------------------------------------- environment record
pk = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "shap", "scipy", "matplotlib", "joblib"]
env = {"python": platform.python_version(), "os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(), "seed": C.SEED}
for k in pk:
    try: env[k] = md.version(k)
    except Exception: env[k] = "n/a"   # not installed
try:
    env["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
except Exception: env["ram_gb"] = "n/a"   # os.sysconf not available on Windows
pd.Series(env).to_csv(C.RESULTS / "environment.csv", header=["value"]); print(pd.Series(env))
