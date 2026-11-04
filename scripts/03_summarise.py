"""Step 3: summarise model results, paired statistical tests and comparison figures.

Reads results/pair_results.csv (step 2) and, if present, results/improvement_results.csv (step 2). Only the
models actually present in those files are summarised/tested, so this also works on a partial pipeline
(e.g. baselines + improvements, before the boosting variants are added).
Run from the project root:  python scripts/03_summarise.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config as C, plotstyle as S
from src.evaluate import paired_test

S.apply()
C.RESULTS.mkdir(exist_ok=True); C.FIGS.mkdir(exist_ok=True)
r = pd.read_csv(C.RESULTS / "pair_results.csv")
imp_path = C.RESULTS / "improvement_results.csv"
if imp_path.exists():
    imp = pd.read_csv(imp_path)
    r = pd.concat([r, imp.drop(columns=["n_train"])], ignore_index=True)
present = set(r.model)
n_pairs = r[["project", "train", "test"]].drop_duplicates().shape[0]
metrics = ["auc", "pr_auc", "mcc_05", "mcc_opt", "f1_opt"]
ALL_MODELS = ["LOC_LR", "LR", "RF", "XGB", "XGB_weighted", "XGB_smote", "LGBM_weighted", "XGB_tuned",
              "RF_tuned", "ENS", "RF_hist", "ENS_hist"]
order = [m for m in ALL_MODELS if m in present]
rng = np.random.default_rng(C.SEED)

def boot_ci(x, n=5000):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

rows = []
for m in order:
    d = r[r.model == m]
    row = {"model": m, "n_pairs": len(d)}
    for k in metrics:
        row[k] = d[k].mean(); row[k + "_sd"] = d[k].std(); lo, hi = boot_ci(d[k].values); row[k + "_lo"], row[k + "_hi"] = lo, hi
    rows.append(row)
summ = pd.DataFrame(rows); summ.to_csv(C.RESULTS / "model_summary.csv", index=False)
print(summ[["model"] + metrics].round(3).to_string(index=False))

def piv(k): return r.pivot_table(index=["project", "train", "test"], columns="model", values=k)
ALL_TESTS = [("LR", "LOC_LR"), ("RF", "LR"), ("XGB_weighted", "RF"), ("XGB_tuned", "RF"), ("XGB_tuned", "XGB_weighted"),
             ("XGB_weighted", "XGB"), ("XGB_smote", "XGB_weighted"), ("LGBM_weighted", "XGB_weighted"), ("XGB_tuned", "LOC_LR"),
             ("RF_tuned", "RF"), ("ENS", "RF"), ("ENS", "XGB_weighted"), ("RF_hist", "RF"), ("ENS_hist", "RF"), ("ENS_hist", "ENS"),
             ("ENS_hist", "LR"), ("ENS_hist", "LOC_LR"), ("ENS", "LOC_LR")]
tests = [(a, b) for a, b in ALL_TESTS if a in present and b in present]
out = []
for a, b in tests:
    for k in ["auc", "pr_auc", "mcc_opt"]:
        P = piv(k); out.append(dict(model_a=a, model_b=b, metric=k, **paired_test(P[a].values, P[b].values)))
st = pd.DataFrame(out); st.to_csv(C.RESULTS / "paired_tests.csv", index=False)
print(st.round(4).to_string(index=False))

# per-project view for the best baseline (RF) vs the best improved model available
challenger = next((m for m in ["XGB_tuned", "RF_tuned"] if m in present), None)
labels = {"LOC_LR": "LOC-only logistic", "LR": "Logistic regression", "RF": "Random forest", "XGB": "XGBoost (plain)",
          "XGB_weighted": "XGBoost + class weights", "XGB_smote": "XGBoost + SMOTE", "LGBM_weighted": "LightGBM + class weights", "XGB_tuned": "XGBoost tuned",
          "RF_tuned": "Random forest tuned", "ENS": "Ensemble (RF + XGBoost)", "RF_hist": "RF, all past releases", "ENS_hist": "Ensemble, all past releases"}
if challenger and "RF" in present:
    pp = r[r.model.isin(["RF", challenger])].groupby(["project", "model"])[["auc", "pr_auc", "mcc_opt"]].mean().unstack("model")
    pp.to_csv(C.RESULTS / f"per_project_rf_vs_{challenger.lower().replace('_', '')}.csv"); print(pp.round(3))

# ---- Figure: model comparison, three metrics as dot + 95% CI (one axis per metric)
colors = {m: S.GREY for m in order}; colors.update({"LR": S.NAVY, "RF": S.NAVY, "LOC_LR": S.NAVY}); colors.update({m: S.TEAL for m in ["XGB", "XGB_weighted", "XGB_smote", "LGBM_weighted"]}); colors["XGB_tuned"] = S.AMBER
for m in ["RF_tuned", "ENS", "RF_hist", "ENS_hist"]: colors[m] = S.AMBER
fig, axes = plt.subplots(1, 3, figsize=(9.8, 0.34 * len(order) + 1.3), sharey=True)
for ax, (k, ttl) in zip(axes, [("auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("mcc_opt", "MCC (tuned threshold)")]):
    for i, m in enumerate(order[::-1]):
        row = summ[summ.model == m].iloc[0]
        ax.plot([row[k + "_lo"], row[k + "_hi"]], [i, i], color=colors[m], lw=1.6)
        ax.plot(row[k], i, "o", color=colors[m], ms=6.5, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(row[k + "_hi"] + 0.004, i, f"{row[k]:.2f}", va="center", fontsize=7.5, color=S.INK)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[m] for m in order[::-1]]); ax.set_title(ttl, fontsize=9.5); ax.grid(axis="y", visible=False)
fig.suptitle(f"Mean over {n_pairs} release pairs, with 95% bootstrap intervals (navy: baselines, teal: boosting variants, amber: tuned / ensemble / more history)", x=0.01, ha="left", fontsize=9, color=S.INK)
fig.tight_layout(); fig.savefig(C.FIGS / "model_comparison.png"); plt.close(fig)

# ---- Figure: per-pair paired comparison RF vs the challenger (AUC and MCC)
if challenger and "RF" in present:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for ax, (k, ttl) in zip(axes, [("auc", "ROC AUC"), ("mcc_opt", "MCC (tuned threshold)")]):
        P = piv(k); ax.scatter(P["RF"], P[challenger], s=34, color=S.TEAL, edgecolor="white", linewidth=1, zorder=3)
        lo, hi = min(P["RF"].min(), P[challenger].min()) - 0.03, max(P["RF"].max(), P[challenger].max()) + 0.03
        ax.plot([lo, hi], [lo, hi], color=S.GREY, lw=1, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Random forest (baseline)"); ax.set_ylabel(labels[challenger])
        ax.set_title(ttl + f": above the line = improvement ({int((P[challenger] > P['RF']).sum())}/{len(P)})", fontsize=8.5)
    fig.tight_layout(); fig.savefig(C.FIGS / f"rf_vs_{challenger.lower()}_pairs.png"); plt.close(fig)
print("saved model_summary.csv, paired_tests.csv and figures to", C.RESULTS.parent)
