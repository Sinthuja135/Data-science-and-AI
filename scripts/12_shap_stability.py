"""Step 12: explanation stability.

Within- vs across-project SHAP stability, dependence plots, SHAP vs permutation importance.
Needs results/shap_importance_by_pair.csv (from step 04).
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from src import config as C, data as D, plotstyle as S
from src.models import build

# ----------------------------------------------------------------------------- within vs across project stability
S.apply(); df = D.load_jira(); feats = D.feature_columns(df)
I = pd.read_csv(C.RESULTS / "shap_importance_by_pair.csv", index_col=0)
proj = {c: c.split(":")[0] for c in I.columns}   # "camel:1.4.0->2.9.0" -> "camel"
within, across = [], []
for x, y in combinations(I.columns, 2):
    r = spearmanr(I[x], I[y]).statistic; (within if proj[x] == proj[y] else across).append(r)
pd.DataFrame([dict(kind="within project", mean_spearman=np.mean(within), n=len(within)),
              dict(kind="across projects", mean_spearman=np.mean(across), n=len(across))]).to_csv(C.RESULTS / "shap_within_across.csv", index=False)
print("within", np.mean(within), len(within), "across", np.mean(across), len(across))
tops = pd.Series({p: I[[c for c in I.columns if proj[c] == p]].mean(1).idxmax() for p in sorted(set(proj.values()))})   # top feature per project
tops.to_csv(C.RESULTS / "shap_top_feature_by_project.csv", header=["top_feature"]); print(tops)

# ----------------------------------------------------------------------------- SHAP vs permutation importance (XGB)
agree, allsv, allX = [], [], []
for p, a, b, tr, te in D.release_pairs(df):
    m = build("XGB_weighted").fit(tr[feats], tr[C.LABEL_REAL].values)
    prep, clf = m.named_steps["prep"], m.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(te[feats]), columns=prep.feature_names_)
    sv = shap.TreeExplainer(clf.model_).shap_values(Xt.values)
    shap_imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns)
    yte = te[C.LABEL_REAL].values
    pi = permutation_importance(clf, Xt.values, yte, scoring=lambda est, X, y: average_precision_score(y, est.predict_proba(X)[:, 1]), n_repeats=3, random_state=C.SEED, n_jobs=1)   # PR-AUC drop
    perm = pd.Series(pi.importances_mean, index=Xt.columns)
    top = shap_imp.nlargest(10).index
    agree.append(dict(project=p, pair=f"{a}->{b}", spearman_all=spearmanr(shap_imp, perm).statistic,
                      top10_overlap=len(set(top) & set(perm.nlargest(10).index)) / 10))
    idx = np.random.default_rng(C.SEED).choice(len(Xt), min(400, len(Xt)), replace=False)   # sample <=400 rows for plots
    allsv.append(pd.DataFrame(sv[idx], columns=Xt.columns)); allX.append(Xt.iloc[idx].reset_index(drop=True))
A = pd.DataFrame(agree); A.to_csv(C.RESULTS / "shap_vs_permutation.csv", index=False)
print(A[["spearman_all", "top10_overlap"]].mean())

# ----------------------------------------------------------------------------- figure: dependence plots (top 3)
SV, XX = pd.concat(allsv, ignore_index=True), pd.concat(allX, ignore_index=True)
imp = SV.abs().mean().sort_values(ascending=False); top3 = list(imp.index[:3])
fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9))
for ax, f in zip(axes, top3):
    ax.scatter(XX[f], SV[f], s=4, alpha=0.35, color=S.NAVY, edgecolors="none")
    ax.axhline(0, color=S.GREY, lw=0.7); ax.set_xlabel(f"{f} (log-scaled)"); ax.set_title(f, fontsize=9.5)
axes[0].set_ylabel("SHAP value (log-odds)"); fig.tight_layout(); fig.savefig(C.FIGS / "shap_dependence.png"); plt.close(fig)

# direction check: Spearman(feature, SHAP) for top 10
pd.Series({f: spearmanr(XX[f], SV[f], nan_policy='omit').statistic for f in imp.index[:10]}).to_csv(C.RESULTS / "shap_direction.csv", header=["spearman_feature_vs_shap"])
print(pd.read_csv(C.RESULTS / "shap_direction.csv", index_col=0).round(2))

# ----------------------------------------------------------------------------- figure: within vs across bar
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(["within\nproject", "across\nprojects"], [np.mean(within), np.mean(across)], color=[S.NAVY, S.TEAL], width=0.55)
ax.set_ylabel("mean Spearman of importance profiles"); ax.set_title("Explanation stability", fontsize=9.5); ax.grid(axis="x", visible=False)
for i, v in enumerate([np.mean(within), np.mean(across)]): ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(C.FIGS / "shap_stability_within_across.png"); plt.close(fig)
