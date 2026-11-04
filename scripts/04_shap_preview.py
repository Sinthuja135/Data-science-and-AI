"""SHAP importance of the class-weighted XGBoost model across release pairs.

For each release pair the model is trained on release t and explained on release t+1; importances are
normalised to shares of total mean |SHAP| so pairs can be compared. Writes shap_importance_by_pair.csv,
shap_stability.csv and figures/shap_top_features.png.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from src import config as C, data as D, plotstyle as S
from src.models import build

S.apply()
C.RESULTS.mkdir(exist_ok=True); C.FIGS.mkdir(exist_ok=True)
df = D.load_jira(); feats = D.feature_columns(df)
imps = []
pairs = list(D.release_pairs(df))
for i, (proj, a, b, tr, te) in enumerate(pairs, 1):
    m = build("XGB_weighted").fit(tr[feats], tr[C.LABEL_REAL].values)
    prep, clf = m.named_steps["prep"], m.named_steps["clf"].model_
    Xt = prep.transform(te[feats]); sv = shap.TreeExplainer(clf).shap_values(Xt)
    imp = pd.Series(np.abs(sv).mean(0), index=prep.feature_names_); imp = imp / imp.sum()
    imps.append(imp.rename(f"{proj}:{a}->{b}"))
    print(f"[{i}/{len(pairs)}] {proj} {a}->{b} done", flush=True)
I = pd.concat(imps, axis=1).fillna(0); I.to_csv(C.RESULTS / "shap_importance_by_pair.csv")
mean = I.mean(1).sort_values(ascending=False)
grp = lambda f: "process" if f in C.PROCESS else "ownership" if f in C.OWNERSHIP else "code"
top = mean.head(15)
rho = [spearmanr(I[x], I[y]).statistic for x, y in combinations(I.columns, 2)]
t10 = [len(set(I[x].nlargest(10).index) & set(I[y].nlargest(10).index)) / 10 for x, y in combinations(I.columns, 2)]
share = pd.Series({g: sum(v for f, v in mean.items() if grp(f) == g) for g in ["code", "process", "ownership"]})
stab = dict(mean_pairwise_spearman=float(np.mean(rho)), mean_top10_overlap=float(np.mean(t10)), **{f"share_{k}": float(v) for k, v in share.items()})
pd.Series(stab).to_csv(C.RESULTS / "shap_stability.csv", header=["value"]); print(stab); print(mean.head(12).round(4))

colors = {"code": S.NAVY, "process": S.TEAL, "ownership": S.AMBER}
fig, ax = plt.subplots(figsize=(6.4, 4.4)); tt = top.iloc[::-1]
# interquartile range across release pairs; the mean can fall outside it for skewed importances,
# so widen the interval to include the mean (matplotlib rejects negative error lengths)
lo = np.minimum(I.loc[tt.index].quantile(0.25, axis=1), tt); hi = np.maximum(I.loc[tt.index].quantile(0.75, axis=1), tt)
ax.barh(tt.index, tt.values, color=[colors[grp(f)] for f in tt.index], height=0.62)
ax.errorbar(tt.values, range(len(tt)), xerr=[tt.values - lo.values, hi.values - tt.values], fmt="none", ecolor=S.INK, elinewidth=0.8, capsize=2)
ax.set_xlabel(f"Share of mean |SHAP| (bars: mean over {I.shape[1]} pairs; whiskers: interquartile range)"); ax.grid(axis="y", visible=False)
ax.set_title("What the boosted model relies on")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], labels=["code metrics", "process metrics", "ownership metrics"], loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3)
fig.savefig(C.FIGS / "shap_top_features.png"); plt.close(fig)
print("saved shap_importance_by_pair.csv, shap_stability.csv and shap_top_features.png")
