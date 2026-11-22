"""Step 13: cross-project vs within-project training.

Within: RF trained on the previous release. Cross: RF pooled on the other 8 projects
(percentile-normalised per release). Same test release and RealBug labels.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src import config as C, data as D, plotstyle as S
from src.models import build
from src.preprocess import Preprocessor
from src.evaluate import oof_threshold, score, paired_test

S.apply(); df = D.load_jira(); feats = D.feature_columns(df)
pct = df.copy(); pct[feats] = df.groupby(["project", "release"])[feats].rank(pct=True)   # per-release percentiles

def pooled():
    """Regularised RF for the large pooled training set."""
    return Pipeline([("prep", Preprocessor()), ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_samples=0.3,
                     n_jobs=2, class_weight="balanced_subsample", random_state=C.SEED))])

# ----------------------------------------------------------------------------- within vs cross per pair
rows = []
for proj, a, b, tr, te in D.release_pairs(df):
    y = te[C.LABEL_REAL].values
    m = build("RF").fit(tr[feats], tr[C.LABEL_REAL].values); p = m.predict_proba(te[feats])[:, 1]
    rows.append(dict(project=proj, train=a, test=b, setting="within-project", **score(y, p, oof_threshold(m, tr[feats], tr[C.LABEL_REAL].values))))
    other = pct[pct.project != proj]; te_p = pct[(pct.project == proj) & (pct.release == b)]
    m2 = pooled().fit(other[feats], other[C.LABEL_REAL].values); p2 = m2.predict_proba(te_p[feats])[:, 1]
    sub = other.sample(min(20000, len(other)), random_state=C.SEED)   # subsample for OOF threshold (runtime)
    thr = oof_threshold(pooled(), sub[feats], sub[C.LABEL_REAL].values)
    rows.append(dict(project=proj, train=a, test=b, setting="cross-project (pooled)", **score(y, p2, thr)))
    print(proj, a, b, round(rows[-2]["auc"], 3), round(rows[-1]["auc"], 3), flush=True)
r = pd.DataFrame(rows); r.to_csv(C.RESULTS / "cross_project_pairs.csv", index=False)
s = r.groupby("setting")[["auc", "pr_auc", "mcc_opt", "f1_opt"]].mean().round(3); s.to_csv(C.RESULTS / "cross_project_summary.csv"); print(s)

# ----------------------------------------------------------------------------- paired tests
out = []
for k in ["auc", "pr_auc", "mcc_opt"]:
    P = r.pivot_table(index=["project", "train", "test"], columns="setting", values=k)
    out.append(dict(metric=k, **paired_test(P["cross-project (pooled)"].values, P["within-project"].values)))
t = pd.DataFrame(out); t.to_csv(C.RESULTS / "cross_project_tests.csv", index=False); print(t.round(4).to_string(index=False))

# ----------------------------------------------------------------------------- figure: AUC per project
P = r.pivot_table(index=["project", "test"], columns="setting", values="auc"); pp = P.groupby(level=0).mean()
fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(len(pp))
ax.bar(x - .2, pp["within-project"], .4, color=S.NAVY, label="within-project"); ax.bar(x + .2, pp["cross-project (pooled)"], .4, color=S.TEAL, label="cross-project (pooled)")
ax.axhline(0.5, color=S.GREY, lw=.7, ls="--"); ax.set_xticks(x); ax.set_xticklabels(pp.index, rotation=30, ha="right"); ax.set_ylabel("ROC AUC"); ax.set_ylim(0.4, 1); ax.legend(fontsize=7.5)
ax.set_title("Cross-project vs within-project (mean over releases)", fontsize=9.5); fig.tight_layout(); fig.savefig(C.FIGS / "cross_project.png"); plt.close(fig)
