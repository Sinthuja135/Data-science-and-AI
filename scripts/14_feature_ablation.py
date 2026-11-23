"""Step 14: metric-family ablation.

Random forest on subsets of metrics (code / process / ownership / ...) vs all metrics.
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
from src.preprocess import Preprocessor
from src.evaluate import oof_threshold, score, paired_test

# ----------------------------------------------------------------------------- feature sets
S.apply(); df = D.load_jira(); feats = D.feature_columns(df)
proc = [f for f in feats if f in C.PROCESS]; own = [f for f in feats if f in C.OWNERSHIP]; code = [f for f in feats if f not in proc + own]
sets = {"all metrics": feats, "code only": code, "process only": proc, "ownership only": own, "code + process": code + proc, "COMM only": ["COMM"]}

def rf(cols):
    """RF on `cols` only (no correlation pruning for a single metric)."""
    return Pipeline([("prep", Preprocessor(only=cols, prune=len(cols) > 1)),
                     ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=2, class_weight="balanced_subsample", random_state=C.SEED))])

# ----------------------------------------------------------------------------- run every set on every pair
rows = []
for proj, a, b, tr, te in D.release_pairs(df):
    y = te[C.LABEL_REAL].values
    for name, cols in sets.items():
        m = rf(cols).fit(tr[cols], tr[C.LABEL_REAL].values); p = m.predict_proba(te[cols])[:, 1]
        rows.append(dict(project=proj, train=a, test=b, feature_set=name, n_metrics=len(cols), **score(y, p, oof_threshold(m, tr[cols], tr[C.LABEL_REAL].values))))
r = pd.DataFrame(rows); r.to_csv(C.RESULTS / "ablation_pairs.csv", index=False)
s = r.groupby("feature_set")[["auc", "pr_auc", "mcc_opt", "f1_opt"]].mean().round(3).reindex(list(sets)); s.insert(0, "n_metrics", [len(v) for v in sets.values()]); s.to_csv(C.RESULTS / "ablation_summary.csv"); print(s)

# ----------------------------------------------------------------------------- paired tests vs all metrics
out = []
for name in list(sets)[1:]:
    for k in ["auc", "pr_auc", "mcc_opt"]:
        P = r.pivot_table(index=["project", "train", "test"], columns="feature_set", values=k)
        out.append(dict(comparison=f"{name} vs all metrics", metric=k, **paired_test(P[name].values, P["all metrics"].values)))
t = pd.DataFrame(out); t.to_csv(C.RESULTS / "ablation_tests.csv", index=False); print(t.round(4).to_string(index=False))

# ----------------------------------------------------------------------------- figure: AUC and MCC per set
fig, ax = plt.subplots(figsize=(6.6, 3.2)); x = np.arange(len(s)); w = 0.38
ax.bar(x - w / 2, s.auc, w, color=S.NAVY, label="ROC AUC"); ax.bar(x + w / 2, s.mcc_opt, w, color=S.AMBER, label="MCC")
for i, (a_, m_) in enumerate(zip(s.auc, s.mcc_opt)): ax.text(i - w / 2, a_ + .01, f"{a_:.2f}", ha="center", fontsize=7); ax.text(i + w / 2, m_ + .01, f"{m_:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({k})" for n, k in zip(s.index, s.n_metrics)], fontsize=7.5); ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="x", visible=False)
ax.set_title("Random forest on subsets of metrics (number of metrics in brackets)", fontsize=9.5); fig.tight_layout(); fig.savefig(C.FIGS / "feature_ablation.png"); plt.close(fig)
