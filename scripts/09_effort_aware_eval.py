"""Step 9: effort-aware evaluation on all release pairs.

Share of defective files found when reviewing 20% of files or 20% of LOC,
models vs. size-based rules (largest / smallest files first).
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config as C, data as D, plotstyle as S
from src.models import build
from src.score import recall_at
from src.evaluate import paired_test

# ----------------------------------------------------------------------------- recall@20% per pair
S.apply(); df = D.load_jira(); feats = D.feature_columns(df); rows = []
for proj, a, b, tr, te in D.release_pairs(df):
    y = te[C.LABEL_REAL].values; loc = te["CountLineCode"].values.astype(float); n = len(y)
    scores = {}
    for name in ["LOC_LR", "LR", "RF", "XGB_weighted"]:
        scores[name] = build(name).fit(tr[feats], tr[C.LABEL_REAL].values).predict_proba(te[feats])[:, 1]
    rules = {"Largest files first": np.argsort(-loc, kind="stable"), "Smallest files first": np.argsort(loc, kind="stable")}
    for name, p in scores.items():
        rows.append(dict(project=proj, train=a, test=b, method=name,
                         recall_20pct_files=recall_at(np.argsort(-p, kind="stable"), y, np.ones(n)),
                         recall_20pct_loc=recall_at(np.argsort(-(p / np.maximum(loc, 1)), kind="stable"), y, loc)))   # LOC budget: rank by risk/LOC
    for name, order in rules.items():
        rows.append(dict(project=proj, train=a, test=b, method=name,
                         recall_20pct_files=recall_at(order, y, np.ones(n)), recall_20pct_loc=recall_at(order, y, loc)))
r = pd.DataFrame(rows); r.to_csv(C.RESULTS / "effort_aware_results.csv", index=False)
summ = r.groupby("method")[["recall_20pct_files", "recall_20pct_loc"]].mean().round(3).sort_values("recall_20pct_loc", ascending=False)
summ.to_csv(C.RESULTS / "effort_aware_summary.csv"); print(summ)

# ----------------------------------------------------------------------------- paired tests
out = []
for a_, b_ in [("RF", "Largest files first"), ("RF", "Smallest files first"), ("RF", "LOC_LR"), ("RF", "LR"), ("XGB_weighted", "RF")]:
    for k in ["recall_20pct_files", "recall_20pct_loc"]:
        P = r.pivot_table(index=["project", "train", "test"], columns="method", values=k)
        out.append(dict(method_a=a_, method_b=b_, metric=k, **paired_test(P[a_].values, P[b_].values)))
t = pd.DataFrame(out); t.to_csv(C.RESULTS / "effort_aware_tests.csv", index=False); print(t.round(4).to_string(index=False))

# ----------------------------------------------------------------------------- figure: mean + 95% bootstrap CI
order = ["Largest files first", "Smallest files first", "LOC_LR", "LR", "RF", "XGB_weighted"]
labels = {"Largest files first": "Largest files first", "Smallest files first": "Smallest files first", "LOC_LR": "LOC-only logistic",
          "LR": "Logistic regression", "RF": "Random forest", "XGB_weighted": "XGBoost + class weights"}
colors = {m: S.GREY for m in order}; colors.update({"LOC_LR": S.NAVY, "LR": S.NAVY, "RF": S.TEAL, "XGB_weighted": S.TEAL})
rng = np.random.default_rng(C.SEED)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for ax, (k, ttl, rand) in zip(axes, [("recall_20pct_files", "Review 20% of the files", 0.2), ("recall_20pct_loc", "Review 20% of the code (lines)", 0.2)]):
    for i, m in enumerate(order[::-1]):
        v = r[r.method == m][k].values; ci = np.percentile([rng.choice(v, len(v)).mean() for _ in range(3000)], [2.5, 97.5])
        ax.plot(ci, [i, i], color=colors[m], lw=1.6); ax.plot(v.mean(), i, "o", color=colors[m], ms=6.5, markeredgecolor="white", markeredgewidth=1.2)
        ax.text(ci[1] + 0.01, i, f"{v.mean():.2f}", va="center", fontsize=7.5, color=S.INK)
    ax.axvline(rand, color="#C9D1DA", ls="--", lw=1); ax.set_title(ttl, fontsize=9.5); ax.set_xlabel("Share of defective files found"); ax.grid(axis="y", visible=False)   # dashed = random order
    ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[m] for m in order[::-1]])
fig.suptitle("Effort-aware evaluation, mean over release pairs with 95% bootstrap intervals (dashed line: random order)", x=0.01, ha="left", fontsize=8.6)
fig.tight_layout(); fig.savefig(C.FIGS / "effort_aware.png"); plt.close(fig)
