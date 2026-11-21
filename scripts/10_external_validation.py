"""Step 10: external validation on TeraPromise using the ~9 metrics shared with JIRA.

JIRA -> TeraPromise, TeraPromise -> JIRA (camel/lucene excluded), and in-domain (TP v -> v+1).
Variants: raw (signed log) and norm (within-release percentile ranks).
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config as C, plotstyle as S
from src.external import load_tp, load_jira_shared, cross_dataset, in_domain, SHARED, OVERLAP, MAPPING
from src.evaluate import paired_test

S.apply(); C.RESULTS.mkdir(exist_ok=True)
tp, jr = load_tp(), load_jira_shared()
print("TeraPromise:", len(tp), "classes,", tp.release.nunique(), "releases,", tp.project.nunique(), "projects | JIRA:", len(jr), "files")

# ----------------------------------------------------------------------------- metric comparison between datasets
desc = pd.DataFrame({"jira_median": jr[SHARED].median(), "tp_median": tp[SHARED].median(), "jira_p90": jr[SHARED].quantile(0.9), "tp_p90": tp[SHARED].quantile(0.9)})
desc["tp_metric"] = {v: k for k, v in MAPPING.items()}
desc["spearman_with_loc_jira"] = [jr[[m, "CountLineCode"]].corr("spearman").iloc[0, 1] for m in SHARED]
desc["spearman_with_loc_tp"] = [tp[[m, "CountLineCode"]].corr("spearman").iloc[0, 1] for m in SHARED]
desc.to_csv(C.RESULTS / "external_metric_comparison.csv"); print(desc.round(2))

# ----------------------------------------------------------------------------- run all three setups
tp_t = tp[~tp.project.isin(OVERLAP)].reset_index(drop=True); jr_t = jr[~jr.project.isin(OVERLAP)].reset_index(drop=True)
fwd = cross_dataset(jr, tp_t, direction="JIRA->TeraPromise"); print("forward done", len(fwd), flush=True)
rev = cross_dataset(tp_t, jr_t, direction="TeraPromise->JIRA"); print("reverse done", len(rev), flush=True)
ind = in_domain(tp_t); print("in-domain done", len(ind), flush=True)
res = pd.concat([fwd, rev, ind], ignore_index=True); res.to_csv(C.RESULTS / "external_results.csv", index=False)

metrics = ["auc", "pr_auc", "mcc_top20", "recall_20files", "recall_20loc"]
summ = res.groupby(["direction", "variant", "model"])[metrics + ["n"]].agg({**{m: "mean" for m in metrics}, "n": "count"}).rename(columns={"n": "n_targets"}).round(3)
summ.to_csv(C.RESULTS / "external_summary.csv"); print(summ.to_string())

# ----------------------------------------------------------------------------- paired tests on per-project means
def per_project(d, k): return d.groupby("project")[k].mean()
out = []
def add(desc_, a, b, k):
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) >= 5: out.append(dict(comparison=desc_, metric=k, n_projects=len(j), **paired_test(j.iloc[:, 0].values, j.iloc[:, 1].values)))
f = fwd
for k in metrics:
    P = lambda d, v, m: per_project(d[(d.variant == v) & (d.model == m)], k)
    add("JIRA->TP: RF norm vs RF raw", P(f, "norm", "RF"), P(f, "raw", "RF"), k)
    add("JIRA->TP: RF norm vs LOC-only norm", P(f, "norm", "RF"), P(f, "norm", "LOC_LR"), k)
    add("JIRA->TP: RF norm vs LR norm", P(f, "norm", "RF"), P(f, "norm", "LR"), k)
    add("JIRA->TP: XGB norm vs RF norm", P(f, "norm", "XGB_weighted"), P(f, "norm", "RF"), k)
    for v in ("raw", "norm"):   # cross-dataset vs in-domain, same targets
        a = f[(f.variant == v) & (f.model == "RF")].set_index("release")[k]; b = ind[(ind.variant == v) & (ind.model == "RF")].set_index("release")[k]
        pr = f[["release", "project"]].drop_duplicates().set_index("release")["project"]
        j = pd.concat([a, b], axis=1, join="inner").dropna(); j["project"] = pr.reindex(j.index).values
        j = j.groupby("project").mean()
        if len(j) >= 5: out.append(dict(comparison=f"RF {v}: JIRA->TP vs in-domain (same targets)", metric=k, n_projects=len(j), **paired_test(j.iloc[:, 0].values, j.iloc[:, 1].values)))
tests = pd.DataFrame(out); tests.to_csv(C.RESULTS / "external_tests.csv", index=False)
print(tests.round(4).to_string())

# ----------------------------------------------------------------------------- figure: JIRA -> TP, raw vs norm
order = ["LOC_LR", "LR", "RF", "XGB_weighted"]; names = {"LOC_LR": "LOC-only logistic", "LR": "Logistic regression", "RF": "Random forest", "XGB_weighted": "XGBoost + class weights"}
rng = np.random.default_rng(C.SEED)
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), sharey=True)
for ax, (k, ttl) in zip(axes, [("auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("recall_20files", "Defects found in top 20% of files")]):
    for i, m in enumerate(order[::-1]):
        for off, v, col in [(-0.16, "raw", S.NAVY), (0.16, "norm", S.TEAL)]:
            d = per_project(f[(f.variant == v) & (f.model == m)], k).values
            ci = np.percentile([rng.choice(d, len(d)).mean() for _ in range(3000)], [2.5, 97.5])
            ax.plot(ci, [i + off] * 2, color=col, lw=1.6); ax.plot(d.mean(), i + off, "o", color=col, ms=6, markeredgecolor="white", markeredgewidth=1)
            ax.text(ci[1] + 0.008, i + off, f"{d.mean():.2f}", va="center", fontsize=7, color=S.INK)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[m] for m in order[::-1]]); ax.set_title(ttl, fontsize=9.5); ax.grid(axis="y", visible=False)
axes[0].axvline(0.5, color="#C9D1DA", ls="--", lw=1)
fig.legend(handles=[plt.Line2D([], [], color=S.NAVY, marker="o", lw=1.6), plt.Line2D([], [], color=S.TEAL, marker="o", lw=1.6)],
           labels=["raw metrics (log)", "within-release percentiles"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Train on JIRA, test on TeraPromise (mean of per-project means, 95% bootstrap interval)", x=0.01, ha="left", fontsize=9.5, fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIGS / "external_validation.png"); plt.close(fig)

# ----------------------------------------------------------------------------- figure: RF AUC by direction
fig, ax = plt.subplots(figsize=(6.4, 3.0))
rows_ = [("JIRA -> TeraPromise", fwd), ("TeraPromise -> JIRA", rev), ("In-domain (TeraPromise v -> v+1)", ind)]
for i, (lab, d) in enumerate(rows_[::-1]):
    for off, v, col in [(-0.16, "raw", S.NAVY), (0.16, "norm", S.TEAL)]:
        x = per_project(d[(d.variant == v) & (d.model == "RF")], "auc").values
        ci = np.percentile([rng.choice(x, len(x)).mean() for _ in range(3000)], [2.5, 97.5])
        ax.plot(ci, [i + off] * 2, color=col, lw=1.6); ax.plot(x.mean(), i + off, "o", color=col, ms=6, markeredgecolor="white", markeredgewidth=1)
        ax.text(ci[1] + 0.008, i + off, f"{x.mean():.2f}", va="center", fontsize=7.5)
ax.set_yticks(range(3)); ax.set_yticklabels([l for l, _ in rows_[::-1]]); ax.axvline(0.5, color="#C9D1DA", ls="--", lw=1); ax.grid(axis="y", visible=False)
ax.set_xlabel("ROC AUC (random forest)"); ax.set_title("Random forest across datasets", fontsize=9.5)
ax.legend(handles=[plt.Line2D([], [], color=S.NAVY, marker="o", lw=1.6), plt.Line2D([], [], color=S.TEAL, marker="o", lw=1.6)], labels=["raw", "percentile"], loc="center left")
fig.savefig(C.FIGS / "external_directions.png"); plt.close(fig)
