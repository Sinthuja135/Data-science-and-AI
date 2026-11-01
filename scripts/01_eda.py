"""preprocessing + exploratory data analysis """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config as C, data as D, plotstyle as S
from src.evaluate import cliffs_delta
from src.preprocess import Preprocessor

S.apply(); C.RESULTS.mkdir(exist_ok=True); C.FIGS.mkdir(exist_ok=True)
df = D.load_jira(); feats = D.feature_columns(df)
print("rows", len(df), "features", len(feats), "projects", df.project.nunique(), "releases", df.groupby(['project','release']).ngroups)

# ---- data quality summary
q = pd.DataFrame({"n_missing": df[feats].isna().sum(), "n_unique": df[feats].nunique(),
                  "min": df[feats].min(), "max": df[feats].max(), "skew": df[feats].skew()})
q["group"] = ["process" if f in C.PROCESS else "ownership" if f in C.OWNERSHIP else "code" for f in q.index]
q.to_csv(C.RESULTS / "eda_feature_quality.csv")
print("missing cells:", int(q.n_missing.sum()), "| constant metrics:", int((q.n_unique <= 1).sum()),
      "| negative-valued metrics:", int((q["min"] < 0).sum()), "| duplicated file rows within release:",
      int(df.duplicated(["project", "release", "File"]).sum()))

# ---- release-level label summary
rel = df.groupby(["project", "release_idx", "release"]).agg(
    files=("File", "size"), real_pos=(C.LABEL_REAL, "sum"), heu_pos=(C.LABEL_HEU, "sum")).reset_index()
rel["real_rate"] = rel.real_pos / rel.files; rel["heu_rate"] = rel.heu_pos / rel.files
both = df.groupby(["project", "release"]).apply(lambda g: pd.Series({
    "heu_only_fp": ((g.HeuBug == 1) & (g.RealBug == 0)).sum(), "real_only_fn": ((g.HeuBug == 0) & (g.RealBug == 1)).sum(),
    "both_pos": ((g.HeuBug == 1) & (g.RealBug == 1)).sum()}), include_groups=False).reset_index()
rel = rel.merge(both, on=["project", "release"]); rel.to_csv(C.RESULTS / "eda_release_summary.csv", index=False)
tot = rel[["files", "real_pos", "heu_pos", "heu_only_fp", "real_only_fn", "both_pos"]].sum()
print(tot.to_dict())
pd.DataFrame([{"total_files": tot.files, "real_defective": tot.real_pos, "heu_defective": tot.heu_pos,
               "heu_false_pos": tot.heu_only_fp, "heu_missed": tot.real_only_fn, "agree_pos": tot.both_pos,
               "heu_miss_rate": tot.real_only_fn / tot.real_pos, "heu_fp_share": tot.heu_only_fp / tot.heu_pos}]
             ).to_csv(C.RESULTS / "eda_label_agreement.csv", index=False)

# ---- Fig 1: defect rate per release, RealBug vs HeuBug
fig, ax = plt.subplots(figsize=(9.2, 3.6))
def _short(r):
    r = r.replace("1.3.0-incubating-beta-1", "1.3.0b1").replace("1.3.0-beta2", "1.3.0b2").replace("1_6_BETA_", "1.6b").replace("1_5_7", "1.5.7").replace("1.7.0.preview1", "1.7.0p1")
    return r
rel["lab"] = rel.project + " " + rel.release.map(_short)
x = np.arange(len(rel)); w = 0.4
ax.bar(x - w / 2, rel.real_rate * 100, w - 0.04, color=S.TEAL, label="RealBug (affected releases)")
ax.bar(x + w / 2, rel.heu_rate * 100, w - 0.04, color=S.AMBER, label="HeuBug (time-window heuristic)")
ax.set_xticks(x); ax.set_xticklabels(rel.lab, rotation=75, ha="right", fontsize=7)
ax.set_ylabel("Defective files (%)"); ax.set_title("Defect rate per release: two labelling methods disagree")
ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
fig.savefig(C.FIGS / "eda_defect_rate_per_release.png"); plt.close(fig)

# ---- Fig 2: label agreement (stacked)
fig, ax = plt.subplots(figsize=(5.4, 2.2))
parts = [("Both label defective", tot.both_pos, S.NAVY), ("Only RealBug (heuristic misses)", tot.real_only_fn, S.TEAL),
         ("Only HeuBug (heuristic false alarm)", tot.heu_only_fp, S.AMBER)]
left = 0
for name, v, c in parts:
    ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", linewidth=1.5)
    ax.text(left + v / 2, 0, f"{int(v):,}", ha="center", va="center", color="white" if c != S.AMBER else S.INK, fontweight="bold")
    left += v
ax.set_yticks([]); ax.set_xlabel("Files"); ax.grid(visible=False); ax.spines["left"].set_visible(False)
ax.set_title("Where the two labels disagree (all 32 releases)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], labels=[n for n, _, _ in parts],
          loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=1, fontsize=8)
fig.savefig(C.FIGS / "eda_label_agreement.png"); plt.close(fig)

# ---- Fig 3: distributions of key metrics, defective vs clean
key = ["CountLineCode", "SumCyclomatic", "CountClassCoupled", "COMM", "ADEV", "Added_lines"]
fig, axes = plt.subplots(2, 3, figsize=(9, 4.6))
for ax, f in zip(axes.ravel(), key):
    d0 = np.log1p(df.loc[df[C.LABEL_REAL] == 0, f].clip(lower=0)); d1 = np.log1p(df.loc[df[C.LABEL_REAL] == 1, f].clip(lower=0))
    bp = ax.boxplot([d0, d1], widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="white", lw=1.6))
    for patch, c in zip(bp["boxes"], [S.TEAL, S.AMBER]): patch.set_facecolor(c); patch.set_edgecolor(c)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["clean", "defective"]); ax.set_title(f, fontsize=9); ax.set_ylabel("log(1 + value)", fontsize=7.5)
    ax.grid(axis="x", visible=False)
fig.suptitle("Defective files are larger, more complex and more heavily changed", x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIGS / "eda_metric_distributions.png"); plt.close(fig)

# ---- Fig 4: Cliff's delta per metric (effect size, defective vs clean)
rows = []
samp = df.sample(20000, random_state=C.SEED)
pos = samp[samp[C.LABEL_REAL] == 1]; neg = samp[samp[C.LABEL_REAL] == 0].sample(min(4000, (samp[C.LABEL_REAL] == 0).sum()), random_state=C.SEED)
pos = pos.sample(min(len(pos), 2000), random_state=C.SEED)
for f in feats:
    if df[f].nunique() > 1: rows.append((f, cliffs_delta(pos[f].values, neg[f].values)))
eff = pd.DataFrame(rows, columns=["metric", "cliffs_delta"]).sort_values("cliffs_delta", key=abs, ascending=False)
eff["group"] = eff.metric.map(q.group); eff.to_csv(C.RESULTS / "eda_effect_sizes.csv", index=False)
top = pd.concat([eff[eff.group == g].head(5) for g in ["code", "process", "ownership"]]).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.2, 4))
colors = {"code": S.NAVY, "process": S.TEAL, "ownership": S.AMBER}
ax.barh(top.metric, top.cliffs_delta, color=[colors[g] for g in top.group], height=0.62)
ax.set_xlabel("Cliff's delta (defective vs clean; larger = more separating)"); ax.grid(axis="y", visible=False)
ax.set_title("Metrics that best separate defective files")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], labels=["code metrics", "process metrics", "ownership metrics"],
          loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3)
ax.set_title("Top 5 separating metrics in each group")
fig.savefig(C.FIGS / "eda_effect_sizes.png"); plt.close(fig)

# ---- Fig 5: correlation before/after pruning
prep = Preprocessor().fit(df[feats]); kept = prep.feature_names_
print("metrics: raw", len(feats), "-> after constant-drop + Spearman<=0.9 pruning", len(kept))
pd.Series(kept, name="kept_metric").to_csv(C.RESULTS / "eda_kept_metrics_all_data.csv", index=False)
corr = np.log1p(df[feats].clip(lower=0)).rank().corr().abs()
order = corr.mean().sort_values().index
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, cols, ttl in [(axes[0], list(order), f"Before: {len(feats)} metrics"), (axes[1], [c for c in order if c in kept], f"After pruning (|rho| > 0.9): {len(kept)} metrics")]:
    im = ax.imshow(corr.loc[cols, cols].values, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(ttl, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.8, label="|Spearman rho|"); fig.savefig(C.FIGS / "eda_correlation_pruning.png"); plt.close(fig)

# ---- Fig 6: files and imbalance per release pair used in modelling
pairs = pd.DataFrame([(p, a, b, len(tr), tr[C.LABEL_REAL].sum(), len(te), te[C.LABEL_REAL].sum()) for p, a, b, tr, te in D.release_pairs(df)],
                     columns=["project", "train", "test", "n_train", "pos_train", "n_test", "pos_test"])
pairs["test_rate"] = pairs.pos_test / pairs.n_test; pairs.to_csv(C.RESULTS / "eda_release_pairs.csv", index=False)
print("release pairs:", len(pairs), "| test positive rate range", round(pairs.test_rate.min() * 100, 1), "-", round(pairs.test_rate.max() * 100, 1),
      "| min positives train/test", pairs.pos_train.min(), pairs.pos_test.min())
print(eff.head(8).to_string(index=False))
