"""Step 5 (early preview for Milestone 3): train on heuristic vs affected-release labels, always test on RealBug.

For each release pair (selected with HeuBug on the training side), RF and class-weighted XGBoost are trained
once on HeuBug and once on RealBug, and both are scored against RealBug on the next release. Thresholds come
from out-of-fold predictions on the training release only.
Run from the project root:  python scripts/05_label_noise_preview.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from src import config as C, data as D, plotstyle as S
from src.models import build
from src.evaluate import oof_threshold, score, paired_test

S.apply()
C.RESULTS.mkdir(exist_ok=True); C.FIGS.mkdir(exist_ok=True)
df = D.load_jira(); feats = D.feature_columns(df)
MODELS = [("RF", "Random forest", S.NAVY), ("XGB_weighted", "XGBoost", S.TEAL)]

rows = []
pairs = list(D.release_pairs(df, label=C.LABEL_HEU))
for i, (proj, a, b, tr, te) in enumerate(pairs, 1):
    Xtr, Xte, yte = tr[feats], te[feats], te[C.LABEL_REAL].values
    for name, _, _ in MODELS:
        for lab in [C.LABEL_HEU, C.LABEL_REAL]:
            ytr = tr[lab].values
            m = build(name).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            rows.append(dict(project=proj, train=a, test=b, model=name, train_label=lab,
                             **score(yte, p, oof_threshold(m, Xtr, ytr))))
    print(f"[{i}/{len(pairs)}] {proj} {a}->{b} done", flush=True)
r = pd.DataFrame(rows); r.to_csv(C.RESULTS / "label_noise_pair_results.csv", index=False)
s = r.groupby(["model", "train_label"])[["auc", "pr_auc", "mcc_opt", "f1_opt"]].mean().round(3)
print(s); s.to_csv(C.RESULTS / "label_noise_summary.csv")

out = []
for name, _, _ in MODELS:
    for k in ["auc", "pr_auc", "mcc_opt"]:
        P = r[r.model == name].pivot_table(index=["project", "train", "test"], columns="train_label", values=k)
        out.append(dict(model=name, metric=k, **paired_test(P[C.LABEL_REAL].values, P[C.LABEL_HEU].values)))
t = pd.DataFrame(out); t.to_csv(C.RESULTS / "label_noise_tests.csv", index=False); print(t.round(4).to_string(index=False))

# ---- Figure: per model, left bar = trained on HeuBug (amber), right bar = trained on RealBug (model colour)
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
for ax, (k, ttl) in zip(axes, [("auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("mcc_opt", "MCC (tuned threshold)")]):
    for j, (name, _, c) in enumerate(MODELS):
        v = [s.loc[(name, C.LABEL_HEU), k], s.loc[(name, C.LABEL_REAL), k]]
        ax.bar([j * 3, j * 3 + 1], v, color=[S.AMBER, c], width=0.9)
        for x, val in zip([j * 3, j * 3 + 1], v): ax.text(x, val + 0.005, f"{val:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks([0.5, 3.5]); ax.set_xticklabels([lbl for _, lbl, _ in MODELS]); ax.set_title(ttl, fontsize=9.5); ax.grid(axis="x", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=S.AMBER)] + [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS]
labels = ["trained on HeuBug (left bar)"] + [f"{lbl} trained on RealBug (right bar)" for _, lbl, _ in MODELS]
fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f"All models tested on RealBug labels (mean over {len(pairs)} release pairs)", x=0.01, ha="left", fontsize=10, fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIGS / "label_noise_preview.png"); plt.close(fig)
print("saved label_noise_pair_results.csv, label_noise_summary.csv, label_noise_tests.csv and label_noise_preview.png")
