"""Step 11: label-noise study (HeuBug vs RealBug).

(a) train x test label matrix  (b) noise-injection dose-response
(c) importance shift           (d) profile of heuristic FP / FN files
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from src import config as C, data as D, plotstyle as S
from src.models import build
from src.evaluate import oof_threshold, score

S.apply(); df = D.load_jira(); feats = D.feature_columns(df)
R, H = C.LABEL_REAL, C.LABEL_HEU
pairs = list(D.release_pairs(df, label=R))
rng = np.random.default_rng(C.SEED)

# ----------------------------------------------------------------------------- (a) train label x test label (RF)
rows = []; imps = []
for proj, a, b, tr, te in pairs:
    for trl in [R, H]:
        m = build("RF").fit(tr[feats], tr[trl].values); thr = oof_threshold(m, tr[feats], tr[trl].values)
        p = m.predict_proba(te[feats])[:, 1]
        for tel in [R, H]:
            if te[tel].sum() >= 1 and te[tel].sum() < len(te):   # both classes present
                rows.append(dict(project=proj, train=a, test=b, train_label=trl, test_label=tel, **score(te[tel].values, p, thr)))
        imp = pd.Series(m.named_steps["clf"].feature_importances_, index=m.named_steps["prep"].feature_names_)
        imps.append(dict(project=proj, pair=f"{a}->{b}", train_label=trl, imp=imp / imp.sum()))
M = pd.DataFrame(rows); M.to_csv(C.RESULTS / "noise_matrix_pairs.csv", index=False)
mat = M.groupby(["train_label", "test_label"])[["auc", "pr_auc", "mcc_opt"]].mean().round(3); mat.to_csv(C.RESULTS / "noise_matrix.csv"); print(mat)

# ----------------------------------------------------------------------------- (b) noise injection (test on clean RealBug)
def inject(y, fn, fp):
    """Flip fn share of positives to 0 and fp share of negatives to 1."""
    y = y.copy(); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    y[rng.choice(pos, int(round(fn * len(pos))), replace=False)] = 0
    y[rng.choice(neg, int(round(fp * len(neg))), replace=False)] = 1
    return y
settings = [("missed defects (FN)", f, 0.0) for f in [0, .2, .4, .6, .71]] + \
           [("spurious defects (FP)", 0.0, q) for q in [0, .01, .02, .04, .08]] + \
           [("both (heuristic-like)", .71, .035)]
rows = []
for proj, a, b, tr, te in pairs:
    yte = te[R].values
    for kind, fn, fp in settings:
        for rep in range(2):
            ytr = inject(tr[R].values, fn, fp) if (fn or fp) else tr[R].values
            if ytr.sum() < 3: continue   # too few positives left
            m = build("RF").fit(tr[feats], ytr); thr = oof_threshold(m, tr[feats], ytr)
            rows.append(dict(project=proj, train=a, test=b, kind=kind, fn=fn, fp=fp, rep=rep, **score(yte, m.predict_proba(te[feats])[:, 1], thr)))
N = pd.DataFrame(rows); N.to_csv(C.RESULTS / "noise_injection_pairs.csv", index=False)
Ns = N.groupby(["kind", "fn", "fp"])[["auc", "pr_auc", "mcc_opt"]].mean().round(3); Ns.to_csv(C.RESULTS / "noise_injection_summary.csv"); print(Ns)

# ----------------------------------------------------------------------------- (c) importance shift Real vs Heu
I = pd.DataFrame({(d["pair"], d["train_label"]): d["imp"] for d in imps}).fillna(0)
rho = [spearmanr(I[(p, R)], I[(p, H)]).statistic for p in sorted({k[0] for k in I.columns})]
mr, mh = I.xs(R, axis=1, level=1).mean(1), I.xs(H, axis=1, level=1).mean(1)
shift = pd.DataFrame({"real": mr, "heu": mh}); shift["diff"] = shift.heu - shift.real
shift.sort_values("real", ascending=False).head(20).to_csv(C.RESULTS / "noise_importance_shift.csv")
top10 = [len(set(I[(p, R)].nlargest(10).index) & set(I[(p, H)].nlargest(10).index)) / 10 for p in sorted({k[0] for k in I.columns})]
pd.Series(dict(mean_spearman_real_vs_heu=np.mean(rho), mean_top10_overlap=np.mean(top10))).to_csv(C.RESULTS / "noise_importance_agreement.csv", header=["value"])
print("importance agreement", np.mean(rho), np.mean(top10))

# ----------------------------------------------------------------------------- (d) heuristic FP / FN profile
d = df.copy(); d["cat"] = np.select([(d[H] == 1) & (d[R] == 1), (d[H] == 1) & (d[R] == 0), (d[H] == 0) & (d[R] == 1)], ["agree_bug", "heuristic_FP", "heuristic_FN"], "agree_clean")
cols = [c for c in ["CountLineCode", "SumCyclomatic", "COMM", "ADEV", "DDEV", "Added_lines", "Del_lines", "OWN_COMM", "MINOR_COMM"] if c in d.columns]
prof = d.groupby("cat")[cols].median().T; prof["n"] = np.nan
prof.loc["n_files"] = d["cat"].value_counts().reindex(prof.columns).values
prof.to_csv(C.RESULTS / "noise_error_profile.csv"); print(prof)
cnt = d["cat"].value_counts(); print(cnt)
d["late"] = d.groupby("project")["release_idx"].transform(lambda s: s >= s.max() - 1)   # last 2 releases
fnrate = d[d[R] == 1].groupby("late")[H].apply(lambda s: 1 - s.mean()); fnrate.to_csv(C.RESULTS / "noise_fn_by_recency.csv", header=["heuristic_miss_rate"]); print(fnrate)

# ----------------------------------------------------------------------------- figure: dose-response + MCC matrix
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
ax = axes[0]
for kind, c in [("missed defects (FN)", S.NAVY), ("spurious defects (FP)", S.AMBER)]:
    s_ = N[N.kind == kind].groupby(["fn", "fp"]).mcc_opt.mean().reset_index(); x = (s_.fn if "FN" in kind else s_.fp) * 100
    ax.plot(x, s_.mcc_opt, "o-", color=c, label=("missed defects: % of true defects relabelled clean" if "FN" in kind else "spurious defects: % of clean files relabelled defective"))
ax.axhline(0.301, color=S.TEAL, ls="--", lw=1); ax.text(72, 0.306, "trained on real HeuBug", color=S.TEAL, fontsize=7, ha="right")   # reference MCC
ax.set_xlabel("% of labels corrupted in that class"); ax.set_ylabel("MCC on clean RealBug"); ax.set_title("Dose-response", fontsize=9.5); ax.legend(fontsize=6.5, loc="lower left")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax = axes[1]; mm = mat.mcc_opt.unstack(); im = ax.imshow(mm.values, cmap="Blues", vmin=0.2, vmax=0.4)
ax.set_xticks(range(len(mm.columns))); ax.set_xticklabels(["test " + ("Real" if c == R else "Heu") for c in mm.columns]); ax.set_yticks(range(len(mm.index))); ax.set_yticklabels(["train " + ("Real" if c == R else "Heu") for c in mm.index])
for i in range(mm.shape[0]):
    for j in range(mm.shape[1]): ax.text(j, i, f"{mm.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("MCC by train / test label", fontsize=9.5); ax.grid(False)
fig.tight_layout(); fig.savefig(C.FIGS / "label_noise_study.png"); plt.close(fig)
