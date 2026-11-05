"""Step 6: system architecture diagram.

Each box lists the files that implement it; a box is drawn solid only if those files exist in this checkout,
otherwise dashed as "planned". So the same script gives an accurate diagram at every stage of the project
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from src import config as C, plotstyle as S

DOCS = C.ROOT / "docs"
DOCS.mkdir(exist_ok=True); C.FIGS.mkdir(exist_ok=True)

def has(*paths):
    """True if every path exists. 'scripts/NN' matches any scripts/NN_*.py."""
    def ok(p):
        if p.startswith("scripts/") and p[8:].isdigit():
            return any((C.ROOT / "scripts").glob(f"{p[8:]}_*.py"))
        return (C.ROOT / p).exists()
    return all(ok(p) for p in paths)

def scripts_present(nums):
    return [n for n in nums if has(f"scripts/{n}")]

fig, ax = plt.subplots(figsize=(13.6, 5.4)); ax.set_xlim(0, 130); ax.set_ylim(0, 58); ax.axis("off")

def box(x, y, w, h, title, lines, fc="white", ec=S.NAVY, tc=S.NAVY, dashed=False, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.1", fc=fc, ec=ec, lw=1.4, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=8.3, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h - 4.4, "\n".join(lines), ha="center", va="top", fontsize=fs, color=S.INK, linespacing=1.4)

def arrow(x1, y1, x2, y2, dashed=False, c=S.NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.4, color=c, ls="--" if dashed else "-"))

PLANNED = dict(fc="white", ec=S.GREY, tc="#5B6673", dashed=True)   # style for components not built yet
W, H, G = 16.8, 20, 1.5
X = [1 + i * (W + G) for i in range(7)]
ax.text(1, 56, "MAIN PIPELINE (implemented)", fontsize=8.5, fontweight="bold", color=S.TEAL)
ax.text(1, 28.2, "SUPPORTING WORK", fontsize=8.5, fontweight="bold", color=S.TEAL)
ax.plot([0, 130], [27.2, 27.2], color="#D5DBE1", lw=1)
TY, BY = 33, 4

models_src = (C.ROOT / "src" / "models.py").read_text(encoding="utf-8") if has("src/models.py") else ""
improved = (["Improved: XGBoost,", "LightGBM, weights,", "SMOTE, tuning"] if "LGBM_weighted" in models_src else
            ["Improved: XGBoost +", "class weights, tuned RF,", "RF + XGB ensemble"])
AMB = dict(fc="#FFF4D6", ec=S.AMBER, tc="#8A6100"); TEA = dict(fc="#E6F3F3", ec=S.TEAL, tc=S.TEAL)
top = [  # (title, lines, style, implemented?)
 ("JIRA dataset", ["Zenodo benchmark v1.0", "9 Apache projects", "32 releases, 73,395 files", "65 metrics, 2 labels"], TEA, True),
 ("Loader + splitter", ["src/data.py", "chronological releases", "train on t, test on t+1", "23 release pairs"], {}, has("src/data.py")),
 ("Preprocessor", ["src/preprocess.py", "fit on train release only", "drop constant, log(1+x)", "prune correlated: 65 to ~41"], {}, has("src/preprocess.py")),
 ("Modelling", ["src/models.py", "Baselines: LOC logistic,", "logistic, random forest"] + improved, AMB, has("src/models.py")),
 ("Evaluation", ["src/evaluate.py", "AUC, PR-AUC, MCC, F1", "OOF-tuned threshold", "Wilcoxon, Cliff's delta,", "bootstrap 95% CI"], {}, has("src/evaluate.py")),
 ("Scoring pipeline", ["src/score.py, scripts 07-08", "save model (joblib)", "score a new release:", "ranked files + SHAP", "reasons"], AMB, has("src/score.py", "scripts/07", "scripts/08")),
 ("Outputs", ["results/*.csv", "figures/*.png", "top-risk file list", "reports, demo video"], {}, True),
]
any_planned = False
for x, (t, l, kw, done) in zip(X, top):
    any_planned |= not done
    box(x, TY, W, H, t, l if done else l + ["(planned)"], **(kw if done else PLANNED))
for i in range(6): arrow(X[i] + W + 0.15, TY + H / 2, X[i + 1] - 0.15, TY + H / 2)

last_script = scripts_present([f"{n:02d}" for n in range(1, 16)])[-1]
shap_scripts = scripts_present(["04", "05", "11", "12"])
bottom = {  # column: (title, lines, style, arrow direction, implemented?)
 1: ("EDA", ["scripts/01_eda.py", "label agreement,", "imbalance, effect sizes"], {}, "down", has("scripts/01")),
 2: ("GitHub release tags", ["architecture / dependency", "metrics (stretch goal)"], dict(dashed=True, ec=S.AMBER, tc="#8A6100"), "up", True),
 3: ("External validation", ["script 10, src/external.py", "TeraPromise, 9 shared", "metrics; both directions;", "cross-project, script 13"], TEA, "up", has("src/external.py", "scripts/10")),
 4: ("SHAP + label noise", ["scripts " + ", ".join(shap_scripts or ["04, 05, 11, 12"]), "explanation stability,", "HeuBug vs RealBug,", "noise injection"], TEA, "down", bool(shap_scripts)),
 5: ("Effort-aware check", ["script 09", "defects found when only", "20% of files / code is", "reviewed"], TEA, "down", has("scripts/09")),
 6: ("Reproducibility", ["seed 42, requirements.txt", f"scripts 01 to {last_script}", "public repo, commits"], {}, "up", has("requirements.txt")),
}
for i, (t, l, kw, d, done) in bottom.items():
    any_planned |= not done
    kw = kw if done else PLANNED
    box(X[i], BY, W, 17, t, l if done else l + ["(planned)"], **kw)
    cx = X[i] + W / 2
    if d == "down": arrow(cx, TY - 0.4, cx, BY + 17.6, dashed=not done, c=S.NAVY if done else S.GREY)
    else: arrow(cx, BY + 17.6, cx, TY - 0.4, dashed=kw.get("dashed", False), c=kw.get("ec", S.NAVY))
legend = "Solid box: done\nDashed box: stretch goal\n(not implemented)\nAmber: improved / final"
if any_planned: legend += "\nGrey dashed: planned"
ax.text(X[0], 12, legend, fontsize=7.2, color="#5B6673", va="center", linespacing=1.5)
for out in [DOCS / "architecture.png", C.FIGS / "architecture.png"]: fig.savefig(out)
print("saved", DOCS / "architecture.png", "and", C.FIGS / "architecture.png")
