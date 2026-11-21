"""score a release with a saved model (the scoring pipeline)
"""
import sys, argparse, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from src import config as C
from src.score import run_scoring, score_new_file

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="camel", choices=list(C.RELEASE_ORDER))
ap.add_argument("--input", default=None, help="CSV of a new release (same columns as the JIRA files)")
ap.add_argument("--new-file", default=None,
                 help="score ONE real .java file on disk instead of a release/folder -- see note above")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--no-explain", action="store_true")
ap.add_argument("--raw-shap", action="store_true", help="show raw metric=value SHAP output instead of plain English")
ap.add_argument("--repo", default=None,
                 help="path to a real git checkout of the scored project; adds a 'likely developer' column "
                      "looked up from git history (not available for the JIRA benchmark CSVs -- see note above)")
ap.add_argument("--changed-since", default=None,
                 help="git ref (tag/commit) in --repo; only score files that changed since it, e.g. the "
                      "previous release tag -- requires --repo")
ap.add_argument("--head", default="HEAD", help="git ref to diff up to with --changed-since (default: HEAD)")
a = ap.parse_args()
if a.changed_since and not a.repo:
    ap.error("--changed-since needs --repo (a git checkout to diff against)")
if not (C.MODELS / f"{a.project}.joblib").exists():
    ap.error(f"no saved model for {a.project}: run  python scripts/07_train_final_models.py  first")
if a.new_file:
    score_new_file(a.project, a.new_file, a.top, explain=not a.no_explain, plain=not a.raw_shap)
else:
    run_scoring(a.project, a.input, a.top, explain=not a.no_explain, plain=not a.raw_shap, repo=a.repo,
                changed_since=a.changed_since, head_ref=a.head)
