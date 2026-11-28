"""Score one .java file against every project's model (helps pick its ci_project_map.json entry).

    python scripts/check_file_projects.py examples/MyNewFile.java
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import joblib
from src import config as C
from src.score import score_new_file

if len(sys.argv) != 2:
    print("Usage: python scripts/check_file_projects.py <path/to/File.java>")
    sys.exit(1)

java_path = Path(sys.argv[1])
if not java_path.exists():
    print(f"[error] file not found: {java_path}")
    sys.exit(1)

print(f"Scoring {java_path} against every project's model:\n")
print(f"{'Project':<12}{'Risk':>8}{'Threshold':>12}   Flagged")
print("-" * 46)
for project in C.RELEASE_ORDER:
    model_path = C.MODELS / f"{project}.joblib"
    if not model_path.exists():
        print(f"{project:<12}{'--':>8}{'--':>12}   (no saved model -- train it first)")
        continue
    art = joblib.load(model_path)
    view = score_new_file(project, java_path, top=1, model_path=model_path, explain=False, quiet=True)
    r = view.iloc[0]
    flag = "warning: yes" if r["flagged"] else "no"
    print(f"{project:<12}{r['risk']:>8.3f}{art['threshold']:>12.3f}   {flag}")

print("\nPick whichever project(s) you want this file to be flagged (or not flagged) under, and add it "
      "to ci_project_map.json, e.g.:\n"
      f'  "{java_path.as_posix()}": "<project>"')
