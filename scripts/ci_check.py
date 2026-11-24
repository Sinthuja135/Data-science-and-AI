"""CI check: score changed .java files in a PR with a saved model; fail if any is flagged.

Writes a Markdown + CSV report. Exit codes: 0 ok, 1 file flagged, 2 setup error.
Needs models/<project>.joblib committed (CI has no raw data to train).

    python scripts/ci_check.py --project camel --repo . --base origin/main
"""
import sys
import argparse
import subprocess
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C
from src.score import score_new_file

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="camel", choices=list(C.RELEASE_ORDER), help="saved model to use")
ap.add_argument("--repo", required=True, help="git checkout to check (often '.')")
ap.add_argument("--base", default="origin/main", help="diff base (PR target)")
ap.add_argument("--head", default="HEAD", help="diff head (PR branch)")
ap.add_argument("--model-path", default=None, help="model path (default: models/<project>.joblib)")
ap.add_argument("--report", default=str(C.RESULTS / "ci_risk_report.md"), help="Markdown report path")
ap.add_argument("--fail-on", choices=["flagged", "never"], default="flagged",
                 help="'flagged': exit 1 if any file is flagged; 'never': report only")
args = ap.parse_args()

repo = Path(args.repo).resolve()
model_path = Path(args.model_path) if args.model_path else (C.MODELS / f"{args.project}.joblib")
if not model_path.exists():
    print(f"[error] no saved model at {model_path} -- train it first (scripts/07_train_final_models.py) "
          f"and commit it, or point --model-path at a cached copy.")
    sys.exit(2)   # setup error


def changed_java_files():
    """(java, other) files changed in base...head; deleted files excluded (ACMR)."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{args.base}...{args.head}"],
        cwd=str(repo), capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        print(f"[error] git diff failed: {out.stderr.strip()}")
        sys.exit(2)
    changed = [c.strip() for c in out.stdout.splitlines() if c.strip()]
    java = [c for c in changed if c.endswith(".java")]
    other = [c for c in changed if not c.endswith(".java")]
    return java, other


def main():
    C.RESULTS.mkdir(exist_ok=True)
    java_files, skipped = changed_java_files()
    print(f"{len(java_files)} changed .java file(s) between {args.base} and {args.head} "
          f"({len(skipped)} non-Java file(s) skipped)")

    # score each changed file
    rows = []
    for rel_path in java_files:
        full_path = repo / rel_path
        if not full_path.exists():
            rows.append({"File": rel_path, "risk": None, "flagged": None, "why it's risky": "file not found on disk (deleted or renamed away?)"})
            continue
        try:
            view = score_new_file(args.project, full_path, top=1, model_path=model_path, quiet=True)
            r = view.iloc[0]
            rows.append({"File": rel_path, "risk": round(float(r["risk"]), 3), "flagged": bool(r["flagged"]),
                         "why it's risky": r.get("why it's risky", ""), "suggested fix": r.get("suggested fix", "")})
        except Exception as e:
            # parse error: report it, don't crash the run
            rows.append({"File": rel_path, "risk": None, "flagged": None, "why it's risky": f"could not score: {e}"})

    df = pd.DataFrame(rows)
    df.to_csv(str(Path(args.report).with_suffix(".csv")), index=False)

    # Markdown report
    lines = [f"### Defect risk check ({args.project} model)\n"]
    if df.empty:
        lines.append("No changed `.java` files to score.\n")
    else:
        lines.append("| File | Risk | Flagged | Why it's risky | Suggested fix |")
        lines.append("|---|---|---|---|---|")
        for _, r in df.iterrows():
            risk_cell = r["risk"] if r["risk"] is not None else "-"
            flagged_cell = "no" if r["flagged"] is None else ("warning: yes" if r["flagged"] else "no")
            why_cell = r.get("why it's risky", "") or "-"
            fix_cell = r.get("suggested fix", "") or "-"
            lines.append(f"| `{r['File']}` | {risk_cell} | {flagged_cell} | {why_cell} | {fix_cell} |")
    if skipped:
        lines.append(f"\n*{len(skipped)} non-Java file(s) changed too, not scored: "
                      + ", ".join(f"`{s}`" for s in skipped[:10]) + ("..." if len(skipped) > 10 else "") + "*")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"Report written to {args.report}")

    # exit code
    n_flagged = int(df["flagged"].fillna(False).sum()) if not df.empty else 0
    if args.fail_on == "flagged" and n_flagged > 0:
        print(f"[FAIL] {n_flagged} changed file(s) flagged as high defect risk.")
        sys.exit(1)
    print(f"[OK] {n_flagged} changed file(s) flagged.")
    sys.exit(0)


if __name__ == "__main__":
    main()
