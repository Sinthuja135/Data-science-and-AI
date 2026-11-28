"""CI check: score every changed .java file in a pull request against a saved model, and fail the build if
any of them looks risky. This is a CI utility, not one of the numbered milestone experiments -- run it from
a pull-request / pre-merge job, not as part of run_all.py.

What it does
    1. Finds every .java file changed between --base and --head (default: origin/main..HEAD) in --repo.
    2. Works out which project's model each changed file should be scored against (see "Multiple projects
       in one PR" below) -- most repos only ever use one project, which is still the default.
    3. Scores each file with score_new_file() -- the same best-effort single-file analysis behind
       `08_score_release.py --new-file` (see src/javafile.py for exactly what's computed vs assumed vs
       imputed for a lone file with no whole-codebase or git-history context).
    4. Writes a Markdown table (for pasting into a PR comment) and a CSV to --report.
    5. Exits 1 (fails the CI job) if any changed file is flagged by its model's own decision threshold, or
       exits 0 otherwise. Use --fail-on never if you only want the report, not a build failure.

Multiple projects in one PR
    A single run of this script used to force every changed file to be scored against the same
    --project. That's wrong for a PR that touches files belonging to different projects (different
    modules of a monorepo, or -- in this capstone's case -- a demo PR that deliberately mixes a
    camel-shaped example with a derby-shaped one). Instead:
      - --project-map points at a JSON file mapping a file path (relative to --repo) or a glob pattern
        to a project name, e.g.:
            {
              "examples/ExtremeRisk.java": "derby",
              "examples/*.java": "camel"
            }
        Patterns are checked in file order, first match wins, so put more specific entries first.
      - A changed file that matches nothing in the map falls back to --project (still "camel" by
        default, so a repo that only ever uses one project doesn't need a map file at all).
      - If --project-map is omitted, the script looks for ci_project_map.json in the repo root and uses
        it automatically if present; otherwise every file just uses --project, exactly as before.
      - The report gets a "Project" column so it's clear which model scored which file.

Example (local, single project, no map needed)
    python scripts/ci_check.py --project camel --repo . --base origin/main

Example (local, mixed projects via map file)
    python scripts/ci_check.py --repo . --base origin/main --project-map ci_project_map.json --project camel

Example (GitHub Actions step) -- see .github/workflows/defect-risk-check.yml for the full workflow:
    - run: |
        pip install -r requirements.txt
        python scripts/ci_check.py --repo . \
            --base ${{ github.event.pull_request.base.sha }} --head ${{ github.event.pull_request.head.sha }}

Note: only .java files are handled (the single-file analyzer parses Java). Non-Java changed files are
listed as skipped in the report, not silently ignored.

Note on models: this script loads an already-trained models/<project>.joblib -- it never trains one itself,
since CI doesn't have the raw benchmark data (data/raw is gitignored and not fetched automatically). Commit
the model file(s) you want CI to use (override the models/*.joblib line in .gitignore for those specific
projects), or have an earlier CI step restore it from a cache/artifact, before this script runs. With
multiple projects in play, commit a model file for each project named in the map (or in --project).
"""
import sys
import json
import fnmatch
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
ap.add_argument("--project", default="camel", choices=list(C.RELEASE_ORDER),
                 help="fallback/default model for any changed file --project-map doesn't cover "
                      "(or the only model used, if you're not scoring multiple projects in this PR)")
ap.add_argument("--repo", required=True, help="path to the git checkout being checked (often '.')")
ap.add_argument("--base", default="origin/main", help="diff base -- the branch/commit this PR is merging into")
ap.add_argument("--head", default="HEAD", help="diff head -- the PR's own branch/commit")
ap.add_argument("--model-path", default=None,
                 help="override path to a single .joblib model, used for every file (default: look up "
                      "models/<project>.joblib per file instead -- don't combine this with --project-map)")
ap.add_argument("--project-map", default=None,
                 help="JSON file mapping a changed file's path or glob pattern to a project name (see "
                      "module docstring). Default: ci_project_map.json in the repo root, if it exists.")
ap.add_argument("--report", default=str(C.RESULTS / "ci_risk_report.md"), help="where to write the Markdown report")
ap.add_argument("--fail-on", choices=["flagged", "never"], default="flagged",
                 help="'flagged' (default): exit 1 if any changed file crosses its model's decision threshold. "
                      "'never': always exit 0, just write the report.")
args = ap.parse_args()

repo = Path(args.repo).resolve()


def load_project_map():
    """Returns an ordered list of (pattern, project) pairs from --project-map, or the repo-root
    ci_project_map.json if --project-map wasn't given and that file exists. Empty list if neither."""
    map_path = Path(args.project_map) if args.project_map else (repo / "ci_project_map.json")
    if not map_path.exists():
        if args.project_map:
            print(f"[error] --project-map given but not found at {map_path}")
            sys.exit(2)
        return []
    try:
        raw = json.loads(map_path.read_text())
    except json.JSONDecodeError as e:
        print(f"[error] {map_path} is not valid JSON: {e}")
        sys.exit(2)
    print(f"Using project map {map_path} ({len(raw)} entr{'y' if len(raw) == 1 else 'ies'})")
    return list(raw.items())


def resolve_project(rel_path, project_map):
    """Which project's model should score this file: first matching pattern in project_map (exact path
    or glob, checked in file order), else the --project fallback."""
    rel_str = rel_path.replace("\\", "/")   # map files are written with forward slashes regardless of OS
    for pattern, project in project_map:
        if rel_str == pattern or fnmatch.fnmatch(rel_str, pattern):
            return project
    return args.project


def changed_java_files():
    """Every file changed between --base and --head in `repo`, split into (.java files, everything else).
    --diff-filter=ACMR: only Added/Copied/Modified/Renamed files -- a file that was just deleted can't be
    scored (there's nothing left on disk to read)."""
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
    project_map = load_project_map()
    java_files, skipped = changed_java_files()
    print(f"{len(java_files)} changed .java file(s) between {args.base} and {args.head} "
          f"({len(skipped)} non-Java file(s) skipped)")

    # Which model file each project needs, checked once up front (not per file) so a missing model gives
    # one clear message instead of repeating the same error on every file that project would have scored.
    projects_needed = sorted({resolve_project(f, project_map) for f in java_files})
    model_paths = {}
    missing = []
    for project in projects_needed:
        mp = Path(args.model_path) if args.model_path else (C.MODELS / f"{project}.joblib")
        model_paths[project] = mp
        if not mp.exists():
            missing.append((project, mp))
    if missing:
        for project, mp in missing:
            print(f"[error] no saved model at {mp} for project '{project}' -- train it first "
                  f"(scripts/07_train_final_models.py), commit it, or fix the mapping in --project-map.")
        if len(missing) == len(projects_needed):
            sys.exit(2)   # nothing at all can be scored -- that's a setup problem, not "risky files found"

    rows = []
    for rel_path in java_files:
        project = resolve_project(rel_path, project_map)
        full_path = repo / rel_path
        if not full_path.exists():
            rows.append({"File": rel_path, "Project": project, "risk": None, "flagged": None,
                         "why it's risky": "file not found on disk (deleted or renamed away?)"})
            continue
        model_path = model_paths.get(project)
        if model_path is None or not model_path.exists():
            rows.append({"File": rel_path, "Project": project, "risk": None, "flagged": None,
                         "why it's risky": f"no saved model for project '{project}' -- see errors above"})
            continue
        try:
            view = score_new_file(project, full_path, top=1, model_path=model_path, quiet=True)
            r = view.iloc[0]
            rows.append({"File": rel_path, "Project": project, "risk": round(float(r["risk"]), 3),
                         "flagged": bool(r["flagged"]), "why it's risky": r.get("why it's risky", ""),
                         "suggested fix": r.get("suggested fix", "")})
        except Exception as e:
            # A file that fails to parse (syntax error, non-standard Java) shouldn't crash the whole CI run --
            # report it as a row the reviewer can see, rather than silently dropping it or failing the build
            # for an unrelated reason.
            rows.append({"File": rel_path, "Project": project, "risk": None, "flagged": None,
                         "why it's risky": f"could not score: {e}"})

    df = pd.DataFrame(rows)
    df.to_csv(str(Path(args.report).with_suffix(".csv")), index=False)

    heading = f"### Defect risk check ({', '.join(projects_needed)} model{'s' if len(projects_needed) != 1 else ''})\n" \
        if projects_needed else "### Defect risk check\n"
    lines = [heading]
    if df.empty:
        lines.append("No changed `.java` files to score.\n")
    else:
        lines.append("| File | Project | Risk | Flagged | Why it's risky | Suggested fix |")
        lines.append("|---|---|---|---|---|---|")
        # Built straight from `rows` (the plain list of dicts), not df.iterrows(): pandas silently upcasts
        # a row's values to one common dtype across ALL its columns when you iterate rows, so a Python
        # `None` sitting in the "flagged" column (file not scored) would turn into float `nan` just
        # because the neighbouring "risk" column is float64 -- and bool(nan) is True in Python, which
        # showed up as a false "warning: yes" on rows that were never actually scored. Reading straight
        # from the original dicts avoids that entirely.
        for r in rows:
            risk_cell = r["risk"] if r["risk"] is not None else "-"
            flagged_cell = "warning: yes" if r["flagged"] else "no"   # None and False both render as "no"
            why_cell = r.get("why it's risky", "") or "-"
            fix_cell = r.get("suggested fix", "") or "-"
            lines.append(f"| `{r['File']}` | {r['Project']} | {risk_cell} | {flagged_cell} | {why_cell} | {fix_cell} |")
    if skipped:
        lines.append(f"\n*{len(skipped)} non-Java file(s) changed too, not scored: "
                      + ", ".join(f"`{s}`" for s in skipped[:10]) + ("..." if len(skipped) > 10 else "") + "*")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"Report written to {args.report}")

    n_flagged = int(df["flagged"].fillna(False).sum()) if not df.empty else 0
    if args.fail_on == "flagged" and n_flagged > 0:
        print(f"[FAIL] {n_flagged} changed file(s) flagged as high defect risk.")
        sys.exit(1)
    print(f"[OK] {n_flagged} changed file(s) flagged.")
    sys.exit(0)


if __name__ == "__main__":
    main()
