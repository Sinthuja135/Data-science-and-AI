"""Final model training and release scoring (used by scripts 07, 08 and ci_check).

train_final     -> fit model + decision threshold on one release
score_release   -> rank files of a release by defect risk
explain_top     -> raw SHAP reasons (metric=value (+shap))
explain_plain   -> plain-English reason + suggested fix
filter_changed  -> keep only files changed since a git ref
select_files    -> keep only named files (no repo needed)
likely_author   -> most frequent recent git author of a file
score_new_file  -> score one new .java file (see src/javafile.py)
ranking_metrics -> precision@k and recall at a review budget
run_scoring     -> load model, score, print, plot, save
"""
from pathlib import Path
from .common import joblib, np, pd, plt
from . import config as C
from . import plotstyle as S
from .models import build
from .evaluate import oof_threshold


# ============================================================================= TRAINING
def train_final(train_df, feats, label=C.LABEL_REAL, seed=C.SEED, name="RF", **meta):
    """Fit model `name` and pick its threshold from OOF predictions. Returns a picklable dict;
    extra **meta (e.g. project, releases) is stored as-is."""
    X, y = train_df[feats], train_df[label].values
    model = build(name, seed).fit(X, y)
    thr = oof_threshold(model, X, y, seed)   # MCC-optimal cutoff
    return dict(model=model, threshold=float(thr), features=list(feats), model_name=name, seed=seed,
                n_train=int(len(train_df)), n_pos=int(y.sum()), **meta)


def save_model(art, path):
    """Save the train_final() dict to disk."""
    joblib.dump(art, path)


# ============================================================================= SCORING
def score_release(art, df, label=C.LABEL_REAL):
    """Score every row of `df` and return them ranked riskiest first.
    Needs "File" + model features; label column (if present) is kept as actual_defective."""
    p = art["model"].predict_proba(df[art["features"]])[:, 1]   # P(defective)
    out = pd.DataFrame({
        "row": np.arange(len(df)),           # original row index
        "File": df["File"].values,
        "risk": p,
        "flagged": p >= art["threshold"],
        "loc": df["CountLineCode"].values,
    })
    if label in df.columns:
        out["actual_defective"] = df[label].astype(int).values
    out = out.sort_values("risk", ascending=False, kind="stable").reset_index(drop=True)   # stable = reproducible ties
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ============================================================================= EXPLANATIONS
def explain_top(art, df, ranked, n=10, k=3):
    """Top-k SHAP metrics pushing risk up for the n riskiest files (raw form)."""
    import shap   # optional, heavy dependency
    rows = ranked["row"].values[:n]
    prep, rf = art["model"].named_steps["prep"], art["model"].named_steps["clf"]
    Xt = prep.transform(df.iloc[rows][art["features"]])
    sv = shap.TreeExplainer(rf).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)   # positive class, any shap version
    names = list(prep.feature_names_)
    reasons = []
    for i, r in enumerate(rows):
        top = np.argsort(-sv[i])[:k]
        reasons.append("; ".join(f"{names[j]}={df.iloc[r][names[j]]:.0f} ({sv[i, j]:+.2f})" for j in top))
    return reasons


# category -> (plain-English phrase, suggested fix)
CATEGORY_INFO = {
    "complexity": ("has complex, branch-heavy logic",
                   "Break the most complex methods into smaller, single-purpose ones and add tests around the branches."),
    "size":       ("is a large file with a lot of code crammed into it",
                   "Consider splitting this file/class into smaller, more focused ones."),
    "comments":   ("is thin on documentation relative to its size",
                   "Add comments/docstrings around the non-obvious parts before merging further changes."),
    "coupling":   ("is tightly coupled to a lot of other classes",
                   "Reduce dependencies on other classes, or introduce an interface to decouple it."),
    "cohesion":   ("does too many unrelated things (low cohesion)",
                   "Split unrelated responsibilities out into separate classes."),
    "paths":      ("has many nested branches and execution paths that are hard to fully test",
                   "Add unit tests covering the edge-case branches, and simplify the control flow where possible."),
    "churn":      ("has changed a lot in this release",
                   "Give it extra code review and regression testing, since it changed heavily this release."),
    "developers": ("was touched by many different developers recently",
                   "Assign one clear owner/reviewer for this file to avoid conflicting changes."),
    "own_single": ("was written mostly by a single main contributor",
                   "Get a second reviewer on this file, since only one person is familiar with it."),
    "own_minor":  ("has a lot of small, scattered edits from occasional contributors",
                   "Have the main owner review the recent minor changes for consistency."),
}
# metric name -> category
METRIC_CATEGORY = {}
for _cat, _names in {
    "complexity": ["MaxCyclomatic", "MaxCyclomaticStrict", "MaxCyclomaticModified", "SumCyclomatic",
                   "SumCyclomaticStrict", "SumCyclomaticModified", "AvgCyclomatic", "AvgCyclomaticStrict",
                   "AvgCyclomaticModified", "SumEssential", "AvgEssential"],
    "size": ["CountLine", "CountLineCode", "CountLineCodeDecl", "CountLineCodeExe", "CountStmt", "CountStmtDecl",
              "CountStmtExe", "CountSemicolon", "AvgLine", "AvgLineCode", "CountDeclFunction", "CountDeclMethod",
              "CountDeclMethodPublic", "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodDefault",
              "CountDeclInstanceMethod", "CountDeclClass", "CountDeclClassMethod", "CountDeclClassVariable",
              "CountDeclInstanceVariable"],
    "comments": ["CountLineComment", "RatioCommentToCode", "AvgLineComment", "CountLineBlank", "AvgLineBlank"],
    "coupling": ["CountClassCoupled", "CountClassDerived", "CountClassBase", "MaxInheritanceTree",
                  "CountInput_Max", "CountInput_Mean", "CountInput_Min", "CountOutput_Max", "CountOutput_Mean",
                  "CountOutput_Min"],
    "cohesion": ["PercentLackOfCohesion"],
    "paths": ["CountPath_Max", "CountPath_Mean", "CountPath_Min", "MaxNesting_Max", "MaxNesting_Mean",
               "MaxNesting_Min"],
    "churn": ["COMM", "Added_lines", "Del_lines"],
    "developers": ["ADEV", "DDEV"],
    "own_single": ["OWN_LINE", "OWN_COMMIT", "MAJOR_COMMIT", "MAJOR_LINE"],
    "own_minor": ["MINOR_COMMIT", "MINOR_LINE"],
}.items():
    for _name in _names:
        METRIC_CATEGORY[_name] = _cat


def explain_plain(art, df, ranked, n=10, k=3):
    """Plain-English reason + suggested fix for the n riskiest files. Returns (reasons, actions)."""
    import shap
    rows = ranked["row"].values[:n]
    prep, rf = art["model"].named_steps["prep"], art["model"].named_steps["clf"]
    Xt = prep.transform(df.iloc[rows][art["features"]])
    sv = shap.TreeExplainer(rf).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    names = list(prep.feature_names_)
    reasons, actions = [], []
    for i in range(len(rows)):
        order = np.argsort(-sv[i])
        cats = []
        for j in order:
            if sv[i, j] <= 0:
                continue   # only risk-increasing metrics
            cat = METRIC_CATEGORY.get(names[j])
            if cat and cat not in cats:
                cats.append(cat)
            if len(cats) >= k:
                break
        if not cats:
            reasons.append("No single risk driver stands out; the overall metric pattern is just mildly unusual for this project.")
            actions.append("Give it a normal review; nothing specific jumps out from the metrics alone.")
            continue
        phrases = [CATEGORY_INFO[c][0] for c in cats]
        lead = "Flagged mainly because it " if ranked["flagged"].values[i] else \
               "Not flagged (below the decision threshold); what raises its risk most is that it "
        reasons.append(lead + (phrases[0] if len(phrases) == 1 else
                        ", ".join(phrases[:-1]) + " and " + phrases[-1]) + ".")   # "a, b and c"
        acts, seen = [], set()
        for c in cats:
            a = CATEGORY_INFO[c][1]
            if a not in seen:
                seen.add(a); acts.append(a)
        actions.append(" ".join(acts[:2]))   # max 2 suggestions
    return reasons, actions


# ============================================================================= FILTERING
def filter_changed(df, repo_path, base_ref, head_ref="HEAD"):
    """Keep only files changed in `git diff base_ref..head_ref` (suffix match).
    Returns `df` unchanged with a warning if git fails or nothing matches."""
    import subprocess
    try:
        out = subprocess.run(["git", "diff", "--name-only", f"{base_ref}..{head_ref}"],
                              cwd=str(repo_path), capture_output=True, text=True, timeout=15)
        changed = [c.strip() for c in out.stdout.splitlines() if c.strip()]
    except Exception as e:
        print(f"  [warning] could not read git diff in {repo_path} ({e}); scoring the full input instead")
        return df
    if not changed:
        print(f"  [warning] git diff {base_ref}..{head_ref} found no changed files; scoring the full input instead")
        return df
    changed = [c.replace("\\", "/") for c in changed]

    def matches(f):
        f = str(f).replace("\\", "/")
        return any(c.endswith(f) or f.endswith(c) for c in changed)   # suffix match either way

    mask = df["File"].apply(matches)
    if not mask.any():
        print(f"  [warning] none of the {len(changed)} changed file(s) matched rows in the input; "
              f"scoring the full input instead")
        return df
    print(f"  Restricting to {int(mask.sum())} changed file(s) out of {len(df)} (git diff {base_ref}..{head_ref})")
    return df[mask].reset_index(drop=True)


def select_files(df, files):
    """Keep only the named file(s) (str or list, suffix match). Raises ValueError if none match."""
    wanted = [files] if isinstance(files, str) else list(files)
    wanted_norm = [str(w).replace("\\", "/") for w in wanted]

    def matches(f):
        f = str(f).replace("\\", "/")
        return any(f.endswith(w) or w.endswith(f) for w in wanted_norm)

    mask = df["File"].apply(matches)
    if not mask.any():
        raise ValueError(f"none of {wanted} matched a File in the input -- check the exact name/path "
                          f"(e.g. from the 'File' column of a full scoring run)")
    return df[mask].reset_index(drop=True)


def likely_author(repo_path, file_path, n_commits=20):
    """Most frequent author in the last n commits of `file_path`. Returns None on any failure."""
    import subprocess
    from collections import Counter
    try:
        out = subprocess.run(
            ["git", "log", f"-n{n_commits}", "--follow", "--format=%an", "--", file_path],
            cwd=str(repo_path), capture_output=True, text=True, timeout=10,
        )
        names = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if not names:
            return None
        top, count = Counter(names).most_common(1)[0]
        return f"{top} ({count}/{len(names)} recent commits)"
    except Exception:
        return None


# ============================================================================= RANKING METRICS
def recall_at(order, y, cost, budget=0.2):
    """Share of defects found when reviewing in `order` until `budget` of total `cost` is spent."""
    y, cost = np.asarray(y), np.asarray(cost, dtype=float)
    c = np.cumsum(cost[order])
    take = order[: max(int((c <= budget * cost.sum()).sum()), 1)]   # at least 1 file
    return float(y[take].sum() / max(y.sum(), 1))


def ranking_metrics(ranked):
    """Precision@k and recall@20% (files / LOC) vs. the "largest files first" baseline. Needs labels."""
    y, loc, risk = ranked["actual_defective"].values, ranked["loc"].values, ranked["risk"].values
    by_risk = np.arange(len(ranked))                                       # already sorted by risk
    by_density = np.argsort(-(risk / np.maximum(loc, 1)), kind="stable")   # risk per LOC
    by_size = np.argsort(-loc, kind="stable")                              # baseline
    m = {"files": len(ranked), "defective": int(y.sum()), "base_rate": float(y.mean())}
    for k in (10, 20, 50):
        m[f"precision@{k}"] = float(y[:k].mean())
    m["recall@20%files (model)"] = recall_at(by_risk, y, np.ones(len(y)))
    m["recall@20%files (largest first)"] = recall_at(by_size, y, np.ones(len(y)))
    m["recall@20%LOC (model risk/LOC)"] = recall_at(by_density, y, loc)
    m["recall@20%LOC (largest first)"] = recall_at(by_size, y, loc)
    return m


def gains_curve(order, y, points=101):
    """(xs, ys): fraction of defects found vs. fraction of files reviewed."""
    y = np.asarray(y)
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    xs = np.linspace(0, 1, points)
    idx = np.minimum((xs * len(y)).astype(int), len(y)) - 1
    return xs, np.where(idx >= 0, cum[np.maximum(idx, 0)], 0.0)


def plot_gains(ranked, title, path):
    """Save cumulative-gains plot: model vs. largest-first vs. random."""
    y, loc = ranked["actual_defective"].values, ranked["loc"].values
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for order, lab, col, lw in [(np.arange(len(y)), "Model ranking", TEAL_, 2.2),
                                (np.argsort(-loc, kind="stable"), "Largest files first", NAVY_, 1.6)]:
        xs, ys = gains_curve(order, y)
        ax.plot(xs * 100, ys * 100, color=col, lw=lw, label=lab)
    ax.plot([0, 100], [0, 100], color=S.GREY, ls="--", lw=1, label="Random order")
    ax.axvline(20, color="#C9D1DA", lw=1)   # 20% review budget
    ax.set_xlabel("Files reviewed (%)"); ax.set_ylabel("Defective files found (%)"); ax.set_title(title, fontsize=9.5)
    ax.legend(loc="lower right"); fig.savefig(path); plt.close(fig)


TEAL_, NAVY_ = S.TEAL, S.NAVY


# ============================================================================= SCORE ONE NEW FILE
def score_new_file(project, java_path, top=1, model_path=None, explain=True, plain=True, quiet=False):
    """Score one new .java file (no CSV/repo). Missing metrics are imputed from the training
    release medians. quiet=True suppresses printing. Returns the one-row result."""
    from .javafile import build_release_row
    from .data import load_jira
    art = joblib.load(model_path or (C.MODELS / f"{project}.joblib"))
    train_rel = art.get("train_release")
    ref = load_jira(); ref = ref[ref.project == project]
    if train_rel is not None:
        ref = ref[ref.release == train_rel]   # medians from the training release only
    medians = ref[art["features"]].median(numeric_only=True).to_dict()

    row, computed_keys, assumed_keys, imputed_keys = build_release_row(java_path, art["features"], medians)
    row["File"] = str(java_path)
    df = pd.DataFrame([row])

    if not quiet:
        print(f"Scoring {java_path} as a new file in {project} (model trained on {train_rel or '?'})")
        print(f"  {len(computed_keys)} metric(s) computed from the file itself, {len(assumed_keys)} assumed "
              f"(new-file defaults: 1 commit, 1 author, fully new code), {len(imputed_keys)} imputed from "
              f"{project}'s typical values (need whole-codebase or git-history context this one file doesn't have)")
        if imputed_keys:
            print(f"  imputed: {', '.join(imputed_keys)}")

    ranked = score_release(art, df)
    view = ranked.head(top).copy()
    if explain:
        if plain:
            reasons, actions = explain_plain(art, df, ranked, n=top)
            view["why it's risky"] = reasons
            view["suggested fix"] = actions
        else:
            view["why (top SHAP metrics)"] = explain_top(art, df, ranked, n=top)
    explain_cols = (["why it's risky", "suggested fix"] if plain else ["why (top SHAP metrics)"]) if explain else []
    cols = ["rank", "File", "risk", "flagged", "loc"] + explain_cols
    view["risk"] = view["risk"].round(3)
    if not quiet:
        with pd.option_context("display.max_colwidth", 90, "display.width", 250):
            print(view[cols].to_string(index=False))
    return view[cols]


# ============================================================================= MAIN ENTRY POINT
def run_scoring(project, input_csv=None, top=10, model_path=None, explain=True, plain=True, repo=None,
                 changed_since=None, head_ref="HEAD"):
    """Load the project's model, score a release (default: held-out latest), print, plot, save.

    explain       -- show why files are risky
    plain         -- plain English (True) or raw SHAP (False)
    repo          -- git checkout path, adds "likely developer"
    changed_since -- git ref; score only files changed since it (needs repo)
    head_ref      -- diff end ref (default "HEAD")"""
    art = joblib.load(model_path or (C.MODELS / f"{project}.joblib"))
    if input_csv is None:
        # default: held-out latest release (never used for training)
        from .data import load_jira
        rel = C.RELEASE_ORDER[project][-1]
        df = load_jira(); df = df[(df.project == project) & (df.release == rel)].reset_index(drop=True)
        source = f"{project} {rel} (held out, never used for training)"
    else:
        df = pd.read_csv(input_csv); source = str(input_csv)
    if changed_since is not None:
        if repo is None:
            raise ValueError("changed_since requires repo (need a git checkout to diff against)")
        df = filter_changed(df, repo, changed_since, head_ref)

    ranked = score_release(art, df)
    print(f"Scoring {source}: {len(df)} files | model trained on {art.get('train_release', '?')} "
          f"| decision threshold {art['threshold']:.2f} | {int(ranked.flagged.sum())} files flagged")

    view = ranked.head(top).copy()
    full_paths = view["File"].values                                   # full paths for git lookup
    view["File"] = view["File"].str.split("/").str[-3:].str.join("/")   # shortened for display
    if explain:
        if plain:
            reasons, actions = explain_plain(art, df, ranked, n=top)
            view["why it's risky"] = reasons
            view["suggested fix"] = actions
        else:
            view["why (top SHAP metrics)"] = explain_top(art, df, ranked, n=top)
    if repo is not None:
        view["likely developer"] = [likely_author(repo, f) or "unknown (no git history for this path)"
                                     for f in full_paths]
    explain_cols = (["why it's risky", "suggested fix"] if plain else ["why (top SHAP metrics)"]) if explain else []
    cols = ["rank", "File", "risk", "loc"] + (["actual_defective"] if "actual_defective" in view else []) + \
           (["likely developer"] if repo is not None else []) + explain_cols
    view["risk"] = view["risk"].round(3)
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(view[cols].to_string(index=False))

    metrics = None
    if "actual_defective" in ranked:   # only for labelled releases
        metrics = ranking_metrics(ranked)
        print("\nRanking quality on this release:")
        for k, v in metrics.items():
            print(f"  {k:36s} {v:.3f}" if isinstance(v, float) else f"  {k:36s} {v}")
        C.FIGS.mkdir(exist_ok=True)
        label = f"{project} {C.RELEASE_ORDER[project][-1]}" if input_csv is None else f"{project} {Path(input_csv).name}"
        plot_gains(ranked, f"{label}: defects found vs files reviewed", C.FIGS / f"scoring_gains_{project}.png")

    C.RESULTS.mkdir(exist_ok=True)
    ranked.drop(columns="row").to_csv(C.RESULTS / f"scored_{project}.csv", index=False)
    return ranked, metrics
