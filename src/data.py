"""Data loading and release-pair construction.

Three things happen here: load the raw JIRA CSVs into one tidy table (load_jira), work out which columns
are actual predictive features vs bookkeeping (feature_columns), and build the "train on release N, test on
release N+1" pairs every experiment in this project runs on (release_pairs).
"""
from .common import pd
from . import config as C


def load_jira() -> pd.DataFrame:
    """Load all 32 JIRA release CSVs into one combined DataFrame.

    Adds three columns not in the raw files, used throughout the rest of the pipeline:
        project      -- which of the 9 Apache projects this row is from
        release      -- the release version string, e.g. "2.11.0"
        release_idx  -- 0, 1, 2, ... this release's position in its project's chronological order
                        (from config.RELEASE_ORDER), so later code can sort/compare releases without
                        re-parsing version strings
    """
    frames = []
    # Walking config.RELEASE_ORDER (rather than globbing the folder) means a missing/misnamed file fails
    # loudly with a clear error, instead of silently loading fewer releases than expected.
    for project, releases in C.RELEASE_ORDER.items():
        for idx, rel in enumerate(releases):
            df = pd.read_csv(C.JIRA_DIR / f"{project}-{rel}.csv")
            df.insert(0, "project", project)
            df.insert(1, "release", rel)
            df.insert(2, "release_idx", idx)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)   # stack all 32 releases into one long table
    # Labels can load as float if pandas sees any NaN; force to int so every later .sum()/==1 check behaves.
    for col in (C.LABEL_REAL, C.LABEL_HEU):
        df[col] = df[col].astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Columns of `df` that are actual predictive metrics -- everything except the project/release
    bookkeeping and the label columns themselves (keeping a label in as a feature would just let the
    model see the answer)."""
    meta = ["project", "release", "release_idx"] + C.NON_FEATURES
    return [c for c in df.columns if c not in meta]


def release_pairs(df: pd.DataFrame, label: str = C.LABEL_REAL):
    """Yield one (project, train_release, test_release, train_df, test_df) tuple per usable consecutive
    release pair, in chronological order within each project.

    A pair is skipped unless both releases have at least config.MIN_POS defective files -- fewer than
    that and any precision/recall computed from it is just noise. 23 pairs survive this out of 32
    releases.

    `label` picks which label decides if the TRAINING side is usable (the label-noise study passes
    label=HeuBug to see how many pairs remain training on the noisier label) -- the TEST side is always
    checked against RealBug, the one target this project treats as ground truth.
    """
    for project, releases in C.RELEASE_ORDER.items():
        for a, b in zip(releases[:-1], releases[1:]):   # every consecutive (earlier, later) release pair
            tr = df[(df.project == project) & (df.release == a)]   # earlier release = train
            te = df[(df.project == project) & (df.release == b)]   # later release = test
            if tr[label].sum() < C.MIN_POS or te[C.LABEL_REAL].sum() < C.MIN_POS:
                continue   # too few defective files on one side to trust this pair
            yield project, a, b, tr, te