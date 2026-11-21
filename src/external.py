"""External validation: map TeraPromise (CK) metrics onto JIRA names and evaluate across datasets.

Train on one dataset, test on the other, using only the ~9 metrics both define (MAPPING/SHARED).
"""
from .common import np, pd
import re
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from . import config as C
from .evaluate import best_threshold
from .models import Boost
from .preprocess import Preprocessor
from .score import recall_at

# TeraPromise (CK) metric -> closest JIRA (Understand) metric
MAPPING = {"loc": "CountLineCode", "wmc": "SumCyclomatic", "cbo": "CountClassCoupled", "dit": "MaxInheritanceTree",
           "noc": "CountClassDerived", "lcom": "PercentLackOfCohesion", "max_cc": "MaxCyclomatic",
           "avg_cc": "AvgCyclomatic", "npm": "CountDeclMethodPublic"}
SHARED = list(MAPPING.values())
OVERLAP = {"camel", "lucene"}   # in both datasets -> excluded as cross-dataset targets


def _version_key(stem):
    """"camel-1.6" -> ("camel", (1, 6)) for numeric version sorting; "-init" -> (-1,)."""
    base, ver = stem, ""
    if "-" in stem:
        base, ver = stem.rsplit("-", 1)
    elif re.search(r"\d+$", stem):
        base, ver = re.match(r"^(.*?)(\d+)$", stem).groups()   # "camel16" -> ("camel", "16")
    if ver == "init":
        return base, (-1,)
    return base, tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.]", ver) if x != "") or (0,)


def load_tp():
    """All TeraPromise CSVs with SHARED columns, binary label y, project/ver/release."""
    frames = []
    for f in sorted(C.TP_DIR.glob("*.csv")):
        d = pd.read_csv(f); d.columns = [c.lower() for c in d.columns]
        base, ver = _version_key(f.stem)
        out = d.rename(columns=MAPPING)[SHARED].copy()
        out["y"] = (d["bug"] > 0).astype(int).values   # bug count -> 0/1
        vnum = sum(v / (1000 ** i) for i, v in enumerate(ver)) if ver != (-1,) else -1.0   # (1,6,2) -> 1.006002
        out["project"] = base
        out["ver"] = float(vnum)
        out["release"] = f.stem
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def load_jira_shared():
    """JIRA data with the same SHARED columns and y/project/release/ver layout."""
    from .data import load_jira
    d = load_jira()
    out = d[SHARED].copy(); out["y"] = d[C.LABEL_REAL].values
    out["project"], out["release"], out["ver"] = d["project"].values, (d["project"] + "-" + d["release"]).values, d["release_idx"].values
    return out


def percentile_normalise(df):
    """Within-release percentile rank of each SHARED metric (removes scale mismatch; no labels used)."""
    out = df.copy()
    out[SHARED] = df.groupby("release")[SHARED].rank(pct=True)
    return out


def make_model(name, normalised, seed=C.SEED):
    """Model on SHARED columns; signed-log prep for raw data, plain float cast for normalised data."""
    pre = FunctionTransformer(lambda X: np.asarray(X, dtype=float)) if normalised else Preprocessor(only=SHARED, prune=False)
    pre = ("pre", pre)
    if name == "LOC_LR":
        cols = ["CountLineCode"]
        sel = FunctionTransformer(lambda X: np.asarray(pd.DataFrame(X, columns=SHARED)[cols], dtype=float)) if normalised \
            else Preprocessor(only=cols, prune=False)
        return Pipeline([("pre", sel), ("scale", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    if name == "LR":
        return Pipeline([pre, ("scale", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5))])
    if name == "RF":
        # more regularised than the main RF: 9 noisy proxy metrics
        return Pipeline([pre, ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_samples=0.5, n_jobs=2,
                                                            class_weight="balanced_subsample", random_state=seed))])
    if name == "XGB_weighted":
        return Pipeline([pre, ("clf", Boost(weighted=True, seed=seed))])
    raise ValueError(name)


def _valid(d, min_pos=5):
    """At least min_pos of each class."""
    return d["y"].sum() >= min_pos and (1 - d["y"]).sum() >= min_pos


def score_target(p, y, thr, loc):
    """AUC, PR-AUC, MCC (tuned threshold and top-20% flag), recall@20% files/LOC for one target."""
    n = len(y); order = np.argsort(-p, kind="stable")
    k = max(int(round(0.2 * n)), 1); flag = np.zeros(n, dtype=bool); flag[order[:k]] = True   # flag top 20%
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), mcc_thr=matthews_corrcoef(y, p >= thr),
                mcc_top20=matthews_corrcoef(y, flag), recall_20files=recall_at(order, y, np.ones(n)),
                recall_20loc=recall_at(np.argsort(-(p / np.maximum(loc, 1)), kind="stable"), y, loc), n=n, base_rate=float(y.mean()))


def cross_dataset(source, targets, models=("LOC_LR", "LR", "RF", "XGB_weighted"), variants=("raw", "norm"), direction=""):
    """Train on pooled `source`, evaluate on each valid release of `targets`, for raw and norm variants."""
    rows = []
    for variant in variants:
        norm = variant == "norm"
        S = percentile_normalise(source) if norm else source
        T = percentile_normalise(targets) if norm else targets
        for name in models:
            m = make_model(name, norm); Xs, ys = S[SHARED], S["y"].values
            m.fit(Xs, ys)
            # threshold from 3-fold OOF predictions on source
            oof = cross_val_predict(clone(m), Xs, ys, cv=StratifiedKFold(3, shuffle=True, random_state=C.SEED), method="predict_proba")[:, 1]
            thr = best_threshold(ys, oof)
            for rel, g in T.groupby("release"):
                if not _valid(g):
                    continue
                p = m.predict_proba(g[SHARED])[:, 1]
                loc = targets.loc[g.index, "CountLineCode"].values.astype(float)   # raw LOC for effort weight
                rows.append(dict(direction=direction, variant=variant, model=name, project=g["project"].iloc[0], release=rel,
                                 **score_target(p, g["y"].values, thr, loc)))
    return pd.DataFrame(rows)


def in_domain(tp, models=("LOC_LR", "LR", "RF", "XGB_weighted"), variants=("raw", "norm")):
    """Reference: train on TeraPromise version v, test on v+1 of the same project."""
    rows = []
    for proj, g in tp.groupby("project"):
        rels = g.sort_values("ver").drop_duplicates("release")[["release", "ver"]].sort_values("ver")["release"].tolist()
        for a, b in zip(rels[:-1], rels[1:]):
            tr, te = tp[tp.release == a], tp[tp.release == b]
            if not (_valid(tr) and _valid(te)):
                continue
            for variant in variants:
                norm = variant == "norm"
                Tr = percentile_normalise(tr) if norm else tr; Te = percentile_normalise(te) if norm else te
                for name in models:
                    m = make_model(name, norm)
                    if name == "RF":
                        m.set_params(clf__min_samples_leaf=2, clf__max_samples=None)   # less regularised for small single-release data
                    m.fit(Tr[SHARED], Tr["y"].values)
                    oof = cross_val_predict(clone(m), Tr[SHARED], Tr["y"].values, cv=StratifiedKFold(3, shuffle=True, random_state=C.SEED), method="predict_proba")[:, 1]
                    thr = best_threshold(Tr["y"].values, oof)
                    p = m.predict_proba(Te[SHARED])[:, 1]
                    rows.append(dict(direction="in-domain", variant=variant, model=name, project=proj, release=b, train_release=a,
                                     **score_target(p, te["y"].values, thr, te["CountLineCode"].values.astype(float))))
    return pd.DataFrame(rows)
