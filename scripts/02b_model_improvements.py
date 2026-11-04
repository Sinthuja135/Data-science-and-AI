"""Extra improvement ideas beyond a single boosted model.

RF_tuned      : random forest with randomised hyper-parameter search (trained on release t only)
ENS           : average of random forest and class-weighted XGBoost probabilities (trained on release t only)
RF_hist/ENS_hist : same models trained on ALL earlier releases of the project (expanding window)

Thresholds are chosen from out-of-fold predictions on the training data only (never the test release).
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from src import config as C, data as D
from src.models import build
from src.evaluate import best_threshold, score

C.RESULTS.mkdir(exist_ok=True)
out = C.RESULTS / "improvement_results.csv"
partial = C.RESULTS / "improvement_results.partial.csv"   # checkpoint, so an interrupted run never clobbers `out`

df = D.load_jira(); feats = D.feature_columns(df)
cv = StratifiedKFold(3, shuffle=True, random_state=C.SEED)


def fit_pred(name, X, y, Xte):
    """Fit `name` on (X, y); return test probabilities and out-of-fold training probabilities."""
    m = build(name).fit(X, y)
    est = m.best_estimator_ if hasattr(m, "best_estimator_") else m
    p = m.predict_proba(Xte)[:, 1]
    oof = cross_val_predict(clone(est), X, y, cv=cv, method="predict_proba")[:, 1]
    return p, oof

rows, t0 = [], time.time()
pairs = list(D.release_pairs(df))
for i, (proj, a, b, tr, te) in enumerate(pairs, 1):
    order = C.RELEASE_ORDER[proj]; earlier = order[: order.index(a) + 1]
    hist = df[(df.project == proj) & (df.release.isin(earlier))]
    Xte, yte = te[feats], te[C.LABEL_REAL].values
    cache = None   # (p_rf, o_rf, p_xg, o_xg) from the release-t fit, reused when history == release t
    for tag, train in [("", tr), ("_hist", hist)]:
        X, y = train[feats], train[C.LABEL_REAL].values
        if tag == "_hist" and len(earlier) == 1:
            p_rf, o_rf, p_xg, o_xg = cache   # first release: expanding window is identical, skip refitting
        else:
            p_rf, o_rf = fit_pred("RF", X, y, Xte)
            p_xg, o_xg = fit_pred("XGB_weighted", X, y, Xte)
            cache = (p_rf, o_rf, p_xg, o_xg)
        outs = {"ENS": ((p_rf + p_xg) / 2, (o_rf + o_xg) / 2)}
        if tag == "":
            outs["RF_tuned"] = fit_pred("RF_tuned", X, y, Xte)
        else:
            outs["RF"] = (p_rf, o_rf)
        for name, (p, o) in outs.items():
            rows.append(dict(project=proj, train=a, test=b, model=name + tag, n_train=len(train),
                             **score(yte, p, best_threshold(y, o))))
    pd.DataFrame(rows).to_csv(partial, index=False)
    print(f"[{i}/{len(pairs)}] {proj} {a}->{b} done  ({time.time()-t0:.0f}s)", flush=True)

partial.replace(out)
print("saved", out)
