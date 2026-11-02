"""Step 2: baselines vs improved models on time-ordered release pairs (train on release t, test on t+1)."""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C, data as D
from src.models import build, MODEL_NAMES
from src.evaluate import oof_threshold, score

C.RESULTS.mkdir(exist_ok=True)
df = D.load_jira(); feats = D.feature_columns(df)
out = C.RESULTS / "pair_results.csv"
rows, t0 = [], time.time()
pairs = list(D.release_pairs(df))
for i, (proj, a, b, tr, te) in enumerate(pairs, 1):
    Xtr, ytr = tr[feats], tr[C.LABEL_REAL].values
    Xte, yte = te[feats], te[C.LABEL_REAL].values
    for name in MODEL_NAMES:
        m = build(name).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        thr = oof_threshold(m, Xtr, ytr)
        est = m.best_estimator_ if hasattr(m, "best_estimator_") else m
        rows.append(dict(project=proj, train=a, test=b, model=name, n_test=len(te), pos_rate_test=yte.mean(),
                         n_features=len(est.named_steps["prep"].feature_names_), **score(yte, p, thr)))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{i}/{len(pairs)}] {proj} {a}->{b} done  ({time.time()-t0:.0f}s)", flush=True)
print("saved", out)
