"""Final model training: the 'system' part of the pipeline.

train_final -> fit the chosen model on one labelled release and remember its decision threshold
save_model  -> persist that trained artifact to models/<project>.joblib

(Release scoring with SHAP reasons is added in the next step, scripts/08_score_release.py.)
"""
from .common import joblib
from . import config as C
from .models import build
from .evaluate import oof_threshold


def train_final(train_df, feats, label=C.LABEL_REAL, seed=C.SEED, name="RF", **meta):
    """Fit `name` (default: random forest, the best model in the comparison) on `train_df`, then pick its
    decision threshold from out-of-fold predictions so it is a fair, deployable estimate.

    Returns a plain dict (so it can be pickled with joblib and loaded back with only this src/ package on the
    path). Extra keyword arguments (**meta) are stored as-is, e.g. project, train_release, scored_release.
    """
    X, y = train_df[feats], train_df[label].values
    model = build(name, seed).fit(X, y)
    thr = oof_threshold(model, X, y, seed)   # MCC-optimal cutoff from cross-validated OOF predictions
    return dict(model=model, threshold=float(thr), features=list(feats), model_name=name, seed=seed,
                n_train=int(len(train_df)), n_pos=int(y.sum()), **meta)


def save_model(art, path):
    """Persist the dict returned by train_final() to disk."""
    joblib.dump(art, path)
