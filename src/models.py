"""Model factory: baseline models.

build(name) returns a fresh, untrained scikit-learn Pipeline by name, so each model is defined in one place.
Every pipeline starts with the same Preprocessor step (see preprocess.py); only the classifier differs.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .preprocess import Preprocessor
from . import config as C

# Every model name build() knows how to construct
MODEL_NAMES = ["LOC_LR", "LR", "RF"]

def build(name: str, seed: int = C.SEED):
    """Return a fresh, UNTRAINED Pipeline for the model called `name`."""
    prep = ("prep", Preprocessor())   # drop constants, signed-log, prune correlated

    if name == "LOC_LR":
        # Simplest baseline: logistic regression on lines of code ONLY -- answers "how much of a fancier
        # model's performance is just 'bigger files are riskier'?"
        return Pipeline([("prep", Preprocessor(only=["CountLineCode"], prune=False)), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
    if name == "LR":
        # Logistic regression on the full (pruned) feature set -- the standard, interpretable baseline.
        return Pipeline([prep, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5))])
    if name == "RF":
        # Random forest, with class weighting for the defect/clean imbalance.
        return Pipeline([prep, ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, n_jobs=2,
                                                              class_weight="balanced_subsample", random_state=seed))])
    raise ValueError(f"Unknown model name: {name!r} (expected one of {MODEL_NAMES})")
