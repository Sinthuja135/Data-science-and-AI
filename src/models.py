"""Model factory: baseline models plus the extra models used by 02b_model_improvements.

build(name) returns a fresh, untrained scikit-learn Pipeline by name, so each model is defined in one place.
Every pipeline starts with the same Preprocessor step (preprocess.py); only the classifier differs.
"""
from .common import np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from .preprocess import Preprocessor
from . import config as C

class Boost(BaseEstimator, ClassifierMixin):
    """Thin scikit-learn wrapper around XGBoost with optional automatic class weighting
    (scale_pos_weight = negatives / positives, computed from whatever data it is fit on)."""

    def __init__(self, weighted=True, n_estimators=300, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0, seed=C.SEED):
        self.weighted = weighted
        self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate
        self.subsample, self.colsample_bytree = subsample, colsample_bytree
        self.min_child_weight, self.reg_lambda, self.seed = min_child_weight, reg_lambda, seed

    def fit(self, X, y):
        y = np.asarray(y)
        spw = (y == 0).sum() / max((y == 1).sum(), 1) if self.weighted else 1.0
        self.model_ = XGBClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                    learning_rate=self.learning_rate, subsample=self.subsample,
                                    colsample_bytree=self.colsample_bytree, min_child_weight=self.min_child_weight,
                                    reg_lambda=self.reg_lambda, random_state=self.seed, scale_pos_weight=spw,
                                    n_jobs=2, verbosity=0, eval_metric="logloss")
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# Search space for RF_tuned ("clf__" addresses the classifier step inside the Pipeline).
RF_SPACE = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 8, 16],
    "clf__min_samples_leaf": [1, 2, 5],
    "clf__max_features": ["sqrt", 0.3, 0.5],
}

# Baseline models compared in 02_run_models.py
MODEL_NAMES = ["LOC_LR", "LR", "RF"]
# Extra models used by 02b_model_improvements.py (buildable, but not part of the baseline comparison)
EXTRA_MODELS = ["XGB_weighted", "RF_tuned"]

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
    if name == "XGB_weighted":
        # Gradient boosting with scale_pos_weight class weighting (see Boost.fit).
        return Pipeline([prep, ("clf", Boost(weighted=True, seed=seed))])
    if name == "RF_tuned":
        # Random forest wrapped in a randomized hyper-parameter search, cross-validated WITHIN the training
        # release only, so it never sees the test release.
        base = Pipeline([prep, ("clf", RandomForestClassifier(n_jobs=2, class_weight="balanced_subsample", random_state=seed))])
        return RandomizedSearchCV(base, RF_SPACE, n_iter=6, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                  scoring="average_precision", random_state=seed, n_jobs=1, refit=True)
    raise ValueError(f"Unknown model name: {name!r} (expected one of {MODEL_NAMES + EXTRA_MODELS})")
