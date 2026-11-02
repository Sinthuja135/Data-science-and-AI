"""Metric selection: drop constant metrics and prune highly correlated ones."""
"""Leak-free preprocessing: fitted on the training release only.

Every decision (which metrics are constant, which are too correlated) is learned once in fit() on the
TRAINING data, then applied as-is by transform() -- so a test release can never influence training.
Being a scikit-learn transformer lets it sit inside a Pipeline.
"""
from .common import np, pd
from sklearn.base import BaseEstimator, TransformerMixin
from . import config as C


class Preprocessor(BaseEstimator, TransformerMixin):
    """Learns which metrics to keep: drops constant metrics."""
    """Drop constant metrics, apply a signed log1p transform (metrics are heavily right-skewed), and
    optionally prune metrics whose | Spearman correlation | with an already-kept metric exceeds `threshold`.

    Parameters
    threshold : float
        |Spearman correlation| above which the later of two correlated metrics is dropped (if prune=True).
    only : list[str] or None
        If given, restrict to exactly these columns first -- used by the LOC-only baseline.
    prune : bool
        Set False to skip correlation pruning.
    """

    def __init__(self, threshold=C.CORR_THRESHOLD, only=None, prune=True):
        # scikit-learn convention: store parameters verbatim so get_params()/clone() work
        self.threshold = threshold
        self.only = only
        self.prune = prune

    @staticmethod
    def _slog(x):
        """Signed log1p: log(1+|x|) with the original sign put back; defined at x=0 and safe for negatives."""
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X, y=None):
        """Learn which columns to keep. Only looks at metric values, never at defect labels."""
        X = pd.DataFrame(X).copy()
        if self.only is not None:
            X = X[self.only] # restrict to the caller's chosen columns first
        X = X.loc[:, X.nunique() > 1] # drop constant metrics
        Xt = self._slog(X)
        keep = list(Xt.columns)
        if self.prune and len(keep) > 1:
            # Greedy correlation pruning: rank-transform each column (Spearman, not Pearson), then walk
            # columns in their original order and keep one only if it is NOT too correlated (> threshold)
            # with any column already kept -- of any correlated cluster, the first-encountered survives.
            corr = Xt.rank().corr().abs().values
            kept = []
            for i in range(len(keep)):
                if all(corr[i, j] <= self.threshold for j in kept):
                    kept.append(i)
            keep = [keep[i] for i in kept]
        self.feature_names_ = keep           # trailing underscore = scikit-learn "set by fit()" convention
        return self

    def transform(self, X):
        """Apply the column selection + signed-log transform learned by fit(). Missing values -> 0."""
        X = pd.DataFrame(X)
        return self._slog(X[self.feature_names_].fillna(0)).values

    def get_feature_names_out(self, input_features=None):
        """Real metric names for the transformed columns instead of anonymous array indices."""
        return np.array(self.feature_names_)
