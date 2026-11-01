"""Metric selection: drop constant metrics and prune highly correlated ones."""
from .common import np, pd
from . import config as C

class Preprocessor:
    """Learns which metrics to keep: drops constant metrics."""

    def __init__(self, threshold=C.CORR_THRESHOLD):
        self.threshold = threshold

    @staticmethod
    def _slog(x):
        """Signed log1p: log(1+|x|) with the original sign put back; defined at x=0 and safe for negatives."""
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, X, y=None):
        """Learn which columns to keep. Only looks at metric values, never at defect labels."""
        X = pd.DataFrame(X).copy()
        X = X.loc[:, X.nunique() > 1] # drop constant metrics
        Xt = self._slog(X)
        keep = list(Xt.columns)
        if len(keep) > 1:
            # Greedy correlation pruning: rank-transform each column (Spearman, not Pearson), then walk
            # columns in their original order and keep one only if it is NOT too correlated (> threshold)
            # with any column already kept -- of any correlated cluster, the first-encountered survives.
            corr = Xt.rank().corr().abs().values
            kept = []
            for i in range(len(keep)):
                if all(corr[i, j] <= self.threshold for j in kept):
                    kept.append(i)
            keep = [keep[i] for i in kept]
        self.feature_names_ = keep
        return self
