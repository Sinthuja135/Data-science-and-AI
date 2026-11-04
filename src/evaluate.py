"""Metrics, threshold selection and simple statistics."""
from .common import np
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from . import config as C

def best_threshold(y, p):
    """The probability cutoff (grid 0.02..0.98) that maximises Matthews Correlation Coefficient (MCC).
    MCC is used rather than F1/accuracy because it stays meaningful at the ~2-10% positive rate of a release."""
    grid = np.linspace(0.02, 0.98, 97) # 97 candidate thresholds, 0.01 apart
    scores = [matthews_corrcoef(y, p >= t) for t in grid]
    return float(grid[int(np.argmax(scores))])

def oof_threshold(model, X, y, seed=C.SEED):
    """Pick the MCC-optimal threshold from OUT-OF-FOLD predictions on the training data only -- the test
    release is never looked at, so the threshold doesn't leak test information.

    3-fold stratified CV within the training release gives one prediction per training row from a model
    that never saw that row; best_threshold() is then run on those predictions.
    """
    # unwrap a fitted hyperparameter search (e.g. RandomizedSearchCV) to its underlying pipeline
    est = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    # clone(): an untrained copy with the same hyperparameters, so no already-fitted state is reused
    p = cross_val_predict(clone(est), X, y, cv=cv, method="predict_proba")[:, 1]
    return best_threshold(y, p)


def score(y, p, thr):
    """Headline metrics for (labels, predicted probabilities, threshold): ROC AUC and PR AUC
    (threshold-free), MCC at 0.5 and at `thr`, and F1 at `thr`."""
    yhat = p >= thr
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                mcc_05=matthews_corrcoef(y, p >= 0.5), mcc_opt=matthews_corrcoef(y, yhat),
                f1_opt=f1_score(y, yhat), thr=thr)


def cliffs_delta(a, b):
    """Cliff's delta: an effect-size measure for "how often is a value from `a` bigger than one from `b`,
    versus the reverse" -- ranges from -1 (b always bigger) to +1 (a always bigger), 0 = no difference."""
    a, b = np.asarray(a), np.asarray(b)
    # a[:, None] > b[None, :] broadcasts into a full (len(a) x len(b)) grid of every pairwise comparison;
    # summing that grid counts how many pairs went each way.
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def paired_test(a, b, n_boot=5000, seed=C.SEED):
    """Compare two models' scores over the same release pairs (a[i], b[i] from pair i): Wilcoxon signed-rank
    p-value, Cliff's delta effect size, and a bootstrap 95% CI on the mean difference."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b   # per-pair difference
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan   # every difference is exactly zero -- nothing to rank
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return dict(mean_diff=d.mean(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                wilcoxon_p=p, cliffs_delta=cliffs_delta(a, b), wins=int((d > 0).sum()), n=len(d))
