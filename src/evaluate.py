"""Simple statistics."""
from .common import np

def cliffs_delta(a, b):
    """Cliff's delta: an effect-size measure for "how often is a value from `a` bigger than one from `b`,
    versus the reverse" -- ranges from -1 (b always bigger) to +1 (a always bigger), 0 = no difference."""
    a, b = np.asarray(a), np.asarray(b)
    # a[:, None] > b[None, :] broadcasts into a full (len(a) x len(b)) grid of every pairwise comparison;
    # summing that grid counts how many pairs went each way.
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))
