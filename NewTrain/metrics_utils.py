"""
Shared metric computation utilities used by evaluation pipelines.

All functions accepting genuine / imposter scores expect similarity semantics
(higher score = more likely a match).  Distance-based scores must be negated
before calling these helpers.
"""

import numpy as np
from scipy.optimize import brentq
from sklearn import metrics


# ---------------------------------------------------------------------------
#  d-prime
# ---------------------------------------------------------------------------

def compute_d_prime(genuine_scores, imposter_scores):
    """
    Compute d-prime (d') — the sensitivity index that quantifies how well
    the genuine and imposter score distributions are separated.

        d' = |μ_genuine − μ_imposter| / sqrt((σ_genuine² + σ_imposter²) / 2)

    Larger d' indicates better discrimination — the two distributions are
    farther apart relative to their combined spread.

    Args:
        genuine_scores: array-like (list / numpy.ndarray / torch.Tensor).
                        Matching scores from same-class (genuine) pairs.
        imposter_scores: array-like (list / numpy.ndarray / torch.Tensor).
                         Matching scores from different-class (imposter) pairs.

    Returns:
        float: d-prime value rounded to 4 decimal places.

    Raises:
        ValueError: if either input is empty, not 1-D, or contains non-finite values.
    """
    # --- input normalisation ------------------------------------------------
    def _to_1d_array(data, name):
        # torch.Tensor -> numpy
        if hasattr(data, 'detach'):
            data = data.detach()
        if hasattr(data, 'cpu'):
            data = data.cpu()
        if hasattr(data, 'numpy'):
            data = data.numpy()

        arr = np.asarray(data, dtype=np.float64)

        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-dimensional, got shape {arr.shape}"
            )
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{name} contains non-finite values (NaN / Inf)"
            )
        return arr

    genuine = _to_1d_array(genuine_scores, "genuine_scores")
    imposter = _to_1d_array(imposter_scores, "imposter_scores")

    # --- d-prime formula ----------------------------------------------------
    mu_gen = np.mean(genuine)
    mu_imp = np.mean(imposter)
    var_gen = np.var(genuine)   # population variance (ddof=0)
    var_imp = np.var(imposter)

    pooled_std = np.sqrt((var_gen + var_imp) / 2.0)
    if pooled_std == 0.0:
        # Perfect separation: all genuine scores identical, all imposter scores
        # identical, and the two groups do not overlap.  Cap at a large finite
        # sentinel so the value survives JSON serialisation (Infinity is not
        # valid JSON per RFC 8259).
        return 9999.0 if mu_gen != mu_imp else 0.0

    d_prime = abs(mu_gen - mu_imp) / pooled_std
    d_prime = round(float(d_prime), 4)
    # Guard against numerically huge values that break downstream consumers.
    return min(d_prime, 9999.0)


# ---------------------------------------------------------------------------
#  ROC helpers (operate on FPR / TPR arrays)
# ---------------------------------------------------------------------------

def roc_upper_envelope(fpr, tpr):
    """Monotonically non-decreasing upper envelope of an ROC curve."""
    order = np.argsort(fpr, kind="mergesort")
    fpr_sorted = np.asarray(fpr, dtype=np.float64)[order]
    tpr_sorted = np.asarray(tpr, dtype=np.float64)[order]
    fpr_unique, start_idx = np.unique(fpr_sorted, return_index=True)
    tpr_unique = np.maximum.reduceat(tpr_sorted, start_idx)
    tpr_unique = np.maximum.accumulate(tpr_unique)
    return fpr_unique, tpr_unique


def tar_at_far(fpr, tpr, far_target):
    """TAR at a specific FAR, evaluated on the upper envelope."""
    fpr_env, tpr_env = roc_upper_envelope(fpr, tpr)
    valid = fpr_env <= far_target
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr_env[valid]))


def compute_eer(fpr, tpr):
    """EER from FPR / TPR via upper envelope + brentq root-finding."""
    fpr_env, tpr_env = roc_upper_envelope(fpr, tpr)
    return float(brentq(lambda x: 1.0 - x - np.interp(x, fpr_env, tpr_env), 0.0, 1.0))


def smooth_roc_curve(fpr, tpr, far_min=1e-6, num_points=2000):
    """
    Logarithmically resample and convolution-smooth an ROC curve for display.

    Returns (x, y) ready for semilogx plotting.
    """
    fpr_unique, idx = np.unique(fpr, return_index=True)
    tpr_unique = tpr[idx]
    lower = max(far_min, float(np.min(fpr_unique[fpr_unique > 0])) if np.any(fpr_unique > 0) else far_min)
    lower = min(lower, 1.0)
    x = np.logspace(np.log10(lower), 0.0, num=num_points)
    y = np.interp(x, fpr_unique, tpr_unique)

    if y.size >= 7:
        win = max(7, num_points // 40)
        if win % 2 == 0:
            win += 1
        win = min(win, 151)
        kernel = np.ones(win, dtype=np.float64) / float(win)
        y = np.convolve(y, kernel, mode="same")
        y = np.clip(y, 0.0, 1.0)
        y = np.maximum.accumulate(y)
        y[-1] = min(1.0, y[-1])
    return x, y


# ---------------------------------------------------------------------------
#  Full evaluation (the single entry-point used by all pipelines)
# ---------------------------------------------------------------------------

def evaluate_from_scores(genuine, imposter):
    """
    Compute all standard verification metrics from genuine & imposter scores.

    Args:
        genuine: 1-D array of same-class pair scores (similarity semantics).
        imposter: 1-D array of different-class pair scores.

    Returns:
        dict with keys:
            fpr, tpr        – raw ROC arrays
            auc             – area under ROC curve
            eer             – equal error rate
            d_prime         – sensitivity index
            TAR_FAR_E6 … E1 – TAR @ FAR = 1e-6 … 1e-1
    """
    genuine = np.asarray(genuine, dtype=np.float64).ravel()
    imposter = np.asarray(imposter, dtype=np.float64).ravel()

    y_true = np.concatenate([
        np.ones(len(genuine), dtype=np.int32),
        np.zeros(len(imposter), dtype=np.int32),
    ])
    y_score = np.concatenate([genuine, imposter])

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc_value = float(metrics.auc(fpr, tpr))
    eer_value = compute_eer(fpr, tpr)
    d_val = compute_d_prime(genuine, imposter)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_value,
        "eer": eer_value,
        "d_prime": d_val,
        "TAR_FAR_E6": tar_at_far(fpr, tpr, 1e-6),
        "TAR_FAR_E5": tar_at_far(fpr, tpr, 1e-5),
        "TAR_FAR_E4": tar_at_far(fpr, tpr, 1e-4),
        "TAR_FAR_E3": tar_at_far(fpr, tpr, 1e-3),
        "TAR_FAR_E2": tar_at_far(fpr, tpr, 1e-2),
        "TAR_FAR_E1": tar_at_far(fpr, tpr, 1e-1),
        "num_genuine": len(genuine),
        "num_imposter": len(imposter),
    }
