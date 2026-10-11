"""UCB(alpha) index restricted to an active set — Algorithm 1 line 4.

`active` is kept sorted ascending by arm index, so a strict `>` comparison
implements the spec C3 tie-break (lowest arm index wins).
"""

from math import sqrt

from numba import njit


@njit(cache=True)
def ucb_argmax(counts, sums, active, alpha, log_t):
    """argmax over the active arms of mu_hat_k + sqrt(alpha*log(t)/T_k).

    Spec C1: an arm never pulled (T_k = 0) has index +inf, so unplayed arms are
    pulled first, in ascending arm order.
    """
    best_arm = active[0]
    best_val = -1.0
    for a in range(active.shape[0]):
        k = active[a]
        c = counts[k]
        if c == 0:
            return k  # +inf beats everything; ascending order gives the lowest arm
        val = sums[k] / c + sqrt(alpha * log_t / c)
        if val > best_val:
            best_val = val
            best_arm = k
    return best_arm


@njit(cache=True)
def phase_argmax(phase, active):
    """Algorithm 1 line 6: most-played active arm during the current phase."""
    best_arm = active[0]
    best_val = -1
    for a in range(active.shape[0]):
        k = active[a]
        val = phase[k]
        if val > best_val:
            best_val = val
            best_arm = k
    return best_arm
