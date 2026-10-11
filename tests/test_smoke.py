"""Phase-0 gate: package imports and the numba hot-loop path works on this interpreter."""

import numba
import numpy as np

import mabamab


def test_import():
    assert mabamab.__version__


def test_numba_jit():
    @numba.njit(cache=True)
    def argmax_restricted(vals: np.ndarray, mask: np.ndarray) -> int:
        best, best_v = -1, -np.inf
        for k in range(vals.shape[0]):
            if mask[k] and vals[k] > best_v:
                best, best_v = k, vals[k]
        return best

    vals = np.array([0.1, 0.9, 0.5])
    mask = np.array([True, False, True])
    assert argmax_restricted(vals, mask) == 2
