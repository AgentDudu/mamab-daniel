"""splitmix64 streams — the kernel RNG (numba 0.68 has no `default_rng` in `@njit`).

Kept in its own module so `engine`, `malicious` and `baselines` can all draw from
the same per-agent stream without importing each other.
"""

import numpy as np
from numba import njit

_U64 = np.uint64
_TWO53_INV = 1.1102230246251565e-16  # 2**-53


@njit(cache=True)
def next_u64(state):
    """splitmix64: one 64-bit draw, returns (new_state, draw).

    Per-agent streams (seeded from a SeedSequence in `engine.agent_seeds`) keep
    runs reproducible and let a single agent's stream be resumed.
    """
    state += _U64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return state, z ^ (z >> _U64(31))


@njit(cache=True)
def uniform(state):
    """One U[0,1) draw (53-bit mantissa)."""
    state, z = next_u64(state)
    return state, (z >> _U64(11)) * _TWO53_INV
