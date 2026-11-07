"""
preflop.py - preflop lookups from the offline tables (bot/data/preflop.npz).

Built by tools/build_preflop.py. Every starting hand maps to one of 19,998
suit-isomorphism classes; for each class we know:
    equity vs a random hand, vs the top 50/30/15/8% of hands, and the
    combo-weighted percentile (1 = best hand).
"""

from __future__ import annotations

import math
import os

import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PATH = os.path.join(_DIR, "preflop.npz")
if not os.path.exists(_PATH):          # development fallback: vs-random only
    _PATH = os.path.join(_DIR, "preflop_provisional.npz")
_D = np.load(_PATH)
HAS_RANGES = bool(_D["has_ranges"]) if "has_ranges" in _D else True
KEYS = _D["keys"]                     # sorted int64
EQ = _D["eq"].astype(np.float64)      # (C, 5): random, top50, top30, top15, top8
PCT = _D["pct"].astype(np.float64)    # (C,)
COMBOS = _D["combos"]
RANGE_X = np.concatenate([[1.0], _D["range_x"].astype(np.float64)])   # 1.0, .5, .3, .15, .08
_LOGX = np.log(RANGE_X)
_KEY_TO_IDX = {int(k): i for i, k in enumerate(KEYS)}
EQ_ROWS = EQ.tolist()                         # plain floats: fast single lookups
PCT_LIST = PCT.tolist()
_LX = [math.log(x) for x in RANGE_X]          # decreasing: 0, log .5, log .3, log .15, log .08


def interp(row, x: float) -> float:
    """Equity vs the top-x range from a table row, linear in log x (pure Python)."""
    if x >= 1.0:
        return row[0]
    if x <= RANGE_X[-1]:
        return row[-1]
    lx = math.log(x)
    for i in range(1, len(_LX)):
        if lx >= _LX[i]:
            t = (lx - _LX[i]) / (_LX[i - 1] - _LX[i])
            return row[i] + t * (row[i - 1] - row[i])
    return row[-1]


def class_key(hole) -> int:
    """Suit-isomorphism key of one 5-card hand."""
    m = [0, 0, 0, 0]
    for c in hole:
        m[c & 3] |= 1 << (c >> 2)
    m.sort()
    return m[0] | (m[1] << 9) | (m[2] << 18) | (m[3] << 27)


def class_index(hole) -> int:
    return _KEY_TO_IDX[class_key(hole)]


def class_index_batch(holes: np.ndarray) -> np.ndarray:
    """(K, 5) hands -> (K,) class indices (vectorized)."""
    holes = np.asarray(holes, dtype=np.int64)
    bits = np.left_shift(1, holes >> 2)
    suit = holes & 3
    m = np.stack([np.where(suit == s, bits, 0).sum(1) for s in range(4)], axis=1)
    m.sort(1)
    keys = m[:, 0] | (m[:, 1] << 9) | (m[:, 2] << 18) | (m[:, 3] << 27)
    return np.searchsorted(KEYS, keys)


def pct(hole) -> float:
    """Combo-weighted percentile of a starting hand (1 = best)."""
    return float(PCT[class_index(hole)])


def eq_random(hole) -> float:
    return float(EQ[class_index(hole), 0])


def eq_vs_top(hole, x: float) -> float:
    """Equity vs the top-x fraction of hands (x in (0, 1]), interpolated in log x."""
    row = EQ[class_index(hole)]
    return float(_interp_row(row, x))


def _interp_row(row: np.ndarray, x: float) -> float:
    lx = np.log(min(1.0, max(RANGE_X[-1], x)))
    # RANGE_X is decreasing; np.interp needs increasing x.
    return np.interp(lx, _LOGX[::-1], row[::-1])
