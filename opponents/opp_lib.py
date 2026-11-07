"""
opp_lib.py - shared helpers for the synthetic practice opponents.

These model what other teams might submit. They are deliberately simple and
independent of our own strategy code (only the exact evaluator and the
preflop table are shared).
"""

from __future__ import annotations

import os
import zlib

import numpy as np

from bot import fasteval as fe

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "bot", "data", "preflop.npz")
if not os.path.exists(_DATA):
    _DATA = _DATA.replace("preflop.npz", "preflop_provisional.npz")
_PF = np.load(_DATA)
_KEY_TO_IDX = {int(k): i for i, k in enumerate(_PF["keys"])}
PF_PCT = _PF["pct"]
PF_EQ = _PF["eq"]


def class_key(hole) -> int:
    m = [0, 0, 0, 0]
    for c in hole:
        m[c & 3] |= 1 << (c >> 2)
    m.sort()
    return m[0] | (m[1] << 9) | (m[2] << 18) | (m[3] << 27)


def preflop_pct(hole) -> float:
    """Combo-weighted percentile of a starting hand (1 = best)."""
    return float(PF_PCT[_KEY_TO_IDX[class_key(hole)]])


def preflop_eq(hole) -> float:
    """Equity vs a random hand."""
    return float(PF_EQ[_KEY_TO_IDX[class_key(hole)], 0])


def hand_rng(obs, salt: str = "") -> np.random.Generator:
    """Deterministic RNG for one decision (same spot -> same choice)."""
    tag = f"{salt}|{obs.hand_index}|{obs.hole_cards}|{obs.board}|{len(obs.action_history)}"
    return np.random.default_rng(zlib.crc32(tag.encode()))


def mc_equity(hole, board, n: int, rng: np.random.Generator) -> float:
    """Monte Carlo equity vs ONE random opponent hand (ties = 1/2)."""
    hole = np.asarray(hole, dtype=np.int64)
    used = np.zeros(36, dtype=bool)
    used[hole] = True
    used[list(board)] = True
    rest = np.flatnonzero(~used)
    need = 5 - len(board)
    draw = rng.permuted(np.tile(rest, (n, 1)), axis=1)[:, :5 + need]
    opp = draw[:, :5]
    if need:
        full = np.concatenate([np.tile(np.asarray(board, dtype=np.int64), (n, 1)), draw[:, 5:]], 1)
    else:
        full = np.tile(np.asarray(board, dtype=np.int64), (n, 1))
    me = fe.omaha_batch(np.tile(hole, (n, 1)), full)
    op = fe.omaha_batch(opp, full)
    return float(((me > op).sum() + 0.5 * (me == op).sum()) / n)


def equity(obs, n: int = 150, rng=None) -> float:
    """Preflop: table equity; postflop: Monte Carlo vs a random hand."""
    if not obs.board:
        return preflop_eq(obs.hole_cards)
    return mc_equity(obs.hole_cards, obs.board, n, rng or hand_rng(obs))


def facing_raise_preflop(obs) -> bool:
    """True if the opponent made a voluntary raise this street."""
    return any((not a.is_me) and a.action.name.startswith("RAISE")
               for a in obs.action_history if a.street == obs.street)
