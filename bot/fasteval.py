"""
fasteval.py - vectorized exact Short Deck PLO5 hand scoring (numpy).

Gives exactly the same scores as evaluator.omaha_score, much faster when
many hands are scored at once.

How it works:
  * The non-flush score of 5 cards depends only on their ranks. The key
    sum(5**rank) is unique for each rank multiset and ADDITIVE, so
    key(hole pair) + key(board triple) = key(5-card hand). One table lookup
    then gives the score.
  * A flush is only possible when a suited hole pair meets a board triple
    of the same suit, so we use the flush table only there.

Card encoding is the same as evaluator.py: rank = c >> 2, suit = c & 3.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, combinations_with_replacement

import numpy as np

from evaluator import _score_ranks

NUM_CARDS = 36

# --- 5-card rank tables ------------------------------------------------------
P5 = np.array([5 ** r for r in range(9)], dtype=np.int64)
_SIZE = 5 ** 9
R_TABLE = np.zeros(_SIZE, np.int32)   # key -> non-flush score
F_TABLE = np.zeros(_SIZE, np.int32)   # key -> flush score (5 distinct ranks only)
for _combo in combinations_with_replacement(range(9), 5):
    if max(Counter(_combo).values()) > 4:
        continue
    _k = sum(5 ** r for r in _combo)
    _rd = tuple(sorted(_combo, reverse=True))
    R_TABLE[_k] = _score_ranks(_rd, False)
    if len(set(_combo)) == 5:
        F_TABLE[_k] = _score_ranks(_rd, True)

PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPS = np.array(list(combinations(range(5), 3)), dtype=np.int64)    # (10, 3)

_CARDS = np.arange(NUM_CARDS)
_RANK = _CARDS >> 2
_SUIT = _CARDS & 3
CARD_KEY = P5[_RANK]                                      # (36,) 5**rank

# Pair lookups for any two cards a, b.
PAIR_KEY = CARD_KEY[:, None] + CARD_KEY[None, :]          # (36, 36)
PAIR_SUIT = np.where(_SUIT[:, None] == _SUIT[None, :], _SUIT[:, None], -1).astype(np.int64)

# Every 2-card pair (a < b), indexed 0..629.
ALL_PAIRS = np.array(list(combinations(range(NUM_CARDS), 2)), dtype=np.int64)   # (630, 2)
PAIR_INDEX = np.full((NUM_CARDS, NUM_CARDS), -1, dtype=np.int64)
PAIR_INDEX[ALL_PAIRS[:, 0], ALL_PAIRS[:, 1]] = np.arange(len(ALL_PAIRS))
PAIR_INDEX[ALL_PAIRS[:, 1], ALL_PAIRS[:, 0]] = np.arange(len(ALL_PAIRS))


def _board_parts(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For (N, 5) boards: (N, 10) triple keys and (N, 10) triple suits (-2 if not monotone)."""
    tc = board[:, TRIPS]                                  # (N, 10, 3)
    bk = CARD_KEY[tc].sum(-1)
    ts = tc & 3
    mono = (ts[:, :, 0] == ts[:, :, 1]) & (ts[:, :, 1] == ts[:, :, 2])
    bsuit = np.where(mono, ts[:, :, 0], -2)
    return bk, bsuit


def omaha_batch(hole: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Scores for N hands: hole (N, 5) and board (N, 5) int arrays -> (N,) int32.

    Same result as evaluator.omaha_score row by row.
    """
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    a = hole[:, PAIRS[:, 0]]
    b = hole[:, PAIRS[:, 1]]
    hk = PAIR_KEY[a, b]                                   # (N, 10)
    hsuit = PAIR_SUIT[a, b]
    bk, bsuit = _board_parts(board)
    key = hk[:, :, None] + bk[:, None, :]                 # (N, 10, 10)
    flush = hsuit[:, :, None] == bsuit[:, None, :]
    sc = np.where(flush, F_TABLE[key], R_TABLE[key])
    return sc.reshape(len(hole), 100).max(1)


def omaha_one_board(holes: np.ndarray, board) -> np.ndarray:
    """Scores for many hands (K, 5) on ONE board (5 cards) -> (K,) int32."""
    holes = np.asarray(holes, dtype=np.int64)
    b = np.asarray(board, dtype=np.int64)[None, :]
    bk, bsuit = _board_parts(b)                           # (1, 10)
    a = holes[:, PAIRS[:, 0]]
    c = holes[:, PAIRS[:, 1]]
    hk = PAIR_KEY[a, c]
    hsuit = PAIR_SUIT[a, c]
    key = hk[:, :, None] + bk[0][None, None, :]
    flush = hsuit[:, :, None] == bsuit[0][None, None, :]
    sc = np.where(flush, F_TABLE[key], R_TABLE[key])
    return sc.reshape(len(holes), 100).max(1)


def pair_table(board) -> np.ndarray:
    """Best score for every 2-card hole pair on a full board.

    Returns a (630,) int32 array indexed like ALL_PAIRS / PAIR_INDEX. Pairs
    that use a board card get a score too (mask them yourself). A hand's
    score is then the max of its 10 pair entries (see score_from_pair_table).
    """
    b = np.asarray(board, dtype=np.int64)[None, :]
    bk, bsuit = _board_parts(b)
    hk = PAIR_KEY[ALL_PAIRS[:, 0], ALL_PAIRS[:, 1]]       # (630,)
    hsuit = PAIR_SUIT[ALL_PAIRS[:, 0], ALL_PAIRS[:, 1]]
    # Pairs that reuse a board card can form impossible keys; clamp them
    # (their entries are garbage and must be masked by the caller anyway).
    key = np.minimum(hk[:, None] + bk[0][None, :], _SIZE - 1)   # (630, 10)
    flush = hsuit[:, None] == bsuit[0][None, :]
    sc = np.where(flush, F_TABLE[key], R_TABLE[key])
    return sc.max(1).astype(np.int32)


_ALL_HK = PAIR_KEY[ALL_PAIRS[:, 0], ALL_PAIRS[:, 1]]      # (630,)
_ALL_HS = PAIR_SUIT[ALL_PAIRS[:, 0], ALL_PAIRS[:, 1]]

# Non-flush scores depend only on the pair's rank multiset: 45 of them.
_RP = sorted({(min(a >> 2, b >> 2), max(a >> 2, b >> 2)) for a, b in ALL_PAIRS})
_RP_KEY = np.array([5 ** a + 5 ** b for a, b in _RP], dtype=np.int64)            # (45,)
_RP_INDEX = {rp: i for i, rp in enumerate(_RP)}
RP_IDX = np.array([_RP_INDEX[(min(a >> 2, b >> 2), max(a >> 2, b >> 2))] for a, b in ALL_PAIRS])
# Flush scores of 5 distinct ranks, indexed by a 9-bit rank mask.
F_MASK = np.zeros(512, np.int32)
for _m in range(512):
    if bin(_m).count("1") == 5:
        _rs = [r for r in range(9) if _m >> r & 1]
        F_MASK[_m] = _score_ranks(tuple(sorted(_rs, reverse=True)), True)
SUITED = np.flatnonzero(_ALL_HS >= 0)                                             # (144,)
SUITED_SUIT = _ALL_HS[SUITED]
SUITED_MASK = ((1 << (ALL_PAIRS[SUITED, 0] >> 2)) | (1 << (ALL_PAIRS[SUITED, 1] >> 2))).astype(np.int64)
_RMASK = (1 << (_CARDS >> 2)).astype(np.int64)


def pair_tables(boards: np.ndarray) -> np.ndarray:
    """pair_table for R full boards at once: (R, 5) -> (R, 630) int32.

    Same values as pair_table() for every pair that doesn't reuse a board card.
    """
    boards = np.asarray(boards, dtype=np.int64)
    tc = boards[:, TRIPS]                                     # (R, 10, 3)
    bk = CARD_KEY[tc].sum(-1)                                 # (R, 10)
    nf = R_TABLE[np.minimum(_RP_KEY[None, None, :] + bk[:, :, None], _SIZE - 1)].max(1)  # (R, 45)
    out = nf[:, RP_IDX]                                       # (R, 630)
    ts = tc & 3
    mono = (ts[:, :, 0] == ts[:, :, 1]) & (ts[:, :, 1] == ts[:, :, 2])
    rows = np.flatnonzero(mono.any(1))
    if len(rows):
        bsuit = np.where(mono[rows], ts[rows, :, 0], -2)      # (R', 10)
        tmask = _RMASK[tc[rows]].sum(-1)                      # (R', 10) distinct ranks when mono
        match = bsuit[:, None, :] == SUITED_SUIT[None, :, None]   # (R', 144, 10)
        fl = np.where(match, F_MASK[(SUITED_MASK[None, :, None] | tmask[:, None, :]) & 511], 0).max(2)
        sub = out[rows]
        sub[:, SUITED] = np.maximum(sub[:, SUITED], fl)
        out[rows] = sub
    return out


_TRIPS_BY_LEN = {n: np.array(list(combinations(range(n), 3)), dtype=np.int64) for n in (3, 4, 5)}


def best_made_score(hole, board) -> int:
    """Best score of exactly 2 hole + 3 board cards for a 3-, 4- or 5-card board."""
    h = np.asarray(hole, dtype=np.int64)
    b = np.asarray(board, dtype=np.int64)
    tc = b[_TRIPS_BY_LEN[len(b)]]                                          # (T, 3)
    tk = CARD_KEY[tc].sum(-1)
    ts = tc & 3
    mono = (ts[:, 0] == ts[:, 1]) & (ts[:, 1] == ts[:, 2])
    tsuit = np.where(mono, ts[:, 0], -2)
    a, c = h[PAIRS[:, 0]], h[PAIRS[:, 1]]
    key = PAIR_KEY[a, c][:, None] + tk[None, :]
    flush = PAIR_SUIT[a, c][:, None] == tsuit[None, :]
    return int(np.where(flush, F_TABLE[key], R_TABLE[key]).max())


# For a (K, 5) hand array: the 10 pair indices of each hand.
def hand_pair_indices(holes: np.ndarray) -> np.ndarray:
    holes = np.asarray(holes, dtype=np.int64)
    return PAIR_INDEX[holes[:, PAIRS[:, 0]], holes[:, PAIRS[:, 1]]]   # (K, 10)


def score_from_pair_table(pt: np.ndarray, pair_idx: np.ndarray) -> np.ndarray:
    """Hand scores from a pair_table and (K, 10) hand_pair_indices -> (K,)."""
    return pt[pair_idx].max(-1)
