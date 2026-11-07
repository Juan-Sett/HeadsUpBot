"""
build_preflop.py - offline: preflop equity tables for every starting-hand class.

    python tools/build_preflop.py [--sims 6000] [--jobs 8]

Writes bot/data/preflop.npz with:
    keys    (C,) int64   sorted suit-isomorphism class keys
    combos  (C,) int32   raw 5-card hands in each class (sums to 376,992)
    rep     (C, 5) int8  one representative hand per class
    eq      (C, 5) float32  equity vs: random, top 50%, top 30%, top 15%, top 8%
    pct     (C,) float32 combo-weighted percentile by equity vs random (1 = best)
    range_x (4,) float32 the X values of eq[:, 1:]

Class key: per-suit 9-bit rank masks, sorted, packed into one int.
Equity counts ties as 1/2. Opponent range hands are drawn uniformly from the
raw hands in the range (so it is combo-weighted) and must not share a card
with ours, so card removal is handled exactly.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

RANGE_X = (0.50, 0.30, 0.15, 0.08)
OUT = os.path.join(ROOT, "bot", "data", "preflop.npz")


def class_keys(hands: np.ndarray) -> np.ndarray:
    """(N, 5) hands -> (N,) int64 suit-isomorphism keys."""
    hands = np.asarray(hands, dtype=np.int64)
    bits = np.left_shift(1, hands >> 2)                       # (N, 5) rank bit
    suit = hands & 3
    m = np.zeros((len(hands), 4), dtype=np.int64)
    for s in range(4):
        m[:, s] = np.where(suit == s, bits, 0).sum(1)
    m.sort(1)
    return m[:, 0] | (m[:, 1] << 9) | (m[:, 2] << 18) | (m[:, 3] << 27)


def all_hands() -> np.ndarray:
    return np.array(list(combinations(range(36), 5)), dtype=np.int64)


# ---- worker state (set once per process) -----------------------------------
_W: dict = {}


def _init(range_hands, sims, seed):
    from bot import fasteval  # noqa: F401  (build tables once per process)
    _W["fe"] = fasteval
    _W["range_hands"] = range_hands            # list of (M_x, 5) arrays, or None
    _W["sims"] = sims
    _W["seed"] = seed


def _eq_random(rep_chunk):
    """Equity vs a random hand for each rep hand (pass 1)."""
    fe = _W["fe"]
    n = _W["sims"]
    out = np.zeros(len(rep_chunk), dtype=np.float64)
    for i, (idx, hero) in enumerate(rep_chunk):
        rng = np.random.default_rng((_W["seed"], 1, int(idx)))
        rest = np.setdiff1d(np.arange(36), hero)
        draw = rng.permuted(np.tile(rest, (n, 1)), axis=1)[:, :10]
        opp, board = draw[:, :5], draw[:, 5:]
        me = fe.omaha_batch(np.tile(hero, (n, 1)), board)
        op = fe.omaha_batch(opp, board)
        out[i] = ((me > op).sum() + 0.5 * (me == op).sum()) / n
    return out


def _eq_ranges(rep_chunk):
    """Equity vs each top-X range for each rep hand (pass 2)."""
    fe = _W["fe"]
    n = _W["sims"]
    out = np.zeros((len(rep_chunk), len(RANGE_X)), dtype=np.float64)
    for i, (idx, hero) in enumerate(rep_chunk):
        rng = np.random.default_rng((_W["seed"], 2, int(idx)))
        hero_mask = np.zeros(36, dtype=bool)
        hero_mask[hero] = True
        hero_tile = np.tile(hero, (n, 1))
        for j, rh in enumerate(_W["range_hands"]):
            opp = np.empty((0, 5), dtype=np.int64)
            while len(opp) < n:
                cand = rh[rng.integers(0, len(rh), size=3 * n)]
                ok = ~hero_mask[cand].any(1)
                opp = np.concatenate([opp, cand[ok]])
            opp = opp[:n]
            keys = rng.random((n, 36))
            keys[:, hero] = 2.0
            np.put_along_axis(keys, opp, 2.0, axis=1)
            board = np.argpartition(keys, 5, axis=1)[:, :5]
            me = fe.omaha_batch(hero_tile, board)
            op = fe.omaha_batch(opp, board)
            out[i, j] = ((me > op).sum() + 0.5 * (me == op).sum()) / n
    return out


def _chunks(items, k):
    size = (len(items) + k - 1) // k
    return [items[i:i + size] for i in range(0, len(items), size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=6000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--pass1-only", action="store_true",
                    help="skip the range pass (range columns copy the vs-random column)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0, help="only the first N classes (testing)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    hands = all_hands()
    keys = class_keys(hands)
    ukeys, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True,
                                              return_counts=True)
    C = len(ukeys)
    reps = hands[first]
    print(f"{len(hands):,} hands -> {C:,} classes ({time.perf_counter() - t0:.1f}s)", flush=True)
    sel = np.arange(C) if not args.limit else np.arange(args.limit)
    items = [(int(i), reps[i]) for i in sel]
    nchunks = args.jobs * 16

    # Pass 1: equity vs a random hand.
    t = time.perf_counter()
    with mp.Pool(args.jobs, initializer=_init, initargs=(None, args.sims, args.seed)) as pool:
        eq_r = np.concatenate(pool.map(_eq_random, _chunks(items, nchunks)))
    print(f"pass 1 (vs random, {args.sims} sims/class): {time.perf_counter() - t:.0f}s", flush=True)
    if args.limit:
        print(eq_r[:10])
        return

    # Combo-weighted percentile (1 = best).
    order = np.argsort(eq_r, kind="stable")
    cum = np.cumsum(counts[order])
    pct = np.empty(C)
    pct[order] = (cum - 0.5 * counts[order]) / cum[-1]
    hand_pct = pct[inverse]
    range_hands = [hands[hand_pct >= 1.0 - x] for x in RANGE_X]
    for x, rh in zip(RANGE_X, range_hands):
        print(f"  top {x:.0%}: {len(rh):,} hands ({len(rh) / len(hands):.1%})", flush=True)

    if args.pass1_only:
        eq_x = np.tile(eq_r[:, None], (1, len(RANGE_X)))
    else:
        # Pass 2: equity vs each top-X range.
        t = time.perf_counter()
        with mp.Pool(args.jobs, initializer=_init, initargs=(range_hands, args.sims, args.seed)) as pool:
            eq_x = np.concatenate(pool.map(_eq_ranges, _chunks(items, nchunks)))
        print(f"pass 2 (vs ranges): {time.perf_counter() - t:.0f}s", flush=True)

    eq = np.column_stack([eq_r, eq_x]).astype(np.float32)
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, keys=ukeys.astype(np.int64), combos=counts.astype(np.int32),
                        rep=reps.astype(np.int8), eq=eq, pct=pct.astype(np.float32),
                        range_x=np.array(RANGE_X, dtype=np.float32),
                        sims=np.array(args.sims), has_ranges=np.array(not args.pass1_only))
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB), total {time.perf_counter() - t0:.0f}s")
    w = counts / counts.sum()
    print(f"check: combo-weighted mean equity vs random = {(eq_r * w).sum():.4f} (should be ~0.5)")
    best = order[::-1][:5]
    for i in best:
        print("  best:", reps[i].tolist(), np.round(eq[i], 3).tolist())


if __name__ == "__main__":
    main()
