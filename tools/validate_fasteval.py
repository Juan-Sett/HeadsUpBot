"""Check bot/fasteval.py against evaluator.omaha_score and benchmark it.

    python tools/validate_fasteval.py [--hands 500000]
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from evaluator import omaha_score, warmup  # noqa: E402
from bot import fasteval as fe  # noqa: E402


def random_deals(rng, n):
    """(n, 10) arrays of 10 distinct cards each: hole = [:5], board = [5:]."""
    return np.argsort(rng.random((n, 36)), axis=1)[:, :10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=500_000)
    args = ap.parse_args()
    warmup()
    rng = np.random.default_rng(12345)
    deals = random_deals(rng, args.hands)
    hole, board = deals[:, :5], deals[:, 5:]

    t = time.perf_counter()
    fast = np.concatenate([fe.omaha_batch(hole[i:i + 20000], board[i:i + 20000])
                           for i in range(0, args.hands, 20000)])
    t_fast = time.perf_counter() - t

    t = time.perf_counter()
    ref = np.array([omaha_score(h.tolist(), b.tolist()) for h, b in zip(hole, board)])
    t_ref = time.perf_counter() - t

    mism = int((fast != ref).sum())
    print(f"omaha_batch vs omaha_score on {args.hands:,} hands: {mism} mismatches")
    print(f"  omaha_score : {t_ref / args.hands * 1e6:6.2f} us/hand")
    print(f"  omaha_batch : {t_fast / args.hands * 1e6:6.2f} us/hand  ({t_ref / t_fast:.0f}x faster)")

    # One board, many hands (river / particle use).
    bad = 0
    t_one = t_pt = 0.0
    for trial in range(300):
        cards = np.argsort(rng.random(36))
        bd = cards[:5]
        rest = cards[5:]
        hands = np.stack([rng.permutation(rest)[:5] for _ in range(200)])
        t = time.perf_counter(); s1 = fe.omaha_one_board(hands, bd); t_one += time.perf_counter() - t
        t = time.perf_counter()
        pt = fe.pair_table(bd); s2 = fe.score_from_pair_table(pt, fe.hand_pair_indices(hands))
        t_pt += time.perf_counter() - t
        r = np.array([omaha_score(h.tolist(), bd.tolist()) for h in hands[:20]])
        bad += int((s1[:20] != r).sum() + (s2 != s1).sum())
    print(f"one-board paths (300 boards x 200 hands): {bad} mismatches")
    print(f"  omaha_one_board      : {t_one / 300 * 1e3:.3f} ms per 200 hands")
    print(f"  pair_table + lookup  : {t_pt / 300 * 1e3:.3f} ms per 200 hands")

    # Small batch latency (what one decision sees).
    for n in (10, 100, 1000):
        h, b = hole[:n], board[:n]
        reps = max(20, 20000 // n)
        t = time.perf_counter()
        for _ in range(reps):
            fe.omaha_batch(h, b)
        dt = (time.perf_counter() - t) / reps
        print(f"  omaha_batch batch={n:5d}: {dt * 1e3:.3f} ms/call = {dt / n * 1e6:.2f} us/hand")
    return 0 if mism == 0 and bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
