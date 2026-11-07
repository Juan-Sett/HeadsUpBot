"""Independent checks of bot/data/preflop.npz.

  1. 19,998 classes, combos sum to C(36,5) = 376,992
  2. combo-weighted mean equity vs random = 0.5 (zero-sum symmetry)
  3. recompute equity vs random for random classes with the ORIGINAL slow
     evaluator.omaha_score and plain random.sample (independent code path);
     the z-scores should look like N(0, 1)
  4. equity vs tighter ranges is lower on average
  5. single-hand and batch lookups agree; lookup speed
"""

import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from evaluator import omaha_score, warmup  # noqa: E402
from bot import preflop as pf  # noqa: E402


def main():
    warmup()
    d = np.load(os.path.join(ROOT, "bot", "data", "preflop.npz"))
    C = len(d["keys"])
    print(f"classes {C:,}, combos {int(d['combos'].sum()):,}, sims/class {int(d['sims'])}")
    w = d["combos"] / d["combos"].sum()
    print(f"combo-weighted mean equity vs random: {float((d['eq'][:, 0] * w).sum()):.4f}")
    for j, x in enumerate(["random", "top50", "top30", "top15", "top8"]):
        print(f"  mean equity vs {x:<6}: {float((d['eq'][:, j] * w).sum()):.4f}")

    rng = random.Random(7)
    zs = []
    n = 3000
    for _ in range(40):
        i = rng.randrange(C)
        hero = [int(c) for c in d["rep"][i]]
        rest = [c for c in range(36) if c not in hero]
        win = 0.0
        for _ in range(n):
            s = rng.sample(rest, 10)
            a, b = omaha_score(hero, s[5:]), omaha_score(s[:5], s[5:])
            win += 1.0 if a > b else 0.5 if a == b else 0.0
        e = win / n
        t = float(d["eq"][i, 0])
        se = (e * (1 - e) / n + t * (1 - t) / int(d["sims"])) ** 0.5
        zs.append((e - t) / se)
    zs = np.array(zs)
    print(f"independent recheck of 40 classes: z mean {zs.mean():+.2f}, z sd {zs.std():.2f}, "
          f"max |z| {np.abs(zs).max():.2f}  (expect ~0, ~1, < 3.3)")

    hands = np.array([random.Random(k).sample(range(36), 5) for k in range(2000)])
    single = np.array([pf.class_index(h) for h in hands])
    batch = pf.class_index_batch(hands)
    print(f"single vs batch lookup mismatches: {int((single != batch).sum())}")
    t = time.perf_counter()
    for h in hands:
        pf.eq_vs_top(h, 0.2)
    print(f"eq_vs_top lookup: {(time.perf_counter() - t) / len(hands) * 1e6:.1f} us")
    t = time.perf_counter()
    pf.class_index_batch(hands[:300])
    print(f"batch class lookup of 300 hands: {(time.perf_counter() - t) * 1e3:.3f} ms")
    order = np.argsort(-d["pct"])
    print("top 5 classes (equity vs random/top50/top30/top15/top8):")
    from evaluator import cards_to_str
    for i in order[:5]:
        print("  ", cards_to_str(sorted(int(c) for c in d["rep"][i])), np.round(d["eq"][i], 3))
    print("bottom 3:")
    for i in order[-3:]:
        print("  ", cards_to_str(sorted(int(c) for c in d["rep"][i])), np.round(d["eq"][i], 3))
    ok = C == 19998 and int(d["combos"].sum()) == 376992 and np.abs(zs).max() < 4 and (single == batch).all()
    print("ALL PREFLOP CHECKS PASSED" if ok else "PREFLOP CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
