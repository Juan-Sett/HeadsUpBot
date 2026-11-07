"""Phase 2 correctness test: with uniform weights (no opponent actions), the
particle-model equity must match a direct Monte Carlo vs a random hand.

For each of 300 random spots (100 flop, 100 turn, 100 river):
  * particle estimate: K=200 particles, the production runout counts
  * its standard error: the spread of 30 independent particle estimates
  * reference: direct MC vs a random hand with 20,000 samples
  * z = (particle - reference) / sqrt(se_p^2 + se_ref^2)
Pass if at least 93% of |z| < 2 (expected ~95%) and z looks like N(0, 1).

    python tools/validate_rangemodel.py
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from bot import fasteval as fe  # noqa: E402
from bot import rangemodel as rm  # noqa: E402

K = 200
R_BY_STREET = {3: 12, 4: 40, 5: 1}     # flop 12 runouts, turn every river, river exact
N_REF_MC = 20000
REPS = 30


def direct_mc(hole, board, n, rng):
    avail = rm.available(hole, board)
    need = 5 - len(board)
    draw = rng.permuted(np.tile(avail, (n, 1)), axis=1)[:, :5 + need]
    full = np.concatenate([np.tile(np.asarray(board), (n, 1)), draw[:, 5:]], 1)
    me = fe.omaha_batch(np.tile(np.asarray(hole), (n, 1)), full)
    op = fe.omaha_batch(draw[:, :5], full)
    wins = (me > op) + 0.5 * (me == op)
    return wins.mean(), wins.std() / np.sqrt(n)


def particle_eq(hole, board, rng):
    hands = rm.sample_hands(rng, rm.available(hole, board), K)
    win, _, _ = rm.evaluate(hole, board, hands, R_BY_STREET[len(board)], rng)
    return win.mean()


def main():
    rng = np.random.default_rng(99)
    zs = {3: [], 4: [], 5: []}
    times = {3: [], 4: [], 5: []}
    for blen in (3, 4, 5):
        for _ in range(100):
            cards = rng.permutation(36)
            hole, board = np.sort(cards[:5]), cards[5:5 + blen]
            t = time.perf_counter()
            est = particle_eq(hole, board, rng)
            times[blen].append(time.perf_counter() - t)
            reps = [particle_eq(hole, board, rng) for _ in range(REPS)]
            se_p = np.std(reps, ddof=1)
            ref, se_r = direct_mc(hole, board, N_REF_MC, rng)
            zs[blen].append((est - ref) / np.sqrt(se_p ** 2 + se_r ** 2 + 1e-12))
    allz = np.concatenate([zs[b] for b in zs])
    names = {3: "flop", 4: "turn", 5: "river"}
    for b in (3, 4, 5):
        z = np.array(zs[b])
        print(f"{names[b]:<5}: within 2 SE {np.mean(np.abs(z) < 2):5.1%}  z mean {z.mean():+.2f}  "
              f"z sd {z.std():.2f}  eval {np.mean(times[b]) * 1e3:.2f} ms (K={K}, R={R_BY_STREET[b]})")
    share = np.mean(np.abs(allz) < 2)
    print(f"all  : within 2 SE {share:5.1%} of 300 spots, z mean {allz.mean():+.2f}, z sd {allz.std():.2f}")
    ok = share >= 0.93 and abs(allz.mean()) < 0.25 and 0.75 < allz.std() < 1.3
    print("RANGE MODEL EQUITY TEST PASSED" if ok else "RANGE MODEL EQUITY TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
