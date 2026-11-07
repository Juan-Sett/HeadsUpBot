"""Phase 6 tuning: coordinate search over strategy parameters.

Every configuration plays the SAME training seeds against the same opponents
(common random numbers), so differences between configurations are mostly
strategy, not card luck. Score = expected tournament points (P(win) + 0.5 P(chop)
per opponent, from the match mean and sd) + 0.05 * average bb/deal (tiebreak).

    python tools/tune.py                       # one pass over the default grid
    python tools/tune.py --base '{"rho_oop": 0.8}' --seeds 4

Results are appended to results/tune_log.jsonl.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import sim  # noqa: E402

OPPONENTS = ["guide", "random", "always_pot", "nit", "overfolder"]
# Deterministic evaluation: fixed compute (no time-dependent particle/runout
# counts, no time gating), so the same seeds replay exactly the same games and
# differences between configurations are purely the parameters.
FIXED = {"_unlimited": True, "time_target": 1000, "max_decision": 1.0, "k": 120, "k_min": 120,
         "r_flop": 8, "r_flop_min": 8, "r_flop_max": 8, "r_turn": 8, "r_turn_min": 8, "r_turn_max": 8}
GRID = {
    "rho_oop": [0.80, 0.95],
    "rho_ip": [0.92, 1.08],
    "steal_w": [0.5, 1.5],
    "sd_pre": [0.12, 0.28],
    "lam_flop": [0.5, 0.9],
    "commit_spr": [1.0, 2.5],
    "tau": [0.05, 0.10],
    "prior_n": [8.0, 25.0],
}


def evaluate(params, seeds, opponents, pool):
    rows = {}
    for opp in opponents:
        r = sim.run_matches("agent", opp, seeds, 1000, params_a=dict(FIXED, **params), pool=pool)
        rows[opp] = {"mean": r["mean"], "pts": r["exp_points"], "p_win": r["p_win"],
                     "think": r["think_a_max"]}
    pts = sum(v["pts"] for v in rows.values())
    avg = sum(v["mean"] for v in rows.values()) / len(rows)
    return pts + 0.05 * avg, pts, avg, rows


def fmt(rows):
    return "  ".join(f"{k[:6]} {v['mean']:+.2f}" for k, v in rows.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="{}", help="JSON starting parameters")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-base", default="tune")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--params", default=",".join(GRID), help="comma list of params to search")
    ap.add_argument("--grid", help="JSON grid overriding the default grid")
    args = ap.parse_args()
    seeds = [f"{args.seed_base}-{i}" for i in range(args.seeds)]
    if args.grid:
        GRID.clear()
        GRID.update(json.loads(args.grid))
        args.params = ",".join(GRID)
    best = json.loads(args.base)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    log = open(os.path.join(ROOT, "results", "tune_log.jsonl"), "a")
    t0 = time.perf_counter()
    with mp.Pool(args.jobs) as pool:
        score, pts, avg, rows = evaluate(best, seeds, OPPONENTS, pool)
        print(f"base {best}: score {score:.3f} (pts {pts:.2f}, avg {avg:+.2f})  {fmt(rows)}", flush=True)
        log.write(json.dumps({"params": best, "score": score, "rows": rows}) + "\n")
        for name in args.params.split(","):
            for v in GRID[name]:
                cand = dict(best, **{name: v})
                s, p, a, rw = evaluate(cand, seeds, OPPONENTS, pool)
                log.write(json.dumps({"params": cand, "score": s, "rows": rw}) + "\n")
                log.flush()
                mark = ""
                if s > score + 0.02:
                    best, score, mark = cand, s, "  <- better"
                print(f"{name}={v}: score {s:.3f} (pts {p:.2f}, avg {a:+.2f})  {fmt(rw)}{mark}", flush=True)
    print(f"best {best}: score {score:.3f}  ({time.perf_counter() - t0:.0f}s)")
    print("BEST_JSON " + json.dumps(best))


if __name__ == "__main__":
    main()
