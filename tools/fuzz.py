"""Phase 5 fuzz test: push many random game states through the strategy.

Each worker plays hands against a random opponent whose aggression changes
every hand (never raises ... always raises), with a fresh half (new memory)
every 1000 hands, hand results delivered late in batches like the judge.
Some workers run with a tiny time budget so the heuristic path is fuzzed too.

Counts: exceptions escaping choose_action / on_hand_end, exceptions caught
inside the strategy (diag errors), and illegal actions.

    python tools/fuzz.py [--decisions 200000] [--jobs 6]
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def worker(args):
    wid, target, low_budget = args
    import numpy as np
    from game import Action, AgentConfig, Game, make_deal
    from bot import strategy
    if low_budget:
        strategy.configure({"time_target": 0.05})
    rng = np.random.default_rng(1000 + wid)
    decisions = escaped = internal = illegal = 0
    first = []
    hand = 0
    mem = {}
    pending = []
    while decisions < target:
        if hand % 1000 == 0:
            mem = {}
            strategy.new_half(AgentConfig(), mem)
        g = Game(make_deal(f"fuzz{wid}", hand), hand % 2, hand_index=hand % 1000)
        rate = float(rng.choice([0.0, 0.2, 0.5, 0.8, 1.0]))
        while not g.is_terminal:
            s = g.to_act
            obs = g.observation(s)
            if s == 0:
                for r in pending:
                    try:
                        strategy.on_hand_end(r, mem)
                    except Exception:
                        escaped += 1
                        first.append(traceback.format_exc())
                pending = []
                try:
                    a = strategy.choose_action(obs, mem)
                except Exception:
                    escaped += 1
                    first.append(traceback.format_exc())
                    a = Action.CHECK_CALL
                if a not in obs.legal_actions:
                    illegal += 1
                    a = Action.CHECK_CALL
                decisions += 1
            else:
                raises = [x for x in obs.legal_actions if x in (Action.RAISE_HALF, Action.RAISE_POT)]
                if raises and rng.random() < rate:
                    a = raises[int(rng.integers(len(raises)))]
                elif Action.FOLD in obs.legal_actions and rng.random() < 0.15 * (1 - rate):
                    a = Action.FOLD
                else:
                    a = Action.CHECK_CALL
            g.apply(a)
        pending.append(g.hand_result(0))
        if hand % 1000 == 999:
            internal += mem["diag"]["errors"]
            first += mem["diag"]["error_msgs"][:1]
        hand += 1
    internal += mem["diag"]["errors"]
    return decisions, escaped, internal, illegal, first[:2], hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=200_000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    per = args.decisions // args.jobs + 1
    tasks = [(i, per, i % 3 == 2) for i in range(args.jobs)]
    t = time.perf_counter()
    with mp.Pool(args.jobs) as pool:
        res = pool.map(worker, tasks)
    dec = sum(r[0] for r in res)
    esc = sum(r[1] for r in res)
    inte = sum(r[2] for r in res)
    ill = sum(r[3] for r in res)
    hands = sum(r[5] for r in res)
    print(f"fuzz: {dec:,} decisions over {hands:,} hands in {time.perf_counter() - t:.0f}s: "
          f"{esc} escaped exceptions, {inte} caught internal errors, {ill} illegal actions")
    for r in res:
        for msg in r[4]:
            print(msg)
            break
    ok = esc == 0 and inte == 0 and ill == 0
    print("FUZZ TEST PASSED" if ok else "FUZZ TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
