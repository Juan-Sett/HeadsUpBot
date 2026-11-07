"""
sim.py - fast in-process duplicate-match simulator (no HTTP).

Mirrors judge_engine.py:
  * deals come from game.make_deal(seed, d); button = d % 2
  * half 1: bot A in seat 0; half 2: seats swapped, same cards
  * each half loads a FRESH copy of each bot module (globals reset) with a
    fresh memory dict, then calls new_half(config, memory)
  * finished hands are queued per bot and delivered (on_hand_end) right
    before that bot's next decision; leftovers are flushed at half end
  * exceptions / non-actions / illegal actions / decisions over 1 s count as
    errors and the default action is played; 5 errors = forfeit
  * thinking time is charged against an 8 s bank per half; once it is empty
    the bot plays check/fold for the rest of the half (not an error)

Thinking time here is pure compute time (no network grace subtracted). Real
HTTP adds some overhead per request on top; see the final judge check.

Usage:
    python sim.py agent random --seeds 5 --deals 1000
    python sim.py agent --pool baselines        # agent vs every bot in the pool
    python sim.py agent --pool all --seeds 5

Bot names: a file path, or a name from opponents/ (e.g. guide, nit),
or a baseline name (random, calling_station, always_half, always_pot),
or "agent" for agent.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game import (  # noqa: E402
    ACTION_TIMEOUT_SECONDS, BIG_BLIND, CHOP_Z, ERROR_LIMIT, NUM_DEALS, TIME_BANK_SECONDS,
    Action, AgentConfig, Game, button_for_deal, default_action, make_deal,
)

BASELINES = ["random", "calling_station", "always_half", "always_pot"]
ARCHETYPES = ["guide", "nit", "maniac", "overfolder", "timebank_death"]
HELDOUT = ["tag", "lag"]           # never used for tuning
POOLS = {
    "baselines": BASELINES,
    "archetypes": ARCHETYPES,
    "all": BASELINES + ARCHETYPES,
    "heldout": HELDOUT,
    "everything": BASELINES + ARCHETYPES + HELDOUT,
}

# Module names that belong to a bot and must be re-imported fresh per half.
_BOT_PACKAGES = ("bot",)


def default_jobs() -> int:
    """Leave two cores free so the machine stays responsive."""
    return max(1, (os.cpu_count() or 4) - 2)


def resolve_bot(name: str) -> str:
    """Turn a bot name into a file path."""
    if os.path.isfile(name):
        return os.path.abspath(name)
    candidates = [
        os.path.join(ROOT, f"{name}.py"),
        os.path.join(ROOT, "opponents", f"{name}.py"),
        os.path.join(ROOT, "opponents", name, "agent.py"),
        os.path.join(ROOT, "opponents", "frozen", name, "agent.py"),
        os.path.join(ROOT, "baseline-bots", f"baseline_{name}.py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"no bot called {name!r}")


def _purge_bot_modules() -> None:
    for mod in list(sys.modules):
        if mod in _BOT_PACKAGES or mod.split(".")[0] in _BOT_PACKAGES:
            del sys.modules[mod]


_LOAD_COUNTER = [0]


def load_bot(path: str, params: Optional[dict] = None):
    """Import a fresh copy of a bot file (its own globals and its own copy
    of the `bot` package next to it). Returns the module."""
    _LOAD_COUNTER[0] += 1
    root = os.path.dirname(path)
    _purge_bot_modules()
    sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location(f"simbot_{_LOAD_COUNTER[0]}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(root)
        _purge_bot_modules()   # the module keeps its own references
    if params and hasattr(module, "configure"):
        module.configure(params)
    return module


@dataclass
class BotStats:
    think: float = 0.0              # seconds charged to the bank
    decisions: int = 0
    max_decision: float = 0.0
    errors: int = 0
    forfeited: bool = False
    timed_out: bool = False
    skipped: int = 0
    startup: float = 0.0            # module import + new_half
    messages: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors += 1
        if len(self.messages) < 5:
            self.messages.append(msg)
        if self.errors >= ERROR_LIMIT:
            self.forfeited = True


class SimBot:
    """One bot in one half, called the way the judge calls it."""

    def __init__(self, path: str, config: AgentConfig, params: Optional[dict] = None):
        self.stats = BotStats()
        t = time.perf_counter()
        self.mod = load_bot(path, params)
        if params and params.get("_unlimited"):
            self.mod.UNLIMITED_TIME = True      # debugging: no time bank
        self.memory: dict = {}
        self.pending: list = []
        self.config = config
        try:
            self._timed(self.mod.new_half, config, self.memory)
        except Exception:
            self.stats.forfeited = True
            self.stats.messages.append("new_half raised\n" + traceback.format_exc())
        self.stats.startup = time.perf_counter() - t

    def _timed(self, fn, *args):
        t = time.perf_counter()
        try:
            return fn(*args)
        finally:
            dt = time.perf_counter() - t
            self.stats.think += dt
            self._last_dt = dt

    def _usable(self) -> bool:
        if self.stats.forfeited:
            return False
        # Reference archetypes can opt out of the time bank (strategy-only bots).
        if getattr(self.mod, "UNLIMITED_TIME", False):
            return True
        if self.stats.think >= self.config.time_bank_seconds:
            self.stats.timed_out = True
            self.stats.skipped += 1
            return False
        return True

    def _deliver(self) -> float:
        """on_hand_end for every queued hand; returns the time spent."""
        spent = 0.0
        results, self.pending = self.pending, []
        for r in results:
            try:
                self._timed(self.mod.on_hand_end, r, self.memory)
            except Exception:
                self.stats.error("on_hand_end raised\n" + traceback.format_exc())
            spent += self._last_dt
        return spent

    def act(self, obs) -> Action:
        fallback = default_action(obs.to_call)
        if not self._usable():
            return fallback
        spent = self._deliver()
        if self.stats.forfeited:
            return fallback
        try:
            action = self._timed(self.mod.choose_action, obs, self.memory)
        except Exception:
            self.stats.error("choose_action raised\n" + traceback.format_exc())
            return fallback
        dt = self._last_dt + spent             # one judge request = deliveries + decision
        self.stats.decisions += 1
        self.stats.max_decision = max(self.stats.max_decision, dt)
        if dt > ACTION_TIMEOUT_SECONDS:
            self.stats.error(f"request took {dt:.2f}s")
            return fallback
        if isinstance(action, bool) or not isinstance(action, (Action, int)):
            self.stats.error(f"returned {action!r}, not an Action")
            return fallback
        try:
            action = Action(action)
        except ValueError:
            self.stats.error(f"returned {action!r}, not an Action")
            return fallback
        if action not in obs.legal_actions:
            self.stats.error(f"illegal {action.name}, legal {[a.name for a in obs.legal_actions]}")
            return fallback
        return action

    def hand_end(self, result) -> None:
        if not self.stats.forfeited:
            self.pending.append(result)

    def flush(self) -> None:
        if self.pending and self._usable():
            self._deliver()


def play_half(path_a: str, path_b: str, seed: str, half: int, deals: int,
              params_a: Optional[dict] = None, params_b: Optional[dict] = None,
              keep_logs: bool = False) -> dict:
    """Play one half. Returns nets for bot A per deal and both bots' stats."""
    config = AgentConfig(num_hands=deals)
    a_seat = 0 if half == 1 else 1
    # Both start fresh, like two new containers.
    bot_a = SimBot(path_a, config, params_a)
    bot_b = SimBot(path_b, config, params_b)
    seats = (bot_a, bot_b) if a_seat == 0 else (bot_b, bot_a)
    nets_a = []
    logs = [] if keep_logs else None
    t0 = time.perf_counter()
    for d in range(deals):
        game = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not game.is_terminal:
            s = game.to_act
            game.apply(seats[s].act(game.observation(s)))
        res = game.result()
        nets_a.append(res.nets[a_seat])
        for s in (0, 1):
            seats[s].hand_end(game.hand_result(s))
        if logs is not None:
            logs.append({"deal": d, "button": button_for_deal(d), "nets": list(res.nets),
                         "actions": [r.to_dict() for r in res.history],
                         "showdown": res.went_to_showdown})
    for b in seats:
        b.flush()
    for b in (bot_a, bot_b):
        b.stats.diag = _diag_summary(b.memory)
    out = {"seed": seed, "half": half, "nets_a": nets_a,
           "stats_a": bot_a.stats.__dict__, "stats_b": bot_b.stats.__dict__,
           "elapsed": time.perf_counter() - t0}
    if logs is not None:
        out["logs"] = logs
    return out


def _diag_summary(memory: dict) -> dict:
    """Small summary of a bot's own diagnostics (if it keeps memory["diag"])."""
    d = memory.get("diag") if isinstance(memory, dict) else None
    if not isinstance(d, dict):
        return {}
    out = {k: v for k, v in d.items() if isinstance(v, (int, float))}
    ess = d.get("ess_min")
    if ess:
        import numpy as np
        e = np.array(ess)
        out["ess_p10"] = float(np.percentile(e, 10))
        out["ess_median"] = float(np.median(e))
    if d.get("error_msgs"):
        out["first_error"] = d["error_msgs"][0][-600:]
    return out


def _half_task(args):
    try:
        return play_half(*args)
    except Exception:
        return {"crash": traceback.format_exc(), "args": [str(a) for a in args[:5]]}


def summarize_match(h1: dict, h2: dict) -> dict:
    """Pair two halves of one seed -> mean bb/deal, sd, CI, verdict for bot A."""
    n = len(h1["nets_a"])
    paired = [(h1["nets_a"][d] + h2["nets_a"][d]) / BIG_BLIND for d in range(n)]
    mean = sum(paired) / n
    sd = statistics.stdev(paired) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    lo, hi = mean - CHOP_Z * se, mean + CHOP_Z * se
    verdict = "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"
    fa = h1["stats_a"]["forfeited"] or h2["stats_a"]["forfeited"]
    fb = h1["stats_b"]["forfeited"] or h2["stats_b"]["forfeited"]
    if fa and fb:
        verdict = "BOTH_FORFEIT"
    elif fa:
        verdict = "LOSS(forfeit)"
    elif fb:
        verdict = "WIN(forfeit)"
    return {"seed": h1["seed"], "n": n, "mean": mean, "sd": sd, "ci": (lo, hi),
            "verdict": verdict, "paired": paired}


def run_matches(bot_a: str, bot_b: str, seeds: list[str], deals: int = NUM_DEALS,
                jobs: int = 0, params_a: Optional[dict] = None,
                params_b: Optional[dict] = None, pool: Optional[mp.pool.Pool] = None) -> dict:
    """Play one duplicate match per seed. Returns per-seed and pooled results."""
    pa, pb = resolve_bot(bot_a), resolve_bot(bot_b)
    tasks = [(pa, pb, s, h, deals, params_a, params_b) for s in seeds for h in (1, 2)]
    if pool is not None:
        halves = pool.map(_half_task, tasks)
    elif jobs == 1:
        halves = [_half_task(t) for t in tasks]
    else:
        with mp.Pool(jobs or default_jobs()) as p:
            halves = p.map(_half_task, tasks)
    crashes = [h for h in halves if "crash" in h]
    if crashes:
        raise RuntimeError("a half crashed inside the simulator:\n" + crashes[0]["crash"])
    matches = [summarize_match(halves[2 * i], halves[2 * i + 1]) for i in range(len(seeds))]
    return aggregate(bot_a, bot_b, matches, halves)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def aggregate(bot_a: str, bot_b: str, matches: list[dict], halves: list[dict]) -> dict:
    all_paired = [x for m in matches for x in m["paired"]]
    n_deal = matches[0]["n"]
    mean = sum(all_paired) / len(all_paired)
    sd = statistics.stdev(all_paired) if len(all_paired) > 1 else 0.0
    se_pool = sd / math.sqrt(len(all_paired))
    # Per-match model: a match mean ~ N(mu, sd^2 / n). P(win) = P(mean > 1.96 se).
    if sd > 0:
        z = mean / (sd / math.sqrt(n_deal))
        p_win, p_loss = _phi(z - CHOP_Z), _phi(-z - CHOP_Z)
    else:  # every deal identical: the interval is a single point
        p_win, p_loss = float(mean > 0), float(mean < 0)
    p_chop = 1.0 - p_win - p_loss
    verdicts = [m["verdict"] for m in matches]
    think_a = [h["stats_a"]["think"] for h in halves]
    think_b = [h["stats_b"]["think"] for h in halves]
    dec_a = sum(h["stats_a"]["decisions"] for h in halves)
    return {
        "bot_a": bot_a, "bot_b": bot_b, "seeds": len(matches), "deals": n_deal,
        "mean": mean, "sd": sd, "ci_pooled": (mean - CHOP_Z * se_pool, mean + CHOP_Z * se_pool),
        "match_means": [m["mean"] for m in matches],
        "verdicts": verdicts,
        "wins": sum(v.startswith("WIN") for v in verdicts),
        "chops": sum(v == "CHOP" for v in verdicts),
        "losses": sum(v.startswith("LOSS") or v == "BOTH_FORFEIT" for v in verdicts),
        "p_win": p_win, "p_chop": p_chop, "p_loss": p_loss,
        "exp_points": p_win + 0.5 * p_chop,
        "think_a_max": max(think_a), "think_a_mean": sum(think_a) / len(think_a),
        "think_b_max": max(think_b),
        "per_decision_a_ms": 1000 * sum(think_a) / max(1, dec_a),
        "max_decision_a_ms": 1000 * max(h["stats_a"]["max_decision"] for h in halves),
        "startup_a_max": max(h["stats_a"]["startup"] for h in halves),
        "errors_a": sum(h["stats_a"]["errors"] for h in halves),
        "errors_b": sum(h["stats_b"]["errors"] for h in halves),
        "forfeits_a": sum(h["stats_a"]["forfeited"] for h in halves),
        "timeouts_a": sum(h["stats_a"]["timed_out"] for h in halves),
        "timeouts_b": sum(h["stats_b"]["timed_out"] for h in halves),
        "error_messages_a": [m for h in halves for m in h["stats_a"]["messages"]][:3],
        "diag_a": _merge_diag([h["stats_a"].get("diag", {}) for h in halves]),
        "paired_by_seed": [m["paired"] for m in matches],
    }


def _merge_diag(ds: list) -> dict:
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            if k == "first_error":
                out.setdefault(k, v)
            elif k.startswith("ess_"):
                out.setdefault(k, []).append(v)
            elif isinstance(v, (int, float)):
                out[k] = out.get(k, 0) + v
    for k in [k for k in out if k.startswith("ess_")]:
        out[k] = sum(out[k]) / len(out[k])
    return out


def format_row(r: dict) -> str:
    lo, hi = r["ci_pooled"]
    half_ci = (hi - lo) / 2
    flags = []
    if r["errors_a"]:
        flags.append(f"ERR {r['errors_a']}")
    if r["timeouts_a"]:
        flags.append("A TIMEOUT")
    if r["timeouts_b"]:
        flags.append("opp timeout")
    return (f"{r['bot_b']:<16} {r['mean']:+7.3f} +- {half_ci:5.3f}  "
            f"W/C/L {r['wins']}/{r['chops']}/{r['losses']}  "
            f"P(win) {r['p_win']:5.1%}  E[pts] {r['exp_points']:.2f}  "
            f"think/half {r['think_a_mean']:.2f}s (max {r['think_a_max']:.2f}s, "
            f"{r['per_decision_a_ms']:.2f} ms/dec, worst {r['max_decision_a_ms']:.0f} ms)  "
            + " ".join(flags))


HEADER = (f"{'opponent':<16} {'bb/deal +- 95% CI':>17}  {'seed verdicts':<13} "
          f"{'':<14}{'':<12}thinking time (bot A)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="In-process duplicate matches.")
    ap.add_argument("bot_a")
    ap.add_argument("bot_b", nargs="*", help="opponents (or use --pool)")
    ap.add_argument("--pool", choices=sorted(POOLS), help="play against a named pool")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", default="eval")
    ap.add_argument("--deals", type=int, default=NUM_DEALS)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--json", help="write all results to this file")
    ap.add_argument("--params", help="JSON parameter overrides for bot A")
    args = ap.parse_args(argv)

    opps = list(args.bot_b)
    if args.pool:
        opps += POOLS[args.pool]
    if not opps:
        ap.error("give at least one opponent or --pool")
    seeds = [f"{args.seed_base}-{i}" for i in range(args.seeds)]
    print(f"{args.bot_a}: {args.seeds} seeds x {args.deals} deals per opponent "
          f"(mean +- 95% CI pooled over all deals)")
    print(HEADER)
    results = []
    t0 = time.perf_counter()
    with mp.Pool(args.jobs or default_jobs()) as pool:
        for opp in opps:
            r = run_matches(args.bot_a, opp, seeds, args.deals, pool=pool,
                            params_a=json.loads(args.params) if args.params else None)
            results.append(r)
            print(format_row(r), flush=True)
            for msg in r["error_messages_a"][:1]:
                print("   first error:", msg.splitlines()[-1])
            d = r.get("diag_a") or {}
            if d.get("post_dec"):
                print(f"   diag: internal errors {d.get('errors', 0)}, postflop decisions {d['post_dec']}, "
                      f"resample share {d.get('resample', 0) / d['post_dec']:.1%}, ESS p10 {d.get('ess_p10', 0):.0f} "
                      f"median {d.get('ess_median', 0):.0f}, quick MC {d.get('quick', 0)}, heuristic {d.get('heuristic_time', 0)}, "
                      f"autopilot decisions {d.get('autopilot_dec', 0)}")
            if d.get("first_error"):
                print("   first internal error:", d["first_error"])
    pts = sum(r["exp_points"] for r in results)
    avg = sum(r["mean"] for r in results) / len(results)
    print(f"expected points {pts:.2f} / {len(results)}   avg bb/deal {avg:+.3f}   "
          f"({time.perf_counter() - t0:.0f}s)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "paired_by_seed"} for r in results],
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
