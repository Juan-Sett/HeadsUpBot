"""
oppmodel.py - what we learn about the opponent during one half.

Contexts (where an opponent action happens):
    pre_open    button's first preflop action (facing the big blind)
    pre_check   big blind after a limp (nothing to call)
    pre_facing  preflop, facing a raise
    {flop,turn,river}_open    postflop, nothing to call
    {flop,turn,river}_bet     postflop, facing a bet or raise
Each context keeps decayed counts of fold / pass (check or call) / raise
plus Beta-style prior pseudo-counts, so frequencies are usable from hand 1.

Also: the autopilot detector (an opponent that only check/folds because
its time bank ran out or it forfeited) and the bluff rate eps learned from
showdowns.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from game import Action, Street

KINDS = ("fold", "pass", "raise")
_K = {k: i for i, k in enumerate(KINDS)}
STREET_NAME = {Street.FLOP: "flop", Street.TURN: "turn", Street.RIVER: "river"}
BOARD_LEN = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}

# Population priors (fold, pass, raise) for each context.
PRIORS = {
    "pre_open": (0.25, 0.35, 0.40),
    "pre_check": (0.0, 0.65, 0.35),
    "pre_facing": (0.35, 0.45, 0.20),
}
for _s in ("flop", "turn", "river"):
    PRIORS[f"{_s}_open"] = (0.0, 0.60, 0.40)
    PRIORS[f"{_s}_bet"] = (0.40, 0.45, 0.15)


def kind_of(action: Action) -> str:
    if action == Action.FOLD:
        return "fold"
    if action == Action.CHECK_CALL:
        return "pass"
    return "raise"


def context(street: Street, to_call: int, first_button_action: bool) -> str:
    if street == Street.PREFLOP:
        if first_button_action:
            return "pre_open"
        return "pre_facing" if to_call > 0 else "pre_check"
    return f"{STREET_NAME[street]}_{'bet' if to_call > 0 else 'open'}"


def parse_history(obs) -> list[dict]:
    """Replay the hand's actions with blinds and label each one.

    Each item: street, is_me, kind, to_call (for the actor), ctx, blen.
    """
    me_button = obs.is_button
    bets = {True: 1 if me_button else 2, False: 2 if me_button else 1}   # key: is_me
    street = Street.PREFLOP
    first_btn = True
    out = []
    for a in obs.action_history:
        if a.street != street:
            street = a.street
            bets = {True: 0, False: 0}
        actor = a.is_me
        to_call = max(0, bets[not actor] - bets[actor])
        is_btn_actor = (actor == me_button)
        fba = street == Street.PREFLOP and first_btn and is_btn_actor
        if street == Street.PREFLOP and is_btn_actor:
            first_btn = False
        out.append({"street": street, "is_me": actor, "kind": kind_of(a.action),
                    "action": a.action, "to_call": to_call,
                    "ctx": context(street, to_call, fba), "blen": BOARD_LEN[street],
                    "facing": to_call > 0})
        bets[actor] = a.street_total
    return out


class OppModel:
    def __init__(self, params: dict):
        self.p = params
        n0 = params["prior_n"]
        self.prior = {c: n0 * np.array(v, dtype=float) for c, v in PRIORS.items()}
        self.counts = {c: np.zeros(3) for c in PRIORS}
        self._fcache: dict = {}
        self.decay = 0.5 ** (1.0 / params["half_life"])
        # autopilot detection
        self.fold_streak = 0                       # opponent folds to our bets in a row
        self.vol_hist: deque = deque(maxlen=params["ap_hands"])
        self.autopilot = False
        # Learned from showdowns, separately for preflop ("pre") and postflop ("post"):
        #   bluff rate = raises among weak-hand decision points
        #   slow-play rate = passes among strong-hand decision points
        self.sd = {ph: np.zeros(4) for ph in ("pre", "post")}   # weak_n, weak_raise, strong_n, strong_pass
        self.hands_seen = 0

    # ---- frequencies ------------------------------------------------------
    def observe(self, ctx: str, kind: str) -> None:
        self.counts[ctx][_K[kind]] += 1.0
        self._fcache.pop(ctx, None)

    def freqs(self, ctx: str) -> tuple:
        """(f_fold, f_pass, f_raise) for a context, smoothed by the prior."""
        f = self._fcache.get(ctx)
        if f is None:
            v = self.counts[ctx] + self.prior[ctx]
            v = v / v.sum()
            f = self._fcache[ctx] = (float(v[0]), float(v[1]), float(v[2]))
        return f

    def n_obs(self, ctx: str) -> float:
        return float(self.counts[ctx].sum())

    def eps(self, phase: str = "post") -> float:
        n0 = self.p["eps_n"]
        v = self.sd[phase]
        e = (v[1] + self.p["eps_prior"] * n0) / (v[0] + n0)
        return float(min(0.4, max(0.02, e)))

    def slow(self, phase: str = "post") -> float:
        n0 = self.p["slow_n"]
        v = self.sd[phase]
        sp = (v[3] + self.p["slow_prior"] * n0) / (v[2] + n0)
        return float(min(0.95, max(0.0, sp)))

    def kappa(self, phase: str = "post") -> float:
        """How much betting ignores hand strength: P(raise | weak) / P(raise | strong)
        from showdowns (1 = raises regardless, 0 = raises only strong hands)."""
        n0 = self.p["kappa_n"]
        v = self.sd[phase]
        r_weak = (v[1] + 0.25 * n0) / (v[0] + n0)
        r_strong = ((v[2] - v[3]) + 0.5 * n0) / (v[2] + n0)
        return float(min(1.0, max(0.0, r_weak / max(0.05, r_strong))))

    # ---- per-hand bookkeeping --------------------------------------------
    def end_hand(self, opp_actions: list[dict]) -> None:
        """Update counts and the autopilot detector with one finished hand's
        opponent actions (in order), then decay."""
        voluntary = False
        for a in opp_actions:
            self.observe(a["ctx"], a["kind"])
            if a["kind"] == "raise" or (a["kind"] == "pass" and a["to_call"] > 0):
                voluntary = True
            if a["facing"] and not a["ctx"].startswith("pre_open"):
                # Facing one of our bets or raises.
                if a["kind"] == "fold":
                    self.fold_streak += 1
                else:
                    self.fold_streak = 0
        self.vol_hist.append(voluntary)
        self._update_autopilot()
        for c in self.counts:
            self.counts[c] *= self.decay
        self._fcache.clear()
        for v in self.sd.values():
            v *= self.decay
        self.hands_seen += 1

    def _update_autopilot(self) -> None:
        full = len(self.vol_hist) == self.vol_hist.maxlen
        self.autopilot = (self.fold_streak >= self.p["ap_streak"] and full
                          and not any(self.vol_hist))

    def live_exit(self, opp_actions: list[dict]) -> None:
        """Leave autopilot mode the moment the opponent calls or raises."""
        if self.autopilot and any(a["kind"] == "raise" or (a["kind"] == "pass" and a["to_call"] > 0)
                                  for a in opp_actions):
            self.autopilot = False
            self.fold_streak = 0

    def showdown(self, strengths: list[tuple[float, str, str]]) -> None:
        """(h, kind, phase) for each opponent decision in a hand that reached showdown."""
        for h, kind, phase in strengths:
            v = self.sd[phase]
            if h < self.p["weak_h"]:
                v[0] += 1.0
                if kind == "raise":
                    v[1] += 1.0
            elif h >= self.p["strong_h"] and kind != "fold":
                v[2] += 1.0
                if kind == "pass":
                    v[3] += 1.0
