"""
decide.py - one-step EV maximization, plus a cheap fallback heuristic.

EV in chips relative to folding now (chips already in the pot are sunk):
    FOLD   0
    CHECK  rho * eq * pot
    CALL   rho * eq * (pot + to_call) - to_call
    RAISE  phi * pot + (1 - phi) * (rho * eq_c * (pot + add + c) - add)
where add = our extra chips, c = the opponent's call, phi = P(opponent
folds), eq_c = our equity vs the hands that continue. rho (equity
realization) is 1 when no betting can follow (river, all-in, committed).
"""

from __future__ import annotations

import numpy as np

from game import Action, Street

from . import fasteval as fe

# Resulting pot size ordering (used to prefer low-variance actions on near-ties).
_SIZE_ORDER = {Action.FOLD: 0, Action.CHECK_CALL: 1, Action.RAISE_HALF: 2, Action.RAISE_POT: 3}


def _future(a_streets) -> float:
    """P(the opponent bets at least once more) from per-street bet probabilities."""
    q = 1.0
    for x in a_streets:
        q *= 1.0 - x
    return 1.0 - q


def _aggr_line(eq_agg, pot: float, n_bets: int, my_left: float, opp_left: float,
               kappa: float = 1.0) -> float:
    """EV (from here) of the line where the opponent bets pot on each of n_bets
    streets: we either call down to the end or fold (worth 0), whichever is
    better. A pot-size bet plus a call triples the pot, so we put in pot*(3^n - 1)/2.

    eq_agg is our equity vs the betting range: a number, or an array of the
    equities we may have on the next street (we decide after seeing it, so
    the fold option is worth something)."""
    x = min(my_left, opp_left, pot * (3 ** n_bets - 1) / 2.0)
    if x <= 0:
        return 0.0
    total = pot + 2.0 * x
    if isinstance(eq_agg, (int, float)):
        return max(eq_agg * total - x, 0.0)
    # Keep the fold option (bad next streets are worth 0, not a loss). The
    # upside above our average equity counts only as far as the opponent's
    # betting ignores its cards (kappa = 1: always_pot / random; kappa = 0: a
    # strength-driven bettor, who stops paying on runouts that turn against it).
    if isinstance(eq_agg, list):
        m = sum(eq_agg) / len(eq_agg)
        return sum(max((e if e <= m else m + kappa * (e - m)) * total - x, 0.0)
                   for e in eq_agg) / len(eq_agg)
    m = float(eq_agg.mean())
    e = np.where(eq_agg <= m, eq_agg, m + kappa * (eq_agg - m))
    return float(np.maximum(e * total - x, 0.0).mean())


def action_evs(obs, eq: float, phi: dict, eq_c: dict, rho_pos: float, commit_spr: float,
               fut: dict | None = None) -> dict:
    """EV of every legal action, aware of future aggression.

    fut (optional):
        agg      per-future-street P(opponent bets), e.g. [turn, river]
        agg_now  P(opponent bets this street after our check) (0 if our check closes it)
        eq_agg   our equity vs the part of the range that keeps betting
        eq_agg_c same, after the opponent called our raise
    With probability a = 1 - prod(1 - agg) the opponent keeps betting pot every
    street; then we stack off or fold, whichever is better. Otherwise the
    hand checks down and we realize rho * eq of the pot.
    """
    fut = fut or {"agg": [], "agg_now": 0.0, "eq_agg": eq, "eq_agg_c": eq}
    pot, tc = obs.pot, obs.to_call
    river = obs.street == Street.RIVER
    agg = [] if river else list(fut["agg"])
    kap = fut.get("kappa", 1.0)
    evs = {}
    for a in obs.legal_actions:
        if a == Action.FOLD:
            evs[a] = 0.0
        elif a == Action.CHECK_CALL:
            if tc == 0:
                p_more = _future(agg + [fut["agg_now"]])
                n = len(agg) + (1 if fut["agg_now"] > 0 else 0)
                passive = (1.0 if river else rho_pos) * eq * pot
                aggr = _aggr_line(fut["eq_agg"], pot, n, obs.my_stack, obs.opp_stack, kap)
                evs[a] = (1.0 - p_more) * passive + p_more * aggr
            else:
                amt = min(tc, obs.my_stack)
                my_left = obs.my_stack - amt
                all_in = my_left == 0 or obs.opp_stack == 0
                P = pot + amt
                if river or all_in:
                    evs[a] = eq * P - amt
                else:
                    p_more = _future(agg)
                    passive = rho_pos * eq * P - amt
                    aggr = _aggr_line(fut["eq_agg"], P, len(agg), my_left, obs.opp_stack, kap) - amt
                    evs[a] = (1.0 - p_more) * passive + p_more * aggr
        else:
            target = obs.raise_to[a]
            eff = min(target, obs.opp_bet + obs.opp_stack)    # what the opponent can match
            add = eff - obs.my_bet
            c = eff - obs.opp_bet
            P = pot + add + c
            my_left = obs.my_stack - add
            opp_left = obs.opp_stack - c
            if river or my_left == 0 or opp_left == 0:
                called = eq_c[a] * P - add
            elif my_left <= commit_spr * P:
                # Committed: value it as all-in.
                x = min(my_left, opp_left)
                called = eq_c[a] * (P + 2 * x) - x - add
            else:
                p_more = _future(agg)
                passive = rho_pos * eq_c[a] * P - add
                aggr = _aggr_line(fut["eq_agg_c"], P, len(agg), my_left, opp_left, kap) - add
                called = (1.0 - p_more) * passive + p_more * aggr
            p = phi[a]
            evs[a] = p * pot + (1.0 - p) * called
    return evs


def pick(evs: dict, pot: int, eq: float, delta: float, mix: float, rng) -> Action:
    """argmax EV; among near-ties (within delta * pot) prefer the smaller pot,
    with a little randomization so adaptive opponents can't read us."""
    best = max(evs.values())
    if eq >= 0.9:                      # near-certain winner: just take the max
        return max(evs, key=lambda a: (evs[a], _SIZE_ORDER[a]))
    near = [a for a, v in evs.items() if v >= best - delta * max(pot, 1)]
    near.sort(key=lambda a: _SIZE_ORDER[a])
    if len(near) > 1 and rng.random() < mix:
        return near[int(rng.integers(len(near)))]
    return near[0]


# ---------------------------------------------------------------------------
# Cheap fallback: preflop table + made-hand rules (no simulation)
# ---------------------------------------------------------------------------

def made_hand_score(hole, board) -> int:
    """Best score of 2 hole + 3 board cards for a 3-, 4- or 5-card board."""
    return fe.best_made_score(hole, board)


def quick_equity(hole, board, n: int, rng) -> float:
    """Monte Carlo equity vs one random hand (n samples, ties count 1/2)."""
    used = np.zeros(36, dtype=bool)
    used[list(hole)] = True
    used[list(board)] = True
    avail = np.flatnonzero(~used)
    need = 5 - len(board)
    keys = rng.random((n, len(avail)))
    draw = avail[np.argpartition(keys, 5 + need - 1, axis=1)[:, :5 + need]]
    b = np.asarray(board, dtype=np.int64)
    full = np.concatenate([np.broadcast_to(b, (n, len(b))), draw[:, 5:]], axis=1) if need else np.broadcast_to(b, (n, 5))
    both = np.concatenate([np.broadcast_to(np.asarray(hole, dtype=np.int64), (n, 5)), draw[:, :5]])
    sc = fe.omaha_batch(both, np.concatenate([full, full]))
    me, op = sc[:n], sc[n:]
    return float(((me > op).sum() + 0.5 * (me == op).sum()) / n)


def quick_action(obs, rng, n: int = 48) -> Action:
    """Cheap postflop fallback (the guide-bot rules): equity vs a random hand,
    pot odds plus a safety margin that grows with the bet size."""
    if not obs.board:
        return heuristic(obs)
    eq = quick_equity(obs.hole_cards, obs.board, n, rng)
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    if obs.to_call > 0:
        needed = obs.to_call / (obs.pot + obs.to_call)
        margin = 0.10 * min(1.0, obs.to_call / max(1, obs.pot - obs.to_call))
        if eq > 0.72 and can_raise:
            return Action.RAISE_POT
        if eq > needed + margin:
            return Action.CHECK_CALL
        return Action.FOLD
    if eq > 0.66 and can_raise:
        return Action.RAISE_POT
    if eq > 0.56 and can_raise:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def heuristic(obs) -> Action:
    """Cheap, safe strategy used when time is short or something failed."""
    from . import preflop as pf
    legal = obs.legal_actions
    can_raise = Action.RAISE_HALF in legal
    if not obs.board:
        eq = pf.eq_random(obs.hole_cards)
        if eq > 0.60 and can_raise and obs.raises_this_street < 2:
            return Action.RAISE_POT
        needed = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
        if obs.to_call == 0 or eq > needed + 0.06:
            return Action.CHECK_CALL
        return Action.FOLD
    cat = made_hand_score(obs.hole_cards, obs.board) >> 20
    # categories: 0 high, 1 pair, 2 two pair, 3 straight, 4 trips, 5 full house, 6 flush, 7 quads, 8 SF
    if cat >= 5 and can_raise:
        return Action.RAISE_POT
    if obs.to_call == 0:
        return Action.RAISE_HALF if (cat >= 4 and can_raise) else Action.CHECK_CALL
    frac = obs.to_call / max(1, obs.pot)
    if cat >= 4 or (cat >= 2 and frac <= 0.6) or (cat >= 1 and frac <= 0.25):
        return Action.CHECK_CALL
    return Action.FOLD
