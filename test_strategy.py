"""
Unit tests for the strategy (bot/): EV layer sanity checks on hand-built spots.

    python -m pytest -q test_strategy.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from evaluator import cards_from_str  # noqa: E402
from game import Action, AgentConfig, Deal, Game, Street  # noqa: E402
from bot import rangemodel as rm  # noqa: E402
from bot import strategy  # noqa: E402
from bot.decide import action_evs, pick  # noqa: E402
from bot.oppmodel import OppModel  # noqa: E402

P = strategy.PARAMS


def spot(hole0, hole1, board, actions, button=0):
    """Play `actions` from the start of a hand and return the Game."""
    deal = Deal(hole_cards=(tuple(sorted(cards_from_str(hole0))), tuple(sorted(cards_from_str(hole1)))),
                board=tuple(cards_from_str(board)))
    g = Game(deal, button_seat=button)
    for a in actions:
        g.apply(a)
    return g


def fresh_memory():
    mem = {}
    strategy.new_half(AgentConfig(), mem)
    return mem


CHECK, HALF, POT, FOLD = Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT, Action.FOLD
# button limps, BB checks -> flop; both check flop and turn -> river, BB to act first.
TO_RIVER = [CHECK, CHECK, CHECK, CHECK, CHECK, CHECK]


def river_obs(ours, theirs, board, extra=()):
    """River spot where `ours` is the hand to act. With no extra actions the
    big blind (seat 1) acts first on the river; after one more action seat 0 does."""
    our_seat = 1 if len(extra) % 2 == 0 else 0
    holes = (theirs, ours) if our_seat == 1 else (ours, theirs)
    g = spot(holes[0], holes[1], board, list(TO_RIVER) + list(extra))
    assert g.to_act == our_seat
    return g, g.observation(our_seat)


# ---------------------------------------------------------------------------
# EV layer (pure functions)
# ---------------------------------------------------------------------------

def test_vs_calling_station_never_bluffs_and_value_bets():
    """phi = 0: raise iff eq_c is above ~0.5-0.57 (rho), never as a bluff."""
    _, obs = river_obs("6c 7d 8h 9s Jc", "Ac Ad Kh Ks Qd", "Th Jh Qh 6s 7c")
    rng = np.random.default_rng(0)
    raises = [a for a in obs.legal_actions if a in (HALF, POT)]
    for eq in (0.05, 0.2, 0.35, 0.45, 0.49):
        evs = action_evs(obs, eq, {a: 0.0 for a in raises}, {a: eq for a in raises}, 0.88, 1.5)
        assert pick(evs, obs.pot, eq, 0.0, 0.0, rng) == CHECK, f"bluffed with eq {eq}"
    for eq in (0.56, 0.7, 0.9):
        evs = action_evs(obs, eq, {a: 0.0 for a in raises}, {a: eq for a in raises}, 0.88, 1.5)
        assert pick(evs, obs.pot, eq, 0.0, 0.0, rng) in (HALF, POT), f"no value bet with eq {eq}"


def test_calls_down_when_equity_beats_pot_odds():
    """Facing a river pot bet: call iff eq > to_call / (pot + to_call)."""
    _, obs = river_obs("6c 7d 8h 9s Jc", "Ac Ad Kh Ks Qd", "Th Jh Qh 6s 7c", [POT])
    needed = obs.to_call / (obs.pot + obs.to_call)
    rng = np.random.default_rng(0)
    raises = [a for a in obs.legal_actions if a in (HALF, POT)]
    for eq, want in ((needed + 0.03, CHECK), (needed - 0.03, FOLD)):
        evs = action_evs(obs, eq, {a: 0.0 for a in raises}, {a: eq for a in raises}, 1.0, 1.5)
        evs = {a: v for a, v in evs.items() if a in (FOLD, CHECK)}
        assert pick(evs, obs.pot, eq, 0.0, 0.0, rng) == want


# ---------------------------------------------------------------------------
# Range model + opponent model
# ---------------------------------------------------------------------------

def _always_raise_model():
    om = OppModel(P)
    for _ in range(300):
        om.end_hand([{"ctx": c, "kind": "raise", "to_call": 1, "facing": c.endswith("bet")}
                     for c in ("pre_open", "flop_open", "flop_bet", "turn_open", "river_open")])
    return om


def test_always_pot_raises_carry_little_information():
    om = _always_raise_model()
    assert om.freqs("flop_open")[2] > 0.9
    rng = np.random.default_rng(1)
    hole = cards_from_str("6c 7d 8h 9s Jc")
    board = cards_from_str("Th Jh Qh")
    parts = rm.Particles(hole, board, 200, rng, P)
    _, h, _ = rm.evaluate(hole, board, parts.hands, 12, rng)
    parts.hcache[3] = h
    for _ in range(3):   # three raises in a row
        parts.apply(board, {"blen": 3, "kind": "raise", "facing": False},
                    om.freqs("flop_open"), P["tau"], om.eps())
    assert parts.ess() > 0.75 * parts.k, parts.ess()


def test_tight_raiser_raises_are_informative():
    om = OppModel(P)
    for _ in range(300):
        om.end_hand([{"ctx": "flop_open", "kind": "pass", "to_call": 0, "facing": False}])
    rng = np.random.default_rng(2)
    hole = cards_from_str("6c 7d 8h 9s Jc")
    board = cards_from_str("Th Jh Qh")
    parts = rm.Particles(hole, board, 200, rng, P)
    _, h, _ = rm.evaluate(hole, board, parts.hands, 12, rng)
    parts.hcache[3] = h
    parts.apply(board, {"blen": 3, "kind": "raise", "facing": False},
                om.freqs("flop_open"), P["tau"], om.eps())
    w = parts.normalized()
    assert (w * h).sum() > h.mean() + 0.1      # the range moved strongly up


def test_autopilot_opponent_gets_raise_half_every_time():
    mem = fresh_memory()
    om = mem["om"]
    for _ in range(12):     # check/folds to everything, no voluntary chips
        om.end_hand([{"ctx": "pre_facing", "kind": "fold", "to_call": 4, "facing": True}])
    assert om.autopilot
    g = spot("6c 6d 7h 7s 8c", "Ac Ad Kh Ks Qd", "Th Jh Qh 6s 7c", [], button=0)
    obs = g.observation(0)
    assert strategy.choose_action(obs, mem) == HALF
    # It leaves the mode the moment the opponent calls or raises.
    g = spot("6c 6d 7h 7s 8c", "Ac Ad Kh Ks Qd", "Th Jh Qh 6s 7c", [HALF], button=0)
    g.apply(POT)   # opponent (BB) re-raises
    strategy.choose_action(g.observation(0), mem)
    assert not mem["om"].autopilot


# ---------------------------------------------------------------------------
# Whole-strategy spots
# ---------------------------------------------------------------------------

def test_never_folds_when_checking_is_free():
    rng = np.random.default_rng(3)
    mem = fresh_memory()
    from game import make_deal
    n = 0
    for d in range(150):
        g = Game(make_deal("free", d), d % 2, hand_index=d)
        while not g.is_terminal:
            s = g.to_act
            obs = g.observation(s)
            if s == 0:
                a = strategy.choose_action(obs, mem)
                assert a in obs.legal_actions
                if obs.to_call == 0:
                    assert a != FOLD
                    n += 1
            else:
                a = rng.choice([x for x in obs.legal_actions if x != FOLD] or [CHECK])
            g.apply(a)
        strategy.on_hand_end(g.hand_result(0), mem)
    assert n > 50
    assert mem["diag"]["errors"] == 0, mem["diag"]["error_msgs"]


@pytest.mark.parametrize("extra", [[], [POT], [HALF], [HALF, POT]])
def test_river_nuts_never_folds_and_raises(extra):
    # Royal flush Ah Kh with Th Jh Qh on board: the absolute nuts.
    mem = fresh_memory()
    g, obs = river_obs("Ah Kh 6c 7d 8s", "9h 9c Tc Td Js", "Th Jh Qh 6s 7c", extra)
    a = strategy.choose_action(obs, mem)
    assert a != FOLD
    if HALF in obs.legal_actions:
        assert a in (HALF, POT), a
