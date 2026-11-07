"""Archetype (b) nit: plays its top ~25% of hands, folds to raises without a strong hand."""

from game import Action, AgentConfig, HandResult, Observation

import opp_lib

PARTICIPANT = "Archetypes"
AGENT_NAME = "Nit"


def new_half(config: AgentConfig, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    if not obs.board:
        pct = opp_lib.preflop_pct(obs.hole_cards)
        if opp_lib.facing_raise_preflop(obs):
            if pct >= 0.96 and can_raise:
                return Action.RAISE_POT
            if pct >= 0.88:
                return Action.CHECK_CALL
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if pct >= 0.90 and can_raise:
            return Action.RAISE_POT
        if pct >= 0.75 or obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.FOLD
    eq = opp_lib.equity(obs)
    if obs.to_call > 0:
        needed = obs.to_call / (obs.pot + obs.to_call)
        if eq > 0.80 and can_raise:
            return Action.RAISE_POT
        if eq > needed + 0.15:
            return Action.CHECK_CALL
        return Action.FOLD
    if eq > 0.70 and can_raise:
        return Action.RAISE_POT
    if eq > 0.62 and can_raise:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
