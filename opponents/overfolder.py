"""Archetype (d) overfolder: folds to any bet unless it holds roughly its top 10%.

When nobody has bet, it plays like the guide bot.
"""

from game import Action, AgentConfig, HandResult, Observation

import opp_lib
from guide import guide_action

PARTICIPANT = "Archetypes"
AGENT_NAME = "Overfolder"


def new_half(config: AgentConfig, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    if not obs.board:
        facing_bet = opp_lib.facing_raise_preflop(obs)
        if facing_bet:
            pct = opp_lib.preflop_pct(obs.hole_cards)
            if pct >= 0.90:
                return Action.CHECK_CALL
            return Action.FOLD
        return guide_action(obs, opp_lib.preflop_eq(obs.hole_cards))
    eq = opp_lib.equity(obs)
    if obs.to_call > 0:
        return Action.CHECK_CALL if eq > 0.75 else Action.FOLD
    return guide_action(obs, eq)


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
