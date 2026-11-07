"""Held-out archetype: loose-aggressive (LAG). Never used for tuning.

The guide bot's value logic, plus ~25% random bluffs when nobody has bet,
and it rarely folds a small bet (calls if equity beats the pot odds at all).
"""

from game import Action, AgentConfig, HandResult, Observation

import opp_lib
from guide import guide_action

PARTICIPANT = "Archetypes"
AGENT_NAME = "LAG"
UNLIMITED_TIME = True


def new_half(config: AgentConfig, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    rng = opp_lib.hand_rng(obs, "lag")
    eq = opp_lib.equity(obs, 150, rng)
    legal = obs.legal_actions
    if obs.to_call == 0 and Action.RAISE_HALF in legal and rng.random() < 0.25:
        return Action.RAISE_HALF if rng.random() < 0.6 else Action.RAISE_POT
    if obs.to_call > 0:
        needed = obs.to_call / (obs.pot + obs.to_call)
        if obs.to_call <= 0.5 * obs.pot and eq > needed:
            a = guide_action(obs, eq)
            return a if a != Action.FOLD else Action.CHECK_CALL
    return guide_action(obs, eq)


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
