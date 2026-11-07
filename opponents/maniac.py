"""Archetype (c) maniac: raises ~60% of the time regardless of cards, rarely folds."""

from game import Action, AgentConfig, HandResult, Observation

import opp_lib

PARTICIPANT = "Archetypes"
AGENT_NAME = "Maniac"


def new_half(config: AgentConfig, memory: dict) -> None:
    pass


def choose_action(obs: Observation, memory: dict) -> Action:
    rng = opp_lib.hand_rng(obs, "maniac")
    u = rng.random()
    if Action.RAISE_POT in obs.legal_actions and u < 0.60:
        return Action.RAISE_POT if rng.random() < 0.5 else Action.RAISE_HALF
    if obs.to_call > 0 and rng.random() < 0.15:
        return Action.FOLD
    return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
