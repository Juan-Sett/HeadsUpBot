"""Archetype (e) time-bank death: plays like the guide bot until a random
hand between 200 and 600, then check/folds forever (as if its 8 s bank ran out).
"""

import zlib

from game import Action, AgentConfig, HandResult, Observation

import opp_lib
from guide import SAMPLES, guide_action

PARTICIPANT = "Archetypes"
AGENT_NAME = "TimebankDeath"


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["death"] = None


def choose_action(obs: Observation, memory: dict) -> Action:
    if memory["death"] is None:
        # Pick the death hand from the first cards we see (depends on the seed).
        memory["death"] = 200 + zlib.crc32(str(obs.hole_cards).encode()) % 401
    if obs.hand_index >= memory["death"]:
        return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    return guide_action(obs, opp_lib.equity(obs, SAMPLES))


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
