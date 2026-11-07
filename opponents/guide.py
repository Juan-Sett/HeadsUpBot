"""Archetype (a) "guide bot": equity vs a RANDOM hand + fixed thresholds.

Probably the most common student design: Monte Carlo equity against a
random opponent hand, pot odds with a margin, raise strong hands.
"""

from game import Action, AgentConfig, HandResult, Observation

import opp_lib

PARTICIPANT = "Archetypes"
AGENT_NAME = "GuideBot"
SAMPLES = 200
# A strategy reference: the simulator does not apply the 8 s time bank to it
# (timebank_death.py models what happens when a bot like this runs out).
UNLIMITED_TIME = True


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands"] = 0


def guide_action(obs: Observation, eq: float) -> Action:
    legal = obs.legal_actions
    can_raise = Action.RAISE_POT in legal
    if obs.to_call > 0:
        needed = obs.to_call / (obs.pot + obs.to_call)
        margin = 0.10 * min(1.0, obs.to_call / max(1, obs.pot - obs.to_call))
        if eq > 0.70 and can_raise:
            return Action.RAISE_POT
        if eq > needed + margin:
            return Action.CHECK_CALL
        return Action.FOLD
    if eq > 0.65 and can_raise:
        return Action.RAISE_POT
    if eq > 0.55 and can_raise:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    return guide_action(obs, opp_lib.equity(obs, SAMPLES))


def on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands"] += 1
