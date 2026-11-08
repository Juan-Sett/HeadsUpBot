"""
strategy.py - the bot: range model + EV decisions + opponent model + time governor.

agent.py calls new_half / choose_action / on_hand_end. Everything here is
wrapped so that no exception ever reaches the judge: on any failure we fall
back to decide.heuristic, and the final answer is always a legal action.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from game import Action, AgentConfig, HandResult, Observation, Street

from . import fasteval as fe
from . import preflop as pf
from . import rangemodel as rm
from .decide import action_evs, heuristic, pick, quick_action
from .oppmodel import BOARD_LEN, OppModel, parse_history

PARAMS = {
    # equity realization when betting can still follow
    "rho_ip": 1.08, "rho_oop": 0.80,      # tuned; checked on held-out seeds vs v1/v2/guide/tag
    # a raise is valued as all-in if our stack left <= this * pot after the call
    "commit_spr": 1.5,
    # spread of our equity on the next street (option value of folding later):
    # preflop -> flop (Gaussian sd), flop runouts shrunk toward the mean by lam
    "sd_pre": 0.20, "lam_flop": 0.90,      # sd_pre > 0.2 made us exploitable by v1
    # weight on our own later-street fold equity in the passive branch
    # (0 = off: on held-out seeds it was neutral vs archetypes)
    "steal_w": 0.0,
    # future-aggression line: True = assume a bet on every remaining street.
    # The "expected number of bets" variant (False) beat the guide bot a bit
    # more but lost ~1 bb/deal head-to-head to v2 and halved the always_pot edge.
    "nbets_all": True,
    # future-betting line preflop: "all" (bet every street), "expected",
    # "blend" (by kappa) or "none"; postflop: "all", "expected" or "blend"
    "pre_fut": "all", "post_fut": "all",
    # button never folds first-in when its equity vs a random hand is at least this (1 = off)
    "limp_floor": 0.46,       # checked on two held-out seed sets vs v1-v3 and archetypes
    # button calls a raise after limping when eq vs the raiser's range >= price + margin (None = off)
    "ip_call_margin": 0.14, "ip_call_max_price": 0.30,
    # action model
    "tau": 0.07, "prior_n": 15.0, "eps_prior": 0.10, "eps_n": 20.0, "weak_h": 0.30,
    "slow_prior": 0.35, "slow_n": 10.0, "strong_h": 0.75, "kappa_n": 6.0,
    "half_life": 150.0,
    # decision rule
    "delta": 0.02, "mix": 0.10,
    # particles
    "k": 150, "k_min": 48,
    # cost model of evaluate(): e0 + c1 * particles * runouts (seconds), times a learned speed factor
    "cost_e0": 0.0003, "cost_c1": 0.6e-6, "n_ref": 32, "r_replay": 4, "n_ref_replay": 24, "min_ess": 30.0,
    "r_flop": 8, "r_flop_min": 2, "r_flop_max": 16,
    "r_turn": 8, "r_turn_min": 2, "r_turn_max": 16,
    # cheap fallback: Monte Carlo samples vs a random hand
    "quick_samples": 48, "quick_min": 12,
    # autopilot exploit
    "ap_streak": 6, "ap_hands": 10,
    # time governor (seconds of thinking per half on this machine)
    "time_target": 3.0, "max_decision": 0.040, "low_budget_frac": 0.05,
    # judge-charged seconds per half we allow (server mode only; the bank is 8 s)
    "bank_target": 6.0,
}


# True when agent.py runs as the real web server (python agent.py). Then the
# governor also watches the judge's view of time: the process CPU time, which
# includes the web framework's work per request (not just our functions).
SERVER = False


def configure(params: dict) -> None:
    """Override parameters (used by the tuner / simulator)."""
    PARAMS.update(params)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    memory["cfg"] = config
    memory["om"] = OppModel(PARAMS)
    memory["hs"] = None
    memory["cpu0"] = time.process_time()
    memory["t"] = {"spent": 0.0, "post_spent": 0.0, "post_dec": 0, "dec": 0,
                   "speed": 1.0, "voi_mean": 1.0, "ovh": 0.0005, "quick_per_sample": 1.2e-5,
                   "cpu_in": 0.0}
    memory["diag"] = {"errors": 0, "fallback": 0, "heuristic_time": 0, "quick": 0, "autopilot_dec": 0,
                      "ess_min": [], "resample": 0, "post_dec": 0, "error_msgs": [],
                      # postflop time by stage: events, evaluate, apply+eq, fold model, future, EV+pick
                      "stage_s": [0.0] * 6}


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        action = _choose(obs, memory)
    except Exception as exc:  # never let a bug reach the judge
        _note_error(memory, exc)
        try:
            action = heuristic(obs)
        except Exception:
            action = Action.CHECK_CALL
    if action not in obs.legal_actions:
        action = Action.CHECK_CALL
    t = memory["t"]
    t["spent"] += time.perf_counter() - t0
    t["cpu_in"] += time.process_time() - c0
    t["dec"] += 1
    return action



def on_hand_end(result: HandResult, memory: dict) -> None:
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        _hand_end(result, memory)
    except Exception as exc:
        _note_error(memory, exc)
    try:
        memory["t"]["spent"] += time.perf_counter() - t0
        memory["t"]["cpu_in"] += time.process_time() - c0
    except Exception:
        pass


def _note_error(memory, exc) -> None:
    try:
        d = memory["diag"]
        d["errors"] += 1
        if len(d["error_msgs"]) < 5:
            import traceback
            d["error_msgs"].append(traceback.format_exc())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def _hand_state(obs, memory) -> dict:
    hs = memory["hs"]
    if hs is None or hs["idx"] != obs.hand_index:
        seed = zlib.crc32(f"{obs.hand_index}|{obs.hole_cards}".encode())
        hs = {"idx": obs.hand_index, "rng": _LazyRng(seed), "parts": None,
              "applied": 0, "hole": obs.hole_cards, "parsed": [], "last": None,
              "is_button": obs.is_button}
        memory["hs"] = hs
    return hs


class _LazyRng:
    """A per-hand numpy Generator, created only when first used (creating
    one costs ~40 us and most preflop decisions never need it)."""

    __slots__ = ("seed", "_g")

    def __init__(self, seed):
        self.seed = seed
        self._g = None

    def __getattr__(self, name):
        if self._g is None:
            self._g = np.random.Generator(np.random.PCG64(self.seed))
        return getattr(self._g, name)


def _choose(obs, memory) -> Action:
    P = PARAMS
    om: OppModel = memory["om"]
    hs = _hand_state(obs, memory)
    parsed = parse_history(obs)
    hs["parsed"] = parsed
    opp_actions = [a for a in parsed if not a["is_me"]]
    om.live_exit(opp_actions)

    if om.autopilot:
        memory["diag"]["autopilot_dec"] += 1
        action = Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL
    elif obs.street == Street.PREFLOP:
        action = _preflop(obs, memory, hs, opp_actions)
    else:
        action = _postflop(obs, memory, hs, opp_actions)

    opp_acted_this_street = any((not a["is_me"]) and a["street"] == obs.street for a in parsed)
    hs["last"] = {"street": obs.street, "action": action, "to_call": obs.to_call,
                  "first": not opp_acted_this_street}
    return action


def _rho(obs) -> float:
    return PARAMS["rho_ip"] if obs.is_button else PARAMS["rho_oop"]


def _preflop(obs, memory, hs, opp_actions) -> Action:
    om: OppModel = memory["om"]
    # The opponent's range as the top x of ALL hands. Preflop frequencies are
    # measured against the whole range, so each action is an absolute cutoff:
    # a raise keeps the top f_raise, a call of a raise keeps the top (1 - f_fold).
    # A limp or a check keeps the range wide.
    x = 1.0
    for a in opp_actions:
        f = om.freqs(a["ctx"])
        if a["kind"] == "raise":
            x = min(x, max(0.03, f[2]))
        elif a["kind"] == "pass" and a["facing"] and a["ctx"] != "pre_open":
            x = min(x, max(0.03, 1.0 - f[0]))
    row = pf.EQ_ROWS[pf.class_index(obs.hole_cards)]
    eq = pf.interp(row, x)
    # If we raise: the opponent continues with the top (1 - f_fold) of all
    # hands, so within its current range (top x) it folds 1 - (1 - f_fold) / x.
    cont = max(0.03, 1.0 - om.freqs("pre_facing")[0])
    fold = max(0.0, 1.0 - cont / x)
    eq_c_val = pf.interp(row, min(x, cont))
    raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
    phi = {a: fold for a in raises}
    eq_c = {a: eq_c_val for a in raises}
    agg = _street_aggression(om, Street.FLOP)
    # The hands that keep betting postflop: roughly the top share of the range.
    # Our flop equity is spread around the preflop number (we can fold the flop).
    a_flop = max(0.08, agg[0])
    sd = PARAMS["sd_pre"]
    e1 = pf.interp(row, x * a_flop)
    e2 = pf.interp(row, min(x, cont) * a_flop)
    fut = {"agg": agg, "agg_now": 0.0, "kappa": _kappa(om),
           "f_next": om.freqs("flop_bet")[0], "steal_w": PARAMS["steal_w"],
           "nbets_mode": PARAMS["pre_fut"] if PARAMS["pre_fut"] != "none" else "all",
           "eq_agg": [min(1.0, max(0.0, e1 + sd * z)) for z in _GH_LIST],
           "eq_agg_c": [min(1.0, max(0.0, e2 + sd * z)) for z in _GH_LIST]}
    if PARAMS["pre_fut"] == "none":
        fut = None          # preflop: equity realization only, no future-betting line
    evs = action_evs(obs, eq, phi, eq_c, _rho(obs), PARAMS["commit_spr"], fut)
    action = pick(evs, obs.pot, eq, PARAMS["delta"], PARAMS["mix"], hs["rng"])
    # Limp floor: as the button facing only the big blind (1 chip into 3), don't
    # fold reasonable hands; the one-step preflop model is too pessimistic there.
    if (action == Action.FOLD and not opp_actions and obs.to_call == 1
            and row[0] >= PARAMS["limp_floor"]):
        action = Action.CHECK_CALL
    # Same idea after we limp the button and get raised: in position at a cheap
    # price, call when our equity vs the raiser's range clears the price.
    m = PARAMS["ip_call_margin"]
    if (m is not None and action == Action.FOLD and obs.is_button and obs.to_call > 0
            and opp_actions and opp_actions[-1]["kind"] == "raise"
            and not any(a["is_me"] and a["kind"] == "raise" for a in hs["parsed"])):
        needed = obs.to_call / (obs.pot + obs.to_call)
        if needed <= PARAMS["ip_call_max_price"] and eq >= needed + m:
            action = Action.CHECK_CALL
    return action


_STREETS = (Street.FLOP, Street.TURN, Street.RIVER)
# 5 equally weighted points with mean 0 and sd 1 (a cheap stand-in for a normal)
_GH_NODES = np.array([-1.6449, -0.6745, 0.0, 0.6745, 1.6449])
_GH_NODES = _GH_NODES / _GH_NODES.std()
_GH_LIST = _GH_NODES.tolist()
_SNAME = {Street.FLOP: "flop", Street.TURN: "turn", Street.RIVER: "river"}
_OPEN_CTX = {s: f"{n}_open" for s, n in _SNAME.items()}


def _kappa(om) -> float:
    """How much the opponent's postflop betting ignores its cards: bluff rate
    over the raise rate of strong hands (1 = bets regardless, 0 = only strong)."""
    return om.kappa("post")


def _street_aggression(om, first: Street) -> list:
    """P(opponent bets) on each street from `first` to the river (open context)."""
    return [om.freqs(_OPEN_CTX[s])[2] for s in _STREETS if s >= first]


def _budget(obs, memory) -> tuple[float, bool]:
    """Seconds we may spend on this postflop decision, and whether to use the
    cheap heuristic because the bank is nearly gone.

    The postflop budget left is spread evenly over the postflop decisions we
    still expect, then weighted by a value-of-information factor (bigger pots
    get more) normalized so that it averages 1."""
    P = PARAMS
    t = memory["t"]
    hands_done = obs.hand_index + 1
    hands_left = max(1, memory["cfg"].num_hands - obs.hand_index)
    # Smoothed (priors worth 30 hands): non-postflop seconds per hand and
    # postflop decisions per hand.
    other_per_hand = (max(0.0, t["spent"] - t["post_spent"]) + 0.0004 * 30) / (hands_done + 30)
    post_rate = (t["post_dec"] + 1.5 * 30) / (hands_done + 30)
    remaining = P["time_target"] - t["spent"] - other_per_hand * hands_left
    if SERVER:
        # The judge charges each request's wall time minus 1 ms, web framework
        # included. We see the framework only as process CPU time outside our
        # functions; under a CPU limit (Kubernetes) wall time runs ahead of CPU
        # time, so scale it by the wall/CPU ratio measured inside our functions.
        calls = max(1, t["dec"])
        cpu_out = max(0.0, time.process_time() - memory["cpu0"] - t["cpu_in"])
        ratio = t["spent"] / t["cpu_in"] if t["cpu_in"] > 0.25 else 1.0
        ratio = min(4.0, max(1.0, ratio))
        plumbing = cpu_out * ratio
        charged = max(t["spent"], t["spent"] + plumbing - 0.0008 * calls)
        calls_per_hand = calls / hands_done
        plumb_per_hand = max(0.0, plumbing / calls - 0.0008) * calls_per_hand
        bank_left = P["bank_target"] - charged - (plumb_per_hand + other_per_hand) * hands_left
        remaining = min(remaining, bank_left)
    if remaining < P["low_budget_frac"] * P["time_target"]:
        return 0.0, True
    per = remaining / max(1.0, post_rate * hands_left)
    voi = min(2.5, max(0.5, 0.5 + obs.pot / 40.0))
    t["voi_mean"] = 0.98 * t["voi_mean"] + 0.02 * voi
    return min(P["max_decision"], per * voi / t["voi_mean"]), False


def _predicted(k: int, r: int) -> float:
    """Cost model of one evaluate() call (seconds, before the speed factor)."""
    return PARAMS["cost_e0"] + PARAMS["cost_c1"] * k * r


def _usable_eval(memory, per_dec: float) -> float:
    """Seconds of evaluate() (in cost-model units) this decision can afford."""
    t = memory["t"]
    return (per_dec - t["ovh"]) / t["speed"] - PARAMS["cost_e0"]


def _floor(memory) -> float:
    """Cost of the cheapest model decision on this machine (seconds)."""
    t = memory["t"]
    return t["ovh"] + t["speed"] * _predicted(PARAMS["k_min"], PARAMS["r_flop_min"])


def _choose_k(memory, per_dec: float) -> int:
    P = PARAMS
    k = int(_usable_eval(memory, per_dec) / (P["cost_c1"] * P["r_flop"]))
    return max(P["k_min"], min(P["k"], k))


def _runouts(obs, memory, per_dec: float, k: int) -> int:
    P = PARAMS
    blen = len(obs.board)
    if blen == 5:
        return 1
    key = "r_flop" if blen == 3 else "r_turn"
    r = int(_usable_eval(memory, per_dec) / (P["cost_c1"] * max(1, k)))
    return max(P[key + "_min"], min(P[key + "_max"], r))


def _runout_equities(wk, wr, valid, blen) -> np.ndarray:
    """Our equity vs a weighted range on each runout (what we may hold on
    the next street). Flop runouts are full boards, so they are shrunk toward
    the mean: the turn reveals only part of the river's information."""
    num = (wk[:, None] * wr * valid).sum(0)
    den = (wk[:, None] * valid).sum(0)
    ok = den > 0
    e = num[ok] / den[ok]
    if len(e) == 0:
        return np.array([0.5])
    if blen == 3:
        m = e.mean()
        e = m + PARAMS["lam_flop"] * (e - m)
    return e


def _postflop(obs, memory, hs, opp_actions) -> Action:
    P = PARAMS
    om: OppModel = memory["om"]
    t_start = time.perf_counter()
    per_dec, low = _budget(obs, memory)
    # Gate: if even the cheapest model run doesn't fit this decision's share of
    # the budget (small pot, many decisions left), use the cheap heuristic.
    # Gate per HAND: the first postflop decision decides whether this hand
    # gets the model (then every later street does too, so nothing has to be
    # replayed). Small pots go to the cheap fallback first when time is short.
    if "model" not in hs:
        hs["model"] = (not low) and per_dec >= _floor(memory)
    if low or not hs["model"]:
        # Tier 2: quick Monte Carlo sized to the budget; tier 3: made-hand rules.
        t = memory["t"]
        n = 0 if low else int(min(P["quick_samples"], per_dec / t["quick_per_sample"]))
        d = memory["diag"]
        if n >= P["quick_min"]:
            tq = time.perf_counter()
            action = quick_action(obs, hs["rng"], n)
            dq = time.perf_counter() - tq
            t["quick_per_sample"] = 0.95 * t["quick_per_sample"] + 0.05 * min(1e-4, dq / n)
            d["quick"] += 1
        else:
            action = heuristic(obs)
            d["heuristic_time"] += 1
        return action
    board = list(obs.board)
    blen = len(board)
    tau, eps, slow = P["tau"], om.eps("post"), om.slow("post")

    def apply(a):
        ph = "pre" if a["blen"] == 0 else "post"
        parts.apply(board, a, om.freqs(a["ctx"]), tau, om.eps(ph), om.slow(ph))

    parts: rm.Particles = hs["parts"]
    pending = opp_actions[hs["applied"]:]
    if parts is None:
        # Particle count from the time budget (the cost scales with it).
        parts = rm.Particles(obs.hole_cards, board, _choose_k(memory, per_dec), hs["rng"], P)
        hs["parts"] = parts
    elif blen > parts.board_len:
        late = [a for a in pending if a["blen"] <= parts.board_len]
        for a in late:
            apply(a)
        pending = pending[len(late):]
        parts.advance_board(board)
    # Actions from earlier streets that are not applied yet (e.g. preflop at creation).
    early = [a for a in pending if a["blen"] < blen]
    for a in early:
        apply(a)
    pending = pending[len(early):]
    st = memory["diag"]["stage_s"]
    t1 = time.perf_counter(); st[0] += t1 - t_start

    eval_dt = None
    cached = hs.get("eval")
    if cached is not None and cached[0] == (blen, parts.version):
        win, wr, valid = cached[1], cached[2], cached[3]   # same street, same particles: reuse
        r = 0
    else:
        r = _runouts(obs, memory, per_dec, len(parts.hands))
        n_ref = 0 if parts.uniform else P["n_ref"]
        te = time.perf_counter()
        win, h, _, wr, valid, ctx = rm.evaluate(obs.hole_cards, board, parts.hands, r, hs["rng"],
                                                n_ref, full_out=True)
        eval_dt = time.perf_counter() - te
        k_eval = len(parts.hands)
        parts.hcache[blen] = h
        if blen == 5:     # keep the river pair table + reference scores for showdown learning
            ptT, Q, qvalid = ctx
            hs["river_ctx"] = (ptT[:, 0], np.sort(Q[qvalid[:, 0], 0]))
    t2 = time.perf_counter(); st[1] += t2 - t1
    for a in pending:
        apply(a)
    hs["applied"] = len(opp_actions)
    idx = parts.maybe_resample(P["min_ess"])
    if idx is not None:
        win, wr, valid = win[idx], wr[idx], valid[idx]
        memory["diag"]["resample"] += 1
    hs["eval"] = ((blen, parts.version), win, wr, valid)
    h = parts.hcache[blen]
    w = parts.normalized()
    eq = float((w * win).sum())
    t3 = time.perf_counter(); st[2] += t3 - t2

    # How the opponent answers a raise: fold model in the facing-bet context.
    street_name = _SNAME[obs.street]
    fr = om.freqs(f"{street_name}_bet")
    nxt = None if obs.street == Street.RIVER else Street(obs.street + 1)
    fo = om.freqs(_OPEN_CTX[nxt]) if nxt is not None else (0.0, 1.0, 0.0)
    # all four frequency-matched thresholds with one sort
    theta_r, theta_f, tr, tf = rm.quantiles(h, w, (1.0 - fr[2], fr[0], 1.0 - fo[2], fo[0]))
    p_f, _, _ = rm.action_probs(h, theta_r, theta_f, True, tau, eps, slow)
    phi_v = float((w * p_f).sum())
    wc = w * (1.0 - p_f)
    eq_c_v = float((wc * win).sum() / max(1e-12, wc.sum()))
    raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
    phi = {a: phi_v for a in raises}
    eq_c = {a: eq_c_v for a in raises}
    t4 = time.perf_counter(); st[3] += t4 - t3

    # Future aggression: who keeps betting, and how often (reverse implied odds).
    fut = None
    if obs.street != Street.RIVER:
        agg = _street_aggression(om, nxt)
        _, _, p_bet = rm.action_probs(h, tr, tf, False, tau, eps, slow)
        wa = w * p_bet
        wac = wc * p_bet
        eq_agg = _runout_equities(wa, wr, valid, blen)
        eq_agg_c = _runout_equities(wac, wr, valid, blen)
        # After our check, does the opponent still act on this street?
        opp_to_act = obs.to_call == 0 and not any(
            (not a["is_me"]) and a["street"] == obs.street for a in hs["parsed"])
        agg_now = float(om.freqs(f"{street_name}_open")[2]) if opp_to_act else 0.0
        fut = {"agg": agg, "agg_now": agg_now, "eq_agg": eq_agg, "eq_agg_c": eq_agg_c,
               "kappa": _kappa(om), "f_next": om.freqs(f"{_SNAME[nxt]}_bet")[0],
               "steal_w": P["steal_w"], "nbets_all": P["nbets_all"],
               "nbets_mode": P["post_fut"]}
    t5 = time.perf_counter(); st[4] += t5 - t4
    evs = action_evs(obs, eq, phi, eq_c, _rho(obs), P["commit_spr"], fut)
    action = pick(evs, obs.pot, eq, P["delta"], P["mix"], hs["rng"])
    st[5] += time.perf_counter() - t5

    # time bookkeeping for the governor
    dt = time.perf_counter() - t_start
    t = memory["t"]
    t["post_spent"] += dt
    t["post_dec"] += 1
    if eval_dt is not None:
        # Learn this machine's speed for evaluate() (clipped, so one slow
        # outlier can't swing it) and the per-decision overhead around it.
        ratio = min(3.0, max(0.33, eval_dt / _predicted(k_eval, r)))
        t["speed"] = min(4.0, max(0.25, 0.95 * t["speed"] + 0.05 * ratio))
        t["ovh"] = 0.95 * t["ovh"] + 0.05 * min(0.01, dt - eval_dt)
    d = memory["diag"]
    d["post_dec"] += 1
    d["ess_min"].append(parts.ess())
    return action


# ---------------------------------------------------------------------------
# End of hand: learning
# ---------------------------------------------------------------------------

def _hand_end(result: HandResult, memory: dict) -> None:
    om: OppModel = memory["om"]
    hs = memory["hs"]
    if hs is None or hs["idx"] != result.hand_index:
        # We never acted: as big blind, the button folded preflop.
        opp = [{"ctx": "pre_open", "kind": "fold", "to_call": 1, "facing": True,
                "blen": 0, "street": Street.PREFLOP}]
        om.end_hand(opp)
        return
    opp = [a for a in hs["parsed"] if not a["is_me"]]
    last = hs["last"]
    resp = None
    if last is not None and last["action"] != Action.FOLD:
        street = last["street"]
        sname = {Street.PREFLOP: "pre", Street.FLOP: "flop", Street.TURN: "turn",
                 Street.RIVER: "river"}[street]
        raised = last["action"] in (Action.RAISE_HALF, Action.RAISE_POT)
        bet_ctx = "pre_facing" if street == Street.PREFLOP else f"{sname}_bet"
        if not result.went_to_showdown and result.winner == "me":
            resp = {"ctx": bet_ctx, "kind": "fold", "to_call": 1, "facing": True,
                    "blen": BOARD_LEN[street], "street": street}
        elif result.went_to_showdown and raised:
            resp = {"ctx": bet_ctx, "kind": "pass", "to_call": 1, "facing": True,
                    "blen": BOARD_LEN[street], "street": street}
        elif (result.went_to_showdown and street == Street.RIVER and last["to_call"] == 0
              and last["first"]):
            resp = {"ctx": "river_open", "kind": "pass", "to_call": 0, "facing": False,
                    "blen": 5, "street": street}
    if resp is not None:
        opp = opp + [resp]
    om.end_hand(opp)

    if result.opp_hole is not None and len(result.board) == 5:
        _showdown_learning(result, hs, opp, om)


def _showdown_learning(result, hs, opp_actions, om) -> None:
    hole_opp = np.array([result.opp_hole], dtype=np.int64)
    board = list(result.board)
    cache = {}
    out = []
    # Cheap on purpose: preflop strength from the table, river strength on
    # the exact board. Flop/turn decisions are skipped (they'd need runouts).
    river_ctx = hs.get("river_ctx")
    if river_ctx is not None:
        pt, refs = river_ctx
        s = int(pt[fe.hand_pair_indices(hole_opp)[0]].max())
        lo = int(np.searchsorted(refs, s, "left"))
        hi = int(np.searchsorted(refs, s, "right"))
        cache[5] = (lo + 0.5 * (hi - lo)) / max(1, len(refs))
    cache[0] = pf.PCT_LIST[pf.class_index(result.opp_hole)]
    for a in opp_actions:
        b = a["blen"]
        if b in cache:
            out.append((cache[b], a["kind"], "pre" if b == 0 else "post"))
    om.showdown(out)
