"""
rangemodel.py - Bayesian opponent range model (a particle filter).

A hand's particles are K possible opponent hands with weights. Every
opponent action multiplies each particle's weight by the likelihood of that
action given the particle's strength h (frequency matched to how often this
opponent takes that action in that context):

    P(raise | h) = eps + (1 - 2 eps) * sigmoid((h - theta_r) / tau)
    P(fold  | h) = eps + (1 - 2 eps) * sigmoid((theta_f - h) / tau)
    P(pass  | h) = the rest (check or call)

theta_r is the (1 - f_raise) weighted quantile of h over the current range,
theta_f the f_fold quantile, so the strongest f_raise share raises and the
weakest f_fold share folds.

Strength h: preflop, the class percentile from the preflop table;
postflop, the mean over runouts of the particle's percentile among random
reference hands (about its equity vs a random hand, draws included).
"""

from __future__ import annotations

import numpy as np

from . import fasteval as fe
from . import preflop as pf

ALL = np.arange(36)


def available(hole, board) -> np.ndarray:
    """Cards that are neither ours nor on the board."""
    used = np.zeros(36, dtype=bool)
    used[list(hole)] = True
    used[list(board)] = True
    return ALL[~used]


def sample_hands(rng: np.random.Generator, avail: np.ndarray, k: int) -> np.ndarray:
    """k random 5-card hands from avail (cards may repeat across hands)."""
    keys = rng.random((k, len(avail)))
    return avail[np.argpartition(keys, 5, axis=1)[:, :5]]


def make_runouts(rng, board, avail, r):
    """Full boards completing `board` with cards from avail.

    Returns (full (R, 5), new (R, need)). Turn spots enumerate every river
    when r >= the number of cards left, otherwise a random subset.
    """
    b = np.asarray(board, dtype=np.int64)
    need = 5 - len(b)
    if need == 0:
        return b[None, :], np.zeros((1, 0), dtype=np.int64)
    if need == 1:
        new = avail[:, None] if r >= len(avail) else rng.choice(avail, r, replace=False)[:, None]
    else:
        new = rng.permuted(np.tile(avail, (r, 1)), axis=1)[:, :need]
    full = np.concatenate([np.tile(b, (len(new), 1)), new], axis=1)
    return full, new


def _onehot(hands: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(hands), 36), dtype=bool)
    oh[np.arange(len(hands))[:, None], hands] = True
    return oh


def evaluate(hole, board, hands: np.ndarray, r: int, rng, n_ref: int = 48, full_out: bool = False):
    """Score our hand and K particle hands over R runouts of `board`.

    Returns (win, h, n_valid):
        win[k]  our win share vs particle k (ties count 1/2)
        h[k]    particle k's strength percentile vs random hands
        n_valid number of valid runouts per particle

    n_ref = 0 uses the particles themselves as the random-hand reference
    (valid while they are still an unweighted uniform sample, i.e. before any
    resampling): this saves scoring a separate reference set.
    """
    hole = np.asarray(hole, dtype=np.int64)
    avail = available(hole, board)
    full, new = make_runouts(rng, board, avail, r)
    ptT = np.ascontiguousarray(fe.pair_tables(full).T)          # (630, R)
    refs = sample_hands(rng, avail, n_ref) if n_ref > 0 else hands[:0]
    k = len(hands)
    allh = np.concatenate([hands, refs, hole[None]])            # (K + n_ref + 1, 5)
    sc = ptT[fe.hand_pair_indices(allh)].max(1)                 # (K + n_ref + 1, R)
    S, Q, my = sc[:k], sc[k:k + n_ref], sc[-1]
    if new.shape[1]:
        bad = _onehot(allh[:-1])[:, new].any(-1)                # (K + n_ref, R)
        valid, qvalid = ~bad[:k], ~bad[k:]
    else:
        valid = np.ones(S.shape, dtype=bool)
        qvalid = np.ones(Q.shape, dtype=bool)
    if n_ref == 0:
        Q, qvalid = S, valid
    pctl = _percentiles(S, Q, qvalid)                            # (K, R)
    nv = valid.sum(1)
    nvs = np.maximum(nv, 1)
    h = (pctl * valid).sum(1) / nvs
    wr = (my > S).astype(np.float64)
    wr += 0.5 * (my == S)
    win = (wr * valid).sum(1) / nvs
    if not nv.all():
        h[nv == 0] = 0.5
        win[nv == 0] = 0.5
    if full_out:
        return win, h, nv, wr, valid, (ptT, Q, qvalid)
    return win, h, nv


_OFF = 1 << 25          # bigger than any score (< 2**24)


def _percentiles(S, Q, qvalid):
    """For each S[k, r]: share of valid reference scores Q[:, r] below it (ties 1/2).

    One sort + two searchsorted over all columns at once: column r is shifted
    by r * _OFF, invalid references are sent below their column (score -1).
    """
    R = S.shape[1]
    off = (np.arange(R, dtype=np.int64) * _OFF)[None, :]
    q = np.where(qvalid, Q, -1).astype(np.int64) + off          # (n_ref, R)
    flat = np.sort(q.ravel())
    s = S.astype(np.int64) + off                                 # (K, R)
    lo = np.searchsorted(flat, s, side="left")
    hi = np.searchsorted(flat, s, side="right")
    n_ref = Q.shape[0]
    start = np.arange(R) * n_ref                                 # sorted: column r occupies [r*n, (r+1)*n)
    n_invalid = (~qvalid).sum(0)                                 # (R,)
    nq = np.maximum(n_ref - n_invalid, 1)
    less = lo - start[None, :] - n_invalid[None, :]
    ties = hi - lo
    return (less + 0.5 * ties) / nq[None, :]


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """The q quantile of x under weights w."""
    q = min(1.0, max(0.0, q))
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return float(np.quantile(x, q))
    i = int(np.searchsorted(cw, q * cw[-1]))
    return float(x[order[min(i, len(x) - 1)]])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


FOLD_FLOOR = 0.02      # nobody folds the nuts, everybody misclicks sometimes


def action_probs(h, theta_r, theta_f, facing, tau, eps, slow=0.0):
    """(p_fold, p_pass, p_raise) per particle for one decision context.

    eps  = bluff rate: P(raise | weak hand)
    slow = slow-play rate: P(no raise | strong hand)
    """
    top = max(eps + 0.02, 1.0 - slow)
    p_r = eps + (top - eps) / (1.0 + np.exp(np.clip((theta_r - h) / tau, -40, 40)))
    if facing:
        p_f = FOLD_FLOOR + (1.0 - 2 * FOLD_FLOOR) / (1.0 + np.exp(np.clip((h - theta_f) / tau, -40, 40)))
        p_p = np.maximum(1.0 - p_r - p_f, 0.01)
        tot = p_r + p_f + p_p
        return p_f / tot, p_p / tot, p_r / tot
    p_p = np.maximum(1.0 - p_r, 0.01)
    tot = p_r + p_p
    return np.zeros_like(h), p_p / tot, p_r / tot


def quantiles(h, w, qs) -> list:
    """Several weighted quantiles of h with one sort."""
    order = np.argsort(h)
    cw = np.cumsum(w[order])
    tot = cw[-1]
    n = len(h) - 1
    idx = np.searchsorted(cw, np.clip(np.asarray(qs, dtype=np.float64), 0.0, 1.0) * tot)
    hs = h[order[np.minimum(idx, n)]]
    return hs.tolist()


def thetas(h, w, freqs):
    """Frequency-matched thresholds from (f_fold, f_pass, f_raise):
    theta_r = (1 - f_raise) quantile, theta_f = f_fold quantile of h under w."""
    ff, _, fr = freqs
    order = np.argsort(h)
    cw = np.cumsum(w[order])
    tot = cw[-1]
    n = len(h) - 1
    i_r = min(n, int(np.searchsorted(cw, (1.0 - min(1.0, max(0.0, fr))) * tot)))
    i_f = min(n, int(np.searchsorted(cw, min(1.0, max(0.0, ff)) * tot)))
    return float(h[order[i_r]]), float(h[order[i_f]])


class Particles:
    """The particle set for one hand."""

    def __init__(self, hole, board, k, rng, params):
        self.hole = np.asarray(hole, dtype=np.int64)
        self.k = k
        self.params = params
        self.rng = rng
        self.board_len = len(board)
        self.hands = sample_hands(rng, available(hole, board), k)
        self.w = np.ones(k)
        self.hcache: dict[int, np.ndarray] = {}
        self.events: list[dict] = []
        self.resamples = 0
        self.version = 0          # bumps whenever the particle set changes
        self.uniform = True       # still an unweighted uniform sample (no resampling yet)?
        self.ess_log: list[float] = []

    # -- strength of (some) particles at an earlier or current board -------
    def strength(self, hands, board, blen, r=None):
        if blen == 0:
            return pf.PCT[pf.class_index_batch(hands)]
        r = r or self.params["r_replay"]
        _, h, _ = evaluate(self.hole, board[:blen], hands, r, self.rng, self.params["n_ref_replay"])
        return h

    def h_at(self, board, blen):
        if blen not in self.hcache:
            self.hcache[blen] = self.strength(self.hands, board, blen)
        return self.hcache[blen]

    # -- apply one observed opponent action ---------------------------------
    def apply(self, board, act, freqs, tau, eps, slow=0.0):
        """act: dict with blen, kind ('fold'/'pass'/'raise'), facing."""
        h = self.h_at(board, act["blen"])
        theta_r, theta_f = thetas(h, self.w, freqs)
        ev = {"blen": act["blen"], "kind": act["kind"], "facing": act["facing"],
              "theta_r": theta_r, "theta_f": theta_f, "tau": tau, "eps": eps, "slow": slow}
        self.w = self.w * self._lik(h, ev)
        self.events.append(ev)

    @staticmethod
    def _lik(h, ev):
        p_f, p_p, p_r = action_probs(h, ev["theta_r"], ev["theta_f"], ev["facing"],
                                     ev["tau"], ev["eps"], ev["slow"])
        return {"fold": p_f, "pass": p_p, "raise": p_r}[ev["kind"]]

    # -- new board cards: drop impossible particles, top up -----------------
    def advance_board(self, board):
        new_cards = list(board[self.board_len:])
        self.board_len = len(board)
        if not new_cards:
            return
        keep = ~_onehot(self.hands)[:, new_cards].any(1)
        self.hands, self.w = self.hands[keep], self.w[keep]
        self.hcache = {}
        n_new = self.k - len(self.hands) if self.params.get("topup", False) else 0
        if n_new > 0:
            fresh = sample_hands(self.rng, available(self.hole, board), n_new)
            wf = np.ones(n_new)
            hcache: dict[int, np.ndarray] = {}
            for ev in self.events:
                if ev["blen"] not in hcache:
                    hcache[ev["blen"]] = self.strength(fresh, board, ev["blen"])
                wf = wf * self._lik(hcache[ev["blen"]], ev)
            self.hands = np.concatenate([self.hands, fresh])
            self.w = np.concatenate([self.w, wf])
        self.version += 1
        if len(self.hands) < 8:      # almost nothing left: start over from uniform
            self.hands = sample_hands(self.rng, available(self.hole, board), self.k)
            self.w = np.ones(self.k)

    # -- degeneracy control ---------------------------------------------------
    def ess(self) -> float:
        s = self.w.sum()
        return float(s * s / max(1e-300, (self.w * self.w).sum()))

    def maybe_resample(self, min_ess):
        e = self.ess()
        self.ess_log.append(e)
        if e >= min_ess:
            return None
        p = self.w / self.w.sum()
        n = len(self.w)
        idx = self.rng.choice(n, size=n, p=p)
        mean_w = self.w.sum() / n               # keep the weight scale for top-ups
        self.hands = self.hands[idx]
        self.w = np.full(n, mean_w)
        self.hcache = {b: v[idx] for b, v in self.hcache.items()}
        self.resamples += 1
        self.uniform = False
        self.version += 1
        return idx

    def normalized(self) -> np.ndarray:
        return self.w / self.w.sum()
