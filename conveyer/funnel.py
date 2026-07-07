"""Purchase-journey (funnel) stage inference.

Two layers, matching research question Q1 ("How can we infer user purchase
journey stage?"):

1. **Per-turn text classification** — keyword/anchor scoring of each user turn
   into the SimilarWeb 6-stage funnel (Awareness → Discovery → Evaluation →
   Intent → Purchase → Post-Purchase). This mirrors the `fact_ai_funnel`
   stage definitions so labels are comparable with the star schema.
2. **Session-level sequence model** — a discrete Hidden Markov Model fitted
   with Baum-Welch on the per-turn stage observations. The latent state is the
   *true* journey stage; the per-turn classifier output is a noisy emission.
   The HMM smooths misclassifications and yields a monotonic-ish decoded
   journey per session plus a learned stage-transition matrix.

Everything is numpy-only (no hmmlearn dependency).
"""

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Stage taxonomy (aligned with dim_ai_funnel in the SimilarWeb star schema)
# --------------------------------------------------------------------------- #
STAGES: List[str] = [
    "Awareness", "Discovery", "Evaluation", "Intent", "Purchase", "Post-Purchase",
]
STAGE_INDEX: Dict[str, int] = {s: i for i, s in enumerate(STAGES)}

# Keyword cues per stage. These are deliberately simple and transparent;
# swap for embedding-anchor k-NN (see notebooks/03) or an LLM for production.
STAGE_PATTERNS: Dict[str, List[str]] = {
    "Awareness": [r"what is\b", r"how does .* work", r"explain", r"what causes",
                  r"is .* bad for", r"why (do|does|is)"],
    "Discovery": [r"what (products|options|brands)", r"recommend", r"best .* for",
                  r"suggest", r"what should i (use|try)", r"any good", r"dupe"],
    "Evaluation": [r"\bvs\b", r"versus", r"compare", r"difference between",
                   r"which (one )?is better", r"better than", r"worth it",
                   r"reviews? (of|for)", r"pros and cons", r"ingredients (in|of)"],
    "Intent": [r"where (can i|to) buy", r"price of", r"how much (is|does)",
               r"in stock", r"coupon", r"discount", r"cheapest", r"under \$?\d+",
               r"should i (buy|get|order)", r"link to"],
    "Purchase": [r"i('m| am) (buying|ordering|purchasing)", r"just (bought|ordered)",
                 r"add(ed)? to cart", r"checkout", r"placed (an|the) order"],
    "Post-Purchase": [r"i('ve| have) been using", r"i (bought|got|ordered) .* (and|but)",
                      r"how (do i|to) (use|apply)", r"is it normal", r"after using",
                      r"return (it|this)", r"refund", r"stopped working",
                      r"irritat", r"breaking me out", r"got worse", r"side effect"],
}


def stage_scores(text: str) -> np.ndarray:
    """Count keyword hits per stage; returns a length-6 score vector."""
    t = str(text).lower()
    return np.array([sum(bool(re.search(p, t)) for p in STAGE_PATTERNS[s]) for s in STAGES],
                    dtype=float)


def classify_stage(text: str, default: str = "Discovery") -> str:
    """Single funnel stage for one user turn (argmax of keyword hits)."""
    sc = stage_scores(text)
    if sc.sum() == 0:
        return default
    # later (deeper) stages win ties: a message with both discovery and intent
    # cues is further down the funnel
    return STAGES[int(np.flatnonzero(sc == sc.max())[-1])]


def add_stage(df: pd.DataFrame, question_col: str = "question") -> pd.DataFrame:
    df = df.copy()
    df["funnel_stage"] = df[question_col].fillna("").apply(classify_stage)
    df["funnel_stage_idx"] = df["funnel_stage"].map(STAGE_INDEX)
    return df


def stage_sequences(df: pd.DataFrame, session_col: str = "session_id",
                    pos_col: str = "session_pos") -> List[np.ndarray]:
    """Per-session observation sequences of stage indices, ordered by turn."""
    d = df.sort_values([session_col, pos_col])
    return [g["funnel_stage_idx"].to_numpy() for _, g in d.groupby(session_col)]


# --------------------------------------------------------------------------- #
# Discrete HMM (Baum-Welch + Viterbi), numpy-only
# --------------------------------------------------------------------------- #
class DiscreteHMM:
    """Multinomial-emission HMM for short categorical sequences.

    States are latent journey stages; observations are the (noisy) per-turn
    classifier labels. `left_right=True` initialises transitions to favour
    forward movement through the funnel, which is the natural prior for a
    purchase journey.
    """

    def __init__(self, n_states: int, n_symbols: int, left_right: bool = True,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_states, self.n_symbols = n_states, n_symbols
        self.pi = np.full(n_states, 1.0 / n_states)
        if left_right:
            A = np.eye(n_states) * 4.0 + np.triu(np.ones((n_states, n_states)), 1)
            A += 0.1  # small backward mass: journeys can revisit earlier stages
        else:
            A = rng.random((n_states, n_states)) + 1.0
        self.A = A / A.sum(axis=1, keepdims=True)
        # emissions: mostly-diagonal (classifier is right more often than not)
        B = np.full((n_states, n_symbols), 0.5) + 4.0 * np.eye(n_states, n_symbols)
        self.B = B / B.sum(axis=1, keepdims=True)

    def _forward_backward(self, obs: np.ndarray):
        T, N = len(obs), self.n_states
        alpha = np.zeros((T, N)); beta = np.zeros((T, N)); c = np.zeros(T)
        alpha[0] = self.pi * self.B[:, obs[0]]
        c[0] = alpha[0].sum() or 1e-300
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = alpha[t].sum() or 1e-300
            alpha[t] /= c[t]
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
        return alpha, beta, c

    def fit(self, sequences: Sequence[np.ndarray], n_iter: int = 50,
            tol: float = 1e-4) -> "DiscreteHMM":
        prev_ll = -np.inf
        seqs = [np.asarray(s, dtype=int) for s in sequences if len(s) > 0]
        for _ in range(n_iter):
            pi_acc = np.zeros(self.n_states)
            A_num = np.zeros_like(self.A)
            B_num = np.zeros_like(self.B)
            ll = 0.0
            for obs in seqs:
                alpha, beta, c = self._forward_backward(obs)
                ll += float(np.log(c).sum())
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                pi_acc += gamma[0]
                for t in range(len(obs) - 1):
                    xi = (alpha[t][:, None] * self.A * self.B[:, obs[t + 1]][None, :]
                          * beta[t + 1][None, :]) / c[t + 1]
                    A_num += xi
                for t, o in enumerate(obs):
                    B_num[:, o] += gamma[t]
            self.pi = pi_acc / pi_acc.sum()
            self.A = (A_num + 1e-6) / (A_num + 1e-6).sum(axis=1, keepdims=True)
            self.B = (B_num + 1e-6) / (B_num + 1e-6).sum(axis=1, keepdims=True)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        self.loglik_ = prev_ll
        return self

    def decode(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi path of latent stages for one observation sequence."""
        obs = np.asarray(obs, dtype=int)
        T, N = len(obs), self.n_states
        logA, logB = np.log(self.A + 1e-300), np.log(self.B + 1e-300)
        delta = np.log(self.pi + 1e-300) + logB[:, obs[0]]
        psi = np.zeros((T, N), dtype=int)
        for t in range(1, T):
            m = delta[:, None] + logA
            psi[t] = m.argmax(axis=0)
            delta = m.max(axis=0) + logB[:, obs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = int(delta.argmax())
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path

    def posterior(self, obs: np.ndarray) -> np.ndarray:
        """P(state_t | full sequence) — soft journey-stage assignment."""
        alpha, beta, _ = self._forward_backward(np.asarray(obs, dtype=int))
        gamma = alpha * beta
        return gamma / gamma.sum(axis=1, keepdims=True)


def fit_journey_hmm(df: pd.DataFrame, session_col: str = "session_id",
                    pos_col: str = "session_pos", n_iter: int = 50,
                    seed: int = 0) -> Tuple[DiscreteHMM, pd.DataFrame]:
    """Fit the journey HMM and return (model, df with `journey_stage` decoded)."""
    if "funnel_stage_idx" not in df.columns:
        df = add_stage(df)
    seqs = stage_sequences(df, session_col, pos_col)
    hmm = DiscreteHMM(len(STAGES), len(STAGES), left_right=True, seed=seed).fit(seqs, n_iter)

    d = df.sort_values([session_col, pos_col]).copy()
    decoded = np.concatenate([hmm.decode(g["funnel_stage_idx"].to_numpy())
                              for _, g in d.groupby(session_col)])
    d["journey_stage_idx"] = decoded
    d["journey_stage"] = [STAGES[i] for i in decoded]
    return hmm, d


def transition_table(hmm: DiscreteHMM) -> pd.DataFrame:
    return pd.DataFrame(hmm.A, index=STAGES[: hmm.n_states],
                        columns=STAGES[: hmm.n_states]).round(3)
