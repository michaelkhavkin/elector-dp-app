"""
experiments/privacy_utility.py  —  The DP-Elector privacy–utility analysis
==========================================================================
OFFLINE experiment (not part of the Streamlit UI). One self-contained script
that produces the paper's core result: under a SINGLE privacy budget ε, the
party gains population-level utility while the individual voter keeps individual
protection. Writes one CSV (every variable per ε) plus the plots.

This file replaces the previous three scripts (design_a.py, privacy_attack.py)
by unifying them around the one threat model the sharpened paper actually argues.

Threat model (the running example)
----------------------------------
The sensitive attribute is an individual's **voting-participation status** (did
they show up?) — NOT which party they support. The party is protected *by
construction*: the Elector never observes the ballot, it only *predicts* support
(`latent_support`). The harm under investigation is a **leak / insider misuse of
the centralized Elector database of participation records** (cf. the real 2021
Elector breach, ~6.5M records). To defend it, every status report is protected
at the point of capture by **binary randomized response** (Local DP); only the
noised value is ever stored.

  Running example (ε = 0.5 ⇒ p = e^0.5/(1+e^0.5) ≈ 0.62). Dana abstains. An
  activist marks her status; RR stores the truth ("didn't vote") with prob ≈0.62
  and flips it to "voted" with prob ≈0.38.

One ε, read two ways
--------------------
A SINGLE leaked-DB realization per round drives both sides at once:

  PRIVACY (the voter)  — plausible deniability of participation status.
    A Bayes-optimal attacker holding the leaked DP record AND the public base
    rates tries to recover Dana's true status. We report attacker accuracy, the
    no-record baseline, and the ADVANTAGE = accuracy − baseline (what the record
    actually adds). The LDP guarantee bounds the attacker's odds shift by e^ε
    regardless of N, so the advantage → 0 at small ε.
    (Secondary: the same attack on a k-RR-protected party signal — an OPTIONAL
    variant where the party support is also DP-protected. Party is not stored in
    the main model, so this is reported but off the critical path.)

  UTILITY (the party) — two readings, both from the same leaked records:
    • Operational: minimize WASTED activist contacts. The party chases its
      believed not-yet-voted supporters; privacy noise misdirects activists onto
      already-voted people, so we measure GOTV recall / precision / wasted, and
      `missed_due_to_privacy` = extra true supporters left unmobilized vs. a
      no-privacy oracle (concrete units: voters).
    • Statistical: accuracy of the dashboard aggregates the manager acts on —
      the de-biased population turnout estimate (and party-share estimate),
      recovered by voting_dp's unbiased estimators. Error shrinks with N, so it
      stays accurate where individual re-identification cannot.

The asymmetry is the paper's point: individual re-identification is bounded by
e^ε independent of N, while the party aggregates over N and its estimate error
shrinks with N — so one small ε can be good for BOTH sides.

Usage
-----
    python experiments/privacy_utility.py
    python experiments/privacy_utility.py --organiser "יש עתיד" --repeats 50 --budget 5000
    python experiments/privacy_utility.py --protect-party --support-noise 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the app package importable whether run from repo root or elsewhere.
_EXP_DIR = Path(__file__).resolve().parent
_APP_DIR = _EXP_DIR.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import population  # noqa: E402
from voting_dp import (  # noqa: E402
    epsilon_to_flip_probability,
    epsilon_to_krr_noise_probability,
    estimate_rr_frequency,
    estimate_krr_frequency,
)

# Default organiser: the party running the Elector. Falls back to the largest
# party by support if this label isn't present in the chosen population.
DEFAULT_ORGANISER = "הליכוד"

# ε sweep on a LINEAR grid: dense in the rising region (0.1–3) where the privacy
# risk climbs, with a few tail anchors (5, 8, 10) showing the saturation plateau.
# np.inf = the no-privacy ORACLE baseline (no noise at all).
DEFAULT_EPSILONS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, np.inf]

# Hebrew → English party labels for plots/filenames (matplotlib has no RTL
# shaping, so Hebrew renders reversed). Covers the 25th-Knesset parties that can
# survive `keep_top_parties`, plus the collapsed "אחר" bucket. Unknown labels
# fall back to a safe ASCII slug so a figure never carries reversed Hebrew.
PARTY_EN = {
    "הליכוד": "Likud",
    "יש עתיד": "Yesh Atid",
    "הציונות הדתית": "Religious Zionism",
    "המחנה הממלכתי": "National Unity",
    "שס": "Shas",
    'ש"ס': "Shas",
    "יהדות התורה": "United Torah Judaism",
    "ישראל ביתנו": "Yisrael Beiteinu",
    "מרצ": "Meretz",
    "העבודה": "Labor",
    'חד"ש-תע"ל': "Hadash-Taal",
    'רע"מ': "Raam",
    'בל"ד': "Balad",
    "הרשימה המשותפת": "Joint List",
    "אחר": "Other",
}


def en_party(label):
    """English display label for a (Hebrew) party name; ASCII-slug fallback."""
    if label in PARTY_EN:
        return PARTY_EN[label]
    slug = "".join(ch if ch.isascii() and (ch.isalnum() or ch in " -") else ""
                   for ch in str(label)).strip()
    return slug or "Party"


def _round_sig(v, sig=2):
    """Round to `sig` significant figures (for tidy auto-budget labels)."""
    if not np.isfinite(v) or v == 0:
        return float(v)
    from math import floor, log10
    return float(round(v, -int(floor(log10(abs(v)))) + (sig - 1)))


# =============================================================================
# DP CHANNELS  (vectorised forms of the voting_dp mechanisms, reusing its
# privacy-probability definitions so the guarantees match exactly)
# =============================================================================

def rr_status(rng, true_voted, epsilon):
    """
    Vectorised binary randomized response on the has-voted flag — THE leaked
    participation record.

    Equivalent to applying `voting_dp.randomized_response` element-wise: a value
    is reported truthfully with prob p = e^ε/(1+e^ε) and flipped otherwise. We
    reuse `epsilon_to_flip_probability` for P(flip) so the calibration is identical.
    ε=inf → P(flip)=0 → exact values.
    """
    p_flip = epsilon_to_flip_probability(epsilon)   # 1/(1+e^ε); 0 at ε=inf
    flips = rng.random(true_voted.shape) < p_flip
    return np.where(flips, ~true_voted, true_voted)


def krr_indices(rng, true_idx, k, epsilon):
    """
    Vectorised k-randomized response on integer category CODES (0..k-1).

    The fully-vectorised core of k-RR: kept truthfully with prob
    p_true = e^ε/(e^ε+k-1), else replaced uniformly by one of the other k-1
    codes. Reuses `epsilon_to_krr_noise_probability` for P(report≠truth).
    Working on codes (not labels) avoids per-element Python loops, which matters
    at national scale (~6.8M voters). ε=inf → codes returned unchanged.
    """
    p_noise = epsilon_to_krr_noise_probability(epsilon, k)   # 0 at ε=inf
    if p_noise <= 0:
        return true_idx
    flip = rng.random(len(true_idx)) < p_noise
    # Draw uniformly among the OTHER k-1 codes: pick r in [0,k-1), skip self.
    other = rng.integers(0, k - 1, size=len(true_idx))
    other = np.where(other >= true_idx, other + 1, other)
    return np.where(flip, other, true_idx)


def krr_party(rng, party, candidates, epsilon):
    """
    Vectorised k-randomized response on a categorical party LABEL.

    Equivalent to `voting_dp.k_randomized_response` element-wise; a thin
    label↔code wrapper around `krr_indices`. `party` values must all be in
    `candidates`. ε=inf → exact values.
    """
    k = len(candidates)
    index = {c: i for i, c in enumerate(candidates)}
    true_idx = np.fromiter((index[p] for p in party), dtype=int, count=len(party))
    out_idx = krr_indices(rng, true_idx, k, epsilon)
    return np.asarray(candidates, dtype=object)[out_idx]


def _p_true_binary(epsilon):
    """P(report = truth) for binary RR. 1.0 at ε=inf."""
    return 1.0 if not np.isfinite(epsilon) else np.exp(epsilon) / (1.0 + np.exp(epsilon))


def _p_true_krr(epsilon, k):
    """P(report = truth) for k-RR. 1.0 at ε=inf."""
    return 1.0 if not np.isfinite(epsilon) else np.exp(epsilon) / (np.exp(epsilon) + k - 1)


# =============================================================================
# BAYES-OPTIMAL ATTACKER DECISION RULES  (depend on ε + base rates, not on draws)
# =============================================================================

def status_decision_rule(q, epsilon):
    """
    Bayes-optimal guess of true status for each observed value, given P(voted)=q.
    Symmetric binary channel: returns (guess_if_observed_1, guess_if_observed_0)
    as booleans, and the no-record baseline max(q, 1-q).
    """
    p = _p_true_binary(epsilon)
    guess_if_1 = (q * p) >= ((1.0 - q) * (1.0 - p))
    guess_if_0 = (q * (1.0 - p)) >= ((1.0 - q) * p)
    baseline = max(q, 1.0 - q)
    return bool(guess_if_1), bool(guess_if_0), baseline


def party_decision_rule(f, epsilon):
    """
    Bayes-optimal guess of true party for each OBSERVED category, given the
    support distribution `f` (length k, sums to 1). Returns (guess_for_obs, base)
    where guess_for_obs[j] = argmax_t P(true=t | observed=j) and base = f.max().
    """
    k = len(f)
    p_true = _p_true_krr(epsilon, k)
    p_other = 0.0 if not np.isfinite(epsilon) else (1.0 - p_true) / (k - 1)
    guess_for_obs = np.empty(k, dtype=int)
    for j in range(k):
        joint = f * p_other
        joint[j] = f[j] * p_true
        guess_for_obs[j] = int(joint.argmax())
    return guess_for_obs, float(f.max())


# =============================================================================
# ONE SIMULATION ROUND
# =============================================================================
# A single leaked-DB realization at budget ε drives every metric: the privacy
# attack, the de-biased aggregates, and the GOTV operation all read the SAME
# noised records — exactly the "one ε, two sides" story.

def simulate_round(state, epsilon, rng, budget=None, protect_party=False,
                   eps_party=None):
    """
    Run one round at privacy budget `epsilon`; return a flat dict of all metrics.

    `state` is the precomputed-per-population context from `build_state`.
    `budget` caps the number of people chased (None = unlimited activist capacity).
    `protect_party` additionally protects the GOTV support signal with k-RR
    (`eps_party`, defaults to `epsilon`) — an imperfect DP-protected predictor.
    """
    true_voted = state["true_voted"]
    support_idx = state["support_idx"]
    candidates = state["candidates"]
    candidates_arr = state["candidates_arr"]
    org_idx = state["org_idx"]
    k = len(candidates)

    # ---- THE leaked records (Local DP at point of capture) ------------------
    # Party is kept as integer codes (not labels) so every round stays fully
    # vectorised — essential at national scale.
    status_obs = rr_status(rng, true_voted, epsilon)                 # binary RR
    party_obs_idx = krr_indices(rng, support_idx, k, epsilon)        # k-RR (optional/secondary)

    # ====================================================================== #
    # PRIVACY (voter): Bayes-optimal re-identification from the leaked records
    # ====================================================================== #
    guess_if_1, guess_if_0, status_baseline = state["status_rule"][epsilon]
    status_guess = np.where(status_obs, guess_if_1, guess_if_0)
    status_attacker_acc = float((status_guess == true_voted).mean())

    party_guess_for_obs, party_baseline = state["party_rule"][epsilon]
    party_attacker_acc = float((party_guess_for_obs[party_obs_idx] == support_idx).mean())

    # ====================================================================== #
    # UTILITY — statistical: de-biased population aggregates (the dashboard)
    # ====================================================================== #
    if np.isfinite(epsilon):
        turnout_est = estimate_rr_frequency(status_obs, epsilon)
        party_obs = candidates_arr[party_obs_idx]   # codes → labels (vectorised)
        share_est = estimate_krr_frequency(party_obs, candidates, epsilon)
        share_vec = np.array([share_est[c] for c in candidates])
    else:
        turnout_est = state["turnout_true"]
        share_vec = state["share_true"]
    turnout_estimate_accuracy = 1.0 - abs(turnout_est - state["turnout_true"])
    party_share_estimate_accuracy = 1.0 - 0.5 * np.abs(share_vec - state["share_true"]).sum()

    # ====================================================================== #
    # UTILITY — operational: GOTV. Chase believed not-yet-voted supporters.
    # ====================================================================== #
    is_target = state["is_target"]                  # ~true_voted & support==organiser
    n_targets = state["n_targets"]

    believed_voted = status_obs
    if protect_party:
        eps_p = epsilon if eps_party is None else eps_party
        believed_supporter = krr_indices(rng, support_idx, k, eps_p) == org_idx
    else:
        believed_supporter = support_idx == org_idx   # organiser's clean prior
    believed_chaseworthy = (~believed_voted) & believed_supporter

    chosen = believed_chaseworthy.copy()
    n_candidates = int(believed_chaseworthy.sum())
    if budget is not None and n_candidates > budget:
        idx = np.flatnonzero(believed_chaseworthy)
        drop = rng.permutation(idx)[budget:]
        chosen[drop] = False

    chased = int(chosen.sum())
    reached = int((chosen & is_target).sum())       # true targets actually chased
    missed = n_targets - reached                    # true targets NOT reached
    wasted = chased - reached                        # effort on non-targets

    return {
        "epsilon": epsilon,
        # --- privacy (voter) ---
        "status_baseline": status_baseline,
        "status_attacker_acc": status_attacker_acc,
        "status_disclosure_advantage": status_attacker_acc - status_baseline,
        "party_baseline": party_baseline,
        "party_attacker_acc": party_attacker_acc,
        "party_disclosure_advantage": party_attacker_acc - party_baseline,
        "odds_bound": np.inf if not np.isfinite(epsilon) else float(np.exp(epsilon)),
        # --- utility: statistical (party dashboard) ---
        "turnout_estimate_accuracy": turnout_estimate_accuracy,
        "party_share_estimate_accuracy": float(party_share_estimate_accuracy),
        # --- utility: operational (GOTV) ---
        "n_targets": n_targets,
        "chased": chased,
        "reached": reached,
        "missed": missed,
        "wasted": wasted,
        "recall": reached / n_targets if n_targets else np.nan,
        "precision": reached / chased if chased else np.nan,
    }


# =============================================================================
# POPULATION CONTEXT  (everything that does not depend on the random draw)
# =============================================================================

def build_state(pop, organiser, candidates, epsilons):
    """Precompute base rates, attacker decision rules and GOTV targets once."""
    true_voted = pop["true_voted"].to_numpy(dtype=bool)
    support = pop["latent_support"].to_numpy(dtype=object)

    index = {c: i for i, c in enumerate(candidates)}
    support_idx = np.fromiter((index[c] for c in support), dtype=int, count=len(support))

    q = float(true_voted.mean())                          # P(voted)
    f = np.bincount(support_idx, minlength=len(candidates)).astype(float)
    f /= f.sum()

    is_target = (~true_voted) & (support == organiser)

    # Bayes rules are deterministic functions of (ε, base rate): cache per ε.
    status_rule = {eps: status_decision_rule(q, eps) for eps in epsilons}
    party_rule = {eps: party_decision_rule(f.copy(), eps) for eps in epsilons}

    return {
        "true_voted": true_voted,
        "support_idx": support_idx,
        "candidates": candidates,
        "candidates_arr": np.asarray(candidates, dtype=object),
        "organiser": organiser,
        "org_idx": index[organiser],
        "turnout_true": q,
        "share_true": f,
        "is_target": is_target,
        "n_targets": int(is_target.sum()),
        "status_rule": status_rule,
        "party_rule": party_rule,
    }


# =============================================================================
# SWEEP + AGGREGATION
# =============================================================================

def run_sweep(pop, organiser, candidates, epsilons, repeats=30,
              budget=None, protect_party=False, eps_party=None, seed=0):
    """Monte-Carlo over (epsilon, repeat). Returns a per-round tidy DataFrame."""
    state = build_state(pop, organiser, candidates, epsilons)
    rng = np.random.default_rng(seed)
    records = []
    for eps in epsilons:
        for r in range(repeats):
            row = simulate_round(state, eps, rng, budget=budget,
                                 protect_party=protect_party, eps_party=eps_party)
            row["repeat"] = r
            records.append(row)
    return pd.DataFrame(records)


def aggregate(df):
    """
    Mean (+95% CI on the headline rates) per epsilon — one row per ε holding
    every variable. `missed_due_to_privacy` = mean missed at this ε minus the
    oracle (ε=inf): the extra supporters left unmobilized BECAUSE of privacy.
    """
    g = df.groupby("epsilon")
    agg = g.agg(
        # privacy (voter)
        status_baseline=("status_baseline", "mean"),
        status_attacker_acc=("status_attacker_acc", "mean"),
        status_attacker_std=("status_attacker_acc", "std"),
        status_disclosure_advantage=("status_disclosure_advantage", "mean"),
        party_baseline=("party_baseline", "mean"),
        party_attacker_acc=("party_attacker_acc", "mean"),
        party_disclosure_advantage=("party_disclosure_advantage", "mean"),
        odds_bound=("odds_bound", "first"),
        # utility: statistical (dashboard aggregates)
        turnout_estimate_accuracy=("turnout_estimate_accuracy", "mean"),
        party_share_estimate_accuracy=("party_share_estimate_accuracy", "mean"),
        # utility: operational (GOTV)
        n_targets=("n_targets", "mean"),
        reached_mean=("reached", "mean"),
        missed_mean=("missed", "mean"),
        missed_std=("missed", "std"),
        wasted_mean=("wasted", "mean"),
        wasted_std=("wasted", "std"),
        chased_mean=("chased", "mean"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        repeats=("repeat", "count"),
    ).reset_index()

    n = agg["repeats"].clip(lower=1)
    agg["status_attacker_ci95"] = 1.96 * agg["status_attacker_std"] / np.sqrt(n)
    agg["recall_ci95"] = 1.96 * agg["recall_std"] / np.sqrt(n)
    agg["precision_ci95"] = 1.96 * agg["precision_std"] / np.sqrt(n)
    agg["missed_ci95"] = 1.96 * agg["missed_std"] / np.sqrt(n)   # ± voters
    agg["wasted_ci95"] = 1.96 * agg["wasted_std"] / np.sqrt(n)   # ± calls

    # Privacy price isolated against the no-privacy oracle (ε=inf). Both mirror
    # each other: extra targets left unmobilized (votes) and extra misdirected
    # calls (contacts) caused specifically by the status-RR noise — not by the
    # support model (which is equally imperfect at the oracle, so it cancels).
    oracle_missed = agg.loc[agg["epsilon"] == np.inf, "missed_mean"]
    oracle_missed = float(oracle_missed.iloc[0]) if len(oracle_missed) else 0.0
    agg["missed_due_to_privacy"] = agg["missed_mean"] - oracle_missed     # lost votes

    oracle_wasted = agg.loc[agg["epsilon"] == np.inf, "wasted_mean"]
    oracle_wasted = float(oracle_wasted.iloc[0]) if len(oracle_wasted) else 0.0
    agg["wasted_due_to_privacy"] = agg["wasted_mean"] - oracle_wasted     # extra calls

    return agg.sort_values("epsilon").reset_index(drop=True)


# =============================================================================
# PLOTS
# =============================================================================

# Shared ε axis label: short name + short interpretation. The ">>>" arrow points
# in the direction of increasing ε (left→right on the axis) = weaker privacy.
XLABEL = "Privacy budget  ε   (higher ε  >>>  weaker privacy)"


def _split(agg):
    finite = agg[np.isfinite(agg["epsilon"])]
    oracle = agg[~np.isfinite(agg["epsilon"])]
    return finite, oracle


def _linear_x(ax, eps_values):
    """
    Linear (natural-interval) ε axis: each point sits at its actual ε value, so
    the gap 2→3 is twice the gap 0.5→1. A *log* axis would make equal ratios
    equally spaced, and an *ordinal* axis would make every gap equal — both
    misread the real spacing of the budget. Ticks are placed at the swept values.
    Returns the x-positions (= the ε values).
    """
    xp = np.asarray(eps_values, dtype=float)
    span = float(xp.max() - xp.min()) or 1.0
    ax.set_xticks(xp)
    # Rotate labels: on a linear axis the small-ε ticks (0.1, 0.25, 0.5) crowd
    # together, so horizontal labels would overlap.
    ax.set_xticklabels([f"{v:g}" for v in eps_values], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlim(xp.min() - 0.04 * span, xp.max() + 0.04 * span)
    return xp


def _feasible_left_edge(xp, feasible):
    """
    Left x-boundary for shading a (high-ε) feasible region on the linear axis:
    midway between the last-infeasible and first-feasible swept ε. If the whole
    range is feasible, return just below the smallest ε.
    """
    star = int(np.flatnonzero(feasible).min())
    if star == 0:
        span = float(xp.max() - xp.min()) or 1.0
        return float(xp.min() - 0.04 * span)
    return float(0.5 * (xp[star - 1] + xp[star]))


def _halfwidth(finite, std_col, kind):
    """
    95% error half-width for a swept metric, in the units of `std_col`.

      kind="ci"       → 1.96·σ/√repeats : CI of the *mean* over Monte-Carlo
                        repeats. Shrinks with both N and repeats, so it is tiny
                        (often sub-pixel) at national scale — our "regular CI".
      kind="sampling" → 1.96·σ          : the 95% spread a *single* campaign
                        actually realises from the RR draws (no √repeats). This
                        is the RR sampling error one election sees; ≈√repeats
                        wider than the CI, hence visible.
    """
    std = finite[std_col].to_numpy()
    if kind == "ci":
        n = np.clip(finite["repeats"].to_numpy().astype(float), 1.0, None)
        return 1.96 * std / np.sqrt(n)
    return 1.96 * std


def plot_attack(agg, out_path):
    """Attacker accuracy vs ε for status (& party) with no-record baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, _ = _split(agg)
    x = finite["epsilon"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xp = _linear_x(ax, x)

    ax.plot(xp, finite["status_attacker_acc"], "o-", color="#C44E52",
            label="Attacker: recover voting status (primary)")
    ax.axhline(finite["status_baseline"].iloc[0], color="#C44E52", ls=":", lw=1.2,
               label="Status baseline (no record)")
    ax.plot(xp, finite["party_attacker_acc"], "s-", color="#3b6cb7",
            label="Attacker: recover party support (secondary)")
    ax.axhline(finite["party_baseline"].iloc[0], color="#3b6cb7", ls=":", lw=1.2,
               label="Party baseline (no record)")
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5,
               label="No DP (certain re-identification)")

    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Attacker accuracy on one individual")
    ax.set_ylim(0, 1.02)
    ax.set_title("Re-identification risk from a leaked record")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gotv(agg, out_path, organiser):
    """Operational utility: GOTV recall & precision vs ε with 95% CI bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, oracle = _split(agg)
    x = finite["epsilon"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xp = _linear_x(ax, x)
    for col, ci, label, color in [
        ("recall_mean", "recall_ci95", "Recall (targets reached)", "#3b6cb7"),
        ("precision_mean", "precision_ci95", "Precision (effort not wasted)", "#C44E52"),
    ]:
        y = finite[col].to_numpy()
        e = finite[ci].to_numpy()
        ax.plot(xp, y, "o-", color=color, label=label)
        ax.fill_between(xp, y - e, y + e, color=color, alpha=0.15)
        if len(oracle):
            ax.axhline(float(oracle[col].iloc[0]), color=color, ls="--", lw=1, alpha=0.6)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Targeting quality vs. privacy budget  ({organiser})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_privacy_utility(agg, out_path, cost_per_call=1.0, error_kind="sampling"):
    """
    Money plot: the privacy–utility asymmetry on one ε axis, now with the cost.

      LEFT  (rate 0–1):
        • turnout_estimate_accuracy     — de-biased COUNT utility (↑ better)
        • party_share_estimate_accuracy — de-biased COUNT utility, 1−TV (↑)
        • recall_mean                   — operational GOTV utility (↑)
        • status_disclosure_advantage   — voter risk (↓ better)
      RIGHT (USD):
        • cost_misdirected_calls        — operational cost (↓ better)

    The point: the de-biased *count* utilities (turnout, party-share) stay ≈1
    even at small ε — they aggregate over N — while the operational cost falls
    only as ε grows and voter risk rises. So the count-based dashboard the party
    acts on is accurate cheaply in privacy terms; the price of privacy is paid in
    operational dollars, not in count accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, _ = _split(agg)
    x = finite["epsilon"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    xp = _linear_x(ax, x)

    handles = []
    handles.append(ax.plot(xp, finite["turnout_estimate_accuracy"], "o-", color="#55A868",
                   label="Party utility: turnout-count estimate  (↑ better)")[0])
    handles.append(ax.plot(xp, finite["party_share_estimate_accuracy"], "^-", color="#3b6cb7",
                   label="Party utility: party-share estimate, 1−TV  (↑ better)")[0])
    handles.append(ax.plot(xp, finite["recall_mean"], "v-", color="#8172B3",
                   label="Party utility: supporters reached, recall  (↑ better)")[0])
    risk = finite["status_disclosure_advantage"].to_numpy()
    handles.append(ax.plot(xp, risk, "s--", color="#C44E52",
                   label="Voter risk: status disclosure advantage  (↓ better)")[0])
    ax.fill_between(xp, 0, risk, color="#C44E52", alpha=0.10)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Rate  (0-1)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # RIGHT axis: the operational cost (dollars), with its error band.
    ax2 = ax.twinx()
    c_cost = "#DD8452"
    cost = finite["cost_misdirected_calls"].to_numpy()
    cost_e = _halfwidth(finite, "wasted_std", error_kind) * cost_per_call
    handles.append(ax2.plot(xp, cost, "D-", color=c_cost,
                   label=f"Party cost: misdirected calls  (↓ better, @ ${cost_per_call:g}/call)")[0])
    ax2.fill_between(xp, cost - cost_e, cost + cost_e, color=c_cost, alpha=0.15)
    ax2.set_ylabel(f"Cost of misdirected calls  (USD @ ${cost_per_call:g}/call)",
                   color=c_cost)
    ax2.tick_params(axis="y", labelcolor=c_cost)
    ax2.set_ylim(bottom=0)

    ax.set_title("Privacy vs. utility asymmetry")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=8, frameon=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_price_of_privacy(agg, out_path, cost_per_call=1.0, max_loss=None,
                          error_kind="sampling"):
    """
    The COST side of privacy alone (the operational price), on ONE ε axis with
    two currencies on twin axes:

      • LEFT  (voters) — potential votes lost: `missed_due_to_privacy`, the extra
        true supporters left unmobilized because the status noise misdirected
        activists away from them.
      • RIGHT (USD)    — cost of misdirected calls: `wasted_due_to_privacy`
        × cost_per_call, the extra contacts spent on already-voted people.

    The two are deliberately kept separate (different units: lost votes vs.
    dollars). Both y-axes share identical numeric limits, so at $1/call — where
    dollars equal the wasted-call count — the curves are directly comparable and
    the wasted-call cost visibly exceeds the votes lost (symmetric RR inflates
    the minority "not-voted" class on election day). Shaded bands are the error
    range (see `error_kind`); the loss-limit line + green band show the ε that
    keep the loss within `max_loss`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, _ = _split(agg)
    x = finite["epsilon"].to_numpy()
    missed = finite["missed_due_to_privacy"].to_numpy()
    missed_e = _halfwidth(finite, "missed_std", error_kind)
    cost = finite["cost_misdirected_calls"].to_numpy()
    cost_e = _halfwidth(finite, "wasted_std", error_kind) * cost_per_call

    c_votes = "#8172B3"
    c_money = "#C44E52"
    c_feas = "#2E8B57"
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    xp = _linear_x(ax, x)
    l1, = ax.plot(xp, missed, "v-", color=c_votes,
                  label="Potential votes lost (supporters not mobilized)")
    ax.fill_between(xp, missed - missed_e, missed + missed_e, color=c_votes, alpha=0.15)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Potential votes lost  (voters)", color=c_votes)
    ax.tick_params(axis="y", labelcolor=c_votes)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    l2, = ax2.plot(xp, cost, "o-", color=c_money,
                   label=f"Cost of misdirected calls  (@ ${cost_per_call:g}/call)")
    ax2.fill_between(xp, cost - cost_e, cost + cost_e, color=c_money, alpha=0.15)
    ax2.set_ylabel(f"Cost of misdirected calls  (USD @ ${cost_per_call:g}/call)",
                   color=c_money)
    ax2.tick_params(axis="y", labelcolor=c_money)

    # Shared numeric limits so the two currencies are visually comparable.
    ymax = 1.05 * max(float(np.nanmax(missed + missed_e)),
                      float(np.nanmax(cost + cost_e)), 1.0)
    ax.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    handles = [l1, l2]
    if max_loss is not None:
        ax2.axhline(max_loss, color=c_feas, ls=":", lw=1.3)
        ax2.text(xp.max(), max_loss, f"max loss ${max_loss:,.0f}",
                 color=c_feas, fontsize=7, ha="right", va="bottom")
        feasible = cost <= max_loss
        if feasible.any():
            span = ax.axvspan(_feasible_left_edge(xp, feasible), ax.get_xlim()[1],
                              color=c_feas, alpha=0.12, label="Within loss limit")
            handles.append(span)

    ax.set_title("The operational price of privacy")
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_utility(agg, out_path, organiser_en, min_recall=None, error_kind="sampling"):
    """
    The UTILITY / privacy side alone (no cost): the voter's status disclosure
    advantage (↓ better) and the party's GOTV recall (↑ better) on a single
    rate axis, each with an error band (see `error_kind`). If `min_recall` is
    given, its threshold line + the green band mark the ε that meet the party's
    minimum recall. Companion to `plot_price_of_privacy` (the cost side); the
    `_combined_*` plots overlay both.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, _ = _split(agg)
    x = finite["epsilon"].to_numpy()
    risk = finite["status_disclosure_advantage"].to_numpy()
    risk_e = _halfwidth(finite, "status_attacker_std", error_kind)
    rec = finite["recall_mean"].to_numpy()
    rec_e = _halfwidth(finite, "recall_std", error_kind)

    c_risk = "#C44E52"
    c_util = "#3b6cb7"
    c_feas = "#2E8B57"
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xp = _linear_x(ax, x)

    l1, = ax.plot(xp, risk, "s--", color=c_risk,
                  label="Voter risk: status disclosure advantage  (↓ better)")
    ax.fill_between(xp, risk - risk_e, risk + risk_e, color=c_risk, alpha=0.15)
    l2, = ax.plot(xp, rec, "v-", color=c_util,
                  label="Party utility: supporters reached, recall  (↑ better)")
    ax.fill_between(xp, rec - rec_e, rec + rec_e, color=c_util, alpha=0.15)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Rate  (0-1)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    handles = [l1, l2]
    if min_recall is not None:
        ax.axhline(min_recall, color=c_util, ls=":", lw=1.3)
        ax.text(xp.min(), min_recall + 0.015, f"min recall {min_recall:g}",
                color=c_util, fontsize=7, ha="left", va="bottom")
        feasible = rec >= min_recall
        if feasible.any():
            span = ax.axvspan(_feasible_left_edge(xp, feasible), ax.get_xlim()[1],
                              color=c_feas, alpha=0.12, label="Meets minimum recall")
            handles.append(span)

    ax.set_title(f"Privacy vs. utility  ({organiser_en})")
    ax.legend(handles=handles, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined(agg, out_path, organiser_en, cost_per_call=1.0,
                  min_recall=None, min_precision=None, max_loss=None, error_kind="ci"):
    """
    One-glance summary of the whole tradeoff on a shared ε axis (twin axes):
    the voter's privacy risk, the party's operational utility, and the dollar
    cost of privacy — together, each with a 95% CI band.

      LEFT  (rate 0–1, three dimensionless metrics share it):
        • status_disclosure_advantage — voter risk            (↓ better)
        • recall_mean                 — targets reached        (↑ better)
        • precision_mean              — effort not wasted      (↑ better)
      RIGHT (USD):
        • cost_misdirected_calls      — privacy-induced wasted calls (↓ better)

    Only the dollar cost needs a second axis, so the figure stays at two axes
    (not three). Shaded bands are 95% CIs over the Monte-Carlo repeats — for the
    cost curve that is ±(wasted-call CI × $/call), i.e. a money error bar.

    Party operating constraints (illustrative): a campaign cannot exceed its
    tolerated misdirected-call loss (`max_loss`) nor fall below a minimum recall
    (`min_recall`) or precision (`min_precision`). All improve as ε rises, so the FEASIBLE region is
    the high-ε band where every active constraint holds (shaded green); the
    **privacy-optimal** choice is its smallest ε — the most protection for the
    voter that still meets the party's needs (dashed marker).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite, _ = _split(agg)
    x = finite["epsilon"].to_numpy()

    c_risk = "#C44E52"   # voter risk
    c_util = "#3b6cb7"   # party utility (recall)
    c_prec = "#55A868"   # party utility (precision)
    c_cost = "#DD8452"   # party cost ($)
    c_feas = "#2E8B57"   # feasible region

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    xp = _linear_x(ax, x)

    # ---- left axis: three rate metrics, each with an error band ----
    risk = finite["status_disclosure_advantage"].to_numpy()
    risk_e = _halfwidth(finite, "status_attacker_std", error_kind)   # baseline const → advantage spread = attacker-acc spread
    rec = finite["recall_mean"].to_numpy()
    rec_e = _halfwidth(finite, "recall_std", error_kind)
    prec = finite["precision_mean"].to_numpy()
    prec_e = _halfwidth(finite, "precision_std", error_kind)

    l1, = ax.plot(xp, risk, "s--", color=c_risk,
                  label="Voter risk: status disclosure advantage  (↓ better)")
    ax.fill_between(xp, risk - risk_e, risk + risk_e, color=c_risk, alpha=0.15)
    l2, = ax.plot(xp, rec, "v-", color=c_util,
                  label="Party utility: supporters reached, recall  (↑ better)")
    ax.fill_between(xp, rec - rec_e, rec + rec_e, color=c_util, alpha=0.15)
    l4, = ax.plot(xp, prec, "^-", color=c_prec,
                  label="Party utility: effort not wasted, precision  (↑ better)")
    ax.fill_between(xp, prec - prec_e, prec + prec_e, color=c_prec, alpha=0.15)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Rate  (0-1)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # ---- right axis: dollar cost, with a money error band ----
    ax2 = ax.twinx()
    cost = finite["cost_misdirected_calls"].to_numpy()
    cost_e = _halfwidth(finite, "wasted_std", error_kind) * cost_per_call
    l3, = ax2.plot(xp, cost, "o-", color=c_cost,
                   label=f"Party cost: misdirected calls  (↓ better, @ ${cost_per_call:g}/call)")
    ax2.fill_between(xp, cost - cost_e, cost + cost_e, color=c_cost, alpha=0.15)
    ax2.set_ylabel(f"Cost of misdirected calls  (USD @ ${cost_per_call:g}/call)",
                   color=c_cost)
    ax2.tick_params(axis="y", labelcolor=c_cost)
    ax2.set_ylim(bottom=0)

    handles = [l1, l2, l4, l3]

    # ---- party operating constraints + feasible-ε region ----
    if min_recall is not None:
        ax.axhline(min_recall, color=c_util, ls=":", lw=1.3)
        ax.text(xp.min(), min_recall + 0.015, f"min recall {min_recall:g}",
                color=c_util, fontsize=7, ha="left", va="bottom")
    if min_precision is not None:
        ax.axhline(min_precision, color=c_prec, ls=":", lw=1.3)
        ax.text(xp.max(), min_precision + 0.015, f"min precision {min_precision:g}",
                color=c_prec, fontsize=7, ha="right", va="bottom")
    if max_loss is not None:
        ax2.axhline(max_loss, color=c_cost, ls=":", lw=1.3)
        ax2.text(xp.max(), max_loss, f"max loss ${max_loss:,.0f}",
                 color=c_cost, fontsize=7, ha="right", va="bottom")

    if min_recall is not None or min_precision is not None or max_loss is not None:
        feasible = np.ones(len(x), dtype=bool)
        if min_recall is not None:
            feasible &= rec >= min_recall
        if min_precision is not None:
            feasible &= prec >= min_precision
        if max_loss is not None:
            feasible &= cost <= max_loss
        if feasible.any():
            left = _feasible_left_edge(xp, feasible)
            span = ax.axvspan(left, ax.get_xlim()[1], color=c_feas, alpha=0.12,
                              label="Party-feasible ε  (meets all active constraints)")
            handles.append(span)
            star = float(xp[np.flatnonzero(feasible).min()])
            vl = ax.axvline(star, color=c_feas, ls="--", lw=1.6,
                            label=f"Privacy-optimal feasible ε = {star:g}")
            handles.append(vl)
        else:
            ax.text(0.5, 0.5, "No ε meets both constraints",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#888", fontsize=9)

    ax.set_title(f"Privacy, utility & cost  ({organiser_en})")
    # Legend BELOW the axes so it never overlaps the curves.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=8, frameon=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="DP-Elector privacy–utility analysis (participation-status threat).")
    ap.add_argument("--organiser", default=DEFAULT_ORGANISER,
                    help="Party running the Elector (default: %(default)s).")
    ap.add_argument("--repeats", type=int, default=30,
                    help="Monte-Carlo repetitions per ε (default: %(default)s).")
    ap.add_argument("--budget", type=int, default=None,
                    help="Calls each activist can make (people one activist chases). "
                         "The EFFECTIVE chase capacity is --activists × --budget. "
                         "Default: unlimited.")
    ap.add_argument("--activists", type=int, default=1,
                    help="Number of activists the party can field (a FIXED count, "
                         "same for every party). Effective chase capacity = "
                         "--activists × --budget (default: %(default)s).")
    ap.add_argument("--frac-activists", type=float, default=None,
                    help="Alternative to --activists: field a fraction (0,1] of the "
                         "organiser's supporter base as activists, so the activist "
                         "count SCALES with party size (same fraction across parties). "
                         "n_activists = round(frac × supporters). Requires --budget; "
                         "mutually exclusive with --activists.")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Population down-sample fraction (default: full size).")
    ap.add_argument("--keep-top-parties", type=int, default=8,
                    help="Collapse minor parties into 'אחר' (default: %(default)s).")
    ap.add_argument("--support-noise", type=float, default=None,
                    help="If set, model an imperfect party predictor of non-voter "
                         "support with this noise level (make_noisy_support_model).")
    ap.add_argument("--protect-party", action="store_true",
                    help="Also protect the GOTV support signal with k-RR.")
    ap.add_argument("--cost-per-call", type=float, default=1.0,
                    help="Monetary cost of one activist contact, used to price the "
                         "misdirected (wasted) calls (default: $%(default)s/call — at "
                         "$1 the dollar cost equals the wasted-call count).")
    ap.add_argument("--min-recall", type=float, default=0.7,
                    help="Party's minimum acceptable GOTV recall; lower bound of the "
                         "feasible-ε region on the combined plot (default: %(default)s; "
                         "set <=0 to disable).")
    ap.add_argument("--min-precision", type=float, default=0.7,
                    help="Party's minimum acceptable GOTV precision (share of effort "
                         "that hits a real target); another lower bound of the "
                         "feasible-ε region on the combined plot (default: %(default)s; "
                         "set <=0 to disable).")
    ap.add_argument("--max-loss", type=float, default=None,
                    help="Party's maximum tolerated DOLLAR LOSS from privacy-induced "
                         "misdirected calls (not the total calling budget); upper "
                         "bound of the feasible-ε region. Default: auto — halfway "
                         "between the cheapest and dearest swept ε, so the limit "
                         "always bisects the loss curve and the region stays "
                         "illustrative at any population scale. Pass a value to fix "
                         "it, or <=0 to disable.")
    ap.add_argument("--national", action="store_true",
                    help="Use ALL settlements in expc.csv (~6.8M voters) instead of "
                         "the default 12-settlement subset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(_EXP_DIR / "output"))
    args = ap.parse_args()

    # Validate the activist knobs up front (before the expensive population synth).
    if args.frac_activists is not None:
        if args.activists != ap.get_default("activists"):
            ap.error("--frac-activists and --activists are mutually exclusive.")
        if not (0.0 < args.frac_activists <= 1.0):
            ap.error("--frac-activists must be in (0, 1].")
        if args.budget is None:
            ap.error("--frac-activists requires --budget (calls per activist); "
                     "without it the chase capacity is unlimited and the fraction "
                     "has no effect.")

    support_model = ("observed" if args.support_noise is None
                     else population.make_noisy_support_model(args.support_noise))

    settlements = None
    if args.national:
        table = population.load_settlements_table()
        settlements = table[population.COL_NAME].tolist()

    print("Building population...")
    pop = population.synthesize_population(
        settlements=settlements,
        fraction=args.fraction, keep_top_parties=args.keep_top_parties,
        support_model=support_model, seed=args.seed,
    )
    candidates = population.party_candidates(pop)
    organiser = args.organiser if args.organiser in candidates else candidates[0]
    organiser_en = en_party(organiser)   # English label for plots/filenames (no RTL)
    if organiser != args.organiser:
        print(f"  '{args.organiser}' not in population; using '{organiser}' instead.")
    print(f"  voters={len(pop):,}  organiser={organiser} ({organiser_en})  "
          f"k={len(candidates)}  turnout={pop['true_voted'].mean():.1%}")

    # Number of activists: either a FIXED count (--activists, same for all parties)
    # or a FRACTION of the organiser's supporter base (--frac-activists, scales with
    # party size). Effective chase capacity = n_activists × per-activist calls.
    n_supporters = int((pop["latent_support"] == organiser).sum())
    if args.frac_activists is not None:
        n_activists = max(1, round(args.frac_activists * n_supporters))
        print(f"  activists = {args.frac_activists:g} × {n_supporters:,} supporters "
              f"= {n_activists:,}")
    else:
        n_activists = args.activists

    effective_budget = (None if args.budget is None
                        else args.budget * n_activists)
    if effective_budget is not None:
        print(f"  chase capacity = {n_activists:,} activists × {args.budget} calls "
              f"= {effective_budget:,} contacts")

    print(f"Running sweep ({args.repeats} repeats × {len(DEFAULT_EPSILONS)} ε values)...")
    raw = run_sweep(pop, organiser, candidates, DEFAULT_EPSILONS,
                    repeats=args.repeats, budget=effective_budget,
                    protect_party=args.protect_party, seed=args.seed + 1)
    agg = aggregate(raw)
    # Price the privacy-attributable misdirected calls (votes stay in voter units).
    agg["cost_misdirected_calls"] = agg["wasted_due_to_privacy"] * args.cost_per_call

    cols = ["epsilon", "odds_bound", "status_baseline", "status_attacker_acc",
            "status_disclosure_advantage", "turnout_estimate_accuracy",
            "party_share_estimate_accuracy", "recall_mean", "precision_mean",
            "missed_due_to_privacy", "wasted_due_to_privacy"]
    print("\nPrivacy–utility vs. ε  (status_disclosure_advantage = info the leaked record adds):")
    print(agg[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{organiser_en}_national" if args.national else organiser_en
    if args.fraction != 1.0:
        tag += f"_f{args.fraction:g}"
    if args.frac_activists is not None:
        tag += f"_fa{args.frac_activists:g}"
    if effective_budget:
        tag += f"_b{effective_budget}"
    # Constraint bounds for the feasible-ε region (<=0 disables a bound).
    min_recall = args.min_recall if args.min_recall and args.min_recall > 0 else None
    min_precision = args.min_precision if args.min_precision and args.min_precision > 0 else None
    if args.max_loss is None:
        # Auto loss limit: midway between the cheapest and dearest swept ε, so it
        # bisects the loss curve and stays illustrative at any population scale.
        fc = agg.loc[np.isfinite(agg["epsilon"]), "cost_misdirected_calls"]
        max_loss = _round_sig(0.5 * (float(fc.min()) + float(fc.max()))) if len(fc) else None
    elif args.max_loss > 0:
        max_loss = args.max_loss
    else:
        max_loss = None
    if max_loss is not None:
        print(f"  max loss (privacy-induced misdirected-call cost) = ${max_loss:,.0f}")
    csv_path = outdir / f"privacy_utility_{tag}.csv"
    agg.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_attack(agg, outdir / f"privacy_utility_{tag}_attack.png")
    plot_gotv(agg, outdir / f"privacy_utility_{tag}_gotv.png", organiser_en)
    plot_privacy_utility(agg, outdir / f"privacy_utility_{tag}_money.png",
                         cost_per_call=args.cost_per_call, error_kind="sampling")
    # SEPARATE-influence plots (sampling error — the visible, one-campaign band):
    plot_utility(agg, outdir / f"privacy_utility_{tag}_utility.png", organiser_en,
                 min_recall=min_recall, error_kind="sampling")
    plot_price_of_privacy(agg, outdir / f"privacy_utility_{tag}_cost.png",
                          cost_per_call=args.cost_per_call, max_loss=max_loss,
                          error_kind="sampling")
    # COMBINED in BOTH error styles:
    plot_combined(agg, outdir / f"privacy_utility_{tag}_combined_ci.png", organiser_en,
                  cost_per_call=args.cost_per_call, min_recall=min_recall,
                  min_precision=min_precision, max_loss=max_loss, error_kind="ci")
    plot_combined(agg, outdir / f"privacy_utility_{tag}_combined_rrerr.png", organiser_en,
                  cost_per_call=args.cost_per_call, min_recall=min_recall,
                  min_precision=min_precision, max_loss=max_loss, error_kind="sampling")
    print(f"\nSaved:\n  {csv_path}\n"
          f"  {outdir / f'privacy_utility_{tag}_attack.png'}\n"
          f"  {outdir / f'privacy_utility_{tag}_gotv.png'}\n"
          f"  {outdir / f'privacy_utility_{tag}_money.png'}\n"
          f"  {outdir / f'privacy_utility_{tag}_utility.png'}\n"
          f"  {outdir / f'privacy_utility_{tag}_cost.png'}\n"
          f"  {outdir / f'privacy_utility_{tag}_combined_ci.png'}\n"
          f"  {outdir / f'privacy_utility_{tag}_combined_rrerr.png'}")


if __name__ == "__main__":
    main()
