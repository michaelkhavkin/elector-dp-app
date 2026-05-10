"""
voting_dp.py
============
Differential Privacy (DP) mechanisms for privacy-preserving voting analytics.

All mechanisms use Local Differential Privacy (LDP), meaning each individual
perturbs their own data before it leaves their device / before it is recorded.

Mechanisms implemented
----------------------
Generic layer:
  - randomized_response        : Binary RR  (ε-LDP, true/false values)
  - k_randomized_response      : k-RR       (ε-LDP, k categorical values)
  - laplace_mechanism          : Laplace    (ε-DP,  numerical aggregates)
  - gaussian_mechanism         : Gaussian   (ε,δ-DP, numerical aggregates)
  - estimate_krr_frequency     : Unbiased frequency estimator for k-RR outputs
  - estimate_rr_frequency      : Unbiased frequency estimator for binary RR outputs
  - laplace_margin_of_error      : 95 % margin of error for a Laplace-noised count
  - laplace_confidence_interval  : 95 % CI for a Laplace-noised count (noisy_value ± margin)
  - gaussian_margin_of_error     : 95 % margin of error for a Gaussian-noised count
  - gaussian_confidence_interval : 95 % CI for a Gaussian-noised count (noisy_value ± margin)

Application layer:
  - protect_party_vote              : (1) Protect which party a voter chose
  - protect_voted_status            : (2) Protect whether a person voted
  - protect_city_vote_counts        : (3a) Protect aggregate voter counts per city
  - protect_city_party_counts       : (3b) Protect per-party vote counts per city
  - protect_potential_voter_status  : (4) Protect whether someone is a potential voter

Usage example (Jupyter)
-----------------------
    from voting_dp import protect_party_vote, protect_voted_status, ...
"""

import numpy as np
import opendp.prelude as dp
from opendp.accuracy import laplacian_scale_to_accuracy, gaussian_scale_to_accuracy

dp.enable_features("contrib")


# =============================================================================
# GENERIC / PRIMITIVE DP MECHANISMS
# =============================================================================

def randomized_response(true_value, epsilon):
    """
    Binary Randomized Response (ε-LDP).

    Each user reports their true binary value with probability p, and flips
    it with probability 1-p.  Satisfies ε-Local Differential Privacy.

        p = e^ε / (1 + e^ε)

    Parameters
    ----------
    true_value : bool
        The real answer (True / False).
    epsilon : float
        Privacy budget (ε > 0). Larger ε → less noise → less privacy.

    Returns
    -------
    bool
        Perturbed (reported) value.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    p_true = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    m = dp.m.make_randomized_response_bool(prob=p_true)
    return m(bool(true_value))


def k_randomized_response(true_value, candidates, epsilon):
    """
    k-Randomized Response (ε-LDP) for a categorical variable with k options.

    With probability p the true value is reported; with probability 1-p a
    value is drawn uniformly at random from the other k-1 candidates.

        p = e^ε / (e^ε + k - 1)

    Parameters
    ----------
    true_value : any
        The real answer; must be an element of `candidates`.
    candidates : list
        All k possible values of the categorical variable.
    epsilon : float
        Privacy budget (ε > 0).

    Returns
    -------
    any
        Perturbed (reported) value, drawn from `candidates`.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    k = len(candidates)
    if k < 2:
        raise ValueError("candidates must contain at least 2 elements.")
    if true_value not in candidates:
        raise ValueError("true_value must be one of the candidates.")

    p_true = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    m = dp.m.make_randomized_response(categories=candidates, prob=p_true)
    return m(true_value)


def laplace_mechanism(true_value, sensitivity, epsilon):
    """
    Laplace Mechanism (ε-DP) for a numerical query.

    Adds Laplace-distributed noise calibrated to global sensitivity / ε.

        Noise ~ Laplace(0, sensitivity / ε)

    Parameters
    ----------
    true_value : float | int
        The true numerical answer.
    sensitivity : float
        Global sensitivity of the query (Δf).
        For a simple count query, sensitivity = 1.
    epsilon : float
        Privacy budget (ε > 0).

    Returns
    -------
    float
        Noisy numerical answer.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive.")

    scale = sensitivity / epsilon
    # Use integer domain to avoid the NaN-exclusion requirement on float domains.
    # Count queries are always integers; noise is still drawn from Laplace(0, scale).
    m = dp.m.make_laplace(dp.atom_domain(T=int), dp.absolute_distance(T=int), scale)
    return float(m(int(round(true_value))))


def gaussian_mechanism(true_value, sensitivity, epsilon, delta):
    """
    Gaussian Mechanism (ε,δ-DP) for a numerical query.

    Adds Gaussian noise with std σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε.

    Note: OpenDP's make_gaussian uses zero-concentrated DP (zCDP) rather than
    (ε,δ)-DP, so we retain the standard (ε,δ) calibration formula here and
    draw noise directly with numpy. The resulting noise magnitude is identical;
    only the formal privacy-accounting semantics differ.

    Parameters
    ----------
    true_value : float | int
        The true numerical answer.
    sensitivity : float
        Global L2-sensitivity of the query.
        For a simple count query, sensitivity = 1.
    epsilon : float
        Privacy budget (ε ∈ (0, 1]).
    delta : float
        Failure probability (δ ∈ (0, 1)).  Typical value: 1e-5.

    Returns
    -------
    float
        Noisy numerical answer.
    """
    if epsilon <= 0 or epsilon > 1:
        raise ValueError("epsilon must be in (0, 1].")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1).")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive.")

    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(loc=0.0, scale=sigma)
    return float(true_value) + noise


def double_randomized_response(true_value, epsilon1, epsilon2):
    """
    Double Binary Randomized Response (sequential composition).

    Applies binary RR twice in sequence.  Used when a boolean value needs
    two independent layers of privacy protection — for example, first
    protecting whether a person voted, then protecting which party they
    voted for, treating each dimension as a separate binary question.

    Layer 1:  r1 = RR(true_value,  epsilon1)
    Layer 2:  r2 = RR(r1,          epsilon2)

    By sequential composition the total privacy budget is epsilon1 + epsilon2.

    The effective probability of the final report being truthful is:
        p_eff = p1 * p2 + (1 - p1) * (1 - p2)
    where p1 = e^ε1 / (1+e^ε1) and p2 = e^ε2 / (1+e^ε2).

    Parameters
    ----------
    true_value : bool
    epsilon1   : float  Privacy budget for the first RR layer.
    epsilon2   : float  Privacy budget for the second RR layer.

    Returns
    -------
    bool   Twice-perturbed reported value.
    """
    r1 = randomized_response(true_value, epsilon1)
    r2 = randomized_response(r1,         epsilon2)
    return r2


def effective_p_double_rr(epsilon1, epsilon2):
    """
    Return the effective truth-keeping probability for double binary RR.

        p_eff = p1 * p2 + (1 - p1) * (1 - p2)

    Parameters
    ----------
    epsilon1, epsilon2 : float

    Returns
    -------
    float
    """
    p1 = np.exp(epsilon1) / (1.0 + np.exp(epsilon1))
    p2 = np.exp(epsilon2) / (1.0 + np.exp(epsilon2))
    return p1 * p2 + (1.0 - p1) * (1.0 - p2)


# =============================================================================
# UNBIASED ESTIMATORS (for aggregated / analysis use)
# =============================================================================

def estimate_rr_frequency(reported_values, epsilon):
    """
    Unbiased frequency estimator for binary Randomized Response outputs.

    Corrects for the deliberate noise introduced by `randomized_response` to
    recover an unbiased estimate of the true proportion of True values.

        p̂_true = (p̂_reported - (1-p)) / (2p - 1)

    where p = e^ε / (1 + e^ε).

    Parameters
    ----------
    reported_values : array-like of bool
        Collection of perturbed values from `randomized_response`.
    epsilon : float
        The ε used when the values were perturbed.

    Returns
    -------
    float
        Estimated true proportion of True in [0, 1] (clamped).
    """
    p = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    reported = np.array(reported_values, dtype=float)
    p_reported = reported.mean()
    estimate = (p_reported - (1.0 - p)) / (2.0 * p - 1.0)
    return float(np.clip(estimate, 0.0, 1.0))


def estimate_double_rr_frequency(reported_values, epsilon1, epsilon2):
    """
    Unbiased frequency estimator for double binary Randomized Response outputs.

    Mirrors `estimate_rr_frequency` but accounts for two sequential RR layers.
    The effective flip probability is derived from p_eff, and the same
    de-biasing formula applies:

        est = (p_reported - (1 - p_eff)) / (2 * p_eff - 1)

    Parameters
    ----------
    reported_values : array-like of bool
        Twice-perturbed values from `double_randomized_response`.
    epsilon1, epsilon2 : float
        The ε values used for layer 1 and layer 2 respectively.

    Returns
    -------
    float
        Estimated true proportion of True, clamped to [0, 1].
    """
    p_eff      = effective_p_double_rr(epsilon1, epsilon2)
    reported   = np.array(reported_values, dtype=float)
    p_reported = reported.mean()
    estimate   = (p_reported - (1.0 - p_eff)) / (2.0 * p_eff - 1.0)
    return float(np.clip(estimate, 0.0, 1.0))


def estimate_krr_frequency(reported_values, candidates, epsilon):
    """
    Unbiased frequency estimator for k-Randomized Response outputs.

    For each candidate c, the unbiased estimate of P(true = c) is:

        p̂_c = (p̂_reported_c - 1/k) * k / (k * p_true - 1)     [equivalent form]
             = (p̂_reported_c * (k - 1) + p_true - 1) / ...

    Simplified closed form per candidate:

        p̂_c = ((k-1)*p̂_reported_c - (1 - p_true)) / (k * p_true - 1)

    where p_true = e^ε / (e^ε + k - 1).

    Parameters
    ----------
    reported_values : array-like
        Collection of perturbed values from `k_randomized_response`.
    candidates : list
        All k possible values (same list used during perturbation).
    epsilon : float
        The ε used when the values were perturbed.

    Returns
    -------
    dict
        {candidate: estimated_true_frequency}, frequencies sum to ≈ 1.
    """
    k = len(candidates)
    p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
    reported = np.array(reported_values)
    n = len(reported)

    estimates = {}
    for c in candidates:
        p_reported_c = np.sum(reported == c) / n
        # Correct closed form derived from inverting the k-RR channel:
        #   q_c = f_c * p + (1 - f_c) * (1-p)/(k-1)
        # Solving for f_c gives:
        #   f_c = [(k-1)*q_c - (1-p)] / [k*p - 1]
        estimate = ((k - 1) * p_reported_c - (1.0 - p)) / (k * p - 1.0)
        estimates[c] = float(estimate)   # keep raw (may be slightly negative)

    # Project onto probability simplex: clip negatives then re-normalise
    for c in candidates:
        estimates[c] = max(0.0, estimates[c])
    total = sum(estimates.values())
    if total > 0:
        estimates = {c: v / total for c, v in estimates.items()}

    return estimates


# =============================================================================
# APPLICATION LAYER
# =============================================================================

# --- (1) Protect which party a voter voted for ---

def protect_party_vote(true_party, all_parties, epsilon):
    """
    Protect the party a single voter voted for using k-Randomized Response.

    This is the primary mechanism to hide individual party preferences while
    still allowing aggregate statistics to be estimated at the population level.

    Parameters
    ----------
    true_party : any
        The party the voter actually voted for (must be in `all_parties`).
    all_parties : list
        All k parties on the ballot.
    epsilon : float
        Privacy budget.

    Returns
    -------
    any
        A (possibly perturbed) reported party name.
    """
    return k_randomized_response(true_party, all_parties, epsilon)


def protect_party_votes_batch(true_parties, all_parties, epsilon):
    """
    Apply `protect_party_vote` to an array of voters.

    Parameters
    ----------
    true_parties : array-like
        Sequence of true party choices, one per voter.
    all_parties : list
        All k parties on the ballot.
    epsilon : float
        Privacy budget.

    Returns
    -------
    list
        Perturbed party choices, one per voter.
    """
    return [protect_party_vote(p, all_parties, epsilon) for p in true_parties]


# --- (2) Protect whether a voter has voted (tick-off from activist list) ---

def protect_voted_status(has_voted, epsilon):
    """
    Protect whether a single person has voted using binary Randomized Response.

    Used by party activists when ticking off voters from their list.  The
    reported status is noisy, preventing the activist from building a perfectly
    accurate "who voted" record.

    Parameters
    ----------
    has_voted : bool
        Whether the person has actually voted.
    epsilon : float
        Privacy budget.

    Returns
    -------
    bool
        Reported (possibly perturbed) voting status.
    """
    return randomized_response(has_voted, epsilon)


def protect_voted_status_batch(has_voted_array, epsilon):
    """
    Apply `protect_voted_status` to an array of voters.

    Parameters
    ----------
    has_voted_array : array-like of bool
        Sequence of true voted statuses.
    epsilon : float
        Privacy budget.

    Returns
    -------
    list of bool
        Perturbed voted statuses.
    """
    return [protect_voted_status(v, epsilon) for v in has_voted_array]


# --- (3a) Protect aggregate voter counts per city ---

def protect_city_vote_counts(city_counts, epsilon, sensitivity=1.0):
    """
    Protect the total number of voters per city using the Laplace Mechanism.

    A dictionary of {city: true_count} is transformed into a dictionary of
    {city: noisy_count}.  Counts are rounded to the nearest integer and
    clipped to be non-negative.

    Parameters
    ----------
    city_counts : dict
        {city_name: true_voter_count}
    epsilon : float
        Total privacy budget.  This is split equally across all cities
        (parallel composition: disjoint datasets → same ε per city is fine
        under parallel composition).
    sensitivity : float
        Global sensitivity per city count.  Defaults to 1 (one person
        can change a city count by at most 1).

    Returns
    -------
    dict
        {city_name: noisy_voter_count}  (non-negative integers)
    """
    noisy = {}
    for city, count in city_counts.items():
        noisy_count = laplace_mechanism(count, sensitivity, epsilon)
        noisy[city] = max(0, int(round(noisy_count)))
    return noisy


# --- (3b) Protect per-party counts per city ---

def protect_city_party_counts(city_party_counts, epsilon, sensitivity=1.0):
    """
    Protect the number of votes each party received per city (Laplace Mechanism).

    Parameters
    ----------
    city_party_counts : dict of dict
        {city_name: {party_name: true_count}}
    epsilon : float
        Privacy budget per (city, party) cell.  Under parallel composition
        (disjoint city datasets), the same ε is used for every city.
    sensitivity : float
        Global sensitivity per cell.  Defaults to 1.

    Returns
    -------
    dict of dict
        {city_name: {party_name: noisy_count}}  (non-negative integers)
    """
    noisy = {}
    for city, party_counts in city_party_counts.items():
        noisy[city] = {}
        for party, count in party_counts.items():
            noisy_count = laplace_mechanism(count, sensitivity, epsilon)
            noisy[city][party] = max(0, int(round(noisy_count)))
    return noisy


# --- (4) Protect potential voter eligibility flag ---

def protect_potential_voter_status(is_potential_voter, epsilon):
    """
    Protect whether a person is flagged as a potential voter (binary RR).

    Prevents precise lists of "who is a potential voter" from being perfectly
    accurate, adding plausible deniability for each individual.

    Parameters
    ----------
    is_potential_voter : bool
        Whether the person is genuinely flagged as a potential voter.
    epsilon : float
        Privacy budget.

    Returns
    -------
    bool
        Reported (possibly perturbed) potential-voter status.
    """
    return randomized_response(is_potential_voter, epsilon)


def protect_potential_voter_status_batch(is_potential_array, epsilon):
    """
    Apply `protect_potential_voter_status` to an array of people.

    Parameters
    ----------
    is_potential_array : array-like of bool
        Sequence of true potential-voter flags.
    epsilon : float
        Privacy budget.

    Returns
    -------
    list of bool
        Perturbed potential-voter flags.
    """
    return [protect_potential_voter_status(v, epsilon) for v in is_potential_array]


# =============================================================================
# CONVENIENCE: PRIVACY BUDGET REFERENCE
# =============================================================================

def epsilon_to_flip_probability(epsilon):
    """
    Return the probability that a binary RR response is *wrong* (i.e. flipped).

    Useful for explaining the privacy-utility tradeoff intuitively:

        P(flip) = 1 / (1 + e^ε)

    Parameters
    ----------
    epsilon : float

    Returns
    -------
    float
        Probability that the reported value differs from the true value.
    """
    return 1.0 / (1.0 + np.exp(epsilon))


def epsilon_to_krr_noise_probability(epsilon, k):
    """
    Return the probability that a k-RR response is *wrong* (i.e. randomised).

        P(report ≠ truth) = (k - 1) / (e^ε + k - 1)

    Parameters
    ----------
    epsilon : float
    k : int
        Number of categories.

    Returns
    -------
    float
        Probability that the reported value differs from the true value.
    """
    return (k - 1) / (np.exp(epsilon) + k - 1)


# =============================================================================
# MARGIN OF ERROR AND CONFIDENCE INTERVALS FOR NUMERICAL DP MECHANISMS
# =============================================================================

def laplace_margin_of_error(sensitivity, epsilon, confidence=0.95, n=1):
    """
    Margin of error for a Laplace-noised count query.

    The noisy value produced by `laplace_mechanism` is an unbiased estimate
    of the true count.  The true value falls within ±margin of the noisy
    value with the given confidence level:

        true_value ∈ (noisy_value − margin, noisy_value + margin)
        with probability = confidence.

    For a single measurement (n=1) the exact Laplace quantile is used via
    opendp.accuracy.laplacian_scale_to_accuracy.  Averaging n independent
    noisy counts shrinks the margin by √n; for n>1 the CLT approximation is
    applied via opendp.accuracy.gaussian_scale_to_accuracy.

        b = sensitivity / epsilon          (Laplace scale)

        n=1  →  margin = b · ln(1/α)          [exact, α = 1 − confidence]
        n>1  →  margin = z_(α/2) · b · √(2/n) [CLT]

    Parameters
    ----------
    sensitivity : float
        Global sensitivity (Δf). For a count query, use 1.
    epsilon : float
        Privacy budget ε > 0.
    confidence : float
        Confidence level (default 0.95 for a 95% margin).
    n : int
        Number of independent noisy measurements averaged (default 1).

    Returns
    -------
    float
        Margin of error m.  The true value lies in
        (noisy_value − m, noisy_value + m) with probability `confidence`.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")
    if n < 1:
        raise ValueError("n must be at least 1.")

    b = sensitivity / epsilon
    alpha = 1.0 - confidence
    if n == 1:
        return float(laplacian_scale_to_accuracy(scale=b, alpha=alpha))
    return float(gaussian_scale_to_accuracy(scale=b * np.sqrt(2.0 / n), alpha=alpha))


def laplace_confidence_interval(noisy_value, sensitivity, epsilon,
                                 confidence=0.95, n=1):
    """
    Confidence interval for the true value underlying a Laplace-noised query.

    Returns (noisy_value − margin, noisy_value + margin) where `margin` is
    computed by `laplace_margin_of_error`.

    Parameters
    ----------
    noisy_value : float
        The DP-protected value from `laplace_mechanism`, or the arithmetic
        mean of `n` such independent outputs.
    sensitivity : float
        Global sensitivity (Δf). For a count query, use 1.
    epsilon : float
        Privacy budget ε > 0.
    confidence : float
        Desired confidence level (default 0.95).
    n : int
        Number of independent noisy measurements averaged (default 1).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) confidence interval for the true value.
    """
    margin = laplace_margin_of_error(sensitivity, epsilon, confidence, n)
    return (noisy_value - margin, noisy_value + margin)


def gaussian_margin_of_error(sensitivity, epsilon, delta, confidence=0.95, n=1):
    """
    Margin of error for a Gaussian-noised count query.

    The true value falls within ±margin of the noisy value with the given
    confidence level.  Because the average of n Gaussians is Gaussian, this
    is exact for any n ≥ 1 (no CLT approximation needed):

        σ = sensitivity · √(2 · ln(1.25/δ)) / ε
        margin = z_(α/2) · σ / √n

    Uses opendp.accuracy.gaussian_scale_to_accuracy.

    Parameters
    ----------
    sensitivity : float
        Global L2-sensitivity. For a count query, use 1.
    epsilon : float
        Privacy budget ε ∈ (0, 1].
    delta : float
        Failure probability δ ∈ (0, 1). Typical value: 1e-5.
    confidence : float
        Confidence level (default 0.95).
    n : int
        Number of independent noisy measurements averaged (default 1).

    Returns
    -------
    float
        Margin of error m.
    """
    if epsilon <= 0 or epsilon > 1:
        raise ValueError("epsilon must be in (0, 1].")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1).")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1).")
    if n < 1:
        raise ValueError("n must be at least 1.")

    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
    return float(gaussian_scale_to_accuracy(scale=sigma / np.sqrt(n),
                                             alpha=1.0 - confidence))


def gaussian_confidence_interval(noisy_value, sensitivity, epsilon, delta,
                                  confidence=0.95, n=1):
    """
    Confidence interval for the true value underlying a Gaussian-noised query.

    Returns (noisy_value − margin, noisy_value + margin) where `margin` is
    computed by `gaussian_margin_of_error`.

    Parameters
    ----------
    noisy_value : float
        The DP-protected value from `gaussian_mechanism`, or the arithmetic
        mean of `n` such independent outputs.
    sensitivity : float
        Global L2-sensitivity. For a count query, use 1.
    epsilon : float
        Privacy budget ε ∈ (0, 1].
    delta : float
        Failure probability δ ∈ (0, 1). Typical value: 1e-5.
    confidence : float
        Desired confidence level (default 0.95).
    n : int
        Number of independent noisy measurements averaged (default 1).

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) confidence interval for the true value.
    """
    margin = gaussian_margin_of_error(sensitivity, epsilon, delta, confidence, n)
    return (noisy_value - margin, noisy_value + margin)


# =============================================================================
# MAIN — USAGE EXAMPLES
# =============================================================================

def main():
    """Demonstrate each mechanism with realistic voting-analytics scenarios."""
    np.random.seed(42)
    W = 60  # column width for section headers

    # =========================================================================
    print("=" * W)
    print("  1. BINARY RANDOMIZED RESPONSE  (voted / did not vote)")
    print("=" * W)
    # =========================================================================
    # A party activist ticks off voters.  Each tick is noised so that no
    # individual's real status can be determined with certainty.
    true_voted = True
    eps = 1.0
    flip_p = epsilon_to_flip_probability(eps)
    print(f"  True status : {true_voted}  |  ε={eps}  |  P(flip)={flip_p:.1%}")
    print(f"  5 independent reports from the same person:")
    for i in range(1, 6):
        reported = randomized_response(true_voted, epsilon=eps)
        tag = "correct" if reported == true_voted else "FLIPPED"
        print(f"    Run {i}: {str(reported):<5}  ({tag})")

    # =========================================================================
    print()
    print("=" * W)
    print("  2. k-RANDOMIZED RESPONSE  (party preference)")
    print("=" * W)
    # =========================================================================
    parties = ["Labour", "Conservative", "LibDem", "Green"]
    true_party = "Labour"
    eps = 1.5
    wrong_p = epsilon_to_krr_noise_probability(eps, k=len(parties))
    print(f"  True party : {true_party}  |  k={len(parties)}  |"
          f"  ε={eps}  |  P(randomised)={wrong_p:.1%}")
    print(f"  5 independent reports from the same voter:")
    for i in range(1, 6):
        reported = k_randomized_response(true_party, parties, epsilon=eps)
        tag = "correct" if reported == true_party else "randomised"
        print(f"    Run {i}: {reported:<14}  ({tag})")

    # =========================================================================
    print()
    print("=" * W)
    print("  3. LAPLACE MECHANISM  (voter count for a city)")
    print("=" * W)
    # =========================================================================
    true_count = 500
    sensitivity = 1   # adding/removing one person changes the count by 1
    eps = 0.5

    noisy = laplace_mechanism(true_count, sensitivity=sensitivity, epsilon=eps)
    margin = laplace_margin_of_error(sensitivity=sensitivity, epsilon=eps)
    lo, hi = laplace_confidence_interval(noisy, sensitivity=sensitivity, epsilon=eps)

    print(f"  True count        : {true_count}")
    print(f"  ε = {eps}  →  Laplace scale b = sensitivity/ε = {sensitivity/eps:.1f}")
    print(f"  Noisy count       : {noisy:.1f}")
    print(f"  Margin of error   : ±{margin:.1f}  (95 % confidence)")
    print(f"  95 % CI           : ({lo:.1f},  {hi:.1f})")
    print()
    print("  Averaging daily counts over n days shrinks the margin by √n:")
    print(f"  {'n':>4}   {'margin of error':>18}   {'95 % CI width':>15}")
    print(f"  {'-'*42}")
    for n in [1, 5, 10, 30]:
        m_n = laplace_margin_of_error(sensitivity=sensitivity, epsilon=eps, n=n)
        print(f"  {n:>4}   ±{m_n:>16.2f}   {2*m_n:>14.2f}")

    # =========================================================================
    print()
    print("=" * W)
    print("  4. GAUSSIAN MECHANISM  (voter count for a city)")
    print("=" * W)
    # =========================================================================
    true_count = 500
    sensitivity = 1
    eps = 0.5
    delta = 1e-5
    sigma = sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / eps

    noisy_g = gaussian_mechanism(true_count, sensitivity=sensitivity,
                                  epsilon=eps, delta=delta)
    margin_g = gaussian_margin_of_error(sensitivity=sensitivity,
                                         epsilon=eps, delta=delta)
    lo_g, hi_g = gaussian_confidence_interval(noisy_g, sensitivity=sensitivity,
                                               epsilon=eps, delta=delta)

    print(f"  True count        : {true_count}")
    print(f"  ε={eps}, δ={delta}  →  σ = {sigma:.2f}")
    print(f"  Noisy count       : {noisy_g:.1f}")
    print(f"  Margin of error   : ±{margin_g:.1f}  (95 % confidence)")
    print(f"  95 % CI           : ({lo_g:.1f},  {hi_g:.1f})")
    print()
    print("  Averaging over n days (Gaussian CI shrinks exactly by √n):")
    print(f"  {'n':>4}   {'margin of error':>18}   {'95 % CI width':>15}")
    print(f"  {'-'*42}")
    for n in [1, 5, 10, 30]:
        m_n = gaussian_margin_of_error(sensitivity=sensitivity,
                                        epsilon=eps, delta=delta, n=n)
        print(f"  {n:>4}   ±{m_n:>16.2f}   {2*m_n:>14.2f}")


if __name__ == "__main__":
    main()
