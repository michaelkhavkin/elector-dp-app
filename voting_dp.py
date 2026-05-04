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
    return bool(true_value) if np.random.random() < p_true else not bool(true_value)


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

    if np.random.random() < p_true:
        return true_value

    other = [c for c in candidates if c != true_value]
    return np.random.choice(other)


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
    noise = np.random.laplace(loc=0.0, scale=scale)
    return float(true_value) + noise


def gaussian_mechanism(true_value, sensitivity, epsilon, delta):
    """
    Gaussian Mechanism (ε,δ-DP) for a numerical query.

    Adds Gaussian noise with std σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε.

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
# MAIN — SIMULATION & USAGE EXAMPLES
# =============================================================================

def _section(title):
    width = 68
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _subsection(title):
    print(f"\n  --- {title} ---")


def main():
    """
    End-to-end simulation that demonstrates every function in this module.

    Simulates a small election with:
      - 1 000 voters
      - 4 parties  : Alpha, Beta, Gamma, Delta
      - 5 cities   : Springfield, Shelbyville, Ogdenville, Capital City, Brockway
      - Two ε values are compared throughout: a tight budget (ε=0.5, more
        private) and a loose budget (ε=3.0, more accurate).

    Run with:  python voting_dp.py
    """
    np.random.seed(42)

    N_VOTERS   = 1_000
    ALL_PARTIES = ["Alpha", "Beta", "Gamma", "Delta"]
    CITIES      = ["Springfield", "Shelbyville", "Ogdenville", "Capital City", "Brockway"]
    EPS_TIGHT   = 0.5   # strong privacy, more noise
    EPS_LOOSE   = 3.0   # weaker privacy, less noise

    # ------------------------------------------------------------------
    # Simulate ground-truth population
    # ------------------------------------------------------------------
    # True party preferences (Alpha is most popular)
    true_party_probs = [0.40, 0.30, 0.20, 0.10]
    true_parties = np.random.choice(ALL_PARTIES, size=N_VOTERS, p=true_party_probs)

    # True voted status (70 % of voters actually voted)
    true_voted = np.random.rand(N_VOTERS) < 0.70

    # True potential-voter flags (80 % are flagged as potential voters)
    true_potential = np.random.rand(N_VOTERS) < 0.80

    # Assign voters to cities (roughly equal split)
    voter_cities = np.random.choice(CITIES, size=N_VOTERS)

    # Build ground-truth city totals and per-city party counts
    true_city_counts = {c: int(np.sum(voter_cities == c)) for c in CITIES}

    true_city_party_counts = {}
    for city in CITIES:
        mask = voter_cities == city
        true_city_party_counts[city] = {
            party: int(np.sum(true_parties[mask] == party))
            for party in ALL_PARTIES
        }

    # ================================================================
    _section("EXAMPLE 1 — protect_party_vote  (k-Randomized Response)")
    # ================================================================
    print("""
  Scenario: Each voter's party choice is locally randomised before being
  recorded.  The aggregate estimated frequencies should still track the
  true distribution closely, especially at higher ε.
    """)

    for eps in [EPS_TIGHT, EPS_LOOSE]:
        _subsection(f"ε = {eps}  |  k = {len(ALL_PARTIES)}  "
                    f"|  P(report≠truth) = "
                    f"{epsilon_to_krr_noise_probability(eps, len(ALL_PARTIES)):.1%}")

        reported = protect_party_votes_batch(true_parties, ALL_PARTIES, eps)
        estimated = estimate_krr_frequency(reported, ALL_PARTIES, eps)
        true_freqs = {p: float(np.mean(true_parties == p)) for p in ALL_PARTIES}

        print(f"  {'Party':<12} {'True %':>8} {'Reported %':>12} {'Estimated %':>13}")
        print(f"  {'-'*47}")
        for party in ALL_PARTIES:
            raw_reported = float(np.mean(np.array(reported) == party))
            print(f"  {party:<12} {true_freqs[party]:>7.1%} "
                  f"{raw_reported:>11.1%}  {estimated[party]:>11.1%}")

    print("""
  Takeaway:
    The raw reported % is pulled toward uniformity (1/k = 25 %) by the noise.
    The estimator corrects for this bias, recovering the true distribution.
    At ε=3.0 the estimate is tight; at ε=0.5 more noise remains but the
    ranking of parties is still recoverable with enough voters.
    """)

    # ================================================================
    _section("EXAMPLE 2 — protect_voted_status  (Binary Randomized Response)")
    # ================================================================
    print("""
  Scenario: Party activists tick voters off their list after confirming they
  voted.  The tick is noisy: the true status is flipped with probability
  1/(1+e^ε), giving each voter plausible deniability.
    """)

    true_voted_rate = float(np.mean(true_voted))
    print(f"  True voted rate: {true_voted_rate:.1%}\n")

    for eps in [EPS_TIGHT, EPS_LOOSE]:
        flip_p = epsilon_to_flip_probability(eps)
        reported_voted = protect_voted_status_batch(true_voted, eps)
        raw_reported_rate   = float(np.mean(reported_voted))
        estimated_rate      = estimate_rr_frequency(reported_voted, eps)

        print(f"  ε = {eps}  |  P(flip) = {flip_p:.1%}")
        print(f"    Raw reported voted rate : {raw_reported_rate:.1%}")
        print(f"    De-biased estimate      : {estimated_rate:.1%}  "
              f"(true = {true_voted_rate:.1%})")
        print()

    print("""
  Takeaway:
    At ε=0.5 nearly one in three reports are flipped, making it hard for
    the activist to know who really voted.  At ε=3.0 only ~5 % are flipped
    yet the true rate is still recoverable from the population statistics.
    """)

    # ================================================================
    _section("EXAMPLE 3a — protect_city_vote_counts  (Laplace Mechanism)")
    # ================================================================
    print("""
  Scenario: Publishing total voter turnout per city.  Laplace noise is added
  to each city count before release.  Under parallel composition each city's
  dataset is disjoint, so the same ε applies to every city independently.
    """)

    print(f"  {'City':<16} {'True count':>11}", end="")
    for eps in [EPS_TIGHT, EPS_LOOSE]:
        print(f"  {'Noisy (ε='+str(eps)+')':>14}", end="")
    print()
    print(f"  {'-'*60}")

    noisy_tight = protect_city_vote_counts(true_city_counts, EPS_TIGHT)
    noisy_loose = protect_city_vote_counts(true_city_counts, EPS_LOOSE)

    for city in CITIES:
        true_c  = true_city_counts[city]
        noisy_t = noisy_tight[city]
        noisy_l = noisy_loose[city]
        err_t   = noisy_t - true_c
        err_l   = noisy_l - true_c
        print(f"  {city:<16} {true_c:>11,}"
              f"  {noisy_t:>8,} ({err_t:+d})"
              f"  {noisy_l:>8,} ({err_l:+d})")

    print("""
  Takeaway:
    At ε=0.5 the Laplace scale is 1/0.5=2, so errors of ±5-10 are common.
    At ε=3.0 the scale drops to 1/3≈0.33, producing near-exact counts most
    of the time.  For large cities (N≈200) the relative error is small even
    at tight budgets.
    """)

    # ================================================================
    _section("EXAMPLE 3b — protect_city_party_counts  (Laplace Mechanism)")
    # ================================================================
    print("""
  Scenario: Publishing how many votes each party got in each city.
  The same Laplace mechanism is applied independently to every (city, party)
  cell.  Parallel composition across cities still gives ε per city.
    """)

    noisy_cpp_tight = protect_city_party_counts(true_city_party_counts, EPS_TIGHT)
    noisy_cpp_loose = protect_city_party_counts(true_city_party_counts, EPS_LOOSE)

    for city in CITIES[:2]:   # show two cities to keep output readable
        print(f"\n  City: {city}")
        print(f"  {'Party':<10} {'True':>7} "
              f"{'Noisy ε='+str(EPS_TIGHT):>14} "
              f"{'Noisy ε='+str(EPS_LOOSE):>14}")
        print(f"  {'-'*48}")
        for party in ALL_PARTIES:
            t  = true_city_party_counts[city][party]
            nt = noisy_cpp_tight[city][party]
            nl = noisy_cpp_loose[city][party]
            print(f"  {party:<10} {t:>7}  {nt:>10} ({nt-t:+d})  {nl:>10} ({nl-t:+d})")

    print("""
  Takeaway:
    Per-party counts within a city are smaller numbers, so relative errors
    look larger.  High-ε budgets keep accuracy reasonable; low-ε budgets may
    reorder parties in small cities — acceptable for public release but not
    for real-time activist use.
    """)

    # ================================================================
    _section("EXAMPLE 4 — protect_potential_voter_status  (Binary RR)")
    # ================================================================
    print("""
  Scenario: A party activist's list of flagged potential voters is perturbed
  before being logged.  This prevents the list from being a perfectly accurate
  record of who the party considers a sympathiser.
    """)

    true_potential_rate = float(np.mean(true_potential))
    print(f"  True potential-voter rate: {true_potential_rate:.1%}\n")

    for eps in [EPS_TIGHT, EPS_LOOSE]:
        flip_p = epsilon_to_flip_probability(eps)
        reported_pot = protect_potential_voter_status_batch(true_potential, eps)
        raw_rate  = float(np.mean(reported_pot))
        est_rate  = estimate_rr_frequency(reported_pot, eps)

        print(f"  ε = {eps}  |  P(flip) = {flip_p:.1%}")
        print(f"    Raw reported potential rate : {raw_rate:.1%}")
        print(f"    De-biased estimate          : {est_rate:.1%}  "
              f"(true = {true_potential_rate:.1%})")
        print()

    print("""
  Takeaway:
    The mechanism injects false positives and false negatives into the
    activist's list.  Individuals gain plausible deniability: "I was never
    flagged as a potential voter."  The true population statistic is still
    recoverable analytically.
    """)

    # ================================================================
    _section("BONUS — Privacy-Utility Tradeoff Summary")
    # ================================================================
    print("""
  How the flip probability for binary RR changes with ε:
    """)
    print(f"  {'ε':>6}  {'P(flip) binary RR':>20}  "
          f"{'P(wrong) k=4 RR':>20}")
    print(f"  {'-'*50}")
    for eps in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        p_flip = epsilon_to_flip_probability(eps)
        p_krr  = epsilon_to_krr_noise_probability(eps, k=4)
        print(f"  {eps:>6.1f}  {p_flip:>19.1%}  {p_krr:>19.1%}")

    print("""
  Rule of thumb:
    ε < 1   → strong privacy, high noise, estimates need large N to be useful.
    1 ≤ ε ≤ 3 → reasonable tradeoff for most electoral analytics use cases.
    ε > 3   → weak privacy; most responses are truthful; only use when
              differential privacy is required by policy, not for strong
              individual protection.
    """)


if __name__ == "__main__":
    main()
