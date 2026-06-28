"""
population.py
=============
Synthetic individual-level voter population, grounded in REAL aggregate results.

The Israeli Central Elections Committee publishes per-settlement results
(`expc.csv`: 25th Knesset, Nov 2022 — eligible voters, turnout, and a vote
count per party-letter, for each of ~1,216 settlements).  Our Elector
simulation, however, reasons about *individuals*: each person has voted or not
yet, supports some party, and (if not yet voted) can be chased by activists.

This module bridges the two via **population synthesis** (a.k.a. synthetic
reconstruction): each settlement row is expanded into individual synthetic
voters whose marginals match the published aggregates **exactly** —
    • exactly `מצביעים` of them have `true_voted = True`
    • the valid-ballot party split matches the per-party columns
while every individual row is fully synthetic (random name) and carries **no
real personal data**.  The result aggregates back to the real CEC numbers,
which is what makes the DP experiments credible.

Output schema (drop-in compatible with app.py's `simulate_voter_list`):
    voter_id      : int    unique across the whole population
    name          : str    synthetic Hebrew name (no real PII)
    city          : str    real settlement name (שם ישוב)
    true_voted    : bool    did this person turn out?
    true_party    : str|NA  party of a VALID ballot; "פסול" if invalid;
                            <NA> for non-voters (they cast no ballot)
    latent_support: str     underlying party preference of EVERY person.
                            For valid voters == true_party.  For non-voters
                            and invalid voters it is *modelled*, not observed —
                            this is the GOTV chase target a party would predict.

NOTE on `latent_support` for non-voters: real published data only reveals the
party split among people who *voted*.  Non-voter preference is unobserved, so
it is an explicit modelling assumption here (default: drawn from the
settlement's own valid-vote distribution).  Treat it as a tunable knob and a
stated limitation, not ground truth.

Usage
-----
    from population import synthesize_population, DEFAULT_SETTLEMENTS
    df = synthesize_population()                       # default subset
    df = synthesize_population(settlements=["חיפה"])   # custom subset
    python population.py                               # sanity-check printout
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# FILE LOCATION
# =============================================================================

_HERE = Path(__file__).resolve().parent
# The CSV lives in the parent research folder during development; for Hugging
# Face deployment, copy it next to this file. We check both locations.
_CSV_CANDIDATES = [_HERE / "expc.csv", _HERE.parent / "expc.csv"]


def _default_csv_path() -> Path:
    for c in _CSV_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError(
        "expc.csv not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in _CSV_CANDIDATES)
        + "\nPass csv_path=... explicitly."
    )


# =============================================================================
# METADATA / PARTY-LETTER MAPPING
# =============================================================================

# Non-party columns in expc.csv.
META_COLS = ["סמל ועדה", "שם ישוב", "סמל ישוב", "בזב", "מצביעים", "פסולים", "כשרים"]
COL_NAME = "שם ישוב"      # settlement name
COL_ELIGIBLE = "בזב"       # eligible voters (בעלי זכות בחירה)
COL_VOTERS = "מצביעים"     # turned out
COL_VALID = "כשרים"        # valid ballots

# Label given to a voter who turned out but cast an INVALID ballot (פסול).
# Kept distinct from any real party and from <NA> (= did not vote at all).
INVALID_LABEL = "פסול"

# Bucket label for minor parties collapsed via `keep_top_parties` (see below).
OTHER_LABEL = "אחר"

# Output columns, in order. Importing code can rely on this schema.
SCHEMA = ["voter_id", "name", "city", "true_voted", "true_party", "latent_support"]

# Best-effort ballot-letter → readable party name, 25th Knesset (Nov 2022).
# Used for display only; the simulation works on whatever labels are present,
# so unmapped letters simply keep their original code. Edit freely.
PARTY_LETTER_TO_NAME = {
    "מחל": "הליכוד",
    "פה":  "יש עתיד",
    "ט":   "הציונות הדתית",
    "כן":  "המחנה הממלכתי",
    "שס":  "שס",
    "ג":   "יהדות התורה",
    "ל":   "ישראל ביתנו",
    "עם":  "רעם",
    "ום":  "חדש-תעל",
    "אמת": "העבודה",
    "מרצ": "מרצ",
    "ד":   "בלד",
}

# Synthetic names — fully made up, NOT linked to any real person.
_FIRST_NAMES = [
    "יוסי", "מיכל", "דוד", "שרה", "אבי", "רחל", "נועה", "אמיר",
    "תמר", "גיל", "לי", "עומר", "דינה", "ניר", "מאיה", "יובל",
    "רון", "הלה", "שי", "אורית", "ידין", "ליאור", "כרמל", "בר",
    "מוחמד", "פאטמה", "אחמד", "ליהאם", "נור", "סלים", "ראמי", "הודא",
]
_LAST_NAMES = [
    "כהן", "לוי", "מזרחי", "פרץ", "ביטון", "אברהם", "דהן", "שמש",
    "פרידמן", "חסן", "ח'ורי", "עבאס", "ברק", "אזולאי", "גבאי", "סבן",
]


# =============================================================================
# DEFAULT SUBSET  (large cities  +  small disclosure-at-risk settlements)
# =============================================================================
# Chosen to exercise both stories at once:
#   • large cities  → DP noise is negligible at scale (utility preserved)
#   • small ones    → many single-vote party cells → real reconstruction risk
DEFAULT_LARGE_SETTLEMENTS = [
    "תל אביב יפו", "ירושלים", "חיפה", "ראשון לציון", "פתח תקווה", "נתניה",
]
DEFAULT_SMALL_SETTLEMENTS = [
    "אלוני הבשן", "גיתה", "מנרה", "נעמה", "גזר", "בארות יצחק",
]
DEFAULT_SETTLEMENTS = DEFAULT_LARGE_SETTLEMENTS + DEFAULT_SMALL_SETTLEMENTS


# =============================================================================
# LOADING / SUBSETTING
# =============================================================================

def _normalize(s: str) -> str:
    """Collapse runs of whitespace so 'תל אביב  יפו' matches 'תל אביב יפו'."""
    return re.sub(r"\s+", " ", str(s)).strip()


def load_settlements_table(csv_path=None) -> pd.DataFrame:
    """Load the raw CEC per-settlement table (one row per settlement)."""
    path = Path(csv_path) if csv_path else _default_csv_path()
    df = pd.read_csv(path)
    df[COL_NAME] = df[COL_NAME].map(_normalize)
    return df


def party_columns(df: pd.DataFrame) -> list[str]:
    """The party-letter columns = everything that isn't metadata."""
    return [c for c in df.columns if c not in META_COLS]


def subset_settlements(df: pd.DataFrame, settlements) -> pd.DataFrame:
    """Return rows for the requested settlement names (whitespace-insensitive)."""
    wanted = {_normalize(s) for s in settlements}
    out = df[df[COL_NAME].isin(wanted)].copy()
    found = set(out[COL_NAME])
    missing = wanted - found
    if missing:
        raise ValueError(
            "Settlement(s) not found in expc.csv: "
            + ", ".join(sorted(missing))
            + ".\nCheck spelling against load_settlements_table()['שם ישוב']."
        )
    return out


# =============================================================================
# SUPPORT MODELS  (how the UNOBSERVED preference of non-/invalid-voters is drawn)
# =============================================================================
# A support model is a callable:
#       model(rng, parties, observed_probs, size) -> np.ndarray[str]
# where `parties` are the (display) party labels and `observed_probs` is the
# settlement's valid-vote distribution over them (or None if no valid votes).
# Real data never reveals non-voter preference, so this is the modelling knob
# distinguishing, e.g., a perfectly-informed party from a noisy predictor.
# Pass a string name (resolved via SUPPORT_MODELS) or your own callable.

def _support_observed(rng, parties, observed_probs, size):
    """Non-voters split exactly like the settlement's actual voters."""
    if observed_probs is None:
        return rng.choice(parties, size=size)
    return rng.choice(parties, size=size, p=observed_probs)


def _support_uniform(rng, parties, observed_probs, size):
    """Non-voters split uniformly across parties (preference is uninformative)."""
    return rng.choice(parties, size=size)


def make_noisy_support_model(noise: float):
    """
    Factory: blend the observed distribution with the uniform one.

        effective = (1 - noise) * observed  +  noise * uniform

    noise=0 → identical to "observed"; noise=1 → identical to "uniform".
    Handy for Design A, where a party's prediction of non-voter support is
    imperfect: raise `noise` to model a worse predictor.
    """
    if not (0.0 <= noise <= 1.0):
        raise ValueError("noise must be in [0, 1].")

    def model(rng, parties, observed_probs, size):
        k = len(parties)
        unif = np.full(k, 1.0 / k)
        base = unif if observed_probs is None else np.asarray(observed_probs)
        probs = (1.0 - noise) * base + noise * unif
        probs = probs / probs.sum()
        return rng.choice(parties, size=size, p=probs)

    return model


# Registry of built-in models, resolvable by name.
SUPPORT_MODELS = {
    "observed": _support_observed,
    "uniform": _support_uniform,
}


def resolve_support_model(support_model):
    """Accept a callable as-is, or look a string up in SUPPORT_MODELS."""
    if callable(support_model):
        return support_model
    try:
        return SUPPORT_MODELS[support_model]
    except KeyError:
        raise ValueError(
            f"Unknown support_model {support_model!r}. "
            f"Use a callable or one of: {sorted(SUPPORT_MODELS)}."
        )


# =============================================================================
# POPULATION SYNTHESIS
# =============================================================================

def _make_names(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorised synthetic 'first last' names."""
    first = rng.choice(_FIRST_NAMES, size=n)
    last = rng.choice(_LAST_NAMES, size=n)
    return np.char.add(np.char.add(first.astype(str), " "), last.astype(str))


def _settlement_scale(eligible_raw, fraction, max_per_settlement):
    """
    Single scale factor folding together proportional down-sampling and an
    optional absolute per-settlement cap. scale == 1.0 ⇒ exact real marginals.
    """
    scale = fraction
    if max_per_settlement is not None and eligible_raw * scale > max_per_settlement:
        scale = max_per_settlement / eligible_raw
    return scale


def _synthesize_one(row, parties, rng, start_id, scale, rename, support):
    """
    Expand a single settlement row into an individual-level DataFrame whose
    marginals match the published aggregates (scaled by `scale`).

    Parameters
    ----------
    row     : the settlement row from the CEC table.
    parties : list of party COLUMN labels (letters) in the table.
    rng     : numpy Generator.
    start_id: first voter_id to assign.
    scale   : proportional size factor (1.0 = exact real counts).
    rename  : {letter: display_name} map (output labels only).
    support : resolved support-model callable for unobserved preferences.
    """
    eligible_raw = int(row[COL_ELIGIBLE])
    if eligible_raw <= 0:
        return None

    eligible = max(1, int(round(eligible_raw * scale)))
    voted_n = min(eligible, int(round(int(row[COL_VOTERS]) * scale)))

    # Per-party VALID vote counts (scaled consistently with voted_n).
    raw_counts = np.array([max(0, int(row[c])) for c in parties], dtype=float)
    party_counts = np.round(raw_counts * scale).astype(int)
    valid_total = int(party_counts.sum())

    display_parties = [rename.get(p, p) for p in parties]
    probs = (party_counts / valid_total) if valid_total > 0 else None

    # ---- voters: assign a ballot label to exactly `voted_n` individuals -----
    # Build the valid-ballot labels by repeating each party by its count, then
    # pad/truncate to voted_n; the remainder are invalid ballots (INVALID_LABEL).
    valid_labels = np.repeat(display_parties, party_counts)
    if valid_labels.size > voted_n:
        valid_labels = rng.permutation(valid_labels)[:voted_n]
    n_invalid = voted_n - valid_labels.size
    voted_party = np.concatenate([
        valid_labels,
        np.full(n_invalid, INVALID_LABEL, dtype=object),
    ])
    rng.shuffle(voted_party)

    n_nonvoters = eligible - voted_n

    # ---- latent support (underlying preference) for everyone ---------------
    # Valid voters: their actual vote. Invalid voters + non-voters: drawn from
    # the chosen support model (the modelled GOTV chase target).
    voted_support = np.where(
        voted_party == INVALID_LABEL,
        support(rng, display_parties, probs, voted_n),
        voted_party,
    )
    nonvoter_support = support(rng, display_parties, probs, n_nonvoters)

    # ---- assemble ----------------------------------------------------------
    true_voted = np.concatenate([
        np.ones(voted_n, dtype=bool),
        np.zeros(n_nonvoters, dtype=bool),
    ])
    true_party = np.concatenate([
        voted_party,
        np.full(n_nonvoters, pd.NA, dtype=object),
    ])
    latent = np.concatenate([voted_support, nonvoter_support])

    n = eligible
    out = pd.DataFrame({
        "voter_id": np.arange(start_id, start_id + n),
        "name": _make_names(rng, n),
        "city": _normalize(row[COL_NAME]),
        "true_voted": true_voted,
        "true_party": pd.array(true_party, dtype="string"),
        "latent_support": pd.array(latent.astype(object), dtype="string"),
    })
    return out


def synthesize_population(
    settlements=None,
    csv_path=None,
    seed: int = 42,
    fraction: float = 1.0,
    max_per_settlement=None,
    support_model="observed",
    keep_top_parties=None,
    rename_parties: bool = True,
) -> pd.DataFrame:
    """
    Build a synthetic individual-level population for the chosen settlements.

    Parameters
    ----------
    settlements : list[str] | None
        Settlement names to include. None → DEFAULT_SETTLEMENTS.
    csv_path : str | Path | None
        Path to expc.csv. None → auto-locate.
    seed : int
        RNG seed for reproducibility.
    fraction : float
        Proportional down-sample factor in (0, 1]. 1.0 = full real size.
        Preserves each settlement's distribution shape; use e.g. 0.01 to keep
        an interactive app responsive.
    max_per_settlement : int | None
        Optional absolute cap on synthetic voters per settlement (applied on
        top of `fraction`). Lets large cities and tiny settlements coexist in
        one responsive population. None = no cap.
    support_model : str | callable
        How to draw the UNOBSERVED preference of non-/invalid-voters. A name in
        SUPPORT_MODELS ("observed", "uniform") or a callable
        (rng, parties, observed_probs, size) -> array. See make_noisy_support_model.
    keep_top_parties : int | None
        If set, keep only the N largest parties (by total vote across the
        subset) and collapse the rest into OTHER_LABEL ("אחר"). Keeps the k of
        k-randomized-response small (better utility, fewer chart bars) for the
        app. None = full granularity, e.g. for the reconstruction experiment.
    rename_parties : bool
        Map ballot letters to readable party names (display only).

    Returns
    -------
    pd.DataFrame
        One row per synthetic voter (schema == SCHEMA), shuffled so settlements
        are interleaved. With fraction == 1.0 and no cap, marginals match
        expc.csv exactly.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1].")
    if max_per_settlement is not None and max_per_settlement < 1:
        raise ValueError("max_per_settlement must be >= 1 or None.")

    settlements = list(settlements) if settlements is not None else DEFAULT_SETTLEMENTS
    support = resolve_support_model(support_model)

    table = load_settlements_table(csv_path)
    sub = subset_settlements(table, settlements)
    parties = party_columns(table)

    # Build the letter → display-label map. Kept parties get a readable name
    # (or their letter); collapsed minor parties all map to OTHER_LABEL.
    rename = dict(PARTY_LETTER_TO_NAME) if rename_parties else {}
    if keep_top_parties is not None:
        totals = sub[parties].sum().sort_values(ascending=False)
        kept = set(totals.index[:int(keep_top_parties)])
        for p in parties:
            if p not in kept:
                rename[p] = OTHER_LABEL

    rng = np.random.default_rng(seed)
    frames, next_id = [], 0
    for _, row in sub.iterrows():
        scale = _settlement_scale(int(row[COL_ELIGIBLE]), fraction, max_per_settlement)
        part = _synthesize_one(row, parties, rng, next_id, scale, rename, support)
        if part is not None and len(part):
            frames.append(part)
            next_id += len(part)

    if not frames:
        raise ValueError("No voters synthesised (empty/zero-eligible subset).")

    pop = pd.concat(frames, ignore_index=True)
    # Interleave settlements and reassign contiguous ids.
    pop = pop.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pop["voter_id"] = np.arange(len(pop))
    return pop


# =============================================================================
# DERIVED VIEWS  (convenience accessors for downstream code, e.g. the app)
# =============================================================================

def settlement_names(pop: pd.DataFrame) -> list[str]:
    """Settlements present in `pop`, ordered by synthetic population (desc)."""
    return pop.groupby("city").size().sort_values(ascending=False).index.tolist()


def party_candidates(pop: pd.DataFrame) -> list[str]:
    """
    The real parties present (the candidate set for k-randomized-response),
    ordered by modelled support (desc). Excludes INVALID_LABEL and <NA>; based
    on `latent_support`, which is defined for every individual.
    """
    counts = pop["latent_support"].dropna().value_counts()
    cand = [c for c in counts.index.tolist() if c != INVALID_LABEL]
    # Keep the collapsed "Other" bucket last so it never becomes the organiser.
    if OTHER_LABEL in cand:
        cand = [c for c in cand if c != OTHER_LABEL] + [OTHER_LABEL]
    return cand


# =============================================================================
# VERIFICATION / SANITY CHECK
# =============================================================================

def verify_against_source(pop: pd.DataFrame, csv_path=None, settlements=None):
    """
    Compare synthesised marginals to the real CEC table, per settlement.
    Returns a tidy DataFrame; intended for the __main__ printout / tests.
    Only meaningful when fraction == 1.0 (exact match expected).
    """
    table = load_settlements_table(csv_path)
    names = settlements or sorted(pop["city"].unique())
    sub = subset_settlements(table, names)

    rows = []
    for _, r in sub.iterrows():
        city = _normalize(r[COL_NAME])
        p = pop[pop.city == city]
        rows.append({
            "city": city,
            "eligible_real": int(r[COL_ELIGIBLE]),
            "eligible_syn": len(p),
            "voted_real": int(r[COL_VOTERS]),
            "voted_syn": int(p.true_voted.sum()),
            "valid_real": int(r[COL_VALID]),
            "valid_syn": int(((p.true_voted) & (p.true_party != INVALID_LABEL)
                              & (p.true_party.notna())).sum()),
        })
    return pd.DataFrame(rows)


def _main():
    import sys
    # UTF-8 stdout for Hebrew on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Synthesising default subset (full real size)...")
    pop = synthesize_population()
    print(f"\nTotal synthetic voters: {len(pop):,}")
    print(f"Settlements: {pop.city.nunique()}")
    print(f"Overall turnout: {pop.true_voted.mean():.1%}")

    print("\nSample rows:")
    print(pop.sample(8, random_state=1).to_string(index=False))

    print("\nMarginals vs. real CEC data (should match exactly):")
    chk = verify_against_source(pop)
    print(chk.to_string(index=False))
    ok = (chk.eligible_real == chk.eligible_syn).all() and \
         (chk.voted_real == chk.voted_syn).all()
    print("\n✅ marginals match" if ok else "\n❌ MISMATCH — investigate")

    print("\nTop latent-support parties (incl. non-voters' modelled preference):")
    print(pop.latent_support.value_counts().head(8).to_string())


if __name__ == "__main__":
    _main()
