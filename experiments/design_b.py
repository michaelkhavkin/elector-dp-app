"""
experiments/design_b.py  —  DP for the official release (Design B)
==================================================================
OFFLINE experiment (not part of the Streamlit UI). Run from the command line;
it writes a CSV + a plot for the short paper.

What it measures
----------------
Treat the published CEC table (`expc.csv`) itself as the sensitive release —
the Israeli analogue of the U.S. Census Bureau's reconstruction concern that
drove DP adoption for the 2020 census. Two halves:

1. DISCLOSURE RISK in the raw counts (independent of ε): per-settlement×party
   cells equal to 1 single out one identifiable voter (their party is exposed);
   cells equal to 2 are near-unique. These concentrate in small settlements.

2. UTILITY under a DP release: re-release every per-settlement party count with
   the Laplace mechanism at budget ε, then ask **does the noisy release still
   call the right winner** — per settlement (overall, and split by size) and
   nationally — plus per-cell error and the national vote-share distance.
   Sweeping ε traces the privacy–utility tradeoff.

Privacy accounting: within a settlement, releasing the whole party histogram has
L1-sensitivity 1 under add/remove-one-voter, so adding Laplace(scale=1/ε) to
each cell makes that settlement's histogram ε-DP. Settlements are disjoint, so
the full national release is ε-DP by parallel composition.

It is generic: it runs on any subset of the real table and reuses the Laplace
calibration (scale = sensitivity/ε) and accuracy tooling from `voting_dp.py`.

Usage
-----
    python experiments/design_b.py
    python experiments/design_b.py --repeats 40 --small-threshold 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import population  # noqa: E402
from voting_dp import laplace_margin_of_error  # noqa: E402

# ε sweep. np.inf = no-noise baseline (perfect release).
DEFAULT_EPSILONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf]

# Settlements at/below this many eligible voters are "small" (higher risk).
DEFAULT_SMALL_THRESHOLD = 1000


# =============================================================================
# DATA
# =============================================================================

def load_counts(csv_path=None, settlements=None, drop_empty_parties=True):
    """
    Return (counts, parties, meta):
      counts : int array [n_settlements, n_parties] of per-party valid votes.
      parties: party labels for the columns (readable names where known).
      meta   : DataFrame with city, eligible, valid per settlement.
    `settlements=None` uses the whole national table.
    """
    table = population.load_settlements_table(csv_path)
    if settlements is not None:
        table = population.subset_settlements(table, settlements)

    letters = population.party_columns(table)
    counts = table[letters].fillna(0).to_numpy()
    counts = np.clip(np.rint(counts), 0, None).astype(int)

    if drop_empty_parties:
        keep = counts.sum(axis=0) > 0
        counts = counts[:, keep]
        letters = [p for p, k in zip(letters, keep) if k]

    parties = [population.PARTY_LETTER_TO_NAME.get(p, p) for p in letters]
    meta = pd.DataFrame({
        "city": table[population.COL_NAME].to_numpy(),
        "eligible": table[population.COL_ELIGIBLE].fillna(0).astype(int).to_numpy(),
        "valid": table[population.COL_VALID].fillna(0).astype(int).to_numpy(),
    })
    return counts, parties, meta


# =============================================================================
# 1. DISCLOSURE RISK  (no DP)
# =============================================================================

def disclosure_risk(counts, meta, small_threshold=DEFAULT_SMALL_THRESHOLD):
    """Singling-out / near-unique cell statistics in the RAW published counts."""
    small = meta["eligible"].to_numpy() < small_threshold
    singling = counts == 1                      # one identifiable voter
    pair = counts == 2                          # near-unique
    return {
        "settlements": int(counts.shape[0]),
        "parties": int(counts.shape[1]),
        "nonzero_cells": int((counts > 0).sum()),
        "singling_cells": int(singling.sum()),
        "singling_cells_in_small": int(singling[small].sum()),
        "pair_cells": int(pair.sum()),
        "settlements_with_singling": int(singling.any(axis=1).sum()),
        "small_settlements": int(small.sum()),
        "voters_exposed_by_singling": int(counts[singling].sum()),  # == #singling
    }


# =============================================================================
# 2. DP RELEASE + UTILITY
# =============================================================================

def release_laplace(rng, counts, epsilon):
    """
    Laplace release of every cell: count + Laplace(0, sensitivity/ε), sensitivity=1.
    Identical calibration to voting_dp.laplace_mechanism (scale = 1/ε), vectorised.
    ε=inf → scale 0 → exact counts.
    """
    scale = 0.0 if not np.isfinite(epsilon) else 1.0 / epsilon
    noise = rng.laplace(0.0, scale, size=counts.shape) if scale > 0 else 0.0
    return counts + noise


def utility_metrics(counts, released, meta, small_threshold=DEFAULT_SMALL_THRESHOLD):
    """Winner-call accuracy and error of a DP-released table vs. the truth."""
    rel = np.clip(np.rint(released), 0, None).astype(int)

    true_w = counts.argmax(axis=1)
    rel_w = rel.argmax(axis=1)
    correct = true_w == rel_w

    valid = meta["valid"].to_numpy().astype(float)
    small = meta["eligible"].to_numpy() < small_threshold

    nat_true = counts.sum(axis=0)
    nat_rel = rel.sum(axis=0)

    return {
        "winner_acc": float(correct.mean()),
        "winner_acc_weighted": float((correct * valid).sum() / max(valid.sum(), 1)),
        "winner_acc_small": float(correct[small].mean()) if small.any() else np.nan,
        "winner_acc_large": float(correct[~small].mean()) if (~small).any() else np.nan,
        "cell_mae": float(np.abs(rel - counts).mean()),
        "national_winner_ok": bool(nat_true.argmax() == nat_rel.argmax()),
        "national_tv": float(0.5 * np.abs(
            nat_true / nat_true.sum() - nat_rel / max(nat_rel.sum(), 1)).sum()),
    }


def run_sweep(counts, meta, epsilons, repeats=20,
              small_threshold=DEFAULT_SMALL_THRESHOLD, seed=0):
    """Monte-Carlo over (epsilon, repeat). Returns a per-round tidy DataFrame."""
    rng = np.random.default_rng(seed)
    records = []
    for eps in epsilons:
        # Theoretical 95% margin of error per cell (reuses voting_dp).
        margin = (0.0 if not np.isfinite(eps)
                  else laplace_margin_of_error(sensitivity=1.0, epsilon=eps))
        for r in range(repeats):
            released = release_laplace(rng, counts, eps)
            m = utility_metrics(counts, released, meta, small_threshold)
            m.update(epsilon=eps, repeat=r, cell_margin95=margin)
            records.append(m)
    return pd.DataFrame(records)


def aggregate(df):
    """Mean + 95% CI per epsilon over the Monte-Carlo repeats."""
    metrics = ["winner_acc", "winner_acc_weighted", "winner_acc_small",
               "winner_acc_large", "cell_mae", "national_tv", "national_winner_ok"]
    g = df.groupby("epsilon")
    agg = g[metrics].mean().reset_index()
    agg["cell_margin95"] = g["cell_margin95"].first().to_numpy()
    agg["repeats"] = g.size().to_numpy()

    n = np.clip(agg["repeats"].to_numpy(), 1, None)
    agg["winner_acc_ci95"] = 1.96 * g["winner_acc"].std().to_numpy() / np.sqrt(n)
    return agg.sort_values("epsilon").reset_index(drop=True)


# =============================================================================
# PLOT
# =============================================================================

def plot_results(agg, out_path):
    """Winner-call accuracy (overall / small / large) vs ε, with 95% CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = agg[np.isfinite(agg["epsilon"])]
    oracle = agg[~np.isfinite(agg["epsilon"])]
    x = finite["epsilon"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [
        ("winner_acc", "All settlements", "#3b6cb7", "winner_acc_ci95"),
        ("winner_acc_small", "Small settlements", "#C44E52", None),
        ("winner_acc_large", "Large settlements", "#55A868", None),
    ]
    for col, label, color, ci in series:
        y = finite[col].to_numpy()
        ax.plot(x, y, "o-", color=color, label=label)
        if ci:
            e = finite[ci].to_numpy()
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.15)
        if len(oracle):
            ax.axhline(float(oracle[col].iloc[0]), color=color, ls="--", lw=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget  ε   (log scale; dashed = no-noise baseline)")
    ax.set_ylabel("Correct-winner rate (per settlement)")
    ax.set_ylim(0, 1.02)
    ax.set_title("DP official release: winner-call accuracy vs. privacy budget")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Design B — DP official-release experiment.")
    ap.add_argument("--repeats", type=int, default=20,
                    help="Monte-Carlo repetitions per ε (default: %(default)s).")
    ap.add_argument("--small-threshold", type=int, default=DEFAULT_SMALL_THRESHOLD,
                    help="Eligible-voter cutoff for 'small' settlements (default: %(default)s).")
    ap.add_argument("--national", action="store_true",
                    help="Use ALL settlements (default). Kept for clarity.")
    ap.add_argument("--subset", action="store_true",
                    help="Use only population.DEFAULT_SETTLEMENTS instead of the full table.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "output"))
    args = ap.parse_args()

    settlements = population.DEFAULT_SETTLEMENTS if args.subset else None
    print("Loading published counts...")
    counts, parties, meta = load_counts(settlements=settlements)
    print(f"  settlements={counts.shape[0]:,}  parties={counts.shape[1]}  "
          f"total valid votes={counts.sum():,}")

    # --- 1. Disclosure risk (no DP) ---
    risk = disclosure_risk(counts, meta, args.small_threshold)
    print("\nDisclosure risk in the RAW release (no DP):")
    for k, v in risk.items():
        print(f"  {k:32s} {v:,}")
    nat_true = counts.sum(axis=0)
    print(f"  national winner               : {parties[int(nat_true.argmax())]} "
          f"({nat_true.max():,} votes)")

    # --- 2. DP release utility sweep ---
    print(f"\nRunning DP-release sweep ({args.repeats} repeats × {len(DEFAULT_EPSILONS)} ε)...")
    raw = run_sweep(counts, meta, DEFAULT_EPSILONS, repeats=args.repeats,
                    small_threshold=args.small_threshold, seed=args.seed)
    agg = aggregate(raw)

    cols = ["epsilon", "cell_margin95", "winner_acc", "winner_acc_small",
            "winner_acc_large", "national_winner_ok", "cell_mae", "national_tv"]
    print("\nUtility vs. ε:")
    print(agg[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "subset" if args.subset else "national"
    csv_path = outdir / f"design_b_{tag}.csv"
    png_path = outdir / f"design_b_{tag}.png"
    agg.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_results(agg, png_path)
    print(f"\nSaved:\n  {csv_path}\n  {png_path}")


if __name__ == "__main__":
    main()
