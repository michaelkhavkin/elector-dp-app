"""
prepare_geo.py — one-time, offline builder for `settlement_coords.csv`.

The dashboard map needs a latitude/longitude for each settlement in `expc.csv`.
The CEC table gives us the official locality code (`סמל ישוב`) and the Hebrew
name (`שם ישוב`) but NO coordinates. We attach coordinates from an open
community dataset and bundle the result next to `app.py`.

Provenance (cite in the paper):
  - Names + codes: Israeli CEC 25th-Knesset per-settlement results (`expc.csv`).
  - Coordinates:  yuvadm/geolocations-il `cities.csv`
                  https://github.com/yuvadm/geolocations-il  (Hebrew name → lat/lon;
                  ~1,364 rows, sourced from data.gov.il). MIT-licensed.

Matching strategy: the coordinate source has no locality code, so we join on the
Hebrew name after *aggressive* normalization (collapse whitespace, drop hyphens
and quotes, strip trailing `שבט`/`יישוב`). That lifts coverage to ~91% and, in
particular, resolves the largest cities (e.g. "תל אביב יפו" ↔ "תל אביב - יפו").
A small manual patch fills known misses that are simply absent from the source
(e.g. נעמה). Settlements that still can't be matched are written with empty
coordinates and are silently skipped by the map (they remain in the table).

Usage:
    python experiments/prepare_geo.py                 # download coords, write CSV
    python experiments/prepare_geo.py --coords a.csv  # use a local coords file
    python experiments/prepare_geo.py --out b.csv     # custom output path

Runtime deps: pandas (already required). Download uses only the stdlib.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent                       # elector-dp-app/
sys.path.insert(0, str(_APP_DIR))
import population  # noqa: E402  (reuse _normalize / _default_csv_path / COL_NAME)

COORDS_URL = "https://raw.githubusercontent.com/yuvadm/geolocations-il/master/cities.csv"
DEFAULT_OUT = _APP_DIR / "settlement_coords.csv"

COL_NAME = population.COL_NAME       # "שם ישוב"
COL_CODE = "סמל ישוב"

# Manual patch: settlements absent from (or mis-spelled beyond matching in) the
# coordinate source. Keyed by the CEC name; values are (lat, lon). Extend as
# needed — kept intentionally small (mainly the default-subset settlements).
MANUAL_COORDS = {
    "נעמה": (31.9057, 35.4561),      # Na'ama, Jordan Valley (Bik'at HaYarden)
}


def aggressive_normalize(name: str) -> str:
    """
    A tougher normalizer than population._normalize, for *matching only*.

    Collapses whitespace, removes hyphens and quote marks (Hebrew gershayim and
    ASCII), and drops a trailing "שבט"/"יישוב"/"ישוב" token. This reconciles
    punctuation/spelling drift between the CEC names and the coordinate source
    (e.g. "תל אביב יפו" vs "תל אביב - יפו", "אבו רובייעה שבט" vs "אבו רובייעה").
    """
    s = population._normalize(name)                 # collapse internal whitespace
    s = s.replace("-", " ").replace('"', "").replace("'", "")
    s = s.replace("׳", "").replace("״", "")   # Hebrew geresh / gershayim
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r" (שבט|יישוב|ישוב)$", "", s)
    return s


def load_coords(coords_path: str | None) -> pd.DataFrame:
    """Load the coordinate source (local file if given, else download)."""
    if coords_path:
        raw = Path(coords_path).read_text(encoding="utf-8")
    else:
        print(f"Downloading coordinates from {COORDS_URL} ...")
        with urllib.request.urlopen(COORDS_URL) as resp:      # noqa: S310
            raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip() for c in df.columns]
    # Expected columns: City, Latitude, Longitude
    df = df.rename(columns={"City": "name", "Latitude": "lat", "Longitude": "lon"})
    df["key"] = df["name"].map(aggressive_normalize)
    # Keep the first occurrence of each key (source has a few variant duplicates).
    return df.drop_duplicates("key")[["key", "lat", "lon"]]


def build(coords_path: str | None, out_path: Path) -> pd.DataFrame:
    settlements = population.load_settlements_table()          # name already normalized
    settlements = settlements[[COL_CODE, COL_NAME]].copy()
    settlements["key"] = settlements[COL_NAME].map(aggressive_normalize)

    coords = load_coords(coords_path)
    merged = settlements.merge(coords, on="key", how="left")

    # Manual patch for known misses (keyed by CEC name).
    patch_key = {aggressive_normalize(k): v for k, v in MANUAL_COORDS.items()}
    for i, row in merged.iterrows():
        if pd.isna(row["lat"]) and row["key"] in patch_key:
            merged.at[i, "lat"], merged.at[i, "lon"] = patch_key[row["key"]]

    out = merged.rename(columns={COL_CODE: "code", COL_NAME: "name"})[
        ["code", "name", "lat", "lon"]
    ]
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out


def report(out: pd.DataFrame) -> None:
    total = len(out)
    matched = int(out["lat"].notna().sum())
    print(f"\nSettlements:      {total}")
    print(f"With coordinates: {matched} ({matched / total:.1%})")
    print(f"Unmatched:        {total - matched}")

    default = getattr(population, "DEFAULT_SETTLEMENTS", [])
    if default:
        keys = out.set_index("name")
        print("\nDefault-subset coverage:")
        for name in default:
            hit = name in keys.index and pd.notna(keys.loc[name, "lat"])
            print(f"  {'OK ' if hit else 'MISS'}  {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build settlement_coords.csv for the map.")
    ap.add_argument("--coords", default=None,
                    help="local path to the coordinate CSV (default: download from GitHub)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    out = build(args.coords, Path(args.out))
    report(out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
