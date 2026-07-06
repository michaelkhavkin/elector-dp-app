"""
geo.py — the geographic bridge for the dashboard map.

Loads `settlement_coords.csv` (built once, offline, by
`experiments/prepare_geo.py`) and exposes settlement centroids keyed by the same
*normalized* Hebrew name the app already uses for cities (`population._normalize`,
which is what `st.session_state.cities` and `voters.city` hold). Purely a data
bridge — no Streamlit, no privacy logic. See `prepare_geo.py` for provenance.

The map is a best-effort overlay: settlements without a coordinate (≈8% of the
CEC table — small or mis-spelled localities absent from the open source) are
simply omitted from the map. They remain in every table/chart, so no information
is lost, only the geographic marker.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
# Mirror population.py: bundled next to app.py for deployment, or the parent
# research folder during local dev.
_COORDS_CANDIDATES = [_HERE / "settlement_coords.csv", _HERE.parent / "settlement_coords.csv"]

# Rough geographic center + zoom for framing an Israel map.
ISRAEL_CENTER = (31.4, 35.0)
ISRAEL_ZOOM = 6.3

# 1 degree of latitude ≈ 111 km; used to convert a jitter radius in km to degrees.
_KM_PER_DEG_LAT = 111.0


def _normalize(s: str) -> str:
    """Match population._normalize so city names line up exactly."""
    return re.sub(r"\s+", " ", str(s)).strip()


def coords_path() -> Path | None:
    for c in _COORDS_CANDIDATES:
        if c.exists():
            return c
    return None


def coords_available() -> bool:
    """True if the bundled coordinates file is present (map can be drawn)."""
    return coords_path() is not None


def load_settlement_coords() -> pd.DataFrame:
    """
    Load the coordinates table: columns `code, name, lat, lon`. Rows without a
    coordinate (empty lat/lon) are dropped. Raises FileNotFoundError if the file
    is missing — callers should gate on `coords_available()` first.
    """
    path = coords_path()
    if path is None:
        raise FileNotFoundError(
            "settlement_coords.csv not found. Looked in:\n  "
            + "\n  ".join(str(c) for c in _COORDS_CANDIDATES)
            + "\nRun: python experiments/prepare_geo.py"
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["name"] = df["name"].map(_normalize)
    return df


def city_latlon() -> dict[str, tuple[float, float]]:
    """
    Map normalized settlement name → (lat, lon), keyed to match
    `st.session_state.cities` / `voters.city`. Only settlements with a known
    coordinate are included.
    """
    df = load_settlement_coords()
    return {row["name"]: (float(row["lat"]), float(row["lon"]))
            for _, row in df.iterrows()}


def jitter(lat: float, lon: float, n: int, rng: np.random.Generator,
           radius_km: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Scatter `n` points uniformly in a disc of `radius_km` around a centroid.

    Points are synthetic placement only — the app has no per-voter address, just
    a settlement — so this is purely to spread the dots for readability, not a
    claim about where anyone lives. Deterministic given `rng`.
    """
    r = radius_km / _KM_PER_DEG_LAT * np.sqrt(rng.random(n))  # sqrt → uniform in area
    theta = rng.random(n) * 2.0 * np.pi
    # Correct longitude spread for latitude (degrees of lon shrink toward the poles).
    lon_scale = max(np.cos(np.radians(lat)), 1e-6)
    lats = lat + r * np.sin(theta)
    lons = lon + r * np.cos(theta) / lon_scale
    return lats, lons
