"""Retrieve real light curves from MAST through Lightkurve, with a local cache.

Two cache layers keep the app usable offline and fast during demonstrations:

1. Lightkurve/astroquery keep the downloaded FITS files in ``CACHE_DIR/fits``.
2. ExoScope stores the search table (JSON) and a compact copy of each
   downloaded light curve (CSV) in ``CACHE_DIR``, so a repeated request needs
   no network at all.

Only the PDCSAP flux, time, uncertainties and quality mask are kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time as _time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import DataError, LightCurveData
from .targets import MISSIONS, validate_target_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("EXOSCOPE_CACHE_DIR", PROJECT_ROOT / ".cache"))
SEARCH_CACHE_MAX_AGE_S = 7 * 24 * 3600

# Long-cadence Kepler and 2-minute TESS SPOC products: the standard,
# pipeline-detrended (PDCSAP) light curves for each mission.
MISSION_PRODUCTS = {
    "Kepler": {"author": "Kepler", "exptime": 1800},
    "TESS": {"author": "SPOC", "exptime": 120},
}

HEADER_KEYS = (
    "OBJECT", "KEPLERID", "TICID", "QUARTER", "SECTOR", "MISSION", "TELESCOP",
    "RA_OBJ", "DEC_OBJ", "TEFF", "RADIUS", "LOGG", "KEPMAG", "TESSMAG",
    "CDPP3_0", "CDPP6_0", "CDPP12_0", "CROWDSAP", "FLFRCSAP", "PDCMETHD",
)

# In-memory Lightkurve SearchResult objects (needed to download a product).
_SEARCH_RESULTS: dict[tuple[str, str], Any] = {}


class RetrievalError(RuntimeError):
    """A user-presentable failure to find or download data."""


def _key(*parts: str) -> str:
    raw = "|".join(p.lower() for p in parts)
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")[:40]
    return f"{slug}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def _check_mission(mission: str) -> str:
    if mission not in MISSIONS:
        raise ValueError(f"Unsupported mission '{mission}'. Choose one of: {', '.join(MISSIONS)}.")
    return mission


def _dataset_label(mission_field: str) -> str:
    """'Kepler Quarter 02' -> 'Quarter 2'; 'TESS Sector 05' -> 'Sector 5'."""
    m = re.search(r"(Quarter|Sector|Campaign)\s*0*(\d+)", str(mission_field))
    return f"{m.group(1)} {int(m.group(2))}" if m else str(mission_field)


def _lk():
    """Import lightkurve lazily (slow import, and only needed for real data)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import lightkurve as lk
    return lk


def search_target(target: str, mission: str, use_cache: bool = True) -> list[dict[str, Any]]:
    """List the available light-curve products for a target.

    Returns a list of dicts with keys ``index, dataset, year, author,
    exptime_s, target_name, product``. Raises RetrievalError when nothing is
    found or MAST cannot be reached and no cached copy exists.
    """
    target = validate_target_name(target)
    mission = _check_mission(mission)
    cache_file = CACHE_DIR / "searches" / f"{_key(target, mission)}.json"

    if use_cache and cache_file.exists():
        age = _time.time() - cache_file.stat().st_mtime
        if age < SEARCH_CACHE_MAX_AGE_S:
            return json.loads(cache_file.read_text())

    lk = _lk()
    opts = MISSION_PRODUCTS[mission]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr = lk.search_lightcurve(target, mission=mission, author=opts["author"], exptime=opts["exptime"])
    except Exception as exc:  # network errors, name-resolution errors, MAST outages
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        raise RetrievalError(
            f"Could not search MAST for '{target}' ({mission}): {exc}. "
            "Check the name and your internet connection, or use a bundled sample / synthetic data."
        ) from exc

    if len(sr) == 0:
        raise RetrievalError(
            f"No {mission} light curves found for '{target}'. "
            f"Try another identifier (e.g. KIC/TIC number) or the other mission."
        )

    # Keep only the closest object: a name search can also return neighbours.
    table = sr.table
    if "distance" in table.colnames:
        dist = np.asarray(table["distance"], dtype=float)
        keep = np.where(np.isclose(dist, np.nanmin(dist)))[0]
        sr = sr[list(keep)]
        table = sr.table

    products = []
    for i in range(len(sr)):
        row = table[i]
        products.append(
            {
                "index": i,
                "dataset": _dataset_label(row["mission"]),
                "year": int(row["year"]) if "year" in table.colnames else None,
                "author": str(row["author"]),
                "exptime_s": float(row["exptime"]),
                "target_name": str(row["target_name"]),
                "product": str(row["productFilename"]) if "productFilename" in table.colnames else f"{target}-{i}",
            }
        )
    _SEARCH_RESULTS[(target.lower(), mission)] = sr
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(products, indent=1))
    return products


def _as_float_array(x: Any) -> np.ndarray:
    """Convert Quantity / masked arrays to a float ndarray with NaN for masked values."""
    v = getattr(x, "value", x)
    if hasattr(v, "unmasked") and hasattr(v, "mask"):  # astropy Masked
        out = np.array(v.unmasked, dtype=float)
        out[np.asarray(v.mask, dtype=bool)] = np.nan
        return out
    if np.ma.isMaskedArray(v):
        return v.astype(float).filled(np.nan)
    return np.asarray(v, dtype=float)


def _cached_lc_path(product: str) -> tuple[Path, Path]:
    base = CACHE_DIR / "lightcurves" / _key(product)
    return base.with_suffix(".csv.gz"), base.with_suffix(".json")


def load_cached(csv_path: Path, meta_path: Path) -> LightCurveData:
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text())
    return LightCurveData.from_frame(df, meta)


def download_lightcurve(
    target: str, mission: str, index: int, products: list[dict[str, Any]] | None = None
) -> LightCurveData:
    """Download (or load from cache) one light-curve product as LightCurveData."""
    target = validate_target_name(target)
    mission = _check_mission(mission)
    products = products if products is not None else search_target(target, mission)
    if not 0 <= index < len(products):
        raise ValueError(f"Product index {index} is out of range (0..{len(products) - 1}).")
    prod = products[index]
    csv_path, meta_path = _cached_lc_path(prod["product"])
    if csv_path.exists() and meta_path.exists():
        data = load_cached(csv_path, meta_path)
        data.meta["source"] = "mast"
        data.meta["from_cache"] = True
        return data

    sr = _SEARCH_RESULTS.get((target.lower(), mission))
    if sr is None:  # search came from the JSON cache; redo it to get a downloadable object
        search_target(target, mission, use_cache=False)
        sr = _SEARCH_RESULTS[(target.lower(), mission)]

    lk = _lk()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = sr[index].download(quality_bitmask="none", download_dir=str(CACHE_DIR / "fits"))
    except Exception as exc:
        raise RetrievalError(f"Download of {prod['dataset']} for '{target}' failed: {exc}") from exc
    if lc is None or len(lc) == 0:
        raise RetrievalError(f"{prod['dataset']} for '{target}' contains no data.")

    quality = np.asarray(getattr(lc.quality, "value", lc.quality), dtype=int)
    flags = lk.utils.KeplerQualityFlags if mission == "Kepler" else lk.utils.TessQualityFlags
    quality_ok = flags.create_quality_mask(quality, bitmask="default")

    meta: dict[str, Any] = {k: lc.meta.get(k) for k in HEADER_KEYS if lc.meta.get(k) is not None}
    meta.update(
        {
            "target": target,
            "mission": mission,
            "dataset": prod["dataset"],
            "author": prod["author"],
            "exptime_s": prod["exptime_s"],
            "product": prod["product"],
            "flux_column": "pdcsap_flux",
            "flux_unit": str(lc.flux.unit),
            "time_format": str(lc.time.format),  # 'bkjd' or 'btjd'
            "source": "mast",
            "retrieved_on": pd.Timestamp.now().date().isoformat(),
        }
    )
    data = LightCurveData(
        time=_as_float_array(lc.time),
        flux=_as_float_array(lc.flux),
        flux_err=_as_float_array(lc.flux_err),
        quality_ok=quality_ok,
        meta=meta,
    )
    if not np.isfinite(data.flux).any():
        raise DataError(f"{prod['dataset']} for '{target}' has no finite flux values.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_frame().to_csv(csv_path, index=False, float_format="%.8g")
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    return data
