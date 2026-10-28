"""Download a few real light curves from MAST and store compact offline copies.

The bundled samples let the app run a real-data demonstration when the
network or MAST is unavailable (e.g. during a live presentation). They are
exact copies of the archived PDCSAP flux, quality-masked by Lightkurve's
default bitmask, and are always labelled as "offline copy of real data".

Usage:
    python scripts/fetch_samples.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exoscope.data.retrieval import download_lightcurve, search_target  # noqa: E402
from exoscope.data.samples import SAMPLES_DIR  # noqa: E402

# (sample id, target, mission, product index in the search result)
SAMPLES = [
    ("kepler8_q2", "Kepler-8", "Kepler", 2),
    ("kepler10_q3", "Kepler-10", "Kepler", 3),
    ("wasp18_s2", "WASP-18", "TESS", 0),
]


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for sample_id, target, mission, index in SAMPLES:
        print(f"Fetching {target} ({mission}) product #{index} ...", flush=True)
        products = search_target(target, mission)
        lc = download_lightcurve(target, mission, index, products=products)
        df = lc.to_frame()
        df.to_csv(SAMPLES_DIR / f"{sample_id}.csv.gz", index=False, float_format="%.8g")
        meta = dict(lc.meta)
        meta["retrieved_on"] = date.today().isoformat()
        meta["source"] = "bundled"
        (SAMPLES_DIR / f"{sample_id}.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"  saved {len(df)} rows -> {sample_id}.csv.gz")


if __name__ == "__main__":
    main()
