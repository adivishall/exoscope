"""Look up known planets of a host star in the NASA Exoplanet Archive (TAP).

The result is *reference information* used to check ExoScope's output. If the
archive cannot be reached, the curated values in ``targets.py`` are used.
"""

from __future__ import annotations

import csv
import io
import urllib.parse
import urllib.request

from .targets import REFERENCE_SOURCE, ReferencePlanet, reference_planets, validate_target_name

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Archive host names that differ from the common target name.
HOST_ALIASES = {"pi men": "HD 39091"}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def lookup_known_planets(target: str, timeout: float = 12.0) -> tuple[list[ReferencePlanet], str]:
    """Return (planets, provenance string). Never raises for network failures."""
    target = validate_target_name(target)
    host = HOST_ALIASES.get(target.lower(), target)
    query = (
        "select pl_name,pl_orbper,pl_trandep,pl_trandur,pl_rade,st_rad from pscomppars "
        f"where lower(hostname) = '{host.lower()}'"  # host is validated: no quotes possible
    )
    url = f"{TAP_URL}?{urllib.parse.urlencode({'query': query, 'format': 'csv'})}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode()
        planets = []
        for row in csv.DictReader(io.StringIO(text)):
            vals = [_num(row.get(k)) for k in ("pl_orbper", "pl_trandep", "pl_trandur", "pl_rade", "st_rad")]
            planets.append(ReferencePlanet(row["pl_name"], *vals))
        if planets:
            return sorted(planets, key=lambda p: p.name), f"{REFERENCE_SOURCE} (live query)"
    except Exception:  # offline, timeout, archive outage
        pass
    fallback = list(reference_planets(target))
    if fallback:
        return fallback, f"{REFERENCE_SOURCE} (values bundled with ExoScope, retrieved 2026-10-28)"
    return [], "No archive entry found (or archive unreachable)"
