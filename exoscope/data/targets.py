"""Curated demonstration targets and reference planet parameters.

Reference values were retrieved from the NASA Exoplanet Archive
"Planetary Systems Composite Parameters" table (pscomppars) on 2026-10-28.
They are used only as *external reference information* for validating
ExoScope's detections — ExoScope itself never confirms a planet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REFERENCE_RETRIEVED = "2026-10-28"
REFERENCE_SOURCE = "NASA Exoplanet Archive, pscomppars table"


@dataclass(frozen=True)
class ReferencePlanet:
    name: str
    period_days: float
    depth_percent: float  # archive transit depth, in percent
    duration_hours: float
    radius_earth: float
    star_radius_solar: float


@dataclass(frozen=True)
class Target:
    name: str
    mission: str  # "Kepler" or "TESS"
    description: str
    difficulty: str  # "easy" | "moderate" | "hard" for a threshold detector
    planets: tuple[ReferencePlanet, ...] = ()


_P = ReferencePlanet
TARGETS: tuple[Target, ...] = (
    Target("Kepler-8", "Kepler", "Hot Jupiter with deep (~0.9%) transits every 3.5 days.", "easy",
           (_P("Kepler-8 b", 3.5224991, 0.91457, 3.19061, 15.87, 1.486),)),
    Target("TrES-2", "Kepler", "Kepler-1 b: grazing hot Jupiter, 1.6% depth, 2.47-day orbit.", "easy",
           (_P("TrES-2 b", 2.47063, 1.6, 1.74259, 15.24, 1.12),)),
    Target("Kepler-6", "Kepler", "Hot Jupiter, ~1.1% depth, 3.23-day orbit.", "easy",
           (_P("Kepler-6 b", 3.2346996, 1.08141, 3.59385, 14.62, 1.391),)),
    Target("Kepler-17", "Kepler", "Active spotted star with a 2.2% deep hot Jupiter (1.49 d) — detrending matters.", "moderate",
           (_P("Kepler-17 b", 1.4857108, 2.18019, 2.2764, 14.68, 1.010),)),
    Target("HAT-P-7", "Kepler", "Kepler-2 b: 0.6% transits around an F star, 2.2-day orbit.", "easy",
           (_P("HAT-P-7 b", 2.20474, 0.6, 3.88216, 16.93, 2.0),)),
    Target("Kepler-5", "Kepler", "Hot Jupiter, 0.7% depth, 3.55-day orbit.", "easy",
           (_P("Kepler-5 b", 3.5484657, 0.72515, 4.56904, 15.98, 1.793),)),
    Target("Kepler-10", "Kepler", "First rocky Kepler planet: tiny 0.02% (~190 ppm) transits every 0.84 d. "
           "Hard for a threshold detector, easy for BLS.", "hard",
           (_P("Kepler-10 b", 0.8374907, 0.01919, 1.811, 1.47, 1.065),
            _P("Kepler-10 c", 45.294301, 0.04768, 6.8714, 2.355, 1.065))),
    Target("HAT-P-11", "Kepler", "Neptune-size planet, 0.43% depth, 4.89-day orbit.", "moderate",
           (_P("HAT-P-11 b", 4.888, 0.43421, 2.35562, 4.36, 0.683),)),
    Target("WASP-18", "TESS", "Massive hot Jupiter, ~1% depth, 0.94-day orbit.", "easy",
           (_P("WASP-18 b", 0.94145223, 1.041, 2.21, 13.90, 1.319),)),
    Target("WASP-121", "TESS", "Inflated hot Jupiter, 1.5% depth, 1.27-day orbit.", "easy",
           (_P("WASP-121 b", 1.27492504, 1.5018, 2.9053, 19.53, 1.461),)),
    Target("WASP-126", "TESS", "Hot Jupiter in the TESS southern continuous viewing zone, 3.29 d.", "moderate",
           (_P("WASP-126 b", 3.28883, 0.70057, 3.43677, 10.51, 1.27),)),
    Target("KELT-11", "TESS", "Very inflated planet around a subgiant: shallow 0.23%, long 7 h transits.", "moderate",
           (_P("KELT-11 b", 4.73620865, 0.23389, 7.1, 15.13, 2.69),)),
    Target("pi Men", "TESS", "Bright star with a 2 Earth-radius planet: 270 ppm transits every 6.27 d.", "hard",
           (_P("pi Men c", 6.2678399, 0.026759, 2.952024, 2.02, 1.17),)),
    Target("TOI-270", "TESS", "M dwarf with several small planets; b has 0.1% transits every 3.36 d.", "hard",
           (_P("TOI-270 b", 3.35992, 0.09767, 1.294, 1.28, 0.38),)),
)

MISSIONS = ("Kepler", "TESS")

_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 +\-_.]{0,39}$")


def validate_target_name(name: str) -> str:
    """Return a cleaned target identifier or raise ValueError.

    Accepts names such as "Kepler-8", "KIC 11904151", "TIC 100100827",
    "WASP-18", "pi Men". Rejects empty strings and unusual characters so user
    input never reaches the archive query unchecked.
    """
    cleaned = " ".join(str(name).split())
    if not cleaned:
        raise ValueError("Please enter a target name or ID (e.g. Kepler-8, KIC 11904151, TIC 100100827).")
    if not _TARGET_RE.match(cleaned):
        raise ValueError(
            "Target names may contain only letters, digits, spaces and + - _ . (max 40 characters)."
        )
    return cleaned


def find_target(name: str) -> Target | None:
    key = name.strip().lower().replace(" ", "")
    for t in TARGETS:
        if t.name.lower().replace(" ", "") == key:
            return t
    return None


def reference_planets(name: str) -> tuple[ReferencePlanet, ...]:
    t = find_target(name)
    return t.planets if t else ()
