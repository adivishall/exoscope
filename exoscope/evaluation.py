"""Explainable candidate evaluation: a checklist of evidence, not a probability.

Each criterion is a simple, documented test with a pass / caution / fail
outcome. The summary counts how many applicable criteria pass. That count is
a teaching aid that shows *which evidence is present*; it is NOT a
calibrated probability that a planet exists. Real vetting (centroid motion,
secondary eclipses, stellar companions, radial velocities...) is beyond
ExoScope's scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dmath.number_theory import alias_relation, interval_table

SOLAR_RADIUS_IN_EARTH_RADII = 109.1
# Failing any of these points to an astrophysical false positive rather than a planet.
FALSE_POSITIVE_TESTS = ("odd_even", "secondary", "shape", "depth")


@dataclass
class Criterion:
    key: str
    name: str
    status: str  # "pass" | "caution" | "fail" | "n/a"
    value: str
    rule: str
    explanation: str


@dataclass
class CandidateEvaluation:
    criteria: list[Criterion]

    @property
    def applicable(self) -> list[Criterion]:
        return [c for c in self.criteria if c.status != "n/a"]

    @property
    def n_pass(self) -> int:
        return sum(c.status == "pass" for c in self.criteria)

    @property
    def n_fail(self) -> int:
        return sum(c.status == "fail" for c in self.criteria)

    def status_of(self, key: str) -> str:
        return next((c.status for c in self.criteria if c.key == key), "n/a")

    @property
    def verdict(self) -> str:
        """Gated summary: detection first, then false-positive tests, then the count."""
        n = len(self.applicable)
        if n == 0:
            return "No periodic signal to evaluate"
        if self.status_of("snr") == "fail":
            return "No significant periodic signal — the strongest BLS peak is consistent with noise"
        if any(self.status_of(k) == "fail" for k in FALSE_POSITIVE_TESTS):
            return "Significant signal with signs of a non-planetary origin (e.g. eclipsing binary or variability)"
        if self.n_fail == 0 and self.n_pass >= 0.75 * n:
            return "Strong potential transit candidate — follow-up needed to establish its nature"
        return "Possible transit-like signal — several checks need further investigation"

    @property
    def level(self) -> str:
        """'strong' | 'possible' | 'non-planetary' | 'noise' | 'none' (for UI colouring)."""
        v = self.verdict
        return ("strong" if v.startswith("Strong") else "possible" if v.startswith("Possible") else
                "non-planetary" if "non-planetary" in v else "noise" if v.startswith("No significant") else "none")

    def to_dicts(self) -> list[dict]:
        return [asdict(c) for c in self.criteria]


def _grade(value: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    if not np.isfinite(value):
        return "n/a"
    if higher_is_better:
        return "pass" if value >= good else "caution" if value >= ok else "fail"
    return "pass" if value <= good else "caution" if value <= ok else "fail"


def evaluate_candidate(cand, threshold, comparison, data_summary: dict, tolerance_hours: float,
                       star_radius_solar: float | None = None) -> CandidateEvaluation:
    """Build the evidence checklist for the best BLS candidate."""
    c: list[Criterion] = []
    if cand is None:
        return CandidateEvaluation([Criterion("signal", "Periodic signal", "n/a", "BLS not run", "—",
                                              "Run BLS to obtain a periodic candidate to evaluate.")])

    # 1. repeated events
    n_obs = cand.n_transits_observed
    c.append(Criterion("repeated", "Repeated transit-like events", _grade(n_obs, 3, 2), f"{n_obs} observed transits",
                       "pass ≥ 3, caution = 2",
                       "A single dip can be anything; a periodic signal needs at least three repetitions."))

    # 2. interval consistency (number theory)
    if threshold is not None and len(threshold.events) >= 2:
        rows = interval_table([e.t_center for e in threshold.events], cand.period, tolerance_hours / 24)
        frac = np.mean([r["consistent"] for r in rows])
        c.append(Criterion("intervals", "Event intervals are integer multiples of P", _grade(frac, 0.8, 0.5),
                           f"{frac:.0%} of {len(rows)} threshold intervals",
                           f"pass ≥ 80%, caution ≥ 50% (|Δt − nP| ≤ {tolerance_hours:g} h)",
                           "If the dips come from one orbiting body, every gap between them is a whole number "
                           "of orbits (missed transits give n > 1)."))
    else:
        c.append(Criterion("intervals", "Event intervals are integer multiples of P", "n/a",
                           "fewer than 2 threshold events", "needs ≥ 2 threshold events",
                           "The threshold detector found too few individual dips to test spacing."))

    # 3. signal strength
    c.append(Criterion("snr", "Signal strength (BLS depth SNR)", _grade(cand.depth_snr, 10.0, 7.1),
                       f"SNR = {cand.depth_snr:.1f}, SDE = {cand.sde:.1f}",
                       "pass ≥ 10, caution (marginal) 7.1–10, fail < 7.1",
                       "BLS tries ~10⁵ period/duration combinations, so the strongest peak in pure noise typically "
                       "reaches SNR 4–6 (the look-elsewhere effect). 7.1σ mirrors the Kepler pipeline's detection "
                       "threshold; it is a convention, not a calibrated false-alarm rate."))

    # 4. odd/even depths
    oe = cand.odd_even_sigma
    c.append(Criterion("odd_even", "Odd and even transits have equal depth", _grade(oe, 3, 5, higher_is_better=False),
                       f"difference = {oe:.1f}σ" if np.isfinite(oe) else "not measurable",
                       "pass < 3σ, caution < 5σ",
                       "Mean core depth of odd vs even transits, with errors from the epoch-to-epoch scatter. An eclipsing "
                       "binary at twice the period shows alternating primary/secondary eclipse depths."))

    # 5. shape: box vs sinusoid
    h = cand.harmonic_delta_log_likelihood
    c.append(Criterion("shape", "Transit-shaped, not sinusoidal", "pass" if h < 0 else "fail",
                       f"log L(sine) − log L(box) = {h:,.1f}", "pass < 0 (box model preferred)",
                       "Stellar pulsation or spots produce smooth sinusoidal variations rather than flat-bottomed dips."))

    # 5b. secondary eclipse at phase 0.5
    sd, ss = cand.secondary_depth, cand.secondary_snr
    if not np.isfinite(ss):
        sec_status, sec_val = "n/a", "phase 0.5 not covered by data"
    else:
        rel = sd / cand.depth if cand.depth > 0 else float("nan")
        sec_val = f"{sd * 1e6:,.0f} ppm ({ss:.1f}σ, {rel:.0%} of primary)"
        sec_status = "pass" if ss < 3 else ("caution" if rel < 0.1 else "fail")
    c.append(Criterion("secondary", "No deep secondary eclipse at phase 0.5", sec_status, sec_val,
                       "pass < 3σ; caution if significant but < 10% of the primary depth",
                       "A deep dip half-way between transits suggests two stars eclipsing each other (or a true "
                       "period of P/2). Hot Jupiters can show a very shallow secondary eclipse, hence the caution band."))

    # 6. data quality
    completeness = data_summary.get("completeness", float("nan"))
    coverage = n_obs / cand.n_transits_expected if cand.n_transits_expected else float("nan")
    worst = min(completeness, coverage)
    c.append(Criterion("quality", "Data quality and completeness", _grade(worst, 0.85, 0.5),
                       f"{completeness:.0%} usable cadences, {coverage:.0%} of predicted transits covered",
                       "pass ≥ 85%, caution ≥ 50% (worst of the two)",
                       "Gaps and flagged data reduce the number of transits that can be checked."))

    # 7. algorithm agreement (set theory)
    if comparison is not None and comparison.bls_available and threshold is not None:
        j = comparison.jaccard()
        detail = f"Jaccard |A∩B|/|A∪B| = {j:.2f}" if np.isfinite(j) else "no events"
        status = _grade(j, 0.5, 0.2) if np.isfinite(j) else "fail"
        if threshold.period is not None and threshold.period.members and not threshold.period.ambiguous:
            rel = alias_relation(cand.period, threshold.period.period)
            detail += f"; P_thr/P_BLS ≈ {rel['fraction']} ({rel['kind']})"
        if not threshold.events:
            status, detail = "caution", "threshold detector found no events"
        c.append(Criterion("agreement", "Agreement between algorithms", status, detail, "pass ≥ 0.5, caution ≥ 0.2",
                           "Independent methods finding the same epochs is reassuring; BLS alone can still find "
                           "shallow transits a threshold misses, so disagreement is not automatically bad."))
    else:
        c.append(Criterion("agreement", "Agreement between algorithms", "n/a", "run both methods", "—",
                           "Run both detectors to compare their detections."))

    # 8. depth plausibility
    depth = cand.depth
    ratio = np.sqrt(depth) if depth > 0 else float("nan")
    value = f"δ = {depth * 1e6:,.0f} ppm → Rp/Rs ≈ {ratio:.3f}"
    if star_radius_solar and np.isfinite(ratio):
        value += f" → Rp ≈ {ratio * star_radius_solar * SOLAR_RADIUS_IN_EARTH_RADII:.1f} R⊕ (R★ = {star_radius_solar:g} R☉)"
    depth_status = "fail" if depth <= 0 else _grade(depth, 0.03, 0.10, higher_is_better=False)
    c.append(Criterion("depth", "Depth plausible for a planet", depth_status, value, "pass < 3%, caution < 10%",
                       "δ ≈ (Rp/Rs)² assumes a dark planet fully in front of a uniform star; depths of several "
                       "percent or more suggest a stellar companion."))

    # 9. duration vs period (central transit of a Sun-like star)
    expected_h = 13.0 * (cand.period / 365.25) ** (1 / 3)
    r = cand.duration * 24 / expected_h
    c.append(Criterion("duration", "Duration consistent with an orbit", "pass" if 0.15 <= r <= 2.5 else "caution",
                       f"{cand.duration * 24:.1f} h vs ~{expected_h:.1f} h expected", "ratio 0.15 – 2.5",
                       "For a Sun-like star T ≈ 13 h·(P/1 yr)^⅓; very different durations hint at a wrong "
                       "period or a non-planetary source (rough check, ignores stellar density and impact parameter)."))
    return CandidateEvaluation(c)
