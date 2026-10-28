"""Number theory for periodicity: multiples, remainders, congruences, GCDs.

Key idea: if transits repeat with period P, every observed transit time obeys

    t_i = t0 + n_i · P      for some integer epoch n_i,

so every difference t_j − t_i is an (approximate) integer multiple of P, and
all transit times are congruent modulo P:  t_i ≡ t0 (mod P).
Real measurements are noisy, so "divides" and "congruent" are replaced by
tolerance versions:

    P ∣_ε Δ   ⇔   ∃ n ∈ ℤ⁺ :  |Δ − n·P| ≤ ε
    a ≡_ε b (mod P)   ⇔   circular distance of (a − b) mod P is ≤ ε

Finding the period is then an *approximate GCD* problem: the largest P that
(approximately) divides the differences. Exact integer GCDs (Euclid) are
shown for comparison — they break as soon as noise is present, which is why
the tolerance versions are needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np


# ---------------------------------------------------------------- basics
def consecutive_intervals(times: np.ndarray) -> np.ndarray:
    t = np.sort(np.asarray(times, dtype=float))
    return np.diff(t)


def nearest_multiple(delta: float, period: float) -> tuple[int, float]:
    """Return (n, residual) with n = round(Δ/P) (at least 1) and residual = Δ − nP."""
    if period <= 0:
        raise ValueError("period must be positive")
    n = max(1, int(round(delta / period)))
    return n, float(delta - n * period)


def is_approx_multiple(delta: float, period: float, tol: float) -> bool:
    """P ∣_ε Δ : Δ is within ε of a positive integer multiple of P."""
    n, r = nearest_multiple(delta, period)
    return abs(r) <= tol and delta > 0


def circular_distance(x: np.ndarray | float, modulus: float) -> np.ndarray | float:
    """Distance from x mod m to the nearest multiple of m (0 ≤ result ≤ m/2)."""
    r = np.mod(x, modulus)
    return np.minimum(r, modulus - r)


def phase_fold(times: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in [−0.5, 0.5): the fractional part of (t − t0)/P, centred on transit."""
    return np.mod(np.asarray(times) - t0 + 0.5 * period, period) / period - 0.5


def interval_table(times: np.ndarray, period: float, tol: float) -> list[dict[str, Any]]:
    """For each consecutive pair: Δ, Δ/P, nearest integer n, remainder, verdict."""
    t = np.sort(np.asarray(times, dtype=float))
    rows = []
    for i in range(len(t) - 1):
        d = t[i + 1] - t[i]
        n, r = nearest_multiple(d, period)
        rows.append(
            {
                "pair": f"E{i + 1}→E{i + 2}",
                "delta_days": d,
                "ratio": d / period,
                "n": n,
                "residual_days": r,
                "residual_hours": r * 24,
                "consistent": abs(r) <= tol,
                "missed_between": n - 1 if abs(r) <= tol else None,
            }
        )
    return rows


# ---------------------------------------------------------------- exact integers
def euclid_steps(a: int, b: int) -> tuple[list[dict[str, int]], int]:
    """Euclidean algorithm: a = q·b + r repeated until r = 0. Returns (steps, gcd)."""
    a, b = abs(int(a)), abs(int(b))
    if a < b:
        a, b = b, a
    steps = []
    while b:
        q, r = divmod(a, b)
        steps.append({"a": a, "b": b, "q": q, "r": r})
        a, b = b, r
    return steps, a


def gcd_list(values: list[int]) -> int:
    g = 0
    for v in values:
        g = math.gcd(g, int(v))
    return g


def modular_residues(times: np.ndarray, period: float, t0: float, unit_days: float) -> dict[str, Any]:
    """Discretise times into integer units and reduce them modulo the period.

    T_i = round((t_i − t0)/unit),  m = round(P/unit),  r_i = T_i mod m.
    A perfectly periodic sequence gives r_i = 0 for all i (all congruent).
    """
    if unit_days <= 0 or period <= 0:
        raise ValueError("unit and period must be positive")
    T = np.round((np.asarray(times) - t0) / unit_days).astype(int)
    m = int(round(period / unit_days))
    if m < 1:
        raise ValueError("period is shorter than one time unit")
    r = np.mod(T, m)
    return {"T": T, "modulus": m, "residues": r, "circular": np.minimum(r, m - r)}


def rational_approximation(x: float, max_denominator: int = 6) -> tuple[Fraction, float]:
    """Best fraction p/q (q ≤ max_denominator) to x, and its relative error."""
    frac = Fraction(x).limit_denominator(max_denominator)
    return frac, abs(float(frac) - x) / abs(x) if x else 0.0


def alias_relation(p_a: float, p_b: float, tol: float = 0.02, max_denominator: int = 4) -> dict[str, Any]:
    """Describe how two period estimates relate: equal, harmonic (P_b ≈ p/q · P_a) or unrelated."""
    ratio = p_b / p_a
    frac, err = rational_approximation(ratio, max_denominator)
    related = err <= tol
    if related and frac == 1:
        kind = "equal"
    elif related:
        kind = "harmonic"
    else:
        kind = "unrelated"
    return {"ratio": ratio, "fraction": frac, "relative_error": err, "kind": kind}


# ---------------------------------------------------------------- period estimation
@dataclass
class PeriodEstimate:
    period: float
    t0: float
    period_err: float
    n_events: int
    members: list[int]  # indices (into the sorted times) of the largest congruence class
    epochs: list[int]  # integer cycle numbers of the members
    residuals_days: list[float]  # observed − calculated (O−C) for the members
    ambiguous: bool = False  # True when only two events: P could be Δ/k for any k
    note: str = ""
    candidates_tested: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_members(self) -> int:
        return len(self.members)


def fit_ephemeris(times: np.ndarray, epochs: np.ndarray) -> dict[str, float | np.ndarray]:
    """Least-squares line t = t0 + n·P through (epoch, time) points."""
    t = np.asarray(times, dtype=float)
    n = np.asarray(epochs, dtype=float)
    if len(np.unique(n)) < 2:
        raise ValueError("need at least two distinct epochs to fit an ephemeris")
    A = np.vstack([np.ones_like(n), n]).T
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    t0, period = float(coef[0]), float(coef[1])
    resid = t - (t0 + period * n)
    dof = max(len(t) - 2, 1)
    s2 = float(np.sum(resid**2) / dof) if len(t) > 2 else 0.0
    cov = s2 * np.linalg.inv(A.T @ A)
    return {"t0": t0, "period": period, "t0_err": math.sqrt(cov[0, 0]),
            "period_err": math.sqrt(cov[1, 1]), "residuals": resid}


def largest_congruence_class(times: np.ndarray, period: float, tol: float) -> np.ndarray:
    """Indices of the largest set of times that are mutually ≡ (mod P) within tol."""
    t = np.asarray(times, dtype=float)
    d = circular_distance(t[:, None] - t[None, :], period)  # d[i, j]: is t_i ≡ t_j ?
    close = d <= tol
    best = int(np.argmax(close.sum(axis=0)))
    return np.where(close[:, best])[0]


def estimate_period(times: np.ndarray, tol: float, min_period: float | None = None,
                    max_period: float | None = None, max_multiple: int = 10,
                    max_pairs_per_event: int = 4) -> PeriodEstimate | None:
    """Approximate-GCD period estimate robust to missed transits and false events.

    1. Candidates: every pairwise difference divided by k = 1..max_multiple
       (Δ/k covers the case where k−1 transits between the pair were missed).
    2. Score: size of the largest congruence class modulo the candidate,
       i.e. how many events line up on one periodic "comb".
    3. Choose the highest score; among ties choose the *largest* period
       (P/2, P/3 ... also divide everything — like 1 dividing every integer).
    4. Assign integer epochs to the class members and refine P, t0 by a
       least-squares fit (O−C residuals).

    Periods below 4·tol are excluded: a tiny period "divides" anything
    within the tolerance and carries no information.
    """
    t = np.sort(np.asarray(times, dtype=float))
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return None
    baseline = t[-1] - t[0]
    lo = max(min_period or 0.0, 4 * tol)
    hi = max_period or baseline
    if baseline < lo:
        return None

    if len(t) == 2:
        d = baseline
        return PeriodEstimate(period=d, t0=t[0], period_err=float("nan"), n_events=2, members=[0, 1],
                              epochs=[0, 1], residuals_days=[0.0, 0.0], ambiguous=True,
                              note="Only two events: the period may be Δ or any Δ/k (missed transits).")

    diffs = []
    for i in range(len(t)):
        for j in range(i + 1, min(len(t), i + 1 + max_pairs_per_event)):
            diffs.append(t[j] - t[i])
    diffs = np.asarray(diffs)
    cands = (diffs[:, None] / np.arange(1, max_multiple + 1)[None, :]).ravel()
    cands = cands[(cands >= lo) & (cands <= hi)]
    if len(cands) == 0:
        return None
    # de-duplicate candidates closer than a small fraction of the tolerance
    cands = np.unique(np.round(cands / (tol / 20)) * (tol / 20))

    diff = t[:, None] - t[None, :]
    scores = np.zeros(len(cands))
    sizes = np.zeros(len(cands), dtype=int)
    for k, c in enumerate(cands):
        close = circular_distance(diff, c) <= tol  # column j: events congruent to t_j
        m = close.sum(axis=0)
        ep = np.round(diff / c)
        span = np.where(close, ep, -np.inf).max(axis=0) - np.where(close, ep, np.inf).min(axis=0) + 1
        # members × fill-fraction: a sub-harmonic P/k can absorb a stray event,
        # but leaves k−1 empty epochs between real transits and scores lower.
        col_scores = m * (m / span)
        j = int(np.argmax(col_scores))
        scores[k], sizes[k] = col_scores[j], m[j]
    if sizes.max() < 3:
        return PeriodEstimate(period=float(np.median(np.diff(t))), t0=t[0], period_err=float("nan"),
                              n_events=len(t), members=[], epochs=[], residuals_days=[], ambiguous=True,
                              note="No three events share a common period within the tolerance.",
                              candidates_tested=len(cands))
    valid = sizes >= 3
    best_score = scores[valid].max()
    # Among (near-)ties choose the largest candidate (avoids sub-harmonics P/2,
    # P/3 ...). Ties can include values slightly above the true period that
    # still fit within the tolerance; the least-squares refinement corrects that.
    tied = valid & (scores >= best_score * (1 - 1e-9))
    p_guess = float(cands[tied].max())
    members = largest_congruence_class(t, p_guess, tol)
    t0_guess = t[members[0]]
    epochs = np.round((t[members] - t0_guess) / p_guess).astype(int)
    # Two events in one epoch (a split detection) -> keep the first only.
    _, first = np.unique(epochs, return_index=True)
    members, epochs = members[first], epochs[first]
    fit = fit_ephemeris(t[members], epochs)
    epochs = epochs - epochs.min()
    fit = fit_ephemeris(t[members], epochs)
    return PeriodEstimate(
        period=fit["period"], t0=fit["t0"], period_err=fit["period_err"], n_events=len(t),
        members=[int(m) for m in members], epochs=[int(e) for e in epochs],
        residuals_days=[float(r) for r in fit["residuals"]],
        note=f"{len(members)} of {len(t)} events fit one periodic sequence.",
        candidates_tested=len(cands),
        extra={"initial_guess": p_guess, "best_score": round(float(best_score), 3)},
    )
