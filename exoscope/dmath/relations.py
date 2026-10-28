"""Relations and functions describing the analysis pipeline.

A relation R from X to Y is any subset of the Cartesian product X × Y
(a set of ordered pairs). R is a *function* f: X → Y exactly when every
x ∈ X appears in exactly one pair. Examples from ExoScope:

* observation → flux: each timestamp has one measured flux → a function
  (after duplicate timestamps are removed during preprocessing).
* event ↔ algorithm: an event can be found by both detectors and each
  detector finds many events → a many-to-many relation, NOT a function.
* event → epoch number: for fixed P and t0, every event gets one integer
  cycle number → a function into ℤ; injective unless two events share an epoch.
* configuration → number of detections: deterministic → a function, usually
  many-to-one (different settings give the same count).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Sequence

import numpy as np


@dataclass
class Relation:
    name: str
    domain: list[Hashable]
    codomain: list[Hashable]
    pairs: set[tuple[Hashable, Hashable]]
    domain_label: str = "X"
    codomain_label: str = "Y"
    codomain_note: str = ""  # e.g. "codomain is ℝ; only the attained values are listed"
    infinite_codomain: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        dom, cod = set(self.domain), set(self.codomain)
        bad = [(x, y) for x, y in self.pairs if x not in dom or (not self.infinite_codomain and y not in cod)]
        if bad:
            raise ValueError(f"pairs outside X × Y: {bad[:3]}")

    # --- images ---------------------------------------------------------
    def image(self, x: Hashable) -> list[Hashable]:
        return sorted((y for a, y in self.pairs if a == x), key=str)

    def preimage(self, y: Hashable) -> list[Hashable]:
        return sorted((a for a, b in self.pairs if b == y), key=str)

    def out_degree(self) -> dict[Hashable, int]:
        d = {x: 0 for x in self.domain}
        for x, _ in self.pairs:
            d[x] += 1
        return d

    def in_degree(self) -> dict[Hashable, int]:
        d: dict[Hashable, int] = defaultdict(int)
        for y in self.codomain:
            d[y] = 0
        for _, y in self.pairs:
            d[y] += 1
        return dict(d)

    # --- function properties ---------------------------------------------
    def function_violations(self) -> list[str]:
        """Reasons R fails to be a function (empty list = it is a function)."""
        out = []
        for x, k in self.out_degree().items():
            if k == 0:
                out.append(f"{x} has no image (R is not total)")
            elif k > 1:
                out.append(f"{x} is related to {k} elements: {', '.join(map(str, self.image(x)))}")
        return out

    @property
    def is_function(self) -> bool:
        return not self.function_violations()

    @property
    def is_injective(self) -> bool:
        return self.is_function and all(k <= 1 for k in self.in_degree().values())

    @property
    def is_surjective(self) -> bool:
        if self.infinite_codomain:
            return False  # finitely many observations cannot cover an infinite codomain
        return self.is_function and all(k >= 1 for k in self.in_degree().values())

    @property
    def is_bijective(self) -> bool:
        return self.is_injective and self.is_surjective

    def classify(self) -> str:
        """one-to-one, one-to-many, many-to-one or many-to-many (by max degrees)."""
        out_max = max(self.out_degree().values(), default=0)
        in_max = max(self.in_degree().values(), default=0)
        many_out, many_in = out_max > 1, in_max > 1
        if not many_out and not many_in:
            return "one-to-one"
        if many_out and not many_in:
            return "one-to-many"
        if many_in and not many_out:
            return "many-to-one"
        return "many-to-many"

    def matrix(self) -> np.ndarray:
        """Boolean relation matrix M[i, j] = 1 ⇔ (x_i, y_j) ∈ R."""
        xi = {x: i for i, x in enumerate(self.domain)}
        yj = {y: j for j, y in enumerate(self.codomain)}
        m = np.zeros((len(self.domain), len(self.codomain)), dtype=int)
        for x, y in self.pairs:
            if y in yj:
                m[xi[x], yj[y]] = 1
        return m

    def properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "type": self.classify(),
            "function": self.is_function,
            "injective": self.is_injective,
            "surjective": self.is_surjective,
            "bijective": self.is_bijective,
            "|R|": len(self.pairs),
        }
        if self.is_homogeneous:
            props.update(self.homogeneous_properties())
        return props

    # --- relations on a single set -------------------------------------------
    @property
    def is_homogeneous(self) -> bool:
        return set(self.domain) == set(self.codomain) and not self.infinite_codomain

    def homogeneous_properties(self) -> dict[str, bool]:
        X, R = self.domain, self.pairs
        reflexive = all((x, x) in R for x in X)
        irreflexive = all((x, x) not in R for x in X)
        symmetric = all((y, x) in R for x, y in R)
        antisymmetric = all(not ((y, x) in R and x != y) for x, y in R)
        succ = defaultdict(set)
        for x, y in R:
            succ[x].add(y)
        transitive = all(z in succ[x] for x, y in R for z in succ[y])
        return {
            "reflexive": reflexive, "irreflexive": irreflexive, "symmetric": symmetric,
            "antisymmetric": antisymmetric, "transitive": transitive,
            "equivalence": reflexive and symmetric and transitive,
            "partial_order": reflexive and antisymmetric and transitive,
            "strict_order": irreflexive and antisymmetric and transitive,
        }

    def transitive_counterexample(self) -> tuple | None:
        succ = defaultdict(set)
        for x, y in self.pairs:
            succ[x].add(y)
        for x, y in self.pairs:
            for z in succ[y]:
                if z not in succ[x]:
                    return (x, y, z)
        return None

    def compose(self, other: "Relation") -> "Relation":
        """other ∘ self: (x, z) whenever (x, y) ∈ self and (y, z) ∈ other."""
        succ = defaultdict(set)
        for y, z in other.pairs:
            succ[y].add(z)
        pairs = {(x, z) for x, y in self.pairs for z in succ[y]}
        return Relation(f"{other.name} ∘ {self.name}", self.domain, other.codomain, pairs,
                        self.domain_label, other.codomain_label, infinite_codomain=other.infinite_codomain)

    def ordered_pairs(self, limit: int = 30) -> str:
        items = sorted(self.pairs, key=lambda p: (str(p[0]), str(p[1])))
        text = ", ".join(f"({x}, {y})" for x, y in items[:limit])
        return "{" + text + (f", … (+{len(items) - limit})" if len(items) > limit else "") + "}"


# ------------------------------------------------------------------ builders
def observation_flux_relation(time: np.ndarray, flux: np.ndarray, n: int = 8, start: int = 0) -> Relation:
    """f: observation → flux for ``n`` consecutive cadences (a function)."""
    idx = range(start, min(start + n, len(time)))
    dom = [f"t={time[i]:.4f}" for i in idx]
    pairs = {(f"t={time[i]:.4f}", round(float(flux[i]), 6)) for i in idx}
    cod = sorted({y for _, y in pairs})
    return Relation("f: Observation → Flux", dom, cod, pairs, "Observation times", "Relative flux",
                    codomain_note="Codomain is ℝ; only the attained values (the image) are shown.",
                    infinite_codomain=True)


def event_algorithm_relation(comparison) -> Relation:
    """R ⊆ Events × {Threshold, BLS} from a set comparison (many-to-many)."""
    dom, pairs = [], set()
    for k in comparison.sorted_ids(comparison.union()):
        e = comparison.elements[k]
        dom.append(e.label)
        if k in comparison.A:
            pairs.add((e.label, "Threshold"))
        if k in comparison.B:
            pairs.add((e.label, "BLS"))
    return Relation("R: Event ↔ Algorithm", dom, ["Threshold", "BLS"], pairs, "Detected events", "Algorithms")


def event_epoch_function(events: Sequence, period: float, t0: float) -> Relation:
    """g: event → epoch number n = round((t − t0)/P) ∈ ℤ."""
    dom = [e.event_id for e in events]
    pairs = {(e.event_id, int(round((e.t_center - t0) / period))) for e in events}
    cod = sorted({n for _, n in pairs})
    return Relation("g: Event → Epoch", dom, cod, pairs, "Events", "Epoch numbers (ℤ)",
                    codomain_note="Codomain is ℤ; only the attained epochs are shown.", infinite_codomain=True)


def interval_function(events: Sequence, limit: int = 10) -> Relation:
    """h: (Eᵢ, Eⱼ), i<j → Δt (hours, rounded) on consecutive pairs."""
    ev = sorted(events, key=lambda e: e.t_center)[: limit + 1]
    dom, pairs = [], set()
    for a, b in zip(ev[:-1], ev[1:]):
        key = f"({a.event_id},{b.event_id})"
        dom.append(key)
        pairs.add((key, round((b.t_center - a.t_center) * 24, 2)))
    cod = sorted({y for _, y in pairs})
    return Relation("h: Event pair → Δt (hours)", dom, cod, pairs, "Consecutive event pairs", "Δt in hours",
                    infinite_codomain=True, codomain_note="Codomain is ℝ⁺; only attained values are shown.")


def config_count_function(configs: Sequence[str], counts: Sequence[int]) -> Relation:
    dom = list(configs)
    pairs = {(c, int(n)) for c, n in zip(configs, counts)}
    cod = sorted({int(n) for n in counts})
    return Relation("k: Configuration → #detections", dom, cod, pairs, "Configurations", "Detection counts",
                    infinite_codomain=True, codomain_note="Codomain is ℕ; only the attained counts are shown.")


def homogeneous_relation(name: str, items: Sequence[str], pairs: Iterable[tuple[str, str]]) -> Relation:
    items = list(items)
    return Relation(name, items, items, set(pairs), "Events", "Events")


def consistency_relation(events: Sequence, period: float, tol_hours: float, max_multiple: int = 10) -> Relation:
    """u ~ v ⇔ u = v or |Δt − nP| ≤ ε for some 1 ≤ n ≤ N (reflexive, symmetric, maybe not transitive)."""
    from .number_theory import nearest_multiple

    ids = [e.event_id for e in events]
    pairs = {(i, i) for i in ids}
    for a in events:
        for b in events:
            if a is b:
                continue
            n, r = nearest_multiple(abs(b.t_center - a.t_center), period)
            if n <= max_multiple and abs(r) * 24 <= tol_hours:
                pairs.add((a.event_id, b.event_id))
    return homogeneous_relation("~: periodic consistency", ids, pairs)


def precedes_relation(events: Sequence) -> Relation:
    """u ≺ v ⇔ t_u < t_v (a strict total order)."""
    ids = [e.event_id for e in events]
    pairs = {(a.event_id, b.event_id) for a in events for b in events if a.t_center < b.t_center}
    return homogeneous_relation("≺: occurs before", ids, pairs)


def example_relations() -> dict[str, Relation]:
    """Small, hand-checkable examples of the four relationship types."""
    return {
        "one-to-one": Relation(
            "Planet → orbital period (days)",
            ["Kepler-8 b", "TrES-2 b", "WASP-18 b"], [3.5225, 2.4706, 0.9415],
            {("Kepler-8 b", 3.5225), ("TrES-2 b", 2.4706), ("WASP-18 b", 0.9415)},
            "Planets", "Periods",
            notes=["Every planet has exactly one period and no two share one → a bijection onto these values."],
        ),
        "one-to-many": Relation(
            "Star → its known planets",
            ["Kepler-10", "Kepler-8"], ["Kepler-10 b", "Kepler-10 c", "Kepler-8 b"],
            {("Kepler-10", "Kepler-10 b"), ("Kepler-10", "Kepler-10 c"), ("Kepler-8", "Kepler-8 b")},
            "Stars", "Planets",
            notes=["Kepler-10 is related to two planets, so this relation is NOT a function."],
        ),
        "many-to-one": Relation(
            "Planet → host star",
            ["Kepler-10 b", "Kepler-10 c", "Kepler-8 b"], ["Kepler-10", "Kepler-8"],
            {("Kepler-10 b", "Kepler-10"), ("Kepler-10 c", "Kepler-10"), ("Kepler-8 b", "Kepler-8")},
            "Planets", "Stars",
            notes=["Each planet has exactly one host → a function; two planets share a host → not injective."],
        ),
        "many-to-many": Relation(
            "Event ↔ detecting algorithm",
            ["E1", "E2", "E3", "E4"], ["Threshold", "BLS"],
            {("E1", "Threshold"), ("E1", "BLS"), ("E2", "BLS"), ("E3", "Threshold"), ("E3", "BLS"), ("E4", "Threshold")},
            "Events", "Algorithms",
            notes=["E1 is found by both algorithms and BLS finds several events → many-to-many, not a function."],
        ),
    }
