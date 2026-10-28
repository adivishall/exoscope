"""Graph theory: a network of transit events linked by periodic consistency.

G = (V, E) where V = detected events and, for a trial period P, tolerance ε
and maximum multiple N:

    {u, v} ∈ E   ⇔   ∃ n ∈ {1, …, N} :  | |t_v − t_u| − n·P | ≤ ε
                      (and, optionally, depths agree within a ratio r)

i.e. u and v could be transits of the *same* periodic object, n cycles
apart. Consequences students can explore:

* Connected components group events that are linked by chains of
  consistent intervals; isolated vertices are events no other event
  supports (often noise).
* A clique is a set of events that are *pairwise* consistent — stronger
  evidence than a component.
* The relation "is consistent with" is reflexive and symmetric but NOT
  necessarily transitive (small timing errors add up), so it is not an
  equivalence relation. Connected components are the equivalence classes of
  its transitive closure.

Graph connectivity is evidence of periodicity, not proof that a planet exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import networkx as nx
import numpy as np

from .number_theory import nearest_multiple


@dataclass
class EdgeRule:
    period: float  # days
    tolerance_hours: float = 2.0
    max_multiple: int = 10
    depth_ratio: float | None = None  # e.g. 2.0: deeper/shallower must be ≤ 2

    def validate(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.tolerance_hours <= 0:
            raise ValueError("tolerance must be positive")
        if self.tolerance_hours / 24 >= self.period / 2:
            raise ValueError("tolerance must be smaller than half the period")
        if not 1 <= self.max_multiple <= 1000:
            raise ValueError("maximum multiple must be between 1 and 1000")
        if self.depth_ratio is not None and self.depth_ratio < 1:
            raise ValueError("depth ratio must be ≥ 1")

    def describe(self) -> str:
        s = (f"u–v is an edge when |Δt − n·P| ≤ {self.tolerance_hours:g} h for some integer "
             f"1 ≤ n ≤ {self.max_multiple}, with P = {self.period:.5f} d")
        if self.depth_ratio:
            s += f", and the deeper event is at most {self.depth_ratio:g}× the shallower one"
        return s


def build_event_graph(events: Sequence, rule: EdgeRule) -> nx.Graph:
    """Vertices = events (objects with event_id, t_center, depth); edges by ``rule``."""
    rule.validate()
    tol = rule.tolerance_hours / 24
    G = nx.Graph()
    for e in events:
        G.add_node(e.event_id, t=float(e.t_center), depth=float(e.depth), method=getattr(e, "method", ""))
    for u, v in combinations(events, 2):
        dt = abs(v.t_center - u.t_center)
        n, r = nearest_multiple(dt, rule.period)
        if n > rule.max_multiple or abs(r) > tol:
            continue
        if rule.depth_ratio is not None:
            d1, d2 = sorted([max(u.depth, 1e-12), max(v.depth, 1e-12)])
            if d2 / d1 > rule.depth_ratio:
                continue
        G.add_edge(u.event_id, v.event_id, dt=dt, n=n, residual_hours=r * 24)
    return G


@dataclass
class GraphSummary:
    n_vertices: int
    n_edges: int
    density: float
    components: list[list[str]]  # sorted largest first; each sorted by time
    isolated: list[str]
    max_clique: list[str]
    degrees: dict[str, int]
    is_transitive: bool
    transitivity_counterexample: tuple[str, str, str] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "|V|": self.n_vertices, "|E|": self.n_edges, "density": self.density,
            "components": len(self.components),
            "largest_component": len(self.components[0]) if self.components else 0,
            "isolated_vertices": len(self.isolated), "max_clique_size": len(self.max_clique),
            "consistency_relation_transitive": self.is_transitive,
        }


def transitivity_counterexample(G: nx.Graph) -> tuple[str, str, str] | None:
    """Find u–v and v–w edges with no u–w edge (relation not transitive)."""
    for v in G.nodes:
        nbrs = list(G.neighbors(v))
        for u, w in combinations(nbrs, 2):
            if not G.has_edge(u, w):
                return (u, v, w)
    return None


def summarize_graph(G: nx.Graph, clique_limit: int = 60) -> GraphSummary:
    by_time = lambda nodes: sorted(nodes, key=lambda k: G.nodes[k]["t"])  # noqa: E731
    comps = sorted((by_time(c) for c in nx.connected_components(G)), key=len, reverse=True)
    # Maximum clique is NP-hard in general; fine for the small event graphs here.
    if G.number_of_nodes() <= clique_limit:
        max_clique = max(nx.find_cliques(G), key=len, default=[])
    else:
        max_clique = list(nx.algorithms.approximation.max_clique(G))
    counter = transitivity_counterexample(G)
    return GraphSummary(
        n_vertices=G.number_of_nodes(), n_edges=G.number_of_edges(), density=nx.density(G) if len(G) > 1 else 0.0,
        components=comps, isolated=by_time(n for n in G.nodes if G.degree(n) == 0),
        max_clique=by_time(max_clique), degrees=dict(G.degree()),
        is_transitive=counter is None, transitivity_counterexample=counter,
    )


def adjacency_matrix(G: nx.Graph) -> tuple[list[str], np.ndarray]:
    nodes = sorted(G.nodes, key=lambda k: G.nodes[k]["t"])
    return nodes, nx.to_numpy_array(G, nodelist=nodes, dtype=int)


def component_period(G: nx.Graph, component: Sequence[str], period: float) -> dict[str, float] | None:
    """Refit P using only the events of one component (integer epochs + least squares)."""
    from .number_theory import fit_ephemeris

    if len(component) < 3:
        return None
    t = np.sort([G.nodes[k]["t"] for k in component])
    epochs = np.round((t - t[0]) / period).astype(int)
    if len(np.unique(epochs)) < 2:
        return None
    fit = fit_ephemeris(t, epochs)
    return {"period": fit["period"], "period_err": fit["period_err"], "t0": fit["t0"],
            "rms_hours": float(np.sqrt(np.mean(fit["residuals"] ** 2)) * 24)}
