from types import SimpleNamespace as E

import pytest

from exoscope.dmath.graphs import (EdgeRule, adjacency_matrix, build_event_graph, component_period, summarize_graph,
                                   transitivity_counterexample)


def events(times, depth=0.01):
    return [E(event_id=f"E{i + 1}", t_center=t, depth=depth) for i, t in enumerate(times)]


def test_periodic_events_form_one_clique_and_noise_is_isolated():
    G = build_event_graph(events([1.0, 4.0, 7.0, 13.0, 8.6]), EdgeRule(3.0, tolerance_hours=1.0))
    s = summarize_graph(G)
    assert s.n_vertices == 5
    assert s.isolated == ["E5"]
    assert sorted(s.max_clique) == ["E1", "E2", "E3", "E4"]
    assert s.n_edges == 6  # C(4, 2)
    assert G.edges["E1", "E4"]["n"] == 4


def test_tolerance_relation_not_transitive():
    # gaps 3.06 and 3.06 are each within 0.08 d of P = 3, but 6.12 is 0.12 d from 2P
    G = build_event_graph(events([0.0, 3.06, 6.12]), EdgeRule(3.0, tolerance_hours=0.08 * 24))
    s = summarize_graph(G)
    assert len(s.components) == 1 and len(s.max_clique) == 2
    assert not s.is_transitive
    assert transitivity_counterexample(G) == ("E1", "E2", "E3")


def test_depth_rule_removes_edges():
    ev = events([1.0, 4.0])
    ev[1].depth = 0.05
    assert build_event_graph(ev, EdgeRule(3.0, 1.0)).number_of_edges() == 1
    assert build_event_graph(ev, EdgeRule(3.0, 1.0, depth_ratio=2.0)).number_of_edges() == 0


def test_adjacency_symmetric_and_component_refit():
    G = build_event_graph(events([1.0, 4.01, 6.99, 10.0]), EdgeRule(3.0, 2.0))
    nodes, M = adjacency_matrix(G)
    assert (M == M.T).all() and M.trace() == 0
    fit = component_period(G, nodes, 3.0)
    assert fit["period"] == pytest.approx(3.0, abs=0.01)


def test_rule_validation():
    with pytest.raises(ValueError):
        EdgeRule(1.0, tolerance_hours=13).validate()
    with pytest.raises(ValueError):
        EdgeRule(-1.0).validate()
    assert summarize_graph(build_event_graph([], EdgeRule(1.0))).n_vertices == 0
