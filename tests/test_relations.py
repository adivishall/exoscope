from types import SimpleNamespace as E

import numpy as np
import pytest

from exoscope.dmath.relations import (Relation, consistency_relation, event_epoch_function, example_relations,
                                      observation_flux_relation, precedes_relation)


def test_example_types_and_function_checks():
    ex = example_relations()
    assert ex["one-to-one"].classify() == "one-to-one" and ex["one-to-one"].is_bijective
    assert ex["one-to-many"].classify() == "one-to-many" and not ex["one-to-many"].is_function
    assert ex["many-to-one"].classify() == "many-to-one" and ex["many-to-one"].is_function
    assert not ex["many-to-one"].is_injective
    assert ex["many-to-many"].classify() == "many-to-many" and not ex["many-to-many"].is_function


def test_non_total_relation_is_not_function():
    r = Relation("r", ["a", "b"], [1], {("a", 1)})
    assert not r.is_function
    assert any("no image" in v for v in r.function_violations())


def test_pairs_outside_product_rejected():
    with pytest.raises(ValueError):
        Relation("bad", ["a"], [1], {("z", 1)})


def test_observation_flux_is_function_not_surjective():
    r = observation_flux_relation(np.arange(10.0), np.linspace(1, 2, 10), n=5)
    assert r.is_function and r.is_injective and not r.is_surjective


def test_event_epoch_function_detects_split_event():
    ev = [E(event_id="T1", t_center=0.0), E(event_id="T2", t_center=3.0), E(event_id="T3", t_center=3.05)]
    g = event_epoch_function(ev, 3.0, 0.0)
    assert g.is_function and not g.is_injective


def test_homogeneous_properties():
    ev = [E(event_id=f"E{i}", t_center=t) for i, t in enumerate([0.0, 3.06, 6.12])]
    sim = consistency_relation(ev, 3.0, tol_hours=0.08 * 24)
    p = sim.homogeneous_properties()
    assert p["reflexive"] and p["symmetric"] and not p["transitive"] and not p["equivalence"]
    assert sim.transitive_counterexample() is not None
    order = precedes_relation(ev).homogeneous_properties()
    assert order["strict_order"] and order["irreflexive"] and not order["symmetric"]


def test_composition_and_matrix():
    f = Relation("f", ["a", "b"], ["x", "y"], {("a", "x"), ("b", "y")})
    g = Relation("g", ["x", "y"], [1], {("x", 1), ("y", 1)})
    h = f.compose(g)
    assert h.pairs == {("a", 1), ("b", 1)} and h.is_function
    assert f.matrix().tolist() == [[1, 0], [0, 1]]
