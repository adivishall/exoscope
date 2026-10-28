from types import SimpleNamespace as E

import pytest

from exoscope.dmath.sets import compare_detections, detection_metrics, match_events, roster


def ev(i, t, m="threshold"):
    return E(event_id=f"{'T' if m == 'threshold' else 'B'}{i}", t_center=t)


def test_match_events_is_one_to_one_within_tolerance():
    pairs = match_events([1.0, 1.05, 5.0], [1.02, 9.0], tol=0.1)
    assert len(pairs) == 1
    i, j, dt = pairs[0]
    assert (i, j) in {(0, 0), (1, 0)}
    assert abs(dt) <= 0.1
    assert match_events([], [1.0], 0.1) == []


def test_set_operations_and_universe():
    A = [ev(1, 1.0), ev(2, 4.0), ev(3, 6.5)]
    B = [ev(1, 1.02, "bls"), ev(2, 4.01, "bls"), ev(3, 7.0, "bls")]
    comp = compare_detections(A, B, bls_gap_times=[10.0], tolerance_hours=2.0)
    s = comp.summary()
    assert (s["|A|"], s["|B|"], s["|A∩B|"], s["|A−B|"], s["|B−A|"]) == (3, 3, 2, 1, 1)
    assert s["|U|"] == 5 and s["|(A∪B)ᶜ|"] == 1
    assert comp.inclusion_exclusion()["holds"]
    assert comp.jaccard() == pytest.approx(2 / 4)
    assert comp.symmetric_difference() == comp.a_minus_b() | comp.b_minus_a()
    assert comp.complement(comp.union()) == frozenset(k for k, e in comp.elements.items() if e.in_gap)
    # De Morgan: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
    assert comp.complement(comp.union()) == comp.complement(comp.A) & comp.complement(comp.B)


def test_tolerance_changes_membership():
    A, B = [ev(1, 1.0)], [ev(1, 1.1, "bls")]  # 2.4 h apart
    assert len(compare_detections(A, B, tolerance_hours=2.0).intersection()) == 0
    assert len(compare_detections(A, B, tolerance_hours=3.0).intersection()) == 1
    with pytest.raises(ValueError):
        compare_detections(A, B, tolerance_hours=0)


def test_without_bls():
    comp = compare_detections([ev(1, 1.0)], [], tolerance_hours=1.0)
    assert not comp.bls_available and len(comp.A) == 1 and not comp.B


def test_detection_metrics():
    m = detection_metrics([1.0, 4.0, 8.0], [1.01, 4.02, 7.0, 10.0], tol=0.05)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)
    assert detection_metrics([], [1.0], 0.1)["f1"] == 0.0


def test_roster():
    assert roster([]) == "∅"
    assert roster(["a", "b"]) == "{a, b}"
    assert "+2" in roster(list("abcde"), limit=3)
