import numpy as np
import pytest

from exoscope.dmath.number_theory import (alias_relation, circular_distance, estimate_period, euclid_steps, fit_ephemeris,
                                          gcd_list, interval_table, is_approx_multiple, modular_residues,
                                          nearest_multiple, phase_fold, rational_approximation)


def test_nearest_multiple_and_divisibility():
    assert nearest_multiple(7.05, 3.52) == (2, pytest.approx(0.01))
    assert is_approx_multiple(10.6, 3.52, 0.1)  # 3 × 3.52 = 10.56
    assert not is_approx_multiple(5.3, 3.52, 0.1)
    with pytest.raises(ValueError):
        nearest_multiple(1.0, 0)


def test_interval_table_reports_missed_transits():
    rows = interval_table([1.0, 4.0, 10.0, 13.0], 3.0, 0.05)
    assert [r["n"] for r in rows] == [1, 2, 1]
    assert [r["missed_between"] for r in rows] == [0, 1, 0]
    assert all(r["consistent"] for r in rows)


def test_euclid_and_gcd():
    steps, g = euclid_steps(48, 18)
    assert g == 6
    assert steps[0] == {"a": 48, "b": 18, "q": 2, "r": 12}
    assert gcd_list([24, 48, 72]) == 24
    assert gcd_list([24, 23, 48]) == 1  # noise destroys exact GCDs


def test_modular_residues_congruent():
    t = 2.0 + 3.0 * np.array([0, 1, 2, 5])
    res = modular_residues(t, 3.0, 2.0, 1 / 24)
    assert res["modulus"] == 72
    assert np.all(res["residues"] == 0)


def test_circular_distance_and_phase():
    assert circular_distance(2.9, 3.0) == pytest.approx(0.1)
    assert phase_fold(np.array([5.0]), 3.0, 2.0)[0] == pytest.approx(0.0)


@pytest.mark.parametrize("period", [0.84, 3.0, 7.3])
def test_estimate_period_with_missed_and_false_events(period):
    rng = np.random.default_rng(0)
    epochs = np.array([0, 1, 2, 4, 5, 7, 8, 9])  # transits 3 and 6 missed
    t = 10 + epochs * period + rng.normal(0, 0.005, len(epochs))
    t = np.sort(np.append(t, [10 + 2.37 * period]))  # one false event
    est = estimate_period(t, tol=0.05)
    assert est is not None and not est.ambiguous
    assert est.period == pytest.approx(period, rel=2e-3)
    assert est.n_members == len(epochs)
    assert sorted(np.diff(sorted(est.epochs))) == sorted(np.diff(epochs))


def test_estimate_period_prefers_fundamental_over_subharmonic():
    t = 1.2 + 3.0 * np.arange(8)
    est = estimate_period(t, tol=0.05)
    assert est.period == pytest.approx(3.0, rel=1e-6)


def test_estimate_period_edge_cases():
    assert estimate_period(np.array([1.0]), 0.1) is None
    two = estimate_period(np.array([1.0, 4.0]), 0.1)
    assert two.ambiguous and two.period == pytest.approx(3.0)
    random_times = np.array([0.0, 1.7, 5.3, 11.9])
    est = estimate_period(random_times, 0.01)
    assert est is None or est.ambiguous or est.n_members >= 3


def test_fit_ephemeris_exact():
    fit = fit_ephemeris(np.array([2.0, 5.0, 11.0]), np.array([0, 1, 3]))
    assert fit["period"] == pytest.approx(3.0)
    assert fit["t0"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        fit_ephemeris(np.array([1.0, 1.1]), np.array([0, 0]))


def test_rational_approximation_alias():
    frac, err = rational_approximation(0.4999)
    assert str(frac) == "1/2" and err < 1e-3
    assert alias_relation(3.0, 6.01)["kind"] == "harmonic"
    assert alias_relation(3.0, 3.001)["kind"] == "equal"
    assert alias_relation(3.0, 4.37)["kind"] == "unrelated"
