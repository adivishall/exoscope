import math

import pytest

from exoscope.dmath.combinatorics import Constraint, Parameter, ParameterSpace, n_choose_k, select_configs
from exoscope.experiments import build_space, run_configurations
from exoscope.pipeline import default_settings_for


def space():
    return ParameterSpace([Parameter("a", "A", [1, 2, 3], "threshold"), Parameter("b", "B", ["x", "y"], "preprocess"),
                           Parameter("c", "C", [0.5, 1.0], "preprocess")],
                          [Constraint("not (3, y)", lambda c: not (c["a"] == 3 and c["b"] == "y"))])


def test_rule_of_product_and_constraints():
    s = space()
    assert s.size() == 12 == len(list(s.enumerate()))
    assert s.formula() == "3 × 2 × 2 = 12"
    assert len(s.valid_configs()) == 10
    assert s.projection_size("preprocess") == 4
    assert s.projection_size(["preprocess", "threshold"]) == 12


def test_choose_and_select():
    assert n_choose_k(24, 10) == math.comb(24, 10) and n_choose_k(3, 5) == 0
    configs = space().valid_configs()
    assert select_configs(configs, 3, "first") == configs[:3]
    r1 = select_configs(configs, 4, "random", seed=1)
    assert r1 == select_configs(configs, 4, "random", seed=1) and len(r1) == 4
    with pytest.raises(ValueError):
        select_configs(configs, 0)


def test_space_validation():
    with pytest.raises(ValueError):
        ParameterSpace([Parameter("a", "A", [1, 1], "x")]).validate()
    with pytest.raises(ValueError):
        ParameterSpace([Parameter("a", "A", [], "x")]).validate()


def test_experiment_runner_memoises_by_projection(synth):
    base = default_settings_for(synth)
    sp = build_space({"threshold.sigma_threshold": [3.0, 4.0], "threshold.min_points": [1, 2],
                      "preprocess.flatten_window_days": [1.0, 2.0]}, base)
    configs = sp.valid_configs()
    assert len(configs) == 8
    df, stats = run_configurations(synth, base, sp, configs)
    assert stats == {"configs_run": 8, "preprocessing_runs": 2, "bls_runs": 2}
    assert (df["status"] == "ok").all()
    assert df["bls_period_error_%"].max() < 1.0


def test_constraint_excludes_short_flatten_window(synth):
    base = default_settings_for(synth)
    sp = build_space({"preprocess.flatten_window_days": [0.25, 1.0]}, base)
    assert len(sp.valid_configs()) == 1  # 0.25 d = 6 h < 3 × 6 h longest duration
