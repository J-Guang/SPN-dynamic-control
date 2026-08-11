"""Config validation: dt consistency and num_jumps wiring (review P1.2)."""
from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.bsde.schedule import (
    bound_curriculum_kwargs,
    effective_control_bound,
    learning_rate_at,
    lr_schedule_kwargs,
    resolve_training_updates,
)
from diffusion_based_policy.config import load_experiment
from diffusion_based_policy.policies.baselines import baseline_gradient
from diffusion_based_policy.sim.prelimit import simulate_prelimit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
EXPERIMENTS = {
    "crisscross": os.path.join(_ROOT, "configs/net_parameters/bsde_crisscross.yaml"),
    "pesic_williams": os.path.join(_ROOT, "configs/net_parameters/bsde_pesic_williams.yaml"),
    "three_station_bigstep": os.path.join(_ROOT, "configs/net_parameters/bsde_bigstep.yaml"),
}


def test_all_experiments_load_and_dt_consistent():
    for name, path in EXPERIMENTS.items():
        exp = load_experiment(path)
        b = exp.bsde
        if "dt" in b:
            assert abs(b["dt"] - b["horizon"] / b["num_steps"]) < 1e-9, name


def test_inconsistent_dt_rejected(tmp_path):
    cfg = {
        "name": "bad",
        "network": os.path.join(_ROOT, "configs/net_topology/crisscross.yaml"),
        "seed": 0,
        "bsde": {"horizon": 3.0, "num_steps": 64, "dt": 0.05},  # 3/64 = 0.046875
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="dt"):
        load_experiment(str(p))


def test_num_jumps_caps_simulation():
    """max_jumps (from prelimit.num_jumps) is honored as the simulation budget."""
    exp = load_experiment(EXPERIMENTS["three_station_bigstep"])
    model = exp.model()
    grad = baseline_gradient(model)
    res = simulate_prelimit(model, grad, num_paths=32, horizon=3.0, seed=1, max_jumps=50)
    assert res["jumps"] <= 50
    # with a tiny budget the simulated time should stop short of the full horizon
    assert res["max_sim_time"] <= res["T_orig"] + 1e-6


def test_prelimit_num_jumps_present_in_configs():
    for name, path in EXPERIMENTS.items():
        exp = load_experiment(path)
        assert "num_jumps" in exp.prelimit, f"{name}: prelimit.num_jumps missing"


def test_crisscross_pw_use_short_training_segment():
    """Criss-Cross/PW configs train on the short BSDE segment.

    These experiments train on T=0.01 with 64 substeps, then evaluate longer
    rollouts separately. Keeping this in config prevents accidentally training
    on a coarse T=3, N=64 segment.
    """
    expected_ref = {
        "crisscross": [-0.5, -0.5, -0.5],
        "pesic_williams": [-1.0, -1.0, -1.0],
    }
    for name in ("crisscross", "pesic_williams"):
        exp = load_experiment(EXPERIMENTS[name])
        b = exp.bsde
        assert b["horizon"] == pytest.approx(0.01)
        assert b["num_steps"] == 64
        assert b["dt"] == pytest.approx(0.01 / 64)
        assert b["train_steps_per_epoch"] == 5000
        assert b["train_epochs"] == 50
        assert resolve_training_updates(b) == 250000
        np.testing.assert_allclose(b["reference_drift"], expected_ref[name])
        assert b["loss_scale_by_horizon_sq"] is True


def test_bsde_lr_schedule_defaults_to_cosine():
    assert lr_schedule_kwargs({})["lr_schedule"] == "cosine"
    assert learning_rate_at(0, base_lr=5e-4, steps=100) == pytest.approx(5e-4)
    assert learning_rate_at(100, base_lr=5e-4, steps=100) == pytest.approx(5e-6)


def test_experiment_configs_use_cosine_lr_schedule():
    for name, path in EXPERIMENTS.items():
        exp = load_experiment(path)
        b = exp.bsde
        kw = lr_schedule_kwargs(b)
        assert kw["lr_schedule"] == "cosine", name
        assert kw["lr_min_ratio"] == pytest.approx(0.01), name
        assert b["log_interval"] == 200, name


def test_bigstep_uses_short_training_segment():
    exp = load_experiment(EXPERIMENTS["three_station_bigstep"])
    b = exp.bsde
    assert b["horizon"] == pytest.approx(0.01)
    assert b["num_steps"] == 64
    assert b["dt"] == pytest.approx(0.01 / 64)
    assert b["smoke_horizon"] == pytest.approx(0.01)
    assert b["train_steps_per_epoch"] == 5000
    assert b["train_epochs"] == 40
    assert resolve_training_updates(b) == 200000


def test_linear_bound_curriculum_formula():
    b_eff = effective_control_bound(
        0, target_bound=20, dim=3, curriculum="linear", pace=30,
        initial_bound=1.0)
    assert b_eff == pytest.approx(1.0)
    # b=20 reaches full bound after (20-1)*40*3*30 = 68400 updates.
    b_eff = effective_control_bound(
        68400, target_bound=20, dim=3, curriculum="linear", pace=30,
        initial_bound=1.0)
    assert b_eff == pytest.approx(20.0)


def test_crisscross_uses_general_driver_and_newton_lemke_reflection():
    exp = load_experiment(EXPERIMENTS["crisscross"])
    b = exp.bsde
    assert "driver" not in b
    assert exp.reflection["backend"] == "newton_lemke"
    kw = bound_curriculum_kwargs(b)
    assert kw["bound_curriculum"] == "linear"
    assert kw["curriculum_pace"] == pytest.approx(10.0)
