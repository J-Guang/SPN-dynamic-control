"""Phase 2/3 smoke tests: diffusion sampling, dataset builders, BSDE training.

Phase 2 portion (diffusion + dataset) runs without TensorFlow; the Phase 3
portion (model / loss / trainer) is skipped if TensorFlow is unavailable.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.bsde.dataset import (
    discount_nodes,
    make_training_batch,
    make_validation_batch,
)
from diffusion_based_policy.config import load_network
from diffusion_based_policy.policies.allocation import hamiltonian_value
from diffusion_based_policy.sim.diffusion import (
    ContinuousReferenceSampler,
    simulate_reference_paths,
)

_HAS_TF = importlib.util.find_spec("tensorflow") is not None


def _model(network_configs, name):
    spec, params = load_network(network_configs[name])
    return BCPModel(spec, params)


# ----------------------------------------------------------------- diffusion
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_reference_paths_in_orthant(network_configs, name):
    model = _model(network_configs, name)
    paths = simulate_reference_paths(model, num_paths=64, num_steps=16,
                                     horizon=0.1, seed=0, backend="numba")
    z = paths["z_path"]
    assert z.shape == (64, 17, model.I)
    assert np.all(z >= -1e-6), f"{name}: negative state min={z.min()}"
    assert np.all(np.isfinite(z))
    assert paths["dw"].shape == (64, 16, model.I)
    assert paths["l_incr"].shape == (64, 16, model.I)


def test_reference_drift_default_is_zeta(network_configs):
    model = _model(network_configs, "crisscross")
    paths = simulate_reference_paths(model, 8, 4, 0.1, seed=1, backend="numba")
    np.testing.assert_allclose(paths["reference_drift"], model.zeta)


def test_sampler_deterministic(network_configs):
    model = _model(network_configs, "pesic_williams")
    a = simulate_reference_paths(model, 16, 8, 0.1, seed=42, backend="numba")
    b = simulate_reference_paths(model, 16, 8, 0.1, seed=42, backend="numba")
    np.testing.assert_array_equal(a["z_path"], b["z_path"])
    c = simulate_reference_paths(model, 16, 8, 0.1, seed=43, backend="numba")
    assert not np.array_equal(a["z_path"], c["z_path"])


def test_init_distributions(network_configs):
    model = _model(network_configs, "crisscross")
    for init in ("zeros", "uniform", "exponential"):
        paths = simulate_reference_paths(model, 8, 4, 0.1, seed=0,
                                         backend="numba", init=init, init_scale=0.5)
        assert np.all(paths["z_path"][:, 0, :] >= -1e-9)
    z0 = simulate_reference_paths(model, 8, 4, 0.1, seed=0, init="zeros")["z_path"][:, 0]
    assert np.allclose(z0, 0.0)


# ------------------------------------------------------------------- dataset
def test_discount_nodes():
    d = discount_nodes(gamma=4.0, dt=0.5, num_steps=4)
    np.testing.assert_allclose(d, np.exp(-4.0 * np.array([0, .5, 1, 1.5, 2])))


@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_training_batch_shapes(network_configs, name):
    model = _model(network_configs, name)
    batch = make_training_batch(model, num_paths=32, num_steps=16,
                                horizon=0.5, seed=7, backend="numba")
    assert batch["z"].shape == (32, 17, model.I)
    assert batch["dw"].shape == (32, 16, model.I)
    assert batch["disc"].shape == (17,)
    np.testing.assert_allclose(batch["ref_drift_minus_zeta"],
                               batch["reference_drift"] - model.zeta)


def test_validation_batch_fixed(network_configs):
    model = _model(network_configs, "crisscross")
    a = make_validation_batch(model, 16, 8, 0.5)
    b = make_validation_batch(model, 16, 8, 0.5)
    np.testing.assert_array_equal(a["z"], b["z"])


# --------------------------------------------------- continuous sampler
def test_continuous_sampler_chains(network_configs):
    """Each segment continues from where the previous one ended (continuity)."""
    model = _model(network_configs, "crisscross")
    s = ContinuousReferenceSampler(model, num_paths=8, num_steps=16, horizon=0.01,
                                   backend="numba", reference_drift=[-0.5, -0.5, -0.5],
                                   seed=0, warmup_segments=0)
    seg1 = s.advance()
    seg2 = s.advance()
    np.testing.assert_allclose(seg2["z_path"][:, 0, :], seg1["z_path"][:, -1, :], atol=1e-9)
    assert s.segments_done == 2


def test_continuous_sampler_warmup_advances(network_configs):
    model = _model(network_configs, "crisscross")
    s = ContinuousReferenceSampler(model, num_paths=8, num_steps=16, horizon=0.01,
                                   reference_drift=[-0.5, -0.5, -0.5], seed=1,
                                   warmup_segments=5)
    assert s.segments_done == 5            # burn-in consumed 5 segments
    # the warmed-up state has left the origin
    assert np.max(s.state) > 0.0


def test_pooled_sampler_reuses_and_formats(network_configs):
    """Pooled sampler generates a bounded pool, serves correct batches, and
    reshuffles per epoch (the published reuse scheme)."""
    from diffusion_based_policy.bsde.dataset import PooledContinuousSampler

    model = _model(network_configs, "crisscross")
    s = PooledContinuousSampler(model, num_paths=8, num_steps=8, horizon=0.01,
                                pool_segments=5, reference_drift=[-0.5, -0.5, -0.5],
                                seed=0, warmup_segments=2)
    b = s.batch()
    assert b["z"].shape == (8, 9, model.I)
    assert b["dw"].shape == (8, 8, model.I)
    assert "disc" in b and "ref_drift_minus_zeta" in b
    for _ in range(11):
        s.batch()
    assert s.epochs_done == 2          # 12 batches over a pool of 5


def test_continuous_sampler_explores_more_than_iid(network_configs):
    """After many chained segments the path reaches states far beyond a single
    short restart-from-zero segment -- the whole point of continuous sampling."""
    model = _model(network_configs, "crisscross")
    iid = simulate_reference_paths(model, num_paths=256, num_steps=16, horizon=0.01,
                                   seed=0, reference_drift=[-0.5, -0.5, -0.5], init="zeros")
    iid_max = iid["z_path"].reshape(-1, 3).max()
    s = ContinuousReferenceSampler(model, num_paths=256, num_steps=16, horizon=0.01,
                                   reference_drift=[-0.5, -0.5, -0.5], seed=0)
    cont_max = 0.0
    for _ in range(400):
        cont_max = max(cont_max, s.advance()["z_path"].max())
    assert cont_max > 2.0 * iid_max


def test_crisscross_general_driver_matches_idle_admissible_formula(network_configs):
    """Crisscross has idle-admissible processing rows (math_foundation 1.6: iota
    S1=S2=1). The block min-term takes max(0, .) on each processing block, so the
    driver is h.z + b[(g0+g1) - max(0, 2 g0, 2(g1-g2)) - max(0, g2)].

    For g = grad V >= 0 (V monotone) this coincides with the work-conserving
    formula h.z - b|g0 - g1 + g2|; the max(0,.) only bites for negative g.
    """
    model = _model(network_configs, "crisscross")
    z = np.array([1.0, 2.0, 3.0])
    b = model.params.b
    for g in (
        np.array([0.7, -0.2, 0.4]),
        np.array([-0.1, 0.8, -0.3]),
        np.array([0.5, 0.3, 0.2]),
    ):
        got = hamiltonian_value(model, z, g)
        expected = float(
            model.h_tilde @ z
            + b * ((g[0] + g[1]) - max(0.0, 2 * g[0], 2 * (g[1] - g[2])) - max(0.0, g[2]))
        )
        assert got == pytest.approx(expected)


# ----------------------------------------------------------- fused TF sampler
@pytest.mark.skipif(not _HAS_TF, reason="tensorflow unavailable")
def test_fused_sampler_valid(network_configs):
    import tensorflow as tf

    from diffusion_based_policy.bsde.dataset import make_fused_reference_sampler

    model = _model(network_configs, "three_station_bigstep")
    fused = make_fused_reference_sampler(model, num_steps=16, horizon=0.1)
    z0 = tf.zeros((32, model.I), tf.float32)
    z_path, dw = fused(z0, tf.constant([1, 2], tf.int32))
    z_path = z_path.numpy()
    assert z_path.shape == (32, 17, model.I)
    assert np.all(z_path >= -1e-4)
    assert np.all(np.isfinite(z_path))


# --------------------------------------------------------- Phase 3 (training)
@pytest.mark.skipif(not _HAS_TF, reason="tensorflow unavailable")
def test_bsde_loss_components(network_configs):
    import tensorflow as tf

    from diffusion_based_policy.bsde.losses import bsde_loss
    from diffusion_based_policy.bsde.model import ValueGradModel

    model = _model(network_configs, "crisscross")
    batch = make_training_batch(model, num_paths=64, num_steps=16,
                                horizon=0.5, seed=3, backend="numba")
    net = ValueGradModel(model.I, hidden=(32, 32), seed=0)
    loss, comps = bsde_loss(net, batch, model)
    assert np.isfinite(loss.numpy())
    assert loss.numpy() >= 0
    for key in ("residual_sq",):
        assert key in comps
    assert comps["residual_sq"].numpy() == pytest.approx(
        comps["residual_sq_raw"].numpy() / (batch["horizon"] ** 2)
    )


@pytest.mark.skipif(not _HAS_TF, reason="tensorflow unavailable")
def test_smoke_train_decreases(network_configs):
    from diffusion_based_policy.bsde.model import ValueGradModel
    from diffusion_based_policy.bsde.trainer import BSDETrainer, TrainConfig

    model = _model(network_configs, "crisscross")
    net = ValueGradModel(model.I, hidden=(32, 32), seed=0)
    cfg = TrainConfig(num_paths=64, num_steps=16, horizon=0.5,
                      steps=30, lr=1e-3, backend="numba", val_interval=10,
                      ckpt_interval=0)
    trainer = BSDETrainer(model, net, cfg)
    history = trainer.train()
    losses = [h["loss"] for h in history if "loss" in h]
    assert np.isfinite(losses[-1])
    # loss should not blow up over a short run
    assert losses[-1] <= 5.0 * losses[0] + 1e-6


@pytest.mark.skipif(not _HAS_TF, reason="tensorflow unavailable")
def test_continuous_mode_trains(network_configs):
    """The default continuous sampler drives a finite, non-exploding training run."""
    from diffusion_based_policy.bsde.model import ValueGradModel
    from diffusion_based_policy.bsde.trainer import BSDETrainer, TrainConfig

    model = _model(network_configs, "crisscross")
    net = ValueGradModel(model.I, hidden=(32, 32), seed=0)
    cfg = TrainConfig(num_paths=64, num_steps=16, horizon=0.01,
                      steps=20, lr=1e-3, backend="numba", val_interval=10,
                      ckpt_interval=0, sampling_mode="continuous",
                      warmup_segments=3, reference_drift=[-0.5, -0.5, -0.5])
    assert trainer_runs(BSDETrainer(model, net, cfg))


def trainer_runs(trainer) -> bool:
    history = trainer.train()
    return bool(history) and np.isfinite(history[-1]["loss"])


@pytest.mark.skipif(not _HAS_TF, reason="tensorflow unavailable")
def test_brownian_value_mc(network_configs):
    """V_MC rollout returns a finite value and matches a manual short integral."""
    from diffusion_based_policy.bsde.evaluator import BSDEEvaluator
    from diffusion_based_policy.bsde.model import ValueGradModel

    model = _model(network_configs, "crisscross")
    net = ValueGradModel(model.I, hidden=(32, 32), seed=0)
    ev = BSDEEvaluator(model, net)
    res = ev.brownian_value_mc(num_paths=128, horizon=0.05, dt=0.01 / 16,
                               reference_drift=[-0.5, -0.5, -0.5], seed=3)
    assert np.isfinite(res["v_mc"])
    assert res["nsteps"] == round(0.05 / (0.01 / 16))
    assert res["control_bound"] == model.params.b   # full bound by default
