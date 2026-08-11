# Diffusion-Based BCP Policy for SPN

Clean, config-driven implementation of the diffusion-based Brownian Control
Problem (BCP) policy for stochastic processing networks. One general
`NetworkSpec` covers all three paper networks (Criss-Cross, Pesic-Williams,
Three-Station Bigstep); every scientific setting lives in YAML, not in Python.

> EWF (the equivalent workload formulation) is the *theoretical* framing in the
> paper; the implemented and benchmarked policy is the BCP one, compared against a
> finite-state MDP and classical max-pressure baselines.

The mathematical spec is [`docs/math_foundation.md`](docs/math_foundation.md); the
code follows it object-by-object.

## Layout

```
docs/math_foundation.md        the mathematical spec (source of truth)
configs/
  net_topology/               one YAML per network: Section-1 prelimit primitives + a small bcp block
                              (crisscross, pesic_williams, three_station_bigstep)
  net_parameters/             one YAML per training / simulation experiment
src/diffusion_based_policy/   the package
  network, planning, bcp, costs, validation, config, io, metrics
  reflection/                 numba LCP, GPU LCP, M-matrix
  bsde/                       model, dataset, losses, schedule, optim, trainer, evaluator
  policies/                   allocation, lifting, vectorized, baselines
  sim/                        diffusion, prelimit
cli/                          command-line entry points
tests/                        pytest suite + fixtures
```

## Environment

```bash
module load python/booth/3.12 || true
export PATH="/apps/python/RHEL8/3.12/3.12.9/bin:$PATH"
# TensorFlow 2.19, numba 0.62, numpy 1.26, scipy 1.15, pyyaml 6.0
```

`src`-layout package. Tests add `src/` to `sys.path` via `tests/conftest.py` and CLI commands via `cli/_bootstrap.py`, so no install is required; for development
use `pip install -e .` and `import diffusion_based_policy`.

## Quick start

```bash
cd BCP-publication
python -m pytest                                    # full test suite (incl. conformance)
python cli/check_networks.py                    # validate networks + print BCP matrices

# train a BSDE value/gradient network, then evaluate it
python cli/train_bsde.py  --config configs/net_parameters/bsde_bigstep.yaml --b 8
python cli/eval_policy.py --config configs/net_parameters/bsde_bigstep.yaml \
       --ckpt results/checkpoints/bsde_bigstep --b 8        # prelimit cost + diagnostics
python cli/eval_vmc.py    --config configs/net_parameters/bsde_bigstep.yaml \
       --ckpt results/checkpoints/bsde_bigstep --b 8        # diffusion-scale V_MC
```

## CLI commands

| command | purpose |
|---------|---------|
| `cli/train_bsde.py`     | train the BSDE value/gradient network from an experiment config |
| `cli/eval_policy.py`    | prelimit cost + Brownian diagnostics for a trained ckpt or a `--baseline` |
| `cli/eval_vmc.py`       | standalone diffusion-scale BCP value `V_MC` (more rollout paths without retraining) |
| `cli/run_prelimit.py`   | run the prelimit simulator under a baseline or trained policy |
| `cli/check_networks.py` | load + validate the networks, report the derived BCP matrices |
| `cli/make_tables.py`    | assemble result tables from processed outputs |
| `cli/run_phase_checks.py` | end-to-end validation entry point (`--phase {reflection,diffusion,all}`) |
| `cli/make_fixtures.py`  | regenerate the deterministic test fixtures |
| `cli/_bootstrap.py`     | makes `src/` importable when a command is run from the repo root |

## Run options

Every scientific default lives in the config; the flags below override it for a
single run. Run `python cli/<command>.py --help` for the full list.

Out-of-the-box defaults:

| knob | default |
|------|---------|
| control bound `b` | `10` |
| `omega` (Q reflection weight) | `0.99` |
| reflection backend | `numba` |
| b-curriculum | linear, **anneal from `initial_bound = 0`** |
| training reference | **offline** |
| seed | `42` |
| BSDE segment | `T = 0.01`, `num_steps = 64` (`dt = 1.5625e-4`) |
| eval prelimit | `100k` paths, horizon `3.0`, `num_jumps = 6000` |
| V_MC (inline at end of training) | `2048` paths — run `cli/eval_vmc.py --paths 100000` for a tight final value |

**Reflection / LCP solver** — config `reflection.backend`, override `--backend`:

| value | solver |
|-------|--------|
| `numba` (default) | projected Gauss-Seidel for the bulk of the batch, with exact basis enumeration only for the (rare) samples PGS can't solve — fast *and* exact |
| `gpu` | exact basis enumeration in TensorFlow throughout (slower, but robust on near-singular H, e.g. ω → 1) |

**Control bound b** (`cli/train_bsde.py`, `cli/eval_*`):

- `--b B` — single bound (default from `bcp.b`).
- `--b-sched B --b-reject B` — split bound: separate b for scheduling (processing) vs rejection/routing (input) blocks.
- `--omega W` — Q-matrix reflection weight (default `bcp.omega = 0.99`; the feasibility ceiling is 1.0).

**b-annealing (curriculum)** — config `bsde.bound_curriculum` ramps
`b_eff = min(b, initial_bound + step / (40·dim·pace))`:

- *default* — anneal from zero (`initial_bound = 0`); this is what avoids the spurious BSDE fixed point on Pesic-Williams.
- `--curriculum-init X` — start the ramp at `X` instead of 0.
- `--fixed-b` — no ramp, train at constant `b` (for staged warm-restart).

**Training reference (on-policy vs offline)** — `cli/train_bsde.py`:

- *default* — **offline**: training pool sampled from the reference reflected diffusion (drift `bsde.reference_drift`).
- `--on-policy` — **true on-policy**: pool driven by the live net, refreshed every `--on-policy-refresh` steps (default 5000), mixed `--on-policy-ratio` on-policy + offline anchor (default 0.5), pool size `--on-policy-pool`.
- `--behavior-from CKPT [--behavior-bound A]` — pool generated under a **frozen** policy from that checkpoint.

**Warm-start** — `--init-from CKPT` restores net weights before training.

**Evaluation** (`cli/eval_policy.py`):

- `--baseline {identity,linear,constant,zero}` — a benchmark gradient instead of a checkpoint:
  `identity` = MP (`g = z`), `linear` = MP-h (`g = h⊙z`), `constant` = `h`, `zero`.
- `--work-conserving` — processing stations never idle (classical max-pressure); input admission unaffected.
- `--paths N`, `--horizon T`, `--max-jumps J` — prelimit Monte-Carlo budget.

**Other**: `--steps`, `--lr`, `--seed`, `--tag` (output name), `--smoke` (tiny CI run).

## Configs hold only independent parameters

A **net_topology** YAML holds the Section-1 prelimit primitives (`buffer_change`
P, `resource_use` A, `rates` μ, `holding_cost` h, `input_cost` c^I,
`rejection_cost` c^D as the **prelimit per-job charge**, `discount` ρ,
`resource_types` κ, `idle_allowed` ι) and a small `bcp:` block with only
`critical_rates` μ*, `nominal_allocation` β, `n`, `b`, `omega`.

A **net_parameters** YAML points at one topology and adds the training/simulation
settings (`heavy_traffic`, `bsde`, `prelimit`, `reflection`).

**Everything else is derived in code** and checked by
`tests/test_config_consistency.py`:

- `mu_hat = sqrt(n)(mu^(n) − mu*)`, `gamma = n·rho`        (heavy-traffic scaling)
- nonbasic activities `= {j : beta_j = 0}`; critical resources `= {processing k : (A beta)_k = 1}`
- all BCP objects ζ, Γ, R, 𝖪, Q, H, c̃

## Supported scope

`NetworkSpec` is a **general** data model for the prelimit primitives of
`docs/math_foundation.md` Section 1. The **policy extraction and prelimit
simulator** are implemented for the structural class of the three paper networks:

- every activity uses exactly **one** resource (resource blocks partition the
  activities, so the block-decomposed policy is the exact LP optimum — §3.5/§5.4);
- every **no-rejection** input row (idle forbidden) has exactly **one** activity
  with unit consumption and `beta = 1`.

This is enforced two ways: `validation.validate_supported_scope` reports it (run
by `check_networks.py`), and the block hot paths (`policies/vectorized.py`,
`bsde/losses.py`) call `validation.require_single_resource_per_activity`, which
**raises `NotImplementedError`** on a multi-resource activity rather than silently
producing a wrong argmax. A buffer *shared across resources* (Pesic-Williams `b1`,
served by two stations) **is** supported — that contention is resolved exactly by
`policies/vectorized.py`.

## Design rules

- One `NetworkSpec`; no per-network Python classes, no per-network branching.
- Configs hold arrays/labels only; no scientific constants hard-coded in modules.
  Redundant heavy-traffic inputs (μ̂, γ, nonbasic, critical) are **derived**, not stored.
- TensorFlow custom training loops (`tf.Module`, `tf.GradientTape`,
  `tf.train.Checkpoint`); no Keras framework (`model.fit` / `tf.keras.Model`).
- Reflection backends sit behind a clean `ReflectionSolver` interface
  (`get_solver(backend, H)`), so the numba and GPU kernels are interchangeable.
