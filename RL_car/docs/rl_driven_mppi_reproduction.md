# RL-Driven MPPI Reproduction Notes

This document records the first reproducible implementation of the paper-style RL-driven MPPI controller in the existing USV ROS/Gazebo project.

## Algorithm Mapping

The paper uses an offline stochastic RL policy to accelerate online MPPI. In this project the first version uses a Stable-Baselines3 SAC policy through `SB3SacPolicyAdapter`; the adapter boundary is intentionally small so a true DSAC policy can replace it later.

Implemented mechanisms:

- RL initialization: the offline policy provides the initial MPPI mean sequence `U0`; the adapter provides the initial action standard deviation.
- Hybrid sampling strategy: guided rollouts are sampled from the offline policy once at the beginning of each online control step and reused across MPPI iterations.
- Mean and variance update: each MPPI iteration selects top-Z candidate sequences and updates both `U` and `Sigma`, with `sigma_min` as a lower bound.
- Terminal value cost: the adapter tries to query the SAC critic as a terminal cost. If critic evaluation is unavailable, the controller falls back to rollout cost and marks `rlmppi_terminal_q_used=false`.

The USV rollout model reuses the same short-horizon 3-DOF approximation and Frenet/laser costs already used by `mppi_dbas.py`, so the new controller is comparable with the existing low-intervention MPPI and hierarchical SAC-MPPI variants.

## Main Files

- `src/nav_demo/scripts/beam_map/rl_driven_mppi.py`: RL-driven MPPI config, optimizer, SAC adapter, and Gym wrapper.
- `src/nav_demo/scripts/beam_map/compare_sac_mppi.py`: evaluation modes and summary/debug output.
- `src/nav_demo/scripts/beam_map/test01.py`: Gazebo single-run mode wiring.
- `tests/test_rl_driven_mppi.py`: fake-state unit tests for initialization, HSS, top-Z updates, sigma bounds, action bounds, and ablation switches.

## Run Commands

Run commands from:

```bash
cd RL_car/src/nav_demo/scripts/beam_map
```

Train the offline SAC policy:

```bash
python train.py --total-timesteps 300000
```

Quick RL-driven MPPI evaluation:

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode rl_driven_mppi \
  --episode 10 \
  --output-dir ./comparison_rl_driven_mppi \
  --plot
```

Paper-mechanism ablation:

```bash
python compare_sac_mppi.py \
  --model ./training_usv_v2_results/best_model \
  --mode ablation_rlmppi \
  --episode 30 \
  --output-dir ./comparison_rl_driven_mppi_ablation \
  --plot
```

Single Gazebo run:

```bash
python test01.py \
  --mode rl_driven_mppi \
  --model ./training_usv_v2_results/best_model
```

## Evaluation Modes

- `pure_mppi`: MPPI only; no RL initialization, no HSS, no terminal Q.
- `rl_driven_mppi`: full first-version RL-driven MPPI.
- `rl_driven_mppi_no_hss`: disables guided rollouts.
- `rl_driven_mppi_fixed_sigma`: disables variance update.
- `rl_driven_mppi_no_q`: disables terminal critic cost.
- `ablation_rlmppi`: runs `baseline`, `pure_mppi`, full RLMPPI, and all three ablations.

## Output Metrics

Evaluation produces per-mode CSV files, per-step traces, `summary.json`, and plots when `--plot` is enabled.

RLMPPI-specific fields include:

- `rlmppi_hss_enabled`
- `rlmppi_terminal_q_enabled`
- `rlmppi_terminal_q_used`
- `rlmppi_update_sigma`
- `rlmppi_num_rl_rollouts`
- `rlmppi_num_mppi_rollouts`
- `rlmppi_num_iterations`
- `rlmppi_top_z`
- `rlmppi_sigma_mean`
- `rlmppi_cost_best`
- `rlmppi_online_time_ms`

## Current Limitations

- The first version is SAC-compatible rather than a full DSAC implementation.
- The SAC critic terminal value may be unavailable for stacked or wrapped observations; this is reported by `rlmppi_terminal_q_used`.
- Online rollouts use the existing approximate USV model, not a learned neural transition model.
