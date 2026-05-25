# Project Structure

This repository is a ROS catkin workspace for USV navigation experiments with reinforcement learning and MPPI-based controllers.

## Top-Level Layout

```text
RL_car/
  src/
    nav_demo/
      scripts/
        beam_map/
        SAC/
        ENV/
    urdf01_rviz/
    urdf02_gazebo/
    my_urdf/
  tests/
  docs/
  build/
  devel/
```

## Core Navigation Code

`src/nav_demo/scripts/beam_map/` contains the active training, evaluation, and controller code.

- `ros_env.py`: Gymnasium environment backed by ROS/Gazebo. It publishes `/cmd_vel`, reads laser scans and Gazebo model state, computes Frenet observations, rewards, and terminal conditions.
- `train.py`: baseline SAC training entrypoint for the 2D `[surge, yaw]` action policy.
- `compare_sac_mppi.py`: offline evaluation and plotting driver for baseline, shield, hierarchical MPPI, and RL-driven MPPI modes.
- `test01.py`: interactive Gazebo run script for one policy/controller mode.
- `mppi_dbas.py`: existing low-intervention MPPI/DBaS optimizer and hierarchical MPPI helper logic.
- `mppi_dbas_wrapper.py`: Gym wrapper for the low-intervention MPPI/DBaS optimizer.
- `hierarchical_mppi_wrapper.py`: wrappers that map high-level SAC intent to lower-level MPPI control.
- `rl_driven_mppi.py`: paper-style RL-driven MPPI implementation with policy adapter, online optimizer, and wrapper.
- `plot_comparison_curves.py`: generates summary and trace plots from comparison outputs.

## ROS/Gazebo Assets

- `src/urdf02_gazebo/`: Gazebo launch files, worlds, maps, dynamic obstacle scripts, and robot simulation assets.
- `src/urdf01_rviz/`: RViz configuration and visualization URDF examples.
- `src/my_urdf/`: additional URDF package files.

## Tests

`tests/` contains Python unit tests that run without launching ROS/Gazebo.

- `test_mppi_dbas_low_intervention.py`: low-intervention MPPI/DBaS behavior.
- `test_hierarchical_sac_mppi.py`: hierarchical wrappers and intent decoding.
- `test_plot_comparison_curves.py`: plotting color/summary robustness.
- `test_rl_driven_mppi.py`: RL-driven MPPI initialization, guided rollout, top-Z update, sigma bound, action bound, and ablation behavior.

Run from the repository root:

```bash
python -m unittest discover -s RL_car/tests
```

## Training And Evaluation Artifacts

Training scripts write model and log directories under `src/nav_demo/scripts/beam_map/`, for example:

- `training_usv_v2_results/`
- `training_hierarchical_mppi_v*_results/`
- `logs_hierarchical_mppi_v*/`
- `sac_*_log/`

Evaluation writes result folders such as:

- `comparison_results/`
- `comparison_rl_driven_mppi/`
- `comparison_rl_driven_mppi_ablation/`

Each comparison output can contain:

- `<mode>_metrics.csv`: per-episode metrics.
- `traces/<mode>_episode_*.csv`: per-step traces.
- `summary.json`: aggregate metrics and paired comparisons.
- `plots/`: generated figures when `--plot` is used.
