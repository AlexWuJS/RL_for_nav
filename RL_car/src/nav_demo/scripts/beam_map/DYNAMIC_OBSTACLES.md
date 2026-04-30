# AIS Dynamic Obstacles

This folder owns the channel-project dynamic obstacle pipeline:

```text
raw AIS .xls -> aligned scenario JSON -> Gazebo AIS obstacle node -> SAC training
```

## Generate Scenario

Run from `RL_car/src/nav_demo/scripts/beam_map`:

```bash
python tools/convert_ais_xls_to_obstacles.py \
  --input-dir data/raw/trajectories \
  --map-meta data/processed/maps/navigation_map_meta.yaml \
  --affine data/processed/maps/affine_params.yaml \
  --out-json data/processed/trajectories/ais_scenario.json
```

The converter uses affine calibration when available and falls back to the map
metadata bounds otherwise. By default, real map meters are scaled by `0.01` and
centered around the Gazebo world origin; override `--world-scale`,
`--world-origin-x`, or `--world-origin-y` if the Gazebo world changes.

## Launch Gazebo Replay

```bash
roslaunch nav_demo usv_dynamic_avoidance.launch \
  scenario_json:=$(rospack find nav_demo)/scripts/beam_map/data/processed/trajectories/ais_scenario.json \
  max_active_obstacles:=5
```

`ais_dynamic_obstacle_node.py` replays only a bounded number of vessels at once
and hides inactive tracks far away from the training area.

## Reserved Global Path Interface

`ros_env.py` subscribes to `/global_path` (`nav_msgs/Path`) and caches the latest
message. The current stage does not yet follow that path; when no path is
published, the existing sampled start/goal behavior is unchanged.
