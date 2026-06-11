import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rospy
import tf.transformations
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist

from dsac_mppi.controllers.reference_tracker import ReferenceLineTracker, ReferenceTrackerConfig
from dsac_mppi.envs.frenet_utils import FrenetTransform
from dsac_mppi.envs.ros_env import MyCarEnv


def parse_xy(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected x,y")
    return np.asarray(parts, dtype=float)


def configure_straight_reference(env: MyCarEnv, start: np.ndarray, goal: np.ndarray, yaw: float) -> None:
    env.pub_cmd_vel.publish(Twist())
    env.start_pos = np.asarray(start, dtype=float).copy()
    env.target_pos = np.asarray(goal, dtype=float).copy()
    env.current_pos = env.start_pos.copy()
    env.current_yaw = float(yaw)
    env.frenet_transform = FrenetTransform(env.start_pos, env.target_pos, curve_offset=0.0)
    env.last_frenet_s = 0.0
    env.last_abs_frenet_d = 0.0
    env.velocity = np.array([0.0, 0.0], dtype=float)
    env.last_cmd_velocity = np.array([0.0, 0.0], dtype=float)
    env.last_action = np.array([0.0, 0.0], dtype=np.float32)
    env.max_steps = int(max(100, env.frenet_transform.path_length / max(0.15, 1e-6) * 10.0 + 150))

    state_msg = ModelState()
    state_msg.model_name = "usv"
    state_msg.pose.position.x = float(start[0])
    state_msg.pose.position.y = float(start[1])
    state_msg.pose.position.z = 0.05
    q = tf.transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
    state_msg.pose.orientation.x = q[0]
    state_msg.pose.orientation.y = q[1]
    state_msg.pose.orientation.z = q[2]
    state_msg.pose.orientation.w = q[3]
    state_msg.twist.linear.x = 0.0
    state_msg.twist.linear.y = 0.0
    state_msg.twist.angular.z = 0.0
    env.set_state_proxy(state_msg)
    rospy.sleep(0.2)

    env._update_marker("marker_start", start[0], start[1], "Blue")
    env._update_marker("marker_goal", goal[0], goal[1], "Red")
    env._visualize_path()


def summarize(rows: List[Dict[str, object]], path_length: float, goal_tolerance: float) -> Dict[str, object]:
    if not rows:
        return {"success": False}
    first = rows[0]
    last = rows[-1]
    s_values = np.asarray([row["frenet_s"] for row in rows], dtype=float)
    d_values = np.asarray([row["frenet_d"] for row in rows], dtype=float)
    heading_values = np.asarray([abs(row["heading_error"]) for row in rows], dtype=float)
    remaining_values = np.asarray([row["remaining_path"] for row in rows], dtype=float)
    s_backtracks = int(np.sum(np.diff(s_values) < -0.05)) if len(s_values) > 1 else 0
    half = max(1, len(heading_values) // 2)
    heading_first_mean = float(np.mean(heading_values[:half]))
    heading_second_mean = float(np.mean(heading_values[half:])) if len(heading_values) > half else heading_first_mean
    success = bool(
        last["remaining_path"] <= goal_tolerance
        and abs(last["frenet_d"]) <= 0.7
        and last.get("terminal_reason", "running") in ("success", "running")
    )
    return {
        "success": success,
        "steps": len(rows),
        "initial_s": float(first["frenet_s"]),
        "final_s": float(last["frenet_s"]),
        "path_length": float(path_length),
        "initial_d": float(first["frenet_d"]),
        "final_d": float(last["frenet_d"]),
        "max_abs_d": float(np.max(np.abs(d_values))),
        "initial_remaining": float(first["remaining_path"]),
        "final_remaining": float(last["remaining_path"]),
        "heading_first_half_mean": heading_first_mean,
        "heading_second_half_mean": heading_second_mean,
        "s_backtracks": s_backtracks,
    }


def write_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def predict_first_order_velocity(previous_velocity: np.ndarray, applied_command: np.ndarray, env: MyCarEnv) -> np.ndarray:
    previous = np.asarray(previous_velocity, dtype=float).reshape(2)
    command = np.asarray(applied_command, dtype=float).reshape(2)
    dt = float(env.dt)
    tu = max(float(env.surge_time_constant), 1e-6)
    tr = max(float(env.yaw_time_constant), 1e-6)
    return np.array(
        [
            previous[0] + dt * (command[0] - previous[0]) / tu,
            previous[1] + dt * (command[1] - previous[1]) / tr,
        ],
        dtype=float,
    )


def read_gazebo_twist(env: MyCarEnv) -> Dict[str, float]:
    try:
        state = env.get_state_proxy("usv", "world")
        if not getattr(state, "success", True):
            raise RuntimeError(getattr(state, "status_message", "get_model_state failed"))
        twist = state.twist
        linear_x = float(twist.linear.x)
        linear_y = float(twist.linear.y)
        angular_z = float(twist.angular.z)
        return {
            "gazebo_linear_x": linear_x,
            "gazebo_linear_y": linear_y,
            "gazebo_speed_xy": float(math.hypot(linear_x, linear_y)),
            "gazebo_angular_z": angular_z,
            "gazebo_twist_error": "",
        }
    except Exception as exc:
        return {
            "gazebo_linear_x": float("nan"),
            "gazebo_linear_y": float("nan"),
            "gazebo_speed_xy": float("nan"),
            "gazebo_angular_z": float("nan"),
            "gazebo_twist_error": str(exc),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run a non-RL lookahead reference-line tracking check in Gazebo.")
    parser.add_argument("--start", type=parse_xy, default=np.array([-8.0, -16.0], dtype=float), help="Fixed start as x,y.")
    parser.add_argument("--goal", type=parse_xy, default=np.array([8.0, -16.0], dtype=float), help="Fixed goal as x,y.")
    parser.add_argument("--yaw", type=float, default=0.25, help="Initial yaw in radians.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=1, help="Print every N steps. Use 1 to inspect dynamics step-by-step.")
    parser.add_argument("--lookahead", type=float, default=3.0)
    parser.add_argument("--target-speed", type=float, default=0.8)
    parser.add_argument("--heading-gain", type=float, default=1.2)
    parser.add_argument("--lateral-gain", type=float, default=0.25)
    parser.add_argument("--goal-tolerance", type=float, default=0.45)
    parser.add_argument("--output", default="data/reference_tracking/gazebo_reference_tracking_trace.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("reference_tracker_check", anonymous=True, disable_signals=True)
    env = MyCarEnv(control_mode="low_level_velocity", dynamics_model="first_order", curriculum_stage=0)
    tracker = ReferenceLineTracker(
        ReferenceTrackerConfig(
            lookahead_distance=args.lookahead,
            target_speed=args.target_speed,
            goal_tolerance=args.goal_tolerance,
            heading_gain=args.heading_gain,
            lateral_gain=args.lateral_gain,
        ),
        env._usv_dynamics_config(),
    )
    rows: List[Dict[str, object]] = []
    try:
        env.reset()
        configure_straight_reference(env, args.start, args.goal, args.yaw)
        print(
            "Reference tracking check: "
            f"start=({args.start[0]:.2f},{args.start[1]:.2f}) "
            f"goal=({args.goal[0]:.2f},{args.goal[1]:.2f}) "
            f"path_length={env.frenet_transform.path_length:.2f}"
        )
        print(
            "Dynamics config: "
            f"dt={env.dt:.3f}s T_u={env.surge_time_constant:.3f}s T_r={env.yaw_time_constant:.3f}s "
            f"max_du={env.max_du:.3f} max_dr={env.max_dr:.3f}"
        )
        for step in range(int(args.max_steps)):
            position, yaw = env._get_robot_position()
            previous_velocity = env.velocity.copy()
            previous_command = env.last_cmd_velocity.copy()
            action, tracker_debug = tracker.compute_action(position, yaw, env.target_pos, env.frenet_transform)
            _, _, terminated, truncated, info = env.step(action)
            applied_command = np.asarray(info["applied_command"], dtype=float).reshape(2)
            current_velocity = np.asarray(info["current_velocity"], dtype=float).reshape(2)
            expected_velocity = predict_first_order_velocity(previous_velocity, applied_command, env)
            gazebo_twist = read_gazebo_twist(env)
            row = {
                "step": float(step),
                "x": float(info["current_position"][0]),
                "y": float(info["current_position"][1]),
                "yaw": float(info["current_yaw"]),
                "prev_u": float(previous_velocity[0]),
                "prev_r": float(previous_velocity[1]),
                "actual_u": float(current_velocity[0]),
                "actual_r": float(current_velocity[1]),
                "tracker_u_cmd": float(action[0]),
                "tracker_r_cmd": float(action[1]),
                "prev_u_cmd": float(previous_command[0]),
                "prev_r_cmd": float(previous_command[1]),
                "applied_u_cmd": float(applied_command[0]),
                "applied_r_cmd": float(applied_command[1]),
                "published_linear_x": float(current_velocity[0]),
                "published_angular_z": float(current_velocity[1]),
                "expected_u": float(expected_velocity[0]),
                "expected_r": float(expected_velocity[1]),
                "velocity_error_u": float(current_velocity[0] - expected_velocity[0]),
                "velocity_error_r": float(current_velocity[1] - expected_velocity[1]),
                "delta_u_cmd": float(applied_command[0] - previous_command[0]),
                "delta_r_cmd": float(applied_command[1] - previous_command[1]),
                **gazebo_twist,
                "frenet_s": float(info["frenet_s"]),
                "frenet_d": float(info["frenet_d"]),
                "heading_error": float(info["heading_error"]),
                "pursuit_heading_error": float(tracker_debug["pursuit_heading_error"]),
                "lookahead_s": float(info["lookahead_s"]),
                "lookahead_x": float(info["lookahead_point"][0]),
                "lookahead_y": float(info["lookahead_point"][1]),
                "lookahead_body_x": float(info["lookahead_body"][0]),
                "lookahead_body_y": float(info["lookahead_body"][1]),
                "remaining_path": float(info["remaining_path"]),
                "min_laser_dist": float(info["min_laser_dist"]),
                "reward": float(info.get("env_reward", 0.0)),
                "terminal_reason": str(info["terminal_reason"]),
            }
            rows.append(row)
            if args.log_every > 0 and (step % int(args.log_every) == 0 or terminated or truncated):
                print(
                    f"step={step:04d} s={row['frenet_s']:.2f} d={row['frenet_d']:.2f} "
                    f"heading={row['heading_error']:.3f} remain={row['remaining_path']:.2f} "
                    f"lh_s={row['lookahead_s']:.2f} lh_body=({row['lookahead_body_x']:.2f},{row['lookahead_body_y']:.2f}) "
                    f"tracker_cmd=({row['tracker_u_cmd']:.3f},{row['tracker_r_cmd']:.3f}) "
                    f"applied_cmd=({row['applied_u_cmd']:.3f},{row['applied_r_cmd']:.3f}) "
                    f"d_cmd=({row['delta_u_cmd']:.3f},{row['delta_r_cmd']:.3f}) "
                    f"actual=({row['actual_u']:.3f},{row['actual_r']:.3f}) "
                    f"published=({row['published_linear_x']:.3f},{row['published_angular_z']:.3f}) "
                    f"gazebo=({row['gazebo_linear_x']:.3f},{row['gazebo_angular_z']:.3f}) "
                    f"expected=({row['expected_u']:.3f},{row['expected_r']:.3f}) "
                    f"err=({row['velocity_error_u']:.2e},{row['velocity_error_r']:.2e}) "
                    f"term={row['terminal_reason']}"
                )
            if row["remaining_path"] <= args.goal_tolerance and abs(row["frenet_d"]) <= 0.7:
                print("Reference tracker reached the goal region.")
                break
            if terminated or truncated:
                print(f"Environment ended: {row['terminal_reason']}")
                break
    finally:
        env.pub_cmd_vel.publish(Twist())
        write_csv(rows, str(PROJECT_ROOT / args.output) if not os.path.isabs(args.output) else args.output)
        if rows:
            summary = summarize(rows, env.frenet_transform.path_length, args.goal_tolerance)
            print("Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        env.close()


if __name__ == "__main__":
    main()
