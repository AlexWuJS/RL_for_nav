#!/usr/bin/env python3
"""Replay AIS trajectories as dynamic obstacle vessels in Gazebo."""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, GetModelState, SetModelState, SpawnModel
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from visualization_msgs.msg import Marker, MarkerArray

try:
    import tf.transformations as tft
except ImportError:  # pragma: no cover - ROS provides this at runtime.
    tft = None


HIDDEN_POSITION = (100000.0, 100000.0, -100.0)


@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float


@dataclass
class ObstacleTrack:
    source_id: str
    model_name: str
    radius: float
    points: List[TrajectoryPoint]

    @property
    def start_time(self) -> float:
        return self.points[0].t

    @property
    def end_time(self) -> float:
        return self.points[-1].t


def quaternion_from_yaw(yaw: float) -> Quaternion:
    if tft is not None:
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def make_vessel_sdf(name: str, radius: float) -> str:
    length = max(radius * 2.5, 1.0)
    width = max(radius * 0.9, 0.4)
    height = max(radius * 0.35, 0.2)
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="body">
      <gravity>false</gravity>
      <kinematic>true</kinematic>
      <collision name="collision">
        <geometry>
          <box><size>{length} {width} {height}</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{length} {width} {height}</size></box>
        </geometry>
        <material>
          <ambient>0.8 0.25 0.1 1</ambient>
          <diffuse>0.8 0.25 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""".format(name=name, length=length, width=width, height=height)


class AisDynamicObstacleNode:
    def __init__(self) -> None:
        rospy.init_node("ais_dynamic_obstacle_node", anonymous=False)
        self.scenario_json = rospy.get_param("~scenario_json", "")
        if not self.scenario_json:
            raise rospy.ROSException("~scenario_json is required")

        self.max_active = int(rospy.get_param("~max_active_obstacles", 5))
        self.time_scale = float(rospy.get_param("~time_scale", 0.0))
        self.update_rate = float(rospy.get_param("~update_rate", 10.0))
        self.model_prefix = rospy.get_param("~model_prefix", "ais_ship_")
        self.z_height = float(rospy.get_param("~z_height", 0.25))
        self.loop = bool(rospy.get_param("~loop", False))
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))

        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/set_model_state")
        rospy.wait_for_service("/gazebo/get_model_state")
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.delete_model = None
        try:
            rospy.wait_for_service("/gazebo/delete_model", timeout=2.0)
            self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        except rospy.ROSException:
            self.delete_model = None

        self.marker_pub = rospy.Publisher("/dynamic_obstacles/markers", MarkerArray, queue_size=1)
        self.tracks, scenario_scale = self._load_scenario(self.scenario_json)
        if self.time_scale <= 0.0:
            self.time_scale = scenario_scale
        self.scenario_duration = max((track.end_time for track in self.tracks), default=0.0)
        self.spawned_models: Dict[str, bool] = {}
        self.active_tracks: Dict[str, ObstacleTrack] = {}
        self.start_time = rospy.Time.now()

        rospy.loginfo(
            "AIS dynamic obstacle node loaded %d tracks, max_active=%d, time_scale=%.3f",
            len(self.tracks),
            self.max_active,
            self.time_scale,
        )

    def _load_scenario(self, path: str) -> Tuple[List[ObstacleTrack], float]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracks = []
        for index, raw in enumerate(data.get("obstacles", [])):
            points = [
                TrajectoryPoint(
                    t=float(p["t"]),
                    x=float(p["x"]),
                    y=float(p["y"]),
                    yaw=float(p.get("yaw", 0.0)),
                    vx=float(p.get("vx", 0.0)),
                    vy=float(p.get("vy", 0.0)),
                )
                for p in raw.get("trajectory", [])
            ]
            if len(points) < 2:
                continue
            points.sort(key=lambda p: p.t)
            source_id = str(raw.get("id", index))
            model_name = "%s%03d" % (self.model_prefix, index)
            tracks.append(
                ObstacleTrack(
                    source_id=source_id,
                    model_name=model_name,
                    radius=float(raw.get("radius", 1.0)),
                    points=points,
                )
            )
        tracks.sort(key=lambda item: item.start_time)
        return tracks, float(data.get("time_scale", 1.0))

    def _scenario_time(self) -> float:
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        t = elapsed * self.time_scale
        if self.loop and self.scenario_duration > 0:
            t = t % self.scenario_duration
        return t

    def _sample_track(self, track: ObstacleTrack, t: float) -> Optional[TrajectoryPoint]:
        if t < track.start_time or t > track.end_time:
            return None
        points = track.points
        for idx in range(len(points) - 1):
            p0 = points[idx]
            p1 = points[idx + 1]
            if p0.t <= t <= p1.t:
                alpha = (t - p0.t) / max(p1.t - p0.t, 1e-6)
                x = p0.x + alpha * (p1.x - p0.x)
                y = p0.y + alpha * (p1.y - p0.y)
                vx = p0.vx + alpha * (p1.vx - p0.vx)
                vy = p0.vy + alpha * (p1.vy - p0.vy)
                yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-6 else p0.yaw
                return TrajectoryPoint(t=t, x=x, y=y, yaw=yaw, vx=vx, vy=vy)
        return points[-1]

    def _ensure_model(self, track: ObstacleTrack) -> None:
        if self.spawned_models.get(track.model_name):
            return
        pose = Pose()
        pose.position.x = HIDDEN_POSITION[0]
        pose.position.y = HIDDEN_POSITION[1]
        pose.position.z = HIDDEN_POSITION[2]
        pose.orientation.w = 1.0
        try:
            self.spawn_model(track.model_name, make_vessel_sdf(track.model_name, track.radius), "", pose, "world")
        except rospy.ServiceException:
            # The model may already exist from a previous node restart.
            pass
        self.spawned_models[track.model_name] = True

    def _set_track_state(self, track: ObstacleTrack, point: TrajectoryPoint) -> None:
        self._ensure_model(track)
        msg = ModelState()
        msg.model_name = track.model_name
        msg.pose.position.x = point.x
        msg.pose.position.y = point.y
        msg.pose.position.z = self.z_height
        msg.pose.orientation = quaternion_from_yaw(point.yaw)
        msg.twist.linear.x = point.vx
        msg.twist.linear.y = point.vy
        self.set_model_state(msg)

    def _hide_track(self, track: ObstacleTrack) -> None:
        if not self.spawned_models.get(track.model_name):
            return
        msg = ModelState()
        msg.model_name = track.model_name
        msg.pose.position.x = HIDDEN_POSITION[0]
        msg.pose.position.y = HIDDEN_POSITION[1]
        msg.pose.position.z = HIDDEN_POSITION[2]
        msg.pose.orientation.w = 1.0
        self.set_model_state(msg)

    def _active_tracks_for_time(self, t: float) -> List[ObstacleTrack]:
        candidates = [track for track in self.tracks if track.start_time <= t <= track.end_time]
        candidates.sort(key=lambda item: (item.start_time, item.source_id))
        return candidates[: self.max_active]

    def _publish_markers(self, samples: Dict[str, TrajectoryPoint]) -> None:
        if not self.publish_markers:
            return
        markers = MarkerArray()
        stamp = rospy.Time.now()
        for idx, (name, point) in enumerate(samples.items()):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = stamp
            marker.ns = "dynamic_obstacles"
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position = Point(x=point.x, y=point.y, z=self.z_height + 0.4)
            marker.pose.orientation = quaternion_from_yaw(point.yaw)
            marker.scale = Vector3(x=2.0, y=0.35, z=0.35)
            marker.color.r = 1.0
            marker.color.g = 0.35
            marker.color.b = 0.1
            marker.color.a = 0.9
            markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def run(self) -> None:
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            t = self._scenario_time()
            active = self._active_tracks_for_time(t)
            active_names = {track.model_name for track in active}

            for old_name, old_track in list(self.active_tracks.items()):
                if old_name not in active_names:
                    self._hide_track(old_track)
                    del self.active_tracks[old_name]

            samples = {}
            for track in active:
                point = self._sample_track(track, t)
                if point is None:
                    continue
                try:
                    self._set_track_state(track, point)
                    self.active_tracks[track.model_name] = track
                    samples[track.model_name] = point
                except rospy.ServiceException as exc:
                    rospy.logwarn("Failed to update %s: %s", track.model_name, exc)

            self._publish_markers(samples)
            rate.sleep()


if __name__ == "__main__":
    try:
        AisDynamicObstacleNode().run()
    except rospy.ROSInterruptException:
        pass
