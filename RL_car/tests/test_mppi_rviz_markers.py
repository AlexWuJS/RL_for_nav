import sys
import types
import unittest
from pathlib import Path

import numpy as np


BEAM_MAP_DIR = Path(__file__).resolve().parents[1] / "src" / "nav_demo" / "scripts" / "beam_map"
sys.path.insert(0, str(BEAM_MAP_DIR))


class FakeStamp:
    pass


class FakeTime:
    def __init__(self, secs=0):
        self.secs = secs

    @staticmethod
    def now():
        return FakeStamp()


class FakeDuration:
    def __init__(self, secs=0.0):
        self.secs = float(secs)


class FakePublisher:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeMarker:
    LINE_STRIP = 4
    ADD = 0
    DELETEALL = 3
    ARROW = 0

    def __init__(self):
        self.header = types.SimpleNamespace(frame_id="", stamp=None)
        self.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        )
        self.scale = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.color = types.SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
        self.points = []
        self.ns = ""
        self.id = 0
        self.type = 0
        self.action = 0
        self.lifetime = None


class FakeMarkerArray:
    def __init__(self):
        self.markers = []


class FakePoint:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


def install_fake_ros_modules():
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = type("Env", (), {})
    gymnasium.spaces = types.SimpleNamespace(Box=lambda *args, **kwargs: None)
    sys.modules["gymnasium"] = gymnasium

    rospy = types.ModuleType("rospy")
    rospy.Time = FakeTime
    rospy.Duration = FakeDuration
    rospy.Publisher = FakePublisher
    rospy.init_node = lambda *args, **kwargs: None
    rospy.ROSException = Exception
    rospy.wait_for_service = lambda *args, **kwargs: None
    rospy.ServiceProxy = lambda *args, **kwargs: None
    rospy.Subscriber = lambda *args, **kwargs: None
    rospy.Rate = lambda *args, **kwargs: types.SimpleNamespace(sleep=lambda: None)
    sys.modules["rospy"] = rospy

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = type("Twist", (), {})
    geometry_msgs_msg.Point = FakePoint
    geometry_msgs_msg.Pose = type("Pose", (), {})
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

    visualization_msgs = types.ModuleType("visualization_msgs")
    visualization_msgs_msg = types.ModuleType("visualization_msgs.msg")
    visualization_msgs_msg.Marker = FakeMarker
    visualization_msgs_msg.MarkerArray = FakeMarkerArray
    sys.modules["visualization_msgs"] = visualization_msgs
    sys.modules["visualization_msgs.msg"] = visualization_msgs_msg

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.LaserScan = type("LaserScan", (), {})
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    gazebo_msgs = types.ModuleType("gazebo_msgs")
    gazebo_msgs_msg = types.ModuleType("gazebo_msgs.msg")
    gazebo_msgs_msg.ModelState = type("ModelState", (), {})
    gazebo_msgs_msg.ModelStates = type("ModelStates", (), {})
    gazebo_msgs_srv = types.ModuleType("gazebo_msgs.srv")
    gazebo_msgs_srv.SetModelState = type("SetModelState", (), {})
    gazebo_msgs_srv.GetModelState = type("GetModelState", (), {})
    gazebo_msgs_srv.SpawnModel = type("SpawnModel", (), {})
    sys.modules["gazebo_msgs"] = gazebo_msgs
    sys.modules["gazebo_msgs.msg"] = gazebo_msgs_msg
    sys.modules["gazebo_msgs.srv"] = gazebo_msgs_srv

    tf = types.ModuleType("tf")
    tf_transformations = types.ModuleType("tf.transformations")
    tf_transformations.euler_from_quaternion = lambda quat: (0.0, 0.0, 0.0)
    tf_transformations.quaternion_from_euler = lambda roll, pitch, yaw: (0.0, 0.0, 0.0, 1.0)
    tf.transformations = tf_transformations
    sys.modules["tf"] = tf
    sys.modules["tf.transformations"] = tf_transformations


install_fake_ros_modules()
from ros_env import MyCarEnv  # noqa: E402


class MppiRvizMarkerTests(unittest.TestCase):
    def make_env_shell(self):
        env = object.__new__(MyCarEnv)
        env.pub_mppi_trajectories_marker = FakePublisher()
        env.pub_mppi_action_marker = FakePublisher()
        env.current_pos = np.array([0.0, 0.0], dtype=float)
        env.current_yaw = 0.0
        return env

    def test_publish_mppi_trajectories_uses_expected_colors_and_widths(self):
        env = self.make_env_shell()
        sampled = [np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)]
        weighted = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)

        env.publish_mppi_trajectories(sampled, weighted)
        msg = env.pub_mppi_trajectories_marker.messages[-1]

        self.assertEqual(msg.markers[0].action, FakeMarker.DELETEALL)
        sample_marker = msg.markers[1]
        weighted_marker = msg.markers[2]
        self.assertEqual(sample_marker.ns, "mppi_samples")
        self.assertAlmostEqual(sample_marker.color.r, 0.2)
        self.assertAlmostEqual(sample_marker.color.g, 1.0)
        self.assertAlmostEqual(sample_marker.color.b, 0.2)
        self.assertAlmostEqual(sample_marker.color.a, 0.18)
        self.assertAlmostEqual(sample_marker.scale.x, 0.012)
        self.assertAlmostEqual(sample_marker.lifetime.secs, 0.8)
        self.assertEqual(weighted_marker.ns, "mppi_weighted")
        self.assertAlmostEqual(weighted_marker.color.r, 1.0)
        self.assertAlmostEqual(weighted_marker.color.g, 0.0)
        self.assertAlmostEqual(weighted_marker.color.b, 0.0)
        self.assertAlmostEqual(weighted_marker.color.a, 1.0)
        self.assertAlmostEqual(weighted_marker.scale.x, 0.04)
        self.assertAlmostEqual(weighted_marker.lifetime.secs, 0.8)

    def test_publish_empty_trajectories_clears_old_markers(self):
        env = self.make_env_shell()

        env.publish_mppi_trajectories([], None)
        msg = env.pub_mppi_trajectories_marker.messages[-1]

        self.assertEqual(len(msg.markers), 1)
        self.assertEqual(msg.markers[0].action, FakeMarker.DELETEALL)

    def test_publish_mppi_action_commands_draws_raw_and_optimized_arrows(self):
        env = self.make_env_shell()
        planner_state = {"position": np.array([1.0, 2.0], dtype=float), "yaw": 0.0}

        env.publish_mppi_action_commands(
            raw_action=np.array([0.5, 0.0], dtype=float),
            optimized_action=np.array([1.0, 0.4], dtype=float),
            planner_state=planner_state,
        )
        msg = env.pub_mppi_action_marker.messages[-1]

        self.assertEqual(msg.markers[0].action, FakeMarker.DELETEALL)
        raw_marker = msg.markers[1]
        optimized_marker = msg.markers[2]
        self.assertEqual(raw_marker.ns, "rl_raw_action")
        self.assertEqual(raw_marker.type, FakeMarker.ARROW)
        self.assertAlmostEqual(raw_marker.color.r, 0.1)
        self.assertAlmostEqual(raw_marker.color.g, 0.65)
        self.assertAlmostEqual(raw_marker.color.b, 1.0)
        self.assertEqual(len(raw_marker.points), 2)
        self.assertEqual(optimized_marker.ns, "mppi_final_action")
        self.assertAlmostEqual(optimized_marker.color.r, 1.0)
        self.assertAlmostEqual(optimized_marker.color.g, 0.25)
        self.assertAlmostEqual(optimized_marker.color.b, 0.0)
        self.assertEqual(len(optimized_marker.points), 2)
        self.assertGreater(optimized_marker.points[1].x, optimized_marker.points[0].x)


if __name__ == "__main__":
    unittest.main()
