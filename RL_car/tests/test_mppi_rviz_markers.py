import sys
import types
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


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


class FakeTwist:
    def __init__(self):
        self.linear = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)


def install_fake_ros_modules():
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = type("Env", (), {})
    gymnasium.Wrapper = type("Wrapper", (), {"__init__": lambda self, env: setattr(self, "env", env)})
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
    geometry_msgs_msg.Twist = FakeTwist
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
from dsac_mppi.envs.ros_env import MyCarEnv  # noqa: E402


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

    def test_high_level_action_maps_to_frenet_target_and_low_level_command(self):
        env = self.make_env_shell()
        env.control_mode = "high_level_frenet"
        env.dynamics_model = "ideal"
        env.LOW_LEVEL_ACTION_LOW = np.array([0.0, -0.6], dtype=np.float32)
        env.LOW_LEVEL_ACTION_HIGH = np.array([1.5, 0.6], dtype=np.float32)
        env.HIGH_LEVEL_ACTION_LOW = np.array([0.8, -2.2, 0.2], dtype=np.float32)
        env.HIGH_LEVEL_ACTION_HIGH = np.array([3.5, 2.2, 1.2], dtype=np.float32)
        env.dt = 0.1
        env.surge_time_constant = 0.6
        env.yaw_time_constant = 0.4
        env.max_du = 0.15
        env.max_dr = 0.12
        env.frenet_transform = types.SimpleNamespace(
            path_length=10.0,
            cartesian_to_frenet=lambda point: (float(point[0]), float(point[1])),
            frenet_to_cartesian=lambda s, d: np.array([s, d], dtype=float),
        )

        low_level, info = env.high_level_action_to_low_level(np.array([2.0, 1.0, 0.8], dtype=np.float32))

        self.assertEqual(low_level.shape, (2,))
        self.assertGreater(low_level[0], 0.0)
        self.assertGreater(low_level[1], 0.0)
        np.testing.assert_allclose(info["high_level_target_point"], np.array([2.0, 1.0], dtype=np.float32))

    def test_ideal_dynamics_publishes_command_without_integrating_inertia(self):
        env = self.make_env_shell()
        env.dynamics_model = "ideal"
        env.LOW_LEVEL_ACTION_LOW = np.array([0.0, -0.6], dtype=np.float32)
        env.LOW_LEVEL_ACTION_HIGH = np.array([1.5, 0.6], dtype=np.float32)
        env.dt = 0.1
        env.surge_time_constant = 0.6
        env.yaw_time_constant = 0.4
        env.max_du = 0.15
        env.max_dr = 0.12
        env.velocity = np.zeros(2, dtype=float)
        env.last_cmd_velocity = np.zeros(2, dtype=float)
        twist = env._low_level_action_to_twist(np.array([0.7, 0.2], dtype=np.float32))

        self.assertAlmostEqual(twist.linear.x, 0.7, places=6)
        self.assertAlmostEqual(twist.angular.z, 0.2, places=6)
        np.testing.assert_allclose(env.velocity, np.array([0.7, 0.2], dtype=float), atol=1e-6)
        np.testing.assert_allclose(env.last_cmd_velocity, np.array([0.7, 0.2], dtype=float), atol=1e-6)

    def test_first_order_dynamics_rate_limits_and_lags_command(self):
        env = self.make_env_shell()
        env.dynamics_model = "first_order"
        env.LOW_LEVEL_ACTION_LOW = np.array([0.0, -0.6], dtype=np.float32)
        env.LOW_LEVEL_ACTION_HIGH = np.array([1.5, 0.6], dtype=np.float32)
        env.velocity = np.zeros(2, dtype=float)
        env.last_cmd_velocity = np.zeros(2, dtype=float)
        env.dt = 0.1
        env.surge_time_constant = 0.6
        env.yaw_time_constant = 0.4
        env.max_du = 0.15
        env.max_dr = 0.12
        twist = env._low_level_action_to_twist(np.array([1.0, 0.4], dtype=np.float32))

        self.assertAlmostEqual(twist.linear.x, 0.025, places=6)
        self.assertAlmostEqual(twist.angular.z, 0.03, places=6)
        np.testing.assert_allclose(env.last_cmd_velocity, np.array([0.15, 0.12], dtype=float), atol=1e-6)
        np.testing.assert_allclose(env.velocity, np.array([0.025, 0.03], dtype=float), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
