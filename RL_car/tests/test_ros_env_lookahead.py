import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def install_ros_stubs():
    gymnasium = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.full(shape, low, dtype=dtype) if shape is not None and np.isscalar(low) else np.asarray(low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype) if shape is not None and np.isscalar(high) else np.asarray(high, dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

    class Env:
        pass

    gymnasium.Env = Env
    spaces.Box = Box
    gymnasium.spaces = spaces
    sys.modules["gymnasium"] = gymnasium
    sys.modules["gymnasium.spaces"] = spaces

    rospy = types.ModuleType("rospy")
    rospy.ROSException = Exception
    rospy.ServiceException = Exception
    rospy.init_node = lambda *args, **kwargs: None
    rospy.wait_for_service = lambda *args, **kwargs: None
    rospy.ServiceProxy = lambda *args, **kwargs: (lambda *a, **k: types.SimpleNamespace(success=True))
    rospy.Publisher = lambda *args, **kwargs: types.SimpleNamespace(publish=lambda *a, **k: None)
    rospy.Subscriber = lambda *args, **kwargs: None
    rospy.Rate = lambda *args, **kwargs: types.SimpleNamespace(sleep=lambda: None)
    rospy.sleep = lambda *args, **kwargs: None
    sys.modules.setdefault("rospy", rospy)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")

    class Vec3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self):
            self.linear = Vec3()
            self.angular = Vec3()

    class Point(Vec3):
        pass

    class Pose:
        def __init__(self):
            self.position = Point()
            self.orientation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    geometry_msgs_msg.Twist = Twist
    geometry_msgs_msg.Point = Point
    geometry_msgs_msg.Pose = Pose
    geometry_msgs.msg = geometry_msgs_msg
    sys.modules.setdefault("geometry_msgs", geometry_msgs)
    sys.modules.setdefault("geometry_msgs.msg", geometry_msgs_msg)

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.LaserScan = type("LaserScan", (), {})
    sensor_msgs.msg = sensor_msgs_msg
    sys.modules.setdefault("sensor_msgs", sensor_msgs)
    sys.modules.setdefault("sensor_msgs.msg", sensor_msgs_msg)

    gazebo_msgs = types.ModuleType("gazebo_msgs")
    gazebo_msgs_msg = types.ModuleType("gazebo_msgs.msg")
    gazebo_msgs_srv = types.ModuleType("gazebo_msgs.srv")
    gazebo_msgs_msg.ModelState = type("ModelState", (), {"__init__": lambda self: None})
    gazebo_msgs_msg.ModelStates = type("ModelStates", (), {})
    gazebo_msgs_srv.SetModelState = type("SetModelState", (), {})
    gazebo_msgs_srv.GetModelState = type("GetModelState", (), {})
    gazebo_msgs_srv.SpawnModel = type("SpawnModel", (), {})
    gazebo_msgs.msg = gazebo_msgs_msg
    gazebo_msgs.srv = gazebo_msgs_srv
    sys.modules.setdefault("gazebo_msgs", gazebo_msgs)
    sys.modules.setdefault("gazebo_msgs.msg", gazebo_msgs_msg)
    sys.modules.setdefault("gazebo_msgs.srv", gazebo_msgs_srv)

    visualization_msgs = types.ModuleType("visualization_msgs")
    visualization_msgs_msg = types.ModuleType("visualization_msgs.msg")
    visualization_msgs_msg.Marker = type("Marker", (), {})
    visualization_msgs_msg.MarkerArray = type("MarkerArray", (), {})
    visualization_msgs.msg = visualization_msgs_msg
    sys.modules.setdefault("visualization_msgs", visualization_msgs)
    sys.modules.setdefault("visualization_msgs.msg", visualization_msgs_msg)

    tf = types.ModuleType("tf")
    tf_transformations = types.ModuleType("tf.transformations")
    tf_transformations.euler_from_quaternion = lambda quat: (0.0, 0.0, 0.0)
    tf_transformations.quaternion_from_euler = lambda r, p, y: (0.0, 0.0, math.sin(y / 2.0), math.cos(y / 2.0))
    tf.transformations = tf_transformations
    sys.modules.setdefault("tf", tf)
    sys.modules.setdefault("tf.transformations", tf_transformations)


install_ros_stubs()

from gymnasium import spaces  # noqa: E402
from dsac_mppi.envs.frenet_utils import FrenetTransform  # noqa: E402
from dsac_mppi.envs.ros_env import MyCarEnv  # noqa: E402


class RosEnvLookaheadTests(unittest.TestCase):
    def make_env_shell(self):
        env = MyCarEnv.__new__(MyCarEnv)
        env.n_laser_beams = 400
        env.max_laser_range = 10.0
        env.lookahead_distance = 3.0
        env.target_pos = np.array([10.0, 0.0], dtype=float)
        env.current_pos = np.array([0.0, 0.0], dtype=float)
        env.current_yaw = 0.0
        env.frenet_transform = FrenetTransform(np.array([0.0, 0.0]), env.target_pos, curve_offset=0.0)
        return env

    def test_observation_space_uses_six_navigation_features(self):
        env = MyCarEnv.__new__(MyCarEnv)
        env.n_laser_beams = 400
        env.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(env.n_laser_beams + 6,), dtype=np.float32)

        self.assertEqual(env.observation_space.shape, (406,))

    def test_navigation_metrics_provide_body_frame_lookahead(self):
        env = self.make_env_shell()

        nav = env._navigation_metrics(np.array([2.0, 0.0], dtype=float), 0.0)

        self.assertAlmostEqual(nav["lookahead_s"], 5.0, places=5)
        np.testing.assert_allclose(nav["lookahead_point"], np.array([5.0, 0.0]), atol=1e-5)
        self.assertGreater(nav["lookahead_body"][0], 0.0)
        self.assertAlmostEqual(nav["lookahead_body"][1], 0.0, places=5)

    def test_navigation_metrics_are_body_frame_not_world_absolute(self):
        env = self.make_env_shell()

        nav = env._navigation_metrics(np.array([2.0, 1.0], dtype=float), 0.0)
        obs = env._build_obs_from_metrics(None, nav)

        self.assertEqual(obs.shape, (406,))
        self.assertGreater(nav["lookahead_body"][0], 0.0)
        self.assertLess(nav["lookahead_body"][1], 0.0)
        self.assertGreater(obs[-2], 0.0)
        self.assertLess(obs[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
