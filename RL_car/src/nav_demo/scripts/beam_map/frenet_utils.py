import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple

class FrenetTransform:
    """
    改进的曲线 Frenet 坐标系转换类
    内部使用二次贝塞尔曲线将起点和终点连接为平滑曲线，并进行离散化处理。
    """

    def __init__(self, start_point: np.ndarray, end_point: np.ndarray, curve_offset: float = 2.0, num_waypoints: int = 500):
        """
        初始化曲线 Frenet 坐标系

        Args:
            start_point: 路径起点 [x, y]
            end_point: 路径终点 [x, y]
            curve_offset: 曲线偏离直线的程度（正数向左弯，负数向右弯）
            num_waypoints: 离散化路点的数量
        """
        self.start_point = np.array(start_point, dtype=float)
        self.end_point = np.array(end_point, dtype=float)
        self.num_waypoints = num_waypoints

        # 1. 自动生成控制点以构建曲线
        mid_point = (self.start_point + self.end_point) / 2.0
        line_vec = self.end_point - self.start_point
        # 计算垂直于直线段的法向量
        normal_vec = np.array([-line_vec[1], line_vec[0]])
        normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-6)
        
        # 控制点 = 中点 + 法线方向偏移
        control_point = mid_point + normal_vec * curve_offset

        # 2. 生成离散的贝塞尔曲线路点
        t = np.linspace(0, 1, num_waypoints)[:, np.newaxis]
        # 二次贝塞尔曲线公式: P(t) = (1-t)^2*P0 + 2t(1-t)*P1 + t^2*P2
        self.waypoints = (1 - t)**2 * self.start_point + 2 * (1 - t) * t * control_point + t**2 * self.end_point

        # 3. 计算每个路点的累积弧长 (s)、切线 (tangent) 和法线 (normal)
        self.s_values = np.zeros(num_waypoints)
        self.tangents = np.zeros_like(self.waypoints)
        self.normals = np.zeros_like(self.waypoints)
        self.path_angles = np.zeros(num_waypoints)

        # 计算分段长度并累加得到 s
        diffs = np.diff(self.waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.s_values[1:] = np.cumsum(segment_lengths)
        self.path_length = self.s_values[-1]

        # 计算切线和法线 (使用前向差分)
        for i in range(num_waypoints - 1):
            self.tangents[i] = diffs[i] / (segment_lengths[i] + 1e-6)
            self.normals[i] = np.array([-self.tangents[i][1], self.tangents[i][0]])
            self.path_angles[i] = math.atan2(self.tangents[i][1], self.tangents[i][0])
            
        # 最后一个点复制前一个点的属性，防止越界
        self.tangents[-1] = self.tangents[-2]
        self.normals[-1] = self.normals[-2]
        self.path_angles[-1] = self.path_angles[-2]

    def cartesian_to_frenet(self, point: np.ndarray) -> Tuple[float, float]:
        """将笛卡尔坐标转换为Frenet坐标"""
        point = np.array(point)
        
        # 1. 找到距离输入点最近的路点索引
        dists = np.linalg.norm(self.waypoints - point, axis=1)
        closest_idx = np.argmin(dists)

        # 2. 获取该局部的参考信息
        closest_wp = self.waypoints[closest_idx]
        local_tangent = self.tangents[closest_idx]
        local_normal = self.normals[closest_idx]
        local_s = self.s_values[closest_idx]

        # 3. 计算点相对于最近路点的局部向量
        local_vec = point - closest_wp

        # 4. 投影计算真实的 s 和 d
        # s = 最近路点的s + 在切线上的投影偏差
        s = local_s + np.dot(local_vec, local_tangent)
        # d = 在法线上的投影
        d = np.dot(local_vec, local_normal)

        return s, d

    def frenet_to_cartesian(self, s: float, d: float) -> np.ndarray:
        """将Frenet坐标转换为笛卡尔坐标"""
        # 限制 s 在合法范围内
        s = np.clip(s, 0, self.path_length)
        
        # 找到对应的 s 区间
        idx = np.searchsorted(self.s_values, s)
        if idx == 0:
            idx = 1
        elif idx == self.num_waypoints:
            idx = self.num_waypoints - 1

        # 在相邻两个路点之间进行线性插值，使坐标更加平滑
        s0, s1 = self.s_values[idx-1], self.s_values[idx]
        ratio = (s - s0) / (s1 - s0 + 1e-6)

        wp = self.waypoints[idx-1] + ratio * (self.waypoints[idx] - self.waypoints[idx-1])
        normal = self.normals[idx-1] + ratio * (self.normals[idx] - self.normals[idx-1])
        normal = normal / (np.linalg.norm(normal) + 1e-6) # 归一化

        # 笛卡尔坐标 = 路径基准点 + d * 法向量
        cartesian_point = wp + d * normal
        return cartesian_point
    
    def generate_path_points(self, num_points: int = None) -> np.ndarray:
        """
        生成路径上的采样点（兼容你原有的 ROS 可视化接口）

        Args:
            num_points: 采样点数量。传入以兼容旧代码的 num_points=50。
        """
        # 如果没有指定数量，或者指定的数量刚好等于我们离散的500个点，直接返回
        if num_points is None or num_points == self.num_waypoints:
            return self.waypoints
            
        # 为了兼容你 ros_env.py 里的 num_points=50，我们在生成的500个点里均匀降采样
        indices = np.linspace(0, self.num_waypoints - 1, num_points, dtype=int)
        return self.waypoints[indices]

    def get_heading_error(self, robot_yaw: float, s: float) -> float:
        """获取机器人朝向与当前路径方向的角度误差"""
        s = np.clip(s, 0, self.path_length)
        idx = np.searchsorted(self.s_values, s)
        idx = min(idx, self.num_waypoints - 1)
        
        local_path_angle = self.path_angles[idx]
        error = local_path_angle - robot_yaw

        # 归一化到 [-pi, pi]
        while error > math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        return error


def frenet_reward(delta_s: float, frenet_d: float, heading_error: float) -> dict:
    """强化学习奖励函数"""
    components = {}

    # 1. 纵向进度奖励 (稍微提高，鼓励向前探索)
    components['s_progress'] = delta_s * 30.0

    # 2. 横向偏离惩罚 (建立 0.5m 的免罚走廊，超出后给出缓和的线性惩罚)
    abs_d = abs(frenet_d)
    if abs_d <= 0.5:
        raw_lateral_penalty = 0.0
    else:
        raw_lateral_penalty = -(abs_d - 0.5) * 1.5
    components['lateral_deviation'] = max(raw_lateral_penalty, -5.0)

    # 3. 朝向误差惩罚 (保持柔和)
    heading_normalized = abs(heading_error) / math.pi
    components['heading_penalty'] = -heading_normalized * 1.5

    reward = sum(components.values())
    return {'total': reward, 'components': components}


if __name__ == "__main__":
    # ==========================================
    # 交互与可视化测试
    # ==========================================
    
    # 1. 创建曲线路径：起点(0,0)，终点(10,0)，向侧边弯曲2.0米
    frenet = FrenetTransform(start_point=[0, 0], end_point=[10, 0], curve_offset=2.5)

    # 2. 模拟一个自动驾驶车辆当前的位置和朝向
    car_pos = np.array([4.0, 3.5])
    car_yaw = math.radians(45) # 假设车头朝向 45 度

    # 3. 计算 Frenet 坐标和朝向误差
    s, d = frenet.cartesian_to_frenet(car_pos)
    heading_err = frenet.get_heading_error(car_yaw, s)
    
    # 找回车在路径上的投影点 (为了画图)
    proj_point = frenet.frenet_to_cartesian(s, 0)

    print(f"--- 坐标转换结果 ---")
    print(f"车辆坐标: x={car_pos[0]}, y={car_pos[1]}")
    print(f"Frenet坐标: s={s:.2f} (沿路径距离), d={d:.2f} (横向偏离)")
    print(f"朝向误差: {math.degrees(heading_err):.2f} 度")

    # 4. 可视化
    plt.figure(figsize=(10, 6))
    
    # 画出生成的曲线路径
    waypoints = frenet.waypoints
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=2, label='Curved Reference Path')
    
    # 画出起点和终点
    plt.plot(0, 0, 'go', markersize=8, label='Start Point')
    plt.plot(10, 0, 'ro', markersize=8, label='End Point')

    # 画出车辆位置
    plt.plot(car_pos[0], car_pos[1], 'ks', markersize=8, label='Car Position')
    
    # 画出投影点
    plt.plot(proj_point[0], proj_point[1], 'co', markersize=6, label='Projection on Path')

    # 画出 d (横向偏离线)
    plt.plot([car_pos[0], proj_point[0]], [car_pos[1], proj_point[1]], 'k--', linewidth=1.5, label=f'd offset ({d:.2f}m)')

    # 画出车辆朝向箭头
    arrow_len = 1.0
    plt.arrow(car_pos[0], car_pos[1], arrow_len*math.cos(car_yaw), arrow_len*math.sin(car_yaw), 
              head_width=0.2, color='k', label='Car Heading')

    # 图表设置
    plt.title("Curved Frenet Coordinate Transform Visualization")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal') # 保持XY比例一致，确保看到的曲线和垂线是真实的几何关系
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()