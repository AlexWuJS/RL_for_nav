"""
Grid Environment Renderer
使用matplotlib渲染栅格环境
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional, List, Tuple
import os


class SimpleRenderer:
    """简单的matplotlib渲染器"""

    def __init__(self, env, figsize: Tuple[int, int] = (10, 10)):
        """
        Args:
            env: GridDynamicObstacleEnv实例
            figsize: 图形大小
        """
        self.env = env
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.fig.show()
        self.fig.canvas.draw()
        self._pause_time = 0.01

    def render(self):
        """渲染当前帧"""
        env = self.env
        ax = self.ax
        ax.clear()

        # 获取地图信息
        grid_map = env.grid_map
        resolution = grid_map.resolution
        origin = grid_map.origin

        # 计算世界坐标范围
        world_width = grid_map.width * resolution
        world_height = grid_map.height * resolution
        world_x_min = origin[0]
        world_x_max = origin[0] + world_width
        world_y_min = origin[1]
        world_y_max = origin[1] + world_height

        # 绘制栅格地图
        occ = grid_map.occupancy.T  # 转置以匹配x,y坐标系
        extent = [world_x_min, world_x_max, world_y_min, world_y_max]
        ax.imshow(occ, cmap='gray_r', origin='lower', extent=extent, alpha=0.7, vmin=0, vmax=1)

        # 绘制动态障碍物
        if env.obstacle_manager:
            for obs_id, (ox, oy) in env.obstacle_manager.get_obstacle_positions().items():
                circle = patches.Circle((ox, oy), env.dynamic_obstacle_radius,
                                       facecolor='red', edgecolor='darkred', alpha=0.7)
                ax.add_patch(circle)

        # 绘制目标点
        goal_x, goal_y = env.goal_pos
        goal_circle = patches.Circle((goal_x, goal_y), 0.3,
                                     facecolor='green', edgecolor='darkgreen', alpha=0.7)
        ax.add_patch(goal_circle)

        # 绘制机器人
        robot_x, robot_y = env.robot_pos
        robot_yaw = env.robot_yaw

        # 机器人本体（圆形）
        robot_circle = patches.Circle((robot_x, robot_y), env.robot_radius,
                                       facecolor='blue', edgecolor='darkblue', alpha=0.9)
        ax.add_patch(robot_circle)

        # 机器人朝向（箭头）
        arrow_dx = env.robot_radius * 1.5 * np.cos(robot_yaw)
        arrow_dy = env.robot_radius * 1.5 * np.sin(robot_yaw)
        ax.arrow(robot_x, robot_y, arrow_dx, arrow_dy,
                head_width=0.1, head_length=0.05, fc='white', ec='black')

        # 绘制机器人轨迹
        if len(env.position_history) > 1:
            traj = np.array(env.position_history)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)

        # 设置坐标轴
        ax.set_xlim(world_x_min - 1, world_x_max + 1)
        ax.set_ylim(world_y_min - 1, world_y_max + 1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Grid Dynamic Obstacle Env | Step: {env.step_count} | '
                    f'Robot: ({robot_x:.1f}, {robot_y:.1f}) | '
                    f'Goal: ({goal_x:.1f}, {goal_y:.1f})')

        # 添加图例
        legend_elements = [
            patches.Patch(facecolor='gray', alpha=0.7, label='Obstacle'),
            patches.Patch(facecolor='blue', alpha=0.9, label='Robot'),
            patches.Patch(facecolor='green', alpha=0.7, label='Goal'),
            patches.Patch(facecolor='red', alpha=0.7, label='Dynamic Obstacle'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self._pause_time)

    def close(self):
        """关闭渲染器"""
        plt.close(self.fig)


class TrajectoryRecorder:
    """轨迹记录器，用于生成GIF/MP4"""

    def __init__(self, env, output_path: str = 'trajectory.gif', fps: int = 10):
        """
        Args:
            env: GridDynamicObstacleEnv实例
            output_path: 输出文件路径
            fps: 帧率
        """
        self.env = env
        self.output_path = output_path
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self._renderer = None

    def _get_renderer(self):
        """获取渲染器（延迟初始化）"""
        if self._renderer is None:
            self._renderer = SimpleRenderer(self.env, figsize=(8, 8))
        return self._renderer

    def capture_frame(self):
        """捕获当前帧"""
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        env = self.env
        grid_map = env.grid_map
        resolution = grid_map.resolution
        origin = grid_map.origin

        world_width = grid_map.width * resolution
        world_height = grid_map.height * resolution
        extent = [origin[0], origin[0] + world_width, origin[1], origin[1] + world_height]

        ax.imshow(grid_map.occupancy.T, cmap='gray_r', origin='lower', extent=extent, alpha=0.7)

        if env.obstacle_manager:
            for obs_id, (ox, oy) in env.obstacle_manager.get_obstacle_positions().items():
                circle = patches.Circle((ox, oy), env.dynamic_obstacle_radius,
                                       facecolor='red', edgecolor='darkred', alpha=0.7)
                ax.add_patch(circle)

        goal_x, goal_y = env.goal_pos
        goal_circle = patches.Circle((goal_x, goal_y), 0.3, facecolor='green', edgecolor='darkgreen', alpha=0.7)
        ax.add_patch(goal_circle)

        robot_x, robot_y = env.robot_pos
        robot_circle = patches.Circle((robot_x, robot_y), env.robot_radius,
                                       facecolor='blue', edgecolor='darkblue', alpha=0.9)
        ax.add_patch(robot_circle)

        if len(env.position_history) > 1:
            traj = np.array(env.position_history)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.5, linewidth=2)

        ax.set_xlim(origin[0] - 1, origin[0] + world_width + 1)
        ax.set_ylim(origin[1] - 1, origin[1] + world_height + 1)
        ax.set_aspect('equal')
        ax.set_title(f'Step: {env.step_count}')

        plt.tight_layout()
        fig.canvas.draw()

        # 转换为图像数组
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(data)

        plt.close(fig)

    def save(self):
        """保存为GIF"""
        if not self.frames:
            print("No frames captured")
            return

        try:
            import imageio
            imageio.mimsave(self.output_path, self.frames, fps=self.fps)
            print(f"Saved trajectory to {self.output_path}")
        except ImportError:
            print("Warning: imageio not available. Cannot save GIF.")
            # 保存最后一帧为PNG
            if self.frames:
                plt.imsave(self.output_path.replace('.gif', '.png'), self.frames[-1])
                print(f"Saved last frame to {self.output_path.replace('.gif', '.png')}")


def visualize_episode(env, max_steps: Optional[int] = None, save_path: Optional[str] = None):
    """
    可视化一个完整的episode

    Args:
        env: GridDynamicObstacleEnv实例
        max_steps: 最大步数，None表示直到终止
        save_path: 保存路径，None表示不保存
    """
    renderer = SimpleRenderer(env)

    obs, info = env.reset()
    step = 0

    while True:
        renderer.render()

        if max_steps and step >= max_steps:
            break

        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if terminated or truncated:
            renderer.render()
            print(f"Episode ended: step={step}, reward={reward:.2f}, "
                  f"collision={info.get('collision', False)}, "
                  f"goal_reached={info.get('goal_reached', False)}")
            break

    input("Press Enter to close...")
    renderer.close()


if __name__ == "__main__":
    print("Testing Grid Renderer...")

    from grid_env import GridDynamicObstacleEnv

    # 创建环境
    env = GridDynamicObstacleEnv(
        map_path=None,
        trajectory_path=None,
        use_local_patch=False,
        render_mode='human',
        seed=42
    )

    # 可视化一个episode
    visualize_episode(env, max_steps=200)
