#!/usr/bin/env python3
"""
RViz 启动脚本 (Python版本 - ROS Noetic)
用于显示小车运动、雷达扫描和全局路径
"""

import os
import subprocess
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RVIZ_CONFIG = os.path.join(SCRIPT_DIR, "nav_rviz_noetic.rviz")

def main():
    print("=" * 50)
    print("  RViz 启动脚本 - 导航可视化 (ROS Noetic)")
    print("=" * 50)
    print()

    # 检查 RViz 配置文件是否存在
    if not os.path.exists(RVIZ_CONFIG):
        print(f"错误: 找不到 RViz 配置文件: {RVIZ_CONFIG}")
        return 1

    print(f"加载配置文件: {RVIZ_CONFIG}")
    print()
    print("显示内容包括:")
    print("  [✓] 网格背景 (50x50m)")
    print("  [✓] 机器人模型")
    print("  [✓] 激光雷达扫描 (红色点云)")
    print("  [✓] 全局路径 (绿色线条 + 方向箭头)")
    print("  [✓] TF 坐标系树")
    print()
    print("鼠标操作:")
    print("  - 左键拖动: 旋转视角")
    print("  - 中键拖动: 平移视角")
    print("  - 滚轮: 缩放")
    print()
    print("-" * 50)
    print("启动 RViz...")
    print("=" * 50)
    print()

    try:
        # 启动 RViz (ROS Noetic)
        subprocess.run(["rosrun", "rviz", "rviz", "-d", RVIZ_CONFIG], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动 RViz 失败: {e}")
        return 1
    except FileNotFoundError:
        print("错误: 未找到 rosrun 命令，请确保已 source ROS Noetic 环境")
        print("  source /opt/ros/noetic/setup.bash")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
