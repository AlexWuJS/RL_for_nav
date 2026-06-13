#!/bin/bash

# RViz 启动脚本 (ROS Noetic版本)
# 用于显示小车运动、雷达扫描和全局路径

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RVIZ_CONFIG="$SCRIPT_DIR/nav_rviz_noetic.rviz"

echo "========================================"
echo "  RViz 启动脚本 - 导航可视化 (ROS Noetic)"
echo "========================================"
echo ""

# 检查 RViz 配置文件是否存在
if [ ! -f "$RVIZ_CONFIG" ]; then
    echo "错误: 找不到 RViz 配置文件: $RVIZ_CONFIG"
    exit 1
fi

echo "加载配置文件: $RVIZ_CONFIG"
echo ""
echo "启动的显示内容:"
echo "  - 网格背景 (50x50m)"
echo "  - 机器人模型"
echo "  - 激光雷达扫描 (红色点云)"
echo "  - 全局路径 (绿色线条 + 箭头)"
echo "  - TF 坐标系"
echo ""
echo "使用说明:"
echo "  - 鼠标左键拖动: 旋转视角"
echo "  - 鼠标中键拖动: 平移视角"
echo "  - 鼠标滚轮: 缩放"
echo ""
echo "启动 RViz..."
echo "========================================"

# 启动 RViz (ROS Noetic)
rosrun rviz rviz -d "$RVIZ_CONFIG"
