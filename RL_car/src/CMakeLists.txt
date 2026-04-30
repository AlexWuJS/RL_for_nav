cmake_minimum_required(VERSION 3.5)

# catkin's generated toplevel file still declares an older minimum CMake
# version on some ROS Noetic installs. CMake 4 rejects that unless this policy
# floor is set before including the catkin toplevel.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(/opt/ros/noetic/share/catkin/cmake/toplevel.cmake)
