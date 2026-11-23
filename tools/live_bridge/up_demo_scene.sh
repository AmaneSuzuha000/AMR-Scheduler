#!/usr/bin/env bash
# Bring up the scaled workshop scene for the two-machine demo.
# Cleanup is by exact PID: any pattern containing the string gzserver also matches
# this script own command line and would kill the launching shell.
set +u
export DISPLAY=:0 ROS_DOMAIN_ID=42 LIBGL_ALWAYS_SOFTWARE=1
source /opt/ros/humble/setup.bash
source $HOME/scout_gazebo_ws/install/setup.bash
W=$HOME/AMR-Scheduler/tools/live_bridge/worlds/rail_workshop_scout_demo.world
XACRO=$HOME/scout_gazebo_ws/src/scout_gazebo_description/urdf/scout_mini.urdf.xacro
for p in $(pgrep -x gzserver; pgrep -x gzclient; pgrep -x robot_state_publisher; pgrep -f spawn_entity.py); do kill -9 "$p" 2>/dev/null; done
sleep 3
while pgrep -x gzserver >/dev/null; do sleep 1; done
setsid nohup gzserver --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so -s libgazebo_ros_force_system.so "$W" > /tmp/gzs.log 2>&1 < /dev/null &
sleep 18
if grep -q 'Address already in use' /tmp/gzs.log; then echo FATAL_MASTER_PORT_BUSY; tail -3 /tmp/gzs.log; exit 1; fi
setsid nohup gzclient > /tmp/gzc.log 2>&1 < /dev/null &
xacro "$XACRO" use_lidar:=true drive_mode:=planar > /tmp/scout.urdf 2>/tmp/xacro.err
echo "xacro_bytes=$(wc -c < /tmp/scout.urdf)"
setsid nohup ros2 run robot_state_publisher robot_state_publisher --ros-args -p use_sim_time:=true -p robot_description:="$(cat /tmp/scout.urdf)" > /tmp/rsp.log 2>&1 < /dev/null &
sleep 8
ros2 run gazebo_ros spawn_entity.py -entity scout_mini -topic robot_description -x 0.18 -y 1.20 -z 0.12 -Y 0 > /tmp/spawn.log 2>&1
echo "gzserver=$(pgrep -c -x gzserver) gzclient=$(pgrep -c -x gzclient) rsp=$(pgrep -c -x robot_state_publisher)"
echo '--- spawn tail ---'
tail -2 /tmp/spawn.log
