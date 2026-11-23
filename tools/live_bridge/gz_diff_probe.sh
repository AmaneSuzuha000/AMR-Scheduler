#!/bin/bash
# 在 Ubuntu 主机上量 --drive_mode:=diff 的实际运动（MAP-01E 用）。
#
# 为什么需要它：drive_mode:=planar 的 libgazebo_ros_planar_move.so 直接改写平面位姿，
# 纯角速度指令根本不转；而 diff 档虽然能加载，实测行为是否可用必须量出来，不能看日志猜。
#
# 用法：bash gz_diff_probe.sh            # 默认 11348 端口 / ROS_DOMAIN_ID=45，不干扰常驻实例
# 输出：P0 / P1（纯角速度 5 s）/ P2（直行 5 s）三组位姿，直接可与指令值对比。
set +u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-45}
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://127.0.0.1:11348}
source /opt/ros/humble/setup.bash
source "$HOME/scout_gazebo_ws/install/setup.bash" 2>/dev/null
OUT=/tmp/gz_diff_probe.out
exec > "$OUT" 2>&1
ros2 launch scout_gazebo_bringup scout_mini_empty.launch.py gui:=false drive_mode:=diff > /tmp/gz_diff_probe_launch.log 2>&1 &
LPID=$!
cleanup() { kill -INT $LPID 2>/dev/null; sleep 2; pkill -9 -P $LPID 2>/dev/null; kill -9 $LPID 2>/dev/null; }
trap cleanup EXIT
UP=0
for k in $(seq 1 45); do
  if grep -q "Successfully spawned" /tmp/gz_diff_probe_launch.log 2>/dev/null; then UP=1; echo "spawned_after=${k}s"; break; fi
  sleep 1
done
[ "$UP" != "1" ] && { echo SPAWN_FAILED; tail -5 /tmp/gz_diff_probe_launch.log; exit 2; }
sleep 3
pose() { ros2 service call /get_entity_state gazebo_msgs/srv/GetEntityState "{name: 'scout_mini', reference_frame: 'world'}" 2>/dev/null | tr "\n" " " | sed -E "s/.*position: x: (-?[0-9.e-]+) y: (-?[0-9.e-]+) z: (-?[0-9.e-]+).*orientation: x: (-?[0-9.e-]+) y: (-?[0-9.e-]+) z: (-?[0-9.e-]+) w: (-?[0-9.e-]+).*/x=\1 y=\2 z=\3 qz=\6 qw=\7/"; }
echo "P0  $(pose)"
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 1.0}}" > /dev/null 2>&1 &
PP=$!; sleep 5; kill -9 $PP 2>/dev/null; sleep 1
echo "P1  $(pose)   # 指令 1.0 rad/s x 5 s, linear=0"
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}, angular: {z: 0.0}}" > /dev/null 2>&1 &
PP=$!; sleep 5; kill -9 $PP 2>/dev/null; sleep 1
echo "P2  $(pose)   # 指令 0.3 m/s x 5 s, angular=0"
ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{}" > /dev/null 2>&1 &
PP=$!; sleep 1; kill -9 $PP 2>/dev/null
echo DONE