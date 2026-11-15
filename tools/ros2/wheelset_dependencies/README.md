# 轮轴车间双车仿真依赖

`scout_gazebo_description`是本次Ubuntu实跑主机已有的本地派生描述包快照，包括两个xacro、两个DAE、package.xml和CMakeLists.txt。仅复制运行所需文件，不包含旧备份或整个历史vendor仓库；CMake安装列表移除了未随包提供且本执行器不使用的launch/rviz目录。

本地xacro说明其网格改编自AgileX/community资产，package.xml声明BSD。该快照不构成上游授权或硬件标定审计；后续对外发布前仍需补齐准确网格来源及原始许可证。两个留存vendor检出版本为community `d26c75098d34ec2a62bad1b4c91b9576bc4b06c4`、official `27633a956c845903ee630538afeb17fe70afdd84`，尚未逐网格确认二者对应关系。

在Ubuntu22.04、ROS2 Humble、Gazebo Classic11及gazebo_ros/xacro已安装的环境中，从解压工程根目录运行：

```bash
mkdir -p "$PWD/out/wheelset_robot_ws/src"
cp -a tools/ros2/wheelset_dependencies/scout_gazebo_description out/wheelset_robot_ws/src/
cd out/wheelset_robot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
cd ../..
python3 -m tools.wheelset_scene.run_light_material_gazebo --output simulation/wheelset_workshop/process_runs/my_fleet
```

执行器运行时通过`metric_model.py`添加车体碰撞、轮胎摩擦与坐标修正、命令隔离。此helper位于`tools/ros2/amr_gazebo_guard/launch/metric_model.py`。车轮差速插件驱动车体，位姿由Gazebo真值服务返回；未接定位/Nav2。原实跑主机source的是`/home/suzuha/scout_gazebo_ws/install/setup.bash`，该绝对路径无需在新环境沿用。
