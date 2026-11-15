# 轮轴假设夹具的本地力伺服

用于受约束、无重力和接触的 XYZ/旋转/探头模型。它是本项目新增的仿真机构控制器，不是原厂机构、真实压装力传感器或硬件急停。

Python 逐轴调用 Gazebo 力服务的早期前段案例出现过 2～3 秒仿真时间的控制空档，轮对在空档中惯性越过工作高度。此插件在每个物理步调用 `SetForce`，ROS 工艺层只发送目标；它没有调用位置瞬移接口。

在 Ubuntu 22.04 / ROS2 Humble / Gazebo Classic 11 及开发头文件已安装的环境，从工程根目录执行：

```bash
source /opt/ros/humble/setup.bash
python3 -m tools.wheelset_scene.build_axis_servo
```

独立构建目录为 `out/wheelset_axis_servo_build`，不修改已有机器人工作区。构建清单记录源码、编译器和动态库哈希；每次前段运行保存这些源码、动态库与构建清单快照。

每个模型的 `servo_targets` 接收 `Float64MultiArray`：递增序号，随后按 SDF 中轴次序给出 `[目标, 使能]`。XYZ 目标为关节位移，旋转轴目标为角速度，探头目标为行程。数组长度、数值有限性、使能取值和目标界限逐项检查，旧序号被忽略。使能为零只用于明确记录的故障注入。

每步力受 SDF 限值约束。XYZ 使用位置到速度的外环和速度反馈；旋转轴采用速度反馈；探头采用位置/速度反馈。`joint_states` 以约 40Hz 发布关节位置、速度、原生请求力、已接受序号和锁定状态。`effort` 是控制器请求力，未测量接触载荷。

探头可选 `integral_gain_n_per_m_s`（N/(m·s)），默认0，维持原位置/速度控制。非零值提供有界积分力和条件抗饱和，禁用时清零；力上限仍由 `limit` 限制。候选参数只在独立接触试验中启用，未据此校准实车。

最后接受目标超过 1 秒仿真时间时，控制器锁定：XYZ及探头以故障时当前位置作为制动/保持目标，旋转停轴，积分清零且不再累积，并拒绝后续目标。此前版本的探头故障动作是回零；历史实跑记录对应此前源码，不因本次改变获得新的故障保持结论。此锁定在当前模型生命周期内不自动清除。工艺执行器据此失败并保存现场，不将重发目标当作故障恢复。该租约依赖仿真时间，仿真暂停时不构成墙钟急停保证。

独立实跑验证：

```bash
python3 -m tools.wheelset_scene.probe_axis_servo simulation/wheelset_workshop/process_runs/my_native_probe
python3 -m tools.wheelset_scene.probe_axis_servo simulation/wheelset_workshop/process_runs/my_native_probe --verify
```

该测试在运动中停止发目标，检查越界/旧序号拒绝、约 1 秒后本地锁定、持续停稳，以及锁定后新序号仍被拒绝。它使用单独端口 11369 / ROS 域 83，仅关闭自己启动的 Gazebo。
