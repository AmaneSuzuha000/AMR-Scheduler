#!/usr/bin/env python3
"""干净的差速验证：只信 Gazebo 真实位姿，不用 /odom。

上一轮的数据全部作废，原因：
  - `ros2 topic pub` 用 shell timeout 包着会留下孤儿进程持续发 /cmd_vel，
    导致机器人被一路推着飞出去（真实位姿跑到 x=-103, y=343, z=0.5）。
  - 那批「位移 28 m / 32 m」不是模型问题，是我的测试污染。

本脚本的两条硬规则：
  1. 只用 `gz model -m scout_mini -i` 读真实位姿，绝不拿 /odom 当真相
     （/odom 是 diff_drive 插件的轮速积分，脱离实际位姿会漂）。
  2. 发布指令用 `--times` 让进程自己退出，并每次事后 pkill 兜底，
     保证测量之间没有残留发布者。
"""
import subprocess, re, math, time, sys

def sh(cmd, timeout=25):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None

def gt_pose():
    """读 Gazebo 里 scout_mini 的真实位姿 -> (x, y, yaw)。"""
    r = sh("gz model -m scout_mini -i 2>/dev/null")
    if r is None:
        return None
    t = r.stdout
    m = re.search(r'^pose \{\s*position \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)',
                  t, re.S | re.M)
    q = re.search(r'orientation \{\s*x: (-?[0-9.eE+-]+)\s*y: (-?[0-9.eE+-]+)'
                  r'\s*z: (-?[0-9.eE+-]+)\s*w: (-?[0-9.eE+-]+)', t)
    if not (m and q):
        return None
    x, y = float(m.group(1)), float(m.group(2))
    qx, qy, qz, qw = (float(q.group(i)) for i in range(1, 5))
    yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return (x, y, yaw)

def nonzero_pub():
    """返回当前 /cmd_vel 发布者数量，确认测量之间没残留。"""
    r = sh("ros2 topic info /cmd_vel 2>/dev/null | grep -i 'publisher count'")
    if r is None:
        return -1
    m = re.search(r'(\d+)', r.stdout)
    return int(m.group(1)) if m else -1

def stop_and_settle():
    sh('ros2 topic pub --times 5 --rate 20 /cmd_vel geometry_msgs/msg/Twist '
       '"{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1', timeout=20)
    sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
    time.sleep(1.5)

def drive(vx, wz, seconds):
    n = max(1, int(seconds * 20))
    sh('ros2 topic pub --times %d --rate 20 /cmd_vel geometry_msgs/msg/Twist '
       '"{linear: {x: %.3f}, angular: {z: %.3f}}" >/dev/null 2>&1' % (n, vx, wz),
       timeout=int(seconds) + 25)
    sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
    stop_and_settle()

def wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a <= -math.pi: a += 2*math.pi
    return a

print("=== 环境检查 ===")
sh("pkill -9 -f 'ros2 topic pub' 2>/dev/null")
time.sleep(1.5)
stop_and_settle()
print("  /cmd_vel 发布者: %d (应为 0)" % nonzero_pub())
p0 = gt_pose()
print("  起始真实位姿: (%.4f, %.4f, %.4f)" % p0 if p0 else "  读不到位姿！")
if p0 is None:
    sys.exit(1)

# 静止基线：不放任何指令，看 3 秒内漂多少
b = gt_pose(); time.sleep(3.0); a = gt_pose()
drift = math.hypot(a[0]-b[0], a[1]-b[1])
print("  静止 3 s 漂移: %.6f m (应接近 0)" % drift)
print("")

cases = [
    ("纯角速度 +0.8",  0.0,  0.8, 3.0),
    ("纯角速度 -0.8",  0.0, -0.8, 3.0),
    ("纯线速度  0.2",  0.2,  0.0, 3.0),
    ("线+角 0.15/0.5", 0.15, 0.5, 3.0),
]

print("%-16s | 真实位移 | yaw变化 | 期望yaw | 期望位移" % "输入")
print("-" * 68)
for name, vx, wz, sec in cases:
    stop_and_settle()
    nb = nonzero_pub()
    b = gt_pose()
    if b is None:
        print("%-16s 读不到位姿" % name); continue
    drive(vx, wz, sec)
    a = gt_pose()
    if a is None:
        print("%-16s 读不到位姿" % name); continue
    d = math.hypot(a[0]-b[0], a[1]-b[1])
    dyaw = wrap(a[2] - b[2])
    print("%-16s | %7.4f m | %+7.4f | %+7.4f | %6.3f  [pub=%d]" % (
        name, d, dyaw, wz*sec, vx*sec, nb))
print("")
print("判据：")
print("  纯角速度  -> 位移应 ≈ 0，yaw 变化应 ≈ ±2.4 rad")
print("  纯线速度  -> 位移应 ≈ 0.6 m，yaw 变化应 ≈ 0")
