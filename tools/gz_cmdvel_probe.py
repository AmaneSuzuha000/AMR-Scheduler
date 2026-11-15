"""直接在 Ubuntu 上测：给 /cmd_vel 纯角速度，Gazebo 里的车转不转。

这一步绕过桥和调度器，直接对 ROS 话题操作，
把「控制器逻辑有问题」和「底盘转不了」两种情况区分开。

三种输入对比：
  A. 纯线速度 (0.2, 0)
  B. 纯角速度 (0, 0.8)   <- 关键：滑移转向能不能原地转
  C. 线+角   (0.15, 0.5)
"""
import math, time, json, subprocess, sys

def odom():
    """读一次 /odom，返回 (x, y, yaw)。"""
    out = subprocess.run(
        ["ros2", "topic", "echo", "--once", "--field", "pose.pose", "/odom"],
        capture_output=True, text=True, timeout=15)
    txt = out.stdout
    def grab(key):
        for line in txt.splitlines():
            if line.strip().startswith(key):
                return float(line.split(":")[1].strip())
        return None
    x, y = grab("x"), grab("y")
    qz, qw = grab("z"), grab("w")
    if None in (x, y, qz, qw):
        return None
    yaw = math.atan2(2.0*qw*qz, 1.0 - 2.0*qz*qz)
    return (x, y, yaw)

def pub(vx, wz, seconds):
    """持续发布 cmd_vel 一段时间。"""
    cmd = ("ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "
           "'{linear: {x: %.3f}, angular: {z: %.3f}}'" % (vx, wz))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(seconds)
    p.terminate()
    try:
        p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        p.kill()
    # 发零速停下来
    subprocess.run("ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "'"{linear: {x: 0.0}, angular: {z: 0.0}}"'"",
                   shell=True, capture_output=True, timeout=15)
    time.sleep(0.3)

print("初始 odom:");
a = odom()
print(" ", a)
if a is None:
    sys.exit("读不到 /odom")

cases = [
    ("A 纯线速度", 0.2, 0.0, 3.0),
    ("B 纯角速度", 0.0, 0.8, 3.0),
    ("C 线+角", 0.15, 0.5, 3.0),
]

print("")
print("%-12s | 位移 | 转角 | 平均速度 | 平均角速度" % "输入")
print("-" * 66)
for name, vx, wz, sec in cases:
    b = odom()
    if b is None:
        print("%-12s 读不到 odom" % name); continue
    pub(vx, wz, sec)
    c = odom()
    if c is None:
        print("%-12s 读不到 odom" % name); continue
    d = math.hypot(c[0]-b[0], c[1]-b[1])
    dyaw = c[2] - b[2]
    while dyaw > math.pi: dyaw -= 2*math.pi
    while dyaw <= -math.pi: dyaw += 2*math.pi
    print("%-12s | %.3f m | %+.3f rad | %.3f m/s | %.3f rad/s" % (
          name, d, dyaw, d/sec, dyaw/sec))
    time.sleep(0.5)