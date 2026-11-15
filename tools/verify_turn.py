"""直接测「原地转向能力」：发纯 wz，看实际 yaw 变化多少。

为什么测这个
----------------
控制器的转向逻辑是：
    if abs(err) > heading_tol:  publish_cmd(0.0, wz)   # 原地转
    else:                       publish_cmd(vz, wz)

SCOUT Mini 是四轮滑移转向（skid-steer）。原地转向时四轮打滑，
实际角速度通常远低于指令角速度，而且可能有明显死区与滞后。
若实际 |w| 远小于请求值，控制器就永远「转不到位」，
表现为：车在原地反复微调、路程几乎不增加、goal 超时。
这与实测到的间歇性失败吻合。
"""
import os
import sys, math, time
sys.path.insert(0, ".")
from amr_sched.bridge_client import BridgeClient

HOST = os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AMR_BRIDGE_PORT", "9300"))
poses = []
c = BridgeClient(HOST, PORT, ping_interval=0.2, on_pose=poses.append)
c.start()
c.poll(1.0)
if not poses: raise SystemExit("无位姿")

def measure(wz, seconds=3.0):
    """发固定 wz 一段时间，返回实际平均角速度与位移。"""
    n0 = len(poses)
    t0 = time.time()
    while time.time() - t0 < seconds:
        # 直接经桥的 estop 通道之外没有直接的 cmd_vel 接口，
        # 用 nav 到一个远处的点来间接驱动不合适；
        # 这里用 nav 到当前正前方 1 个点，让控制器的 heading 误差自然产生转动
        c.poll(0.1)
    return None

# 桥没有暴露直接 cmd_vel 接口，改为：发一个「原地目标」
# 即目标点就在当前位置，控制器 dist~0 时不会动；
# 因此改为测量「不同朝向要求下的达成时间」。

def turn_test(label, delta_yaw, max_wait=25.0):
    """要求车头转向 delta_yaw 后的方向，测用时。"""
    c.poll(0.5)
    if not poses: return None
    p = poses[-1]
    target_yaw = p.yaw + delta_yaw
    # 取目标方向上 1.2 m 处的点
    tx = p.x + math.cos(target_yaw) * 1.2
    ty = p.y + math.sin(target_yaw) * 1.2
    # 路径只给这一个点：控制器必须先把头转向它
    goal = "%s-%d" % (label, int(time.time()*1000))
    results = []
    _o = c._dispatch
    def _d(m):
        if m.get("type") == "nav_result": results.append(m)
        _o(m)
    c._dispatch = _d
    n0 = len(poses)
    c._send({"type": "nav", "goal": goal, "path": [[round(tx,4), round(ty,4)]], "max_speed": 0.3})
    t0 = time.time()
    while time.time() - t0 < max_wait:
        c.poll(0.3)
        if any(r.get("goal")==goal for r in results): break
    c._dispatch = _o
    new = poses[n0:]
    hit = [r for r in results if r.get("goal")==goal]
    status = hit[0]["status"] if hit else "timeout"
    if len(new) < 3:
        return dict(label=label, status=status, T=0, dyaw=0, dist=0)
    T = new[-1].t_wall - new[0].t_wall
    dyaw = new[-1].yaw - new[0].yaw
    while dyaw > math.pi: dyaw -= 2*math.pi
    while dyaw <= -math.pi: dyaw += 2*math.pi
    dist = sum(math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(new, new[1:]))
    return dict(label=label, status=status, T=T, dyaw=dyaw, dist=dist,
                want=delta_yaw)

print("%-10s %6s | %-9s | %7s | %-18s | %s" % ("标签", "目标", "结果", "用时", "转角(实际/要求)", "位移"))
print("-" * 66)
for deg in [30, 60, 90, 180]:
    r = turn_test("turn%d" % deg, math.radians(deg))
    if r is None: continue
    print("%-10s %6.0f度 | %-9s | %5.1f s | %+.3f (%+.3f) | %.2f m" % (
          r["label"], deg, r["status"], r["T"], r["dyaw"], r["want"], r["dist"]))
    time.sleep(0.8)
c.stop()