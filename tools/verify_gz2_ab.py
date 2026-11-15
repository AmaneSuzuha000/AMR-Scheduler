"""GZ-2 正确版：直线 / L 形 / 圆弧三种路径的跟随精度。

前两次尝试的错误（记录下来避免再犯）：

  错误 1：把 results[0] 当成当前 goal 的结果。
         桥在 nav() 里会为**被替换掉的旧 goal** 发一条 canceled，
         所以第一条 nav_result 往往属于上一个目标。必须按 goal id 过滤。

  错误 2：圆弧参数化写错，首点没落在机器人当前位置，
         机器人一开头就去追一个几米外的点，测出来的数字没有意义。

  错误 3：第一次用了 demo 路网的 80 m 长边，只等 30 s，
         0.3 m/s 下要走 267 s，必然「跑不完」，那是测试设计错误。
"""
import os
import sys, math, time, json
sys.path.insert(0, ".")
from amr_sched.bridge_client import BridgeClient

HOST = os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AMR_BRIDGE_PORT", "9300"))


def run_path(name, make_pts, max_wait=60.0, speed=0.3):
    """跑一条路径，按 goal id 等到真正的结果。"""
    poses, results = [], []
    c = BridgeClient(HOST, PORT, ping_interval=0.2, on_pose=poses.append)
    _o = c._dispatch
    def _d(msg):
        if msg.get("type") == "nav_result":
            results.append(msg)
        _o(msg)
    c._dispatch = _d
    c.start()
    c.poll(1.0)
    if not poses:
        c.stop()
        return None
    start = poses[-1]
    pts = make_pts(start)
    goal = "%s-%d" % (name, int(time.time() * 1000))
    n0 = len(poses)
    c._send({"type": "nav", "goal": goal, "path": pts, "max_speed": speed})
    # 按 goal id 等结果，最多 max_wait 秒
    t_end = time.time() + max_wait
    while time.time() < t_end:
        c.poll(0.5)
        hit = [r for r in results if r.get("goal") == goal]
        if hit:
            break
    c.poll(0.5)
    c.stop()
    hit = [r for r in results if r.get("goal") == goal]
    res = hit[0] if hit else {"status": "no_result"}
    new = poses[n0:]
    if len(new) < 3:
        return dict(name=name, res=res, n=len(new), pts=pts)
    fin = new[-1]
    err = math.hypot(fin.x - pts[-1][0], fin.y - pts[-1][1])
    dist = sum(math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(new, new[1:]))
    T = new[-1].t_wall - new[0].t_wall
    # 计划长度
    plan_len = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(pts, pts[1:]))
    return dict(name=name, res=res, n=len(new), err=err, dist=dist,
                T=T, v=dist/T if T > 0 else 0, plan_len=plan_len, pts=pts)


def straight(s, length=2.0, n=10):
    h = s.yaw
    return [(round(s.x + math.cos(h)*length*i/n, 4),
             round(s.y + math.sin(h)*length*i/n, 4)) for i in range(1, n+1)]


def lshape(s, leg=1.5, n=5):
    h = s.yaw
    pts = [(s.x, s.y)]
    for i in range(1, n+1):
        f = leg*i/n
        pts.append((s.x + math.cos(h)*f, s.y + math.sin(h)*f))
    cx, cy = pts[-1]
    h2 = h + math.pi/2
    for i in range(1, n+1):
        f = leg*i/n
        pts.append((cx + math.cos(h2)*f, cy + math.sin(h2)*f))
    return [(round(x, 4), round(y, 4)) for x, y in pts[1:]]


def arc(s, radius=1.5, n=12):
    """左转 90 度圆弧。圆心在车体左侧 R 处，首点严格等于当前位姿。"""
    h = s.yaw
    cx = s.x - radius * math.sin(h)
    cy = s.y + radius * math.cos(h)
    # 圆上角度：theta0 使点落在 (s.x, s.y)
    theta0 = math.atan2(s.y - cy, s.x - cx)
    pts = []
    for i in range(1, n+1):
        th = theta0 + (math.pi/2) * i / n
        pts.append((round(cx + radius*math.cos(th), 4),
                    round(cy + radius*math.sin(th), 4)))
    return pts


# 自检：三种路径的首点是否都接近当前位置（应当接近，因为控制器会追第一个点）
for nm, fn in [("直线", straight), ("L形90度", lshape), ("圆弧90度", arc)]:
    r = run_path(nm, fn, max_wait=60.0)
    if r is None or "err" not in r:
        print("%-10s 数据不足 res=%s" % (nm, r["res"]["status"] if r else "无"))
        continue
    print("%-10s %-10s | 终点误差 %6.3f m | 路程 %5.2f/%.2f m | 用时 %5.1f s | 平均 %5.3f m/s" % (
          nm, r["res"]["status"], r["err"], r["dist"], r["plan_len"], r["T"], r["v"]))
    time.sleep(1.0)