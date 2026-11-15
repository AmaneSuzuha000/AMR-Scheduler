"""验证「尖角 vs 倒角」假说：离散图路径的直角拐点是否可执行。

背景：上一轮测出直线和圆弧都能走完（终点误差约 0.145 m），
但 L 形尖角 90 度在 60 s 内没走完（只走了 1.92/2.70 m）。

假说：问题不在「转弯」本身，而在「90 度瞬时方向突变」，
控制器必须原地转满 90 度才能继续前进，而原地转过程中容易震荡。
若假说成立，把同一个直角**倒角**（切掉 0.3 m 的斜边）就能走通。

这个结论对论文有意义：它说明调度器给出的图路径
不能原样下发给底盘，必须经过「拐角处理」，
否则抽象层面可行的路径在运动层面执行不了。
"""
import os
import sys, math, time
sys.path.insert(0, ".")
from amr_sched.bridge_client import BridgeClient

HOST = os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AMR_BRIDGE_PORT", "9300"))


def run_path(name, make_pts, max_wait=45.0, speed=0.3):
    poses, results = [], []
    c = BridgeClient(HOST, PORT, ping_interval=0.2, on_pose=poses.append)
    _o = c._dispatch
    def _d(msg):
        if msg.get("type") == "nav_result": results.append(msg)
        _o(msg)
    c._dispatch = _d
    c.start()
    c.poll(1.2)
    if not poses:
        c.stop(); return None
    start = poses[-1]
    pts = make_pts(start)
    goal = "%s-%d" % (name, int(time.time()*1000))
    n0 = len(poses)
    c._send({"type": "nav", "goal": goal, "path": pts, "max_speed": speed})
    t_end = time.time() + max_wait
    while time.time() < t_end:
        c.poll(0.5)
        if any(r.get("goal") == goal for r in results):
            break
    c.poll(0.3)
    c.stop()
    hit = [r for r in results if r.get("goal") == goal]
    res = hit[0]["status"] if hit else "timeout"
    new = poses[n0:]
    if len(new) < 3:
        return dict(name=name, res=res, err=float("nan"), dist=0, plan=0, T=0)
    fin = new[-1]
    err = math.hypot(fin.x-pts[-1][0], fin.y-pts[-1][1])
    dist = sum(math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(new, new[1:]))
    plan = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(pts, pts[1:]))
    return dict(name=name, res=res, err=err, dist=dist, plan=plan,
                T=new[-1].t_wall-new[0].t_wall)


def sharp_corner(s, leg=1.5, n=5):
    """标准 L 形：在拐点处方向突变 90 度。"""
    h = s.yaw
    pts = []
    for i in range(1, n+1):
        f = leg*i/n
        pts.append((s.x + math.cos(h)*f, s.y + math.sin(h)*f))
    cx, cy = pts[-1]
    h2 = h + math.pi/2
    for i in range(1, n+1):
        f = leg*i/n
        pts.append((cx + math.cos(h2)*f, cy + math.sin(h2)*f))
    return [(round(x,4), round(y,4)) for x, y in pts]


def chamfer_corner(s, leg=1.5, cut=0.4, n=4):
    """同样两个腿，但拐点切出一个 cut 米的斜边（倒角）。"""
    h = s.yaw
    h2 = h + math.pi/2
    pts = []
    # 第一段：走到距拐点 cut 处
    for i in range(1, n+1):
        f = (leg - cut) * i / n
        pts.append((s.x + math.cos(h)*f, s.y + math.sin(h)*f))
    # 拐点本体（倒角后的尖点）
    c0 = (s.x + math.cos(h)*(leg-cut), s.y + math.sin(h)*(leg-cut))
    c1 = (s.x + math.cos(h)*leg, s.y + math.sin(h)*leg)
    c2 = (c1[0] + math.cos(h2)*cut, c1[1] + math.sin(h2)*cut)
    # c0 -> c1 -> c2 之间插值成斜边
    for i in range(1, n+1):
        f = i / n
        pts.append((c0[0] + (c2[0]-c0[0])*f, c0[1] + (c2[1]-c0[1])*f))
    # 第二段
    for i in range(1, n+1):
        f = (leg - cut) * i / n
        pts.append((c2[0] + math.cos(h2)*f, c2[1] + math.sin(h2)*f))
    return [(round(x,4), round(y,4)) for x, y in pts]


tests = [("L尖角-A", sharp_corner),
         ("L倒角", chamfer_corner),
         ("L尖角-B", sharp_corner)]

print("%-10s %-9s | 终点误差 | 实际/计划路程 | 用时" % ("路径", "结果"))
print("-" * 62)
for nm, fn in tests:
    r = run_path(nm, fn)
    if r is None:
        print("%-10s 无位姿" % nm); continue
    print("%-10s %-9s | %7.3f m | %5.2f/%.2f m | %5.1f s" % (
          nm, r["res"], r["err"], r["dist"], r["plan"], r["T"]))
    time.sleep(1.0)