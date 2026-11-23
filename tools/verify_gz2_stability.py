#!/usr/bin/env python3
import os
"""GZ-2 稳定性验证：修复后重复跑 L 形与多拐角路径。

修复前：L 形尖角 60 s 超时，只走 1.92/2.70 m。
修复后单次：19.8 s 走完，误差 0.145 m。

但要确认不是运气。这里跑 3 次 L 形 + 2 次多拐角（S 形），
每次独立，看是否都能成功。
"""
import sys, math, time
sys.path.insert(0, ".")
from amr_sched.bridge_client import BridgeClient

HOST = os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AMR_BRIDGE_PORT", "9300"))


def run(name, make_pts, max_wait=70.0, speed=0.3):
    poses, results = [], []
    c = BridgeClient(HOST, PORT, ping_interval=0.2, on_pose=poses.append)
    _o = c._dispatch
    def _d(m):
        if m.get("type") == "nav_result":
            results.append(m)
        _o(m)
    c._dispatch = _d
    c.start()
    c.poll(1.2)
    if not poses:
        c.stop(); return None
    s = poses[-1]
    pts = make_pts(s)
    goal = "%s-%d" % (name, int(time.time()*1000))
    n0 = len(poses)
    c._send({"type": "nav", "goal": goal, "path": pts, "max_speed": speed})
    t_end = time.time() + max_wait
    while time.time() < t_end:
        c.poll(0.5)
        if any(r.get("goal") == goal for r in results):
            break
    c.poll(0.4)
    c.stop()
    hit = [r for r in results if r.get("goal") == goal]
    res = hit[0]["status"] if hit else "TIMEOUT"
    new = poses[n0:]
    if len(new) < 3:
        return dict(name=name, res=res, err=float("nan"), dist=0, plan=0, T=0)
    fin = new[-1]
    err = math.hypot(fin.x-pts[-1][0], fin.y-pts[-1][1])
    dist = sum(math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(new, new[1:]))
    plan = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(pts, pts[1:]))
    return dict(name=name, res=res, err=err, dist=dist, plan=plan,
                T=new[-1].t_wall-new[0].t_wall)


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


def sshape(s, leg=1.2, n=4):
    """S 形：右转 90 度再左转 90 度，两次拐角方向相反。"""
    h = s.yaw
    pts = [(s.x, s.y)]
    cx, cy = s.x, s.y
    for turn in (+math.pi/2, -math.pi/2):
        for i in range(1, n+1):
            f = leg*i/n
            pts.append((cx + math.cos(h)*f, cy + math.sin(h)*f))
        cx, cy = pts[-1]
        h += turn
        for i in range(1, n+1):
            f = leg*i/n
            pts.append((cx + math.cos(h)*f, cy + math.sin(h)*f))
        cx, cy = pts[-1]
    return [(round(x, 4), round(y, 4)) for x, y in pts[1:]]


tests = [("L形-#1", lshape), ("L形-#2", lshape), ("L形-#3", lshape),
         ("S形-#1", sshape), ("S形-#2", sshape)]

print("%-10s %-9s | 终点误差 | 实际/计划 | 用时    | 判定" % ("路径", "结果"))
print("-" * 70)
ok = 0
for nm, fn in tests:
    r = run(nm, fn)
    if r is None:
        print("%-10s 无位姿" % nm); continue
    good = r["res"] == "succeeded" and r["err"] < 0.4
    ok += good
    print("%-10s %-9s | %7.3f m | %4.2f/%.2f | %5.1f s | %s" % (
        nm, r["res"], r["err"], r["dist"], r["plan"], r["T"],
        "PASS" if good else "FAIL"))
    time.sleep(1.2)

print("")
print("成功 %d / %d" % (ok, len(tests)))
