"""连真实 Ubuntu 桥的端到端验证（记录当前现场配置下的实际行为）。"""
import os
import sys, math, json
sys.path.insert(0, ".")
from amr_sched.bridge_client import BridgeClient, AffineCalib, to_map_pose

HOST = os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AMR_BRIDGE_PORT", "9300"))
poses, stats = [], []
c = BridgeClient(HOST, PORT, ping_interval=0.2, on_pose=poses.append, on_status=stats.append)
c.start()
c.poll(10.0)
c.stop()

print("hello:", json.dumps(c.hello, ensure_ascii=False))
print("位姿 %d 条  心跳 %d 次" % (len(poses), c.n_pings))
errs = set(str(s.error) for s in stats)
print("error 取值:", errs)
print("mode 取值:", set(str(s.mode) for s in stats))

move = sum(math.hypot(b.x-a.x, b.y-a.y) for a,b in zip(poses, poses[1:]))
T = poses[-1].t_wall - poses[0].t_wall if len(poses)>1 else 0
print("累计位移 %.4f m / %.2f s = %.4f m/s" % (move, T, move/T if T>0 else 0))
print("seq %d -> %d" % (poses[0].seq, poses[-1].seq))
still = sum(1 for a,b in zip(poses, poses[1:]) if math.hypot(b.x-a.x,b.y-a.y) < 1e-6)
print("静止帧对 %d/%d" % (still, len(poses)-1))
print()
if len(poses) > 1:
    print("首末位姿:")
    p0, p1 = poses[0], poses[-1]
    print("  (%.4f, %.4f) yaw %.4f  ->  (%.4f, %.4f) yaw %.4f" % (p0.x,p0.y,p0.yaw,p1.x,p1.y,p1.yaw))

# 用首末点做一次两点的相似变换标定演示（仅演示接口，不代表真实标定）
print()
print("标定接口演示（两点解相似变换）:")
pairs = [((p0.x,p0.y),(p0.x,p0.y)), ((p1.x,p1.y),(p1.x*2,p1.y*2))]
cal = AffineCalib.from_pairs(pairs)
print("  scale=%.4f theta=%.4f tx=%.3f ty=%.3f" % (cal.scale, cal.theta, cal.tx, cal.ty))
print("  残差:", ["%.6f" % r for r in cal.residuals(pairs)])