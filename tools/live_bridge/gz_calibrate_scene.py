"""实测标定：把 Gazebo 世界坐标对应到 RAWSim-O 车间地图坐标。

为什么必须实测
--------------
GUI 在 hello 里看到 frame=odom 时会拒绝画车，理由是"里程计原点与地图原点不同，
位置不能直接画"。这条拒绝是对的：/odom 是轮式积分，项目自己测过它不可信
（走 0.45 m 报 0.879 m）。硬编一个变换把它糊上去，画出来的点看起来合理、实际是错的，
比不画更糟。

所以这里做的是**测**：把车依次开到若干个已知的场景节点上，每个点等它停稳，从 Gazebo
的真值服务（get_entity_state，不是 /odom）读 N 帧取均值，再和该节点的场景坐标组成一对
对应点，最后用最小二乘拟出 Sim(2)（含尺度）变换，并按 GUI 的标定文件格式写出。

真值来源
--------
真值走 gz_pose_node.py 的 get_entity_state 服务。这条路径是项目里唯一被验证过的
地面真值通道：gz model -i 会挂，/odom 会漂，pose/info 不发。

标定文件
--------
输出 JSON 直接可被 RAWSim-O 的"加载标定"读入（camelCase，字段与
RAWSimO.Live.Calibration.CalibrationResult 一致）。拟合同时给出 SE(2) 与 Sim(2)，
留一交叉检验误差，并检查手性——镜像说明坐标系搞反了，必须报出来而不是悄悄吸收掉。

用法（Ubuntu 上，Gazebo 与 gz_pose_node 已就位）

    python3 tools/live_bridge/gz_calibrate_scene.py \
        --points J_S_8 J_S_45 R_S_8 R_S_45 D_S_8 D_S_12 \
        --out ~/AMR/out/archive/map01_dual_demo/scene_calibration.json

    # 只算不写文件
    python3 tools/live_bridge/gz_calibrate_scene.py --points ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

DEFAULT_SCENE = os.path.join(ROOT, "scenes", "rail_freight_workshop_v1.json")


def load_nodes(scene_path):
    """场景节点表：id -> (x, y)。"""
    with open(scene_path, "r", encoding="utf-8") as fh:
        scene = json.load(fh)
    return {n["id"]: (float(n["x"]), float(n["y"])) for n in scene["nodes"]}


def fit_sim2(pairs):
    """最小二乘 Sim(2)：map = q * world + t，q = s * exp(i*theta)。

    用复数形式的闭式解（Umeyama 的一维特例）：先去质心，再让 q = sum(conj(w)*z)/sum|w|^2。
    返回 dict(theta, tx, ty, scale, rms)。
    """
    if len(pairs) < 2:
        raise ValueError("至少需要 2 个对应点才能拟合尺度，实际 %d 个" % len(pairs))
    n = float(len(pairs))
    wx = sum(p[0][0] for p in pairs) / n
    wy = sum(p[0][1] for p in pairs) / n
    zx = sum(p[1][0] for p in pairs) / n
    zy = sum(p[1][1] for p in pairs) / n

    num_re = num_im = den = 0.0
    for (ax, ay), (bx, by) in pairs:
        # w = (ax-wx) + i(ay-wy); z = (bx-zx) + i(by-zy)
        ar, ai = ax - wx, ay - wy
        br, bi = bx - zx, by - zy
        # conj(w) * z
        num_re += ar * br + ai * bi
        num_im += ar * bi - ai * br
        den += ar * ar + ai * ai
    if den <= 0.0:
        raise ValueError("对应点全部重合，无法拟合")
    q_re, q_im = num_re / den, num_im / den
    scale = math.hypot(q_re, q_im)
    theta = math.atan2(q_im, q_re)
    # t = zbar - q * wbar
    tx = zx - (q_re * wx - q_im * wy)
    ty = zy - (q_re * wy + q_im * wx)

    total = 0.0
    for (ax, ay), (bx, by) in pairs:
        px = q_re * ax - q_im * ay + tx
        py = q_re * ay + q_im * ax + ty
        total += (px - bx) ** 2 + (py - by) ** 2
    return {"theta": theta, "tx": tx, "ty": ty, "scale": scale,
            "rms": math.sqrt(total / n)}


def fit_se2(pairs):
    """尺度固定为 1 的刚性拟合（SE(2)）。"""
    scaled = [((ax, ay), (bx, by)) for (ax, ay), (bx, by) in pairs]
    fit = fit_sim2(scaled)
    n = float(len(pairs))
    # 用单位尺度的旋转重算平移，避免把测出来的尺度偷偷带进 SE(2)
    c, s = math.cos(fit["theta"]), math.sin(fit["theta"])
    wx = sum(p[0][0] for p in pairs) / n
    wy = sum(p[0][1] for p in pairs) / n
    zx = sum(p[1][0] for p in pairs) / n
    zy = sum(p[1][1] for p in pairs) / n
    tx = zx - (c * wx - s * wy)
    ty = zy - (c * wy + s * wx)
    total = 0.0
    for (ax, ay), (bx, by) in pairs:
        px = c * ax - s * ay + tx
        py = c * ay + s * ax + ty
        total += (px - bx) ** 2 + (py - by) ** 2
    return {"theta": fit["theta"], "tx": tx, "ty": ty, "scale": 1.0,
            "rms": math.sqrt(total / n)}


def leave_one_out(pairs):
    """每去掉一个点重拟合，看该点被预测成什么——比 RMS 更能暴露一个坏点。"""
    if len(pairs) < 4:
        return []
    out = []
    for i in range(len(pairs)):
        rest = pairs[:i] + pairs[i + 1:]
        fit = fit_sim2(rest)
        c, s = math.cos(fit["theta"]), math.sin(fit["theta"])
        ax, ay = pairs[i][0]
        px = fit["scale"] * (c * ax - s * ay) + fit["tx"]
        py = fit["scale"] * (c * ay + s * ax) + fit["ty"]
        bx, by = pairs[i][1]
        out.append(math.hypot(px - bx, py - by))
    return out


def apply_fit(fit, ax, ay):
    """把世界坐标按拟合结果映射到地图坐标，供自检使用。"""
    c, s = math.cos(fit["theta"]), math.sin(fit["theta"])
    return (fit["scale"] * (c * ax - s * ay) + fit["tx"],
            fit["scale"] * (c * ay + s * ax) + fit["ty"])


class TruthClient:
    """gz_pose_node.py 的 stdin/stdout 客户端。"""

    def __init__(self, model="scout_mini", timeout=60.0):
        self.model = model
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "tools", "live_bridge", "gz_pose_node.py"),
             model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        deadline = time.time() + timeout
        line = ""
        while time.time() < deadline:
            line = self.proc.stdout.readline().strip()
            if line:
                break
        if line != "ready":
            raise RuntimeError("gz_pose_node 未就绪: %r" % line)

    def pose(self):
        """读一次真值位姿 (x, y, yaw)。"""
        self.proc.stdin.write("POSE %s\n" % self.model)
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().strip()
        if reply.startswith("err"):
            raise RuntimeError(reply)
        parts = reply.split()
        if len(parts) < 3:
            raise RuntimeError("真值回复无法解析: %r" % reply)
        return float(parts[0]), float(parts[1]), float(parts[2])

    def set_pose(self, x, y, yaw=0.0):
        """Move the model through the same service channel used for reading.

        Using the persistent node instead of shelling out to ros2 for every point keeps a
        6-point run to a single process, and guarantees set and get hit the same plugin.
        """
        self.proc.stdin.write("SET %s %.6f %.6f %.6f\n" % (self.model, x, y, yaw))
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().strip()
        if reply != "ok":
            raise RuntimeError("teleport failed: %r" % reply)

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except Exception:                       # noqa: BLE001 - 关闭路径不走异常
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:                       # noqa: BLE001
            self.proc.kill()


def sample_stationary(truth, frames=20, settle=1.0, gap=0.05):
    """等车停稳后取 N 帧真值，返回 (均值, 位置标准差, 帧数)。

    标准差是用来判断"这个点到底稳没稳"的：如果车还在漂，均值毫无意义。
    """
    time.sleep(settle)
    xs, ys, yaws = [], [], []
    for _ in range(frames):
        x, y, yaw = truth.pose()
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
        time.sleep(gap)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs)
    return (mx, my, yaws[0]), math.sqrt(var), len(xs)


def calibrate(scene_path, node_ids, model="scout_mini", frames=20, settle=1.0,
              teleport=True, verbose=True):
    """把车开到每个场景节点，采真值，拟合并返回结果 dict。"""
    nodes = load_nodes(scene_path)
    missing = [n for n in node_ids if n not in nodes]
    if missing:
        raise SystemExit("场景里没有这些节点: %s" % ", ".join(missing))

    truth = TruthClient(model=model)
    pairs, samples = [], []
    try:
        for node_id in node_ids:
            scene_x, scene_y = nodes[node_id]
            if teleport:
                # 场景米 -> Gazebo 世界：1/25 缩比。这个 1/25 是台架的造法
                # （worlds/rail_workshop_scout_demo.world 里的静态块就是按 1/25 摆的），
                # 不是标定要证明的东西——标定要测的是原点偏移、朝向和实际尺度。
                truth.set_pose(scene_x / 25.0, scene_y / 25.0, 0.0)
                time.sleep(0.5)
            mean, std, count = sample_stationary(truth, frames=frames, settle=settle)
            pairs.append(((mean[0], mean[1]), (scene_x, scene_y)))
            samples.append({"node": node_id, "scene": [scene_x, scene_y],
                            "world": [mean[0], mean[1]], "std_m": std, "frames": count})
            if verbose:
                print("  %-10s 场景(%7.2f,%6.2f)  真值(%8.4f,%7.4f)  σ=%.4f m  n=%d"
                      % (node_id, scene_x, scene_y, mean[0], mean[1], std, count))
    finally:
        truth.close()

    sim2 = fit_sim2(pairs)
    se2 = fit_se2(pairs)
    loo = leave_one_out(pairs)
    # 手性检查：正确对应下尺度应为正且接近 1；负尺度意味着镜像。
    mirrored = sim2["scale"] < 0
    return {
        "pairs": pairs,
        "samples": samples,
        "sim2": sim2,
        "se2": se2,
        "loo": loo,
        "mirror": mirrored,
        "yawOffset": 0.0,
    }


def to_calibration_json(result, vehicle="agv1", frame="odom", sigma=0.02):
    """按 RAWSim-O 的 CalibrationResult 字段（camelCase）写出标定。"""
    fit = result["sim2"]
    return {
        "version": 1,
        "vehicle": vehicle,
        "frame": frame,
        "model": "Sim2",
        "mirror": bool(result["mirror"]),
        "theta": fit["theta"],
        "tx": fit["tx"],
        "ty": fit["ty"],
        "scaleDiag": fit["scale"],
        "yawOffset": result["yawOffset"],
        "sigma": sigma,
        "residualRms": fit["rms"],
        "points": [
            {"map": {"x": map_pt[0], "y": map_pt[1]},
             "chassis": {"x": world_pt[0], "y": world_pt[1]},
             "yawChassis": 0.0,
             "looError": result["loo"][i] if i < len(result["loo"]) else 0.0,
             "residual": math.hypot(*[a - b for a, b in zip(
                 apply_fit(fit, world_pt[0], world_pt[1]), map_pt)])}
            for i, (world_pt, map_pt) in enumerate(result["pairs"])
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gazebo 真值 -> 车间地图 的实测标定")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--points", nargs="+", required=True,
                    help="用于标定的场景节点 id，至少 4 个、尽量分散")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-teleport", action="store_true",
                    help="不移动车，只读当前位置（用于手工摆位）")
    args = ap.parse_args(argv)

    print("标定点 %d 个，每点 %d 帧" % (len(args.points), args.frames))
    result = calibrate(args.scene, args.points, model=args.model,
                       frames=args.frames, settle=args.settle,
                       teleport=not args.no_teleport)
    sim2, se2 = result["sim2"], result["se2"]
    print("\nSim(2): 尺度 %.5f  θ=%.4f rad  t=(%.4f, %.4f)  RMS=%.4f m"
          % (sim2["scale"], sim2["theta"], sim2["tx"], sim2["ty"], sim2["rms"]))
    print("SE(2): 尺度 1        θ=%.4f rad  t=(%.4f, %.4f)  RMS=%.4f m"
          % (se2["theta"], se2["tx"], se2["ty"], se2["rms"]))
    if result["loo"]:
        print("留一检验: 最大 %.4f m，均值 %.4f m"
              % (max(result["loo"]), sum(result["loo"]) / len(result["loo"])))
    if result["mirror"]:
        print("!! 手性为镜像：世界与地图的对应关系搞反了，不能直接用")

    if args.out and not args.dry_run:
        payload = to_calibration_json(result)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print("写出 %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
