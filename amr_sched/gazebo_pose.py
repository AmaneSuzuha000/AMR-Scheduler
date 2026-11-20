"""Gazebo 真值位姿入流：让 L2 上的"实际轨迹"真正来自 Gazebo。

数据所有权第②条说 Gazebo 是实际位姿的权威来源。那么 L2 上的轨迹点就必须来自
`/get_entity_state`，而不是拿 Python 仿真里的位置冒充实测。

为什么是常驻管道而不是每次 shell 出去问一次：

* `gz model -m <name> -i` 约 0.41 s 一次，且跑几分钟后彻底不再应答；
* 每次 `ros2 service call` 起一个解释器等服务发现，约 1 s，做跟随完全不够；
* `/odom` 是轮速积分，实测走 0.45 m 报成 0.879 m，用它当"实际位姿"是错的。

所以复用已有的 `gz_pose_node.py`（它常驻、按行问答），在它前面挂一个管道订阅者：
窗口这边开一条 ssh，远端节点打印 `ready` 后，每 `interval` 秒写一行 `POSE <model>`，
读回 `x y yaw`（Gazebo world 米）。

坐标换算只走 `GzFrame.to_network`——和把地图画到界面上用的是同一个标定，不另开一套。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

DEFAULT_HOST = os.environ.get("AMR_GAZEBO_SSH", "ubuntu@127.0.0.1")
DEFAULT_MODEL = "scout_mini"

#: 远端启动串。ROS_DOMAIN_ID 必须和跑 Gazebo 的那个 shell 一致，否则服务发现找不到节点。
REMOTE_COMMAND = (
    "export ROS_DOMAIN_ID=42; source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
    "cd ~/AMR && python3 tools/live_bridge/gz_pose_node.py"
)


def parse_pose_line(line: str):
    """把 `x y yaw` 解析成三元组；`ready`、空行、错误信息一律返回 None。

    远端在服务没起来时会打印异常堆栈，那些行不能被当成坐标——否则一个
    `Traceback ...` 里的数字会被画到地图上。只接受**恰好三个**可解析的浮点数。
    """
    parts = line.split()
    if len(parts) != 3:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None


class GazeboPoseFeed:
    """在后台线程里维护"最新一次真值位姿"。

    读不到就是读不到：`latest()` 返回 None，调用方必须自己决定退回仿真位姿并**标注**
    来源，不能拿上一次的值一直充数——那会在图上画出一条不动的假轨迹。
    """

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
                 interval: float = 0.2, command: str = REMOTE_COMMAND) -> None:
        self.host = host
        self.model = model
        self.interval = max(0.02, float(interval))
        self.command = command
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pose: tuple[float, float, float] | None = None
        self._stamp: float = 0.0
        self.ready = threading.Event()
        self.errors: list[str] = []

    # ---- 生命周期 -------------------------------------------------------
    def start(self) -> "GazeboPoseFeed":
        self._proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-T", self.host, self.command],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._loop, name="gz-pose-feed", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False

    # ---- 读取 -----------------------------------------------------------
    def latest(self, max_age: float = 3.0):
        """最新位姿 `(x, y, yaw, age)`；过旧或没有则 None。"""
        with self._lock:
            if self._pose is None:
                return None
            age = time.monotonic() - self._stamp
            if age > max_age:
                return None
            x, y, yaw = self._pose
        return (x, y, yaw, age)

    def ingest(self, line: str) -> bool:
        """喂一行远端输出。返回 True 表示这行是位姿。"""
        stripped = line.strip()
        if stripped == "ready":
            self.ready.set()
            return False
        pose = parse_pose_line(stripped)
        if pose is None:
            if stripped:
                self.errors.append(stripped[:200])
                del self.errors[:-20]
            return False
        with self._lock:
            self._pose = pose
            self._stamp = time.monotonic()
        return True

    # ---- 线程 -----------------------------------------------------------
    def _loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None or proc.stdin is None:
            return
        while not self._stop.is_set():
            try:
                proc.stdin.write("POSE %s\n" % self.model)
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (BrokenPipeError, ValueError, OSError):
                self.errors.append("pipe closed")
                return
            if not line:
                self.errors.append("remote ended")
                return
            self.ingest(line)
            self._stop.wait(self.interval)


def main(argv=None) -> int:
    """命令行自检：连上远端，打印若干条真值位姿。"""
    import argparse
    ap = argparse.ArgumentParser(description="poll Gazebo ground-truth pose over ssh")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args(argv)

    with GazeboPoseFeed(host=args.host, model=args.model, interval=args.interval) as feed:
        if not feed.ready.wait(timeout=40.0):
            print("远端未就绪", file=sys.stderr)
            for e in feed.errors[-3:]:
                print("  " + e, file=sys.stderr)
            return 1
        for _ in range(args.count):
            time.sleep(args.interval)
            latest = feed.latest(max_age=10.0)
            if latest is None:
                print("no pose")
                continue
            x, y, yaw, age = latest
            print("x=%.4f y=%.4f yaw=%.4f age=%.2fs" % (x, y, yaw, age))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
