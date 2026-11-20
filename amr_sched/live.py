"""实时状态推送服务：把运行中的仿真逐帧推给 C# / RAWSim-O。

为什么是 TCP 而不是 WebSocket
----------------------------
C# 侧现有代码用的是 .NET Framework 4.0，引入 WebSocket 要额外依赖。
按行分隔的 JSON（JSONL over TCP）用 ``TcpClient`` + ``StreamReader.ReadLine``
就能消费，零依赖，也方便用 telnet 直接看。

协议
----
连接后服务端先发一行 ``hello``（schema / 路网 / meta），
随后按 ``--rate`` 指定的节奏每帧发一行 ``frame``。
客户端不需要任何握手或应答。

用法::

    python -m amr_sched.live --port 8770 --n-tasks 50 --n-agvs 4

然后 C# 侧：

    var client = new TcpClient("127.0.0.1", 8770);
    var reader = new StreamReader(client.GetStream(), Encoding.UTF8);
    while (true) { var line = reader.ReadLine(); /* 解析 JSON */ }
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
import threading
import time
from typing import Any

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import FrameBuilder, PoseSource, frame_to_line


class LiveServer:
    """一个极简的广播式 TCP 服务：所有客户端收到同一份帧流。"""

    def __init__(self, sim: Simulator, host: str = "127.0.0.1",
                 port: int = 8770, rate: float = 20.0,
                 steps_per_frame: int = 5,
                 gazebo_feed: object | None = None,
                 gazebo_model: str = "scout_mini",
                 gazebo_frame: object | None = None) -> None:
        self.sim = sim
        self.host = host
        self.port = port
        self.rate = rate
        self.steps_per_frame = steps_per_frame
        self.builder = FrameBuilder(sim, PoseSource.SIM)
        #: 三维验证模式：物理车的真值位姿由 Gazebo 提供，作为独立实体显示。
        self.gazebo_feed = gazebo_feed
        self.gazebo_model = gazebo_model
        self.gz_frame = gazebo_frame
        self.clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _refresh_external(self) -> None:
        """把 Gazebo 真值搬进当前帧的外部位姿表。

        读不到就**清空**而不是沿用上一帧：一条停在原地的"实测轨迹"比没有轨迹更糟，
        它会让看图的人以为车真的没动过。
        """
        if self.gazebo_feed is None:
            return
        latest = self.gazebo_feed.latest()
        if latest is None:
            self.builder.clear_physical()
            return
        x, y, yaw, _age = latest
        nx, ny = self.gz_frame.to_network(x, y)
        self.builder.set_physical(nx, ny, self.gz_frame.to_network_yaw(yaw),
                                  name=self.gazebo_model)

    # ---- 广播 ----
    def _broadcast(self, payload: bytes) -> None:
        dead = []
        with self._lock:
            for c in self.clients:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    # ---- 客户端接入 ----
    def _serve_client(self, conn: socket.socket) -> None:
        try:
            conn.sendall(frame_to_line(self.builder.hello()))
            conn.sendall(frame_to_line(self.builder.frame()))
        except OSError:
            conn.close()
            return
        with self._lock:
            self.clients.append(conn)
        print("[live] 客户端接入，当前 %d 个" % len(self.clients))
        # 连接建立后本线程只负责"探活"：读不到东西说明客户端走了
        try:
            while not self._stop.is_set():
                conn.settimeout(1.0)
                try:
                    data = conn.recv(64)
                    if not data:
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            with self._lock:
                if conn in self.clients:
                    self.clients.remove(conn)
            try:
                conn.close()
            except OSError:
                pass
            print("[live] 客户端断开，当前 %d 个" % len(self.clients))

    # ---- 仿真推进 + 推送 ----
    def wait_for_client(self, timeout: float = 60.0) -> bool:
        """等至少一个客户端接入。

        为什么必须等：仿真在 60 帧/秒下 8 秒就跑完 30 个任务，
        如果不等，人还没打开界面服务端就已经结束了——客户端连上
        只会看到"连接被拒绝"。**实时联动的语义是"有人在看才跑"**。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self.clients:
                    return True
            if self._stop.is_set():
                return False
            time.sleep(0.05)
        return False

    def run(self, max_steps: int | None = None,
            wait_client: float = 60.0) -> None:
        sim = self.sim
        # 先把仿真 reset 好，但**不推进**，等观众到场
        if wait_client > 0:
            print("[live] 等待客户端接入（最长 %.0f 秒）…" % wait_client)
            if not self.wait_for_client(wait_client):
                print("[live] 没有客户端接入，本次不启动仿真")
                self._stop.set()
                return
            print("[live] 客户端已接入，开始推进")
        sim.reset()
        period = 1.0 / max(1e-6, self.rate)
        t0 = time.monotonic()
        frames_sent = 0
        while not sim.is_finished():
            if max_steps is not None and sim._step_cursor >= max_steps:
                break
            for _ in range(self.steps_per_frame):
                if not sim.step_once():
                    break
            self._refresh_external()
            self._broadcast(frame_to_line(self.builder.frame()))
            frames_sent += 1
            target = t0 + frames_sent * period
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._broadcast(frame_to_line({
            "type": "done",
            "step": sim._step_cursor,
            "summary": sim._build_result().summary(),
        }))
        print("[live] 完成，共推送 %d 帧" % frames_sent)
        self._stop.set()
        # 主动断开所有客户端，否则对端 ReadLine 会一直等下去
        with self._lock:
            for c in list(self.clients):
                try:
                    c.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self.clients.clear()

    def serve_forever(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(1.0)
        print("[live] 监听 %s:%d" % (self.host, self.port))
        worker = threading.Thread(target=self.run, daemon=True)
        worker.start()
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._serve_client,
                                 args=(conn,), daemon=True).start()
        except KeyboardInterrupt:
            print("[live] 收到中断")
        finally:
            self._stop.set()
            with self._lock:
                for c in self.clients:
                    try:
                        c.close()
                    except OSError:
                        pass
            srv.close()


def build_sim(n_tasks: int, n_agvs: int, seed: int,
              layout: str = "ring") -> Simulator:
    """Builds the simulator the stream serves.

    The layout argument used to be accepted and ignored, so every stream served the ring
    no matter what was asked for. It is honoured now: "workshop" builds the 62-node
    workshop scene the console loads, "ring" keeps the synthetic study instance.
    """
    if layout == "workshop":
        from amr_sched.workshop import build_scenario

        net, tasks, agvs = build_scenario(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    elif layout == "ring":
        sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
        net, tasks, agvs = generate_scenario(sc)
    else:
        raise ValueError("unknown layout: %s" % layout)
    return Simulator(
        net, tasks, agvs, StaticDijkstraPlanner(), GreedyAllocator(),
        SimulationConfig(max_steps=200_000, seed=seed,
                         record_trajectory=False),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AMR 实时状态推送")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--n-agvs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=20.0,
                    help="每秒推送帧数")
    ap.add_argument("--steps-per-frame", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--layout", choices=("workshop", "ring"), default="workshop",
                    help="场景：车间（默认，与控制台加载的地图一致）或合成环线")
    ap.add_argument("--gazebo-pose", default=None, metavar="HOST",
                    help="三维验证模式：从该 ssh 主机读 Gazebo 真值位姿，"
                         "如 user@host（或设 AMR_GAZEBO_SSH）；不给则整帧都是仿真位姿")
    ap.add_argument("--gazebo-model", default="scout_mini",
                    help="Gazebo 里那台物理车的模型名")
    args = ap.parse_args(argv)

    sim = build_sim(args.n_tasks, args.n_agvs, args.seed, layout=args.layout)

    feed = None
    gz_frame = None
    if args.gazebo_pose:
        # 只在这里 import：算法研究模式不该因为装了 ssh 才能跑
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "live_bridge"))
        from gz_frame import DEMO_FRAME  # noqa: E402
        from amr_sched.gazebo_pose import GazeboPoseFeed  # noqa: E402

        feed = GazeboPoseFeed(host=args.gazebo_pose, model=args.gazebo_model).start()
        gz_frame = DEMO_FRAME
        print("[live] 三维验证模式：物理车 %s 的真值位姿来自 %s"
              % (args.gazebo_model, args.gazebo_pose))

    server = LiveServer(sim, host=args.host, port=args.port,
                        rate=args.rate,
                        steps_per_frame=args.steps_per_frame,
                        gazebo_feed=feed, gazebo_model=args.gazebo_model,
                        gazebo_frame=gz_frame)
    if args.max_steps is not None:
        # 只跑有限步：套一层薄包装，不改 LiveServer 语义
        import threading as _th
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((server.host, server.port))
        srv.listen(8)
        srv.settimeout(1.0)
        print("[live] 监听 %s:%d" % (server.host, server.port))
        _th.Thread(target=server.run, kwargs={"max_steps": args.max_steps},
                   daemon=True).start()
        end = time.time() + 3600
        while time.time() < end and not server._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if server.sim.is_finished():
                    break
                continue
            _th.Thread(target=server._serve_client, args=(conn,),
                       daemon=True).start()
        return 0
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())