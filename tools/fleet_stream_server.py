"""Serve a live scheduling stream over TCP so a client can consume it.

`amr_sched.live` already speaks newline-delimited JSON to a socket, but it drives the
run itself. This server does the same thing under the caller's control: it advances a
real simulation and publishes one frame per step, which is what a GUI actually
consumes. It exists so the C# client can be exercised against the real producer over a
real socket instead of against a captured file.

Usage::

    python -m tools.fleet_stream_server --port 9411 --tasks 20 --agvs 3 --steps 200

Then point a client at 127.0.0.1:9411. The server exits after --steps frames.
"""
from __future__ import annotations

import argparse
import json
import socket
import time

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import FrameBuilder


def build_sim(tasks: int, agvs: int, seed: int) -> Simulator:
    """A simulator configured exactly like the CLI default run."""
    net, task_list, fleet = generate_scenario(
        ScenarioConfig(n_tasks=tasks, n_agvs=agvs, seed=seed))
    sim = Simulator(net, task_list, fleet,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(max_steps=200000))
    sim.reset()
    return sim


def serve(port: int, steps: int, tasks: int, agvs: int, seed: int,
          rate_hz: float, warmup: int) -> int:
    sim = build_sim(tasks, agvs, seed)
    for _ in range(warmup):
        sim.step_once()

    builder = FrameBuilder(sim)
    period = 1.0 / rate_hz if rate_hz > 0 else 0.0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(4)
        print("listening on 127.0.0.1:%d" % port, flush=True)

        # Accept in a loop and prove the peer is real by sending it the hello.
        #
        # A readiness probe (connect, then close) must not consume the one slot a
        # single-shot server has, or the real client is refused while the probe looks
        # like success. getpeername() cannot tell the two apart -- it reports the local
        # socket state and still succeeds for a peer that has already gone away -- so
        # the test is whether the first write lands.
        conn = None
        while conn is None:
            candidate, addr = srv.accept()
            try:
                candidate.settimeout(10.0)
                candidate.sendall(
                    (json.dumps(builder.hello(), ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError:
                candidate.close()
                continue
            conn = candidate
            print("client %s:%d" % addr, flush=True)

        with conn:
            def send(obj) -> None:
                conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

            # The hello already went out during the accept probe.
            for _ in range(steps):
                sim.step_once()
                send(builder.frame())
                if period:
                    time.sleep(period)
            # Closing the socket is the end-of-run signal; the client must survive it
            # rather than treating it as a protocol error.
        print("sent 1 hello + %d frames" % steps, flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=9411)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--agvs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30,
                    help="steps to advance before the first frame, so vehicles are moving")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="frames per second; 0 sends as fast as possible")
    a = ap.parse_args(argv)
    return serve(a.port, a.steps, a.tasks, a.agvs, a.seed, a.rate, a.warmup)


if __name__ == "__main__":
    raise SystemExit(main())
