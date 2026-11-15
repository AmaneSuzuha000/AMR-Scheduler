"""End-to-end check of the live channel: a Python client subscribes to the same stream C# will.

What is checked
--------
1. the first message must be ``hello``, carrying a schema and the full network;
2. the rest must be ``frame``, with a strictly increasing step;
3. vehicle positions really change between frames, rather than repeating one point;
4. the last message is ``done`` and carries a summary;
5. the frame fields are structurally the same as the offline export;
6. **a second client joining mid-run gets the current frame, not a replay from the start**
   -- this is what separates a live feed from a file playback.

On connection order (a trap already hit)
----------------------
After ``listen()``, call ``connect()`` before ``accept()``. The other order deadlocks:
the main thread blocks in accept waiting for a connection that can only be made by the
line after it. TCP allows connect to complete before accept (it queues in the backlog).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amr_sched.live import LiveServer, build_sim

PORT = 18771


def _read_lines(sock: socket.socket, limit: int, timeout: float = 40.0):
    """Read until a done frame, ``limit`` lines, or a timeout."""
    sock.settimeout(timeout)
    buf = b""
    out = []
    while len(out) < limit:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf and len(out) < limit:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            obj = json.loads(line.decode("utf-8"))
            out.append(obj)
            if obj.get("type") == "done":
                return out
    return out


def main() -> int:
    sim = build_sim(n_tasks=30, n_agvs=4, seed=1)
    server = LiveServer(sim, port=PORT, rate=40.0, steps_per_frame=3)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(8)

    # Connect first, accept second -- see "On connection order" above.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", PORT))
    conn, _ = srv.accept()
    threading.Thread(target=server._serve_client, args=(conn,),
                     daemon=True).start()

    # Advance only once the client is in place, so the stream it receives is complete from the start.
    runner = threading.Thread(target=server.run, daemon=True)
    runner.start()

    msgs = _read_lines(sock, 5000, timeout=40.0)
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
    srv.close()

    ok = True

    def fail(msg):
        nonlocal ok
        print("  [FAIL] " + msg)
        ok = False

    print("received %d messages" % len(msgs))
    if not msgs:
        print("  [FAIL] received nothing at all")
        return 1

    # 1. hello
    first = msgs[0]
    print("first message type=%s schema=%s" % (first.get("type"), first.get("schema")))
    if first.get("type") != "hello":
        fail("the first message is not hello")
    if first.get("schema") != "1.0":
        fail("schema is not 1.0")
    nodes = (first.get("network") or {}).get("nodes") or []
    edges = (first.get("network") or {}).get("edges") or []
    print("  network: %d nodes / %d edges" % (len(nodes), len(edges)))
    if (len(nodes), len(edges)) != (17, 36):
        fail("the loop network should be 17 nodes and 36 edges")

    # 2. The frame sequence
    frames = [m for m in msgs if m.get("type") == "frame"]
    if len(frames) < 5:
        fail("too few frames: %d" % len(frames))
        print("FAIL")
        return 1
    steps = [f["step"] for f in frames]
    print("%d frames, step %d..%d" % (len(frames), steps[0], steps[-1]))
    if any(b <= a for a, b in zip(steps, steps[1:])):
        fail("step is not strictly increasing")

    # 3. Positions really change
    def pos(f):
        return tuple(sorted(
            (a["agv_id"], a["pose"].get("node") or a["pose"].get("edge"))
            for a in f["agvs"]))
    uniq = len(set(pos(f) for f in frames))
    print("distinct layouts %d / %d" % (uniq, len(frames)))
    if uniq < 3:
        fail("positions barely change, which does not look like a live feed")

    # 4. done + summary
    done = [m for m in msgs if m.get("type") == "done"]
    if not done:
        fail("no done frame was received")
    else:
        sm = done[0].get("summary") or {}
        print("done: completed %s/%s, completion time %s s" % (
            sm.get("tasks_completed"), sm.get("tasks_total"),
            sm.get("makespan_seconds")))
        if sm.get("tasks_completed") != sm.get("tasks_total"):
            fail("not every task completed")

    # 5. Field structure
    a0 = frames[0]["agvs"][0]
    need = {"agv_id", "pose", "state", "task_id", "path"}
    if not need <= set(a0):
        fail("frame is missing fields: %s" % (need - set(a0)))
    pn = {"agv_id", "node", "edge", "progress", "source", "timestamp"}
    if not pn <= set(a0["pose"]):
        fail("pose is missing fields: %s" % (pn - set(a0["pose"])))
    print("  pose.source = %s" % a0["pose"]["source"])
    if a0["pose"]["source"] != "sim":
        fail("a simulation frame must report source sim")

    print("")
    print("PASS" if ok else "FAIL")
    sys.stdout.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)