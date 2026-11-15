"""Cross-language interop check: can a stream exported by Python be read back by C#?

Why this cannot stay manual
----------------------
"the interface works" is an easy sentence to write in a document. Proving it needs a
step that **actually reads it in the other language and matches the fields**, otherwise
the interface is only two documents that happen to agree with themselves.

Flow::

    Python simulation ──► .stream.json ──► C# reader ──► summary output
         │                                            │
         └──────────── field-by-field compare ◄───────────────────┘

Compared: schema, frame count, the first frame's step and vehicle count, and each
vehicle's node, state, and source. Any mismatch fails the run.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import SCHEMA_VERSION, export_stream


HERE = os.path.dirname(os.path.abspath(__file__))
CS_DIR = os.path.join(HERE, "csharp")
CSC = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
EXE = os.path.join(CS_DIR, "reader.exe")


def build_exe() -> bool:
    """Build the C# reader, skipping when it already exists."""
    if os.path.exists(EXE):
        return True
    if not os.path.exists(CSC):
        print("! csc.exe not found, skipping the C# side")
        return False
    r = subprocess.run(
        [CSC, "/nologo", "/target:exe", "/out:" + EXE,
         "/r:System.Web.Extensions.dll",
         os.path.join(CS_DIR, "StreamReader.cs"),
         os.path.join(CS_DIR, "Program.cs")],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("! the C# build failed:\n" + (r.stdout or "")[:800])
        return False
    return True


def make_stream(path: str) -> dict:
    """Run one simulation, export the exchange file, and return the expected values."""
    net = build_layout("ring", aisle_capacity=2)
    sc = ScenarioConfig(n_tasks=20, n_agvs=3, seed=0, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15)
    _, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=200_000, seed=0,
                                     record_trajectory=True))
    res = sim.run()
    info = export_stream(res, path, dt=1.0, stride=5, network=net,
                         meta={"scenario": "ring/20/3/seed0"})
    first = res.trajectory[0]
    expect = {
        "schema": SCHEMA_VERSION,
        "frames": info["n_frames"],
        "first_step": first.step,
        "n_agvs": len(first.agv_positions),
        "agv_ids": sorted(first.agv_positions.keys()),
        "states": {k: first.agv_states.get(k, "idle") for k in first.agv_positions},
        "nodes": {k: v for k, v in first.agv_positions.items()},
    }
    return expect


def parse_cs(out: str) -> dict:
    """Parse the C# reader's output."""
    d: dict = {"agvs": []}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("schema="):
            d["schema"] = line.split("=", 1)[1]
        elif line.startswith("frames="):
            d["frames"] = int(line.split("=", 1)[1])
        elif line.startswith("first_step="):
            m = re.match(r"first_step=(\d+) agvs=(\d+)", line)
            if m:
                d["first_step"] = int(m.group(1))
                d["n_agvs"] = int(m.group(2))
        elif line.startswith("sources="):
            d["sources"] = line.split("=", 1)[1]
        elif re.match(r"^AGV\d+ ", line):
            toks = line.split()
            parts = dict(t.split("=", 1) for t in toks if "=" in t)
            parts["agv_id"] = toks[0]   # the first token is a bare value, not k=v
            d["agvs"].append(parts)
    return d


def main() -> int:
    print("=== cross-language interop check ===")
    if not build_exe():
        return 0  # an unsupported environment is not a failure, but it must be said out loud

    tmp = os.path.join(tempfile.gettempdir(), "interop_check.stream.json")
    expect = make_stream(tmp)
    print("Python side: %d frames, first frame step %d, %d vehicles"
          % (expect["frames"], expect["first_step"], expect["n_agvs"]))

    run_exe = EXE
    try:
        EXE.encode("ascii")
    except UnicodeEncodeError:
        import shutil
        tmp_exe = os.path.join(tempfile.gettempdir(), "amr_interop_reader.exe")
        if not os.path.exists(tmp_exe) or os.path.getmtime(EXE) > os.path.getmtime(tmp_exe):
            shutil.copy2(EXE, tmp_exe)
        run_exe = tmp_exe

    r = subprocess.run([run_exe, tmp], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("C# read failed:\n" + (r.stdout or "") + (r.stderr or ""))
        return 1

    got = parse_cs(r.stdout)
    print("C# side    : %s frames, first frame step %s, %s vehicles"
          % (got.get("frames"), got.get("first_step"), got.get("n_agvs")))
    print("")

    checks = [
        ("schema", got.get("schema"), expect["schema"]),
        ("frames", str(got.get("frames")), str(expect["frames"])),
        ("first step", str(got.get("first_step")), str(expect["first_step"])),
        ("vehicles", str(got.get("n_agvs")), str(expect["n_agvs"])),
    ]
    ok = True
    for name, a, b in checks:
        good = (a == b)
        ok = ok and good
        print("  %s %-10s C#=%s  Python=%s" % ("✔" if good else "✘", name, a, b))

    # Compare node and state vehicle by vehicle
    by_id = {x["agv_id"]: x for x in got.get("agvs", [])}
    for agv in expect["agv_ids"]:
        row = by_id.get(agv)
        if row is None:
            print("  FAIL %s is missing from the C# output" % agv)
            ok = False
            continue
        node_ok = row.get("node") == expect["nodes"][agv]
        state_ok = row.get("state") == expect["states"][agv]
        ok = ok and node_ok and state_ok
        print("  %s %s node=%s/%s state=%s/%s"
              % ("✔" if (node_ok and state_ok) else "✘", agv,
                 row.get("node"), expect["nodes"][agv],
                 row.get("state"), expect["states"][agv]))

    print("")
    print("interop OK: C# read back every field Python exported" if ok
          else "interop FAILED")
    try:
        os.remove(tmp)
    except OSError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
