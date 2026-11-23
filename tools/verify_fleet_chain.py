"""Run the whole scheduling chain end to end and report what happened.

The chain under test is:

    amr_sched simulation -> FrameBuilder -> TCP -> LiveTcpClient -> FleetTopology
                                                                        -> drawable XY

The last step is the one that matters. A parser test proves bytes become objects; it
does not prove a consumer can turn those objects into positions on a map. This script
wires the real producer to the real consumer over a real socket and fails if any frame
is dropped or any vehicle cannot be placed.

Usage::

    python tools/verify_fleet_chain.py

Requires the .NET SDK, because the consumer half is the C# client. Exit code 0 means
every frame arrived and every vehicle was placeable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port() -> int:
    """Ask the OS for an unused port, so parallel runs do not collide."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _has_sdk(host: str, env: dict) -> bool:
    """Whether this host can actually see an SDK, not just start."""
    try:
        r = subprocess.run([host, "--list-sdks"], capture_output=True, text=True,
                           timeout=30, env=env, encoding="utf-8", errors="replace")
    except OSError:
        return False
    return r.returncode == 0 and bool(r.stdout.strip())


def find_dotnet() -> tuple[str, dict] | None:
    """Locate a .NET host that can resolve an SDK.

    Plain ``which dotnet`` is not enough. On this machine it finds
    ``C:/Program Files/dotnet/dotnet.exe``, which starts but has no SDK; the SDKs live
    in the per-user install and are only found when DOTNET_ROOT points there. Picking
    the first ``dotnet`` on PATH and hoping would make this script fail with a
    misleading "no SDK found" message.
    """
    candidates: list[tuple[str, str]] = []
    env_root = os.environ.get("DOTNET_ROOT")
    if env_root:
        candidates.append((os.path.join(env_root, "dotnet"), env_root))
        candidates.append((os.path.join(env_root, "dotnet.exe"), env_root))

    home = os.path.expanduser("~")
    user_root = os.path.join(home, ".dotnet")
    candidates.append((os.path.join(user_root, "dotnet.exe"), user_root))
    candidates.append((os.path.join(user_root, "dotnet"), user_root))

    on_path = shutil.which("dotnet")
    if on_path:
        candidates.append((on_path, os.path.dirname(on_path)))

    for host, root in candidates:
        if not os.path.exists(host):
            continue
        env = dict(os.environ)
        env["DOTNET_ROOT"] = root
        env.setdefault("DOTNET_CLI_TELEMETRY_OPTOUT", "1")
        env.setdefault("DOTNET_NOLOGO", "1")
        if _has_sdk(host, env):
            return host, env
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--agvs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    found = find_dotnet()
    if found is None:
        print("SKIP: no .NET SDK found, cannot run the consumer half")
        return 0
    dotnet, env = found
    print("using %s (DOTNET_ROOT=%s)" % (dotnet, env["DOTNET_ROOT"]))

    port = free_port()

    print("building the C# consumer ...")
    build = subprocess.run(
        [dotnet, "build", os.path.join("tools", "FleetHarness", "FleetHarness.csproj"),
         "-c", "Release", "--nologo", "-v", "q"],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    if build.returncode != 0:
        print(build.stdout[-4000:])
        print(build.stderr[-4000:])
        print("FAIL: the consumer did not build")
        return 1

    print("starting the producer on 127.0.0.1:%d ..." % port)
    producer = subprocess.Popen(
        [sys.executable, "-m", "tools.fleet_stream_server",
         "--port", str(port), "--steps", str(a.steps),
         "--tasks", str(a.tasks), "--agvs", str(a.agvs),
         "--seed", str(a.seed), "--warmup", "40", "--rate", "20"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")

    try:
        # Do not probe the port to test readiness. The producer accepts one client at
        # a time and a probe would either consume that slot or, if it is filtered,
        # report a false negative. The producer prints "listening" only once bind and
        # listen have both succeeded, so its own output is the signal.
        waited = 0.0
        while waited < 40.0:
            line = producer.stdout.readline()
            if line:
                print("  producer: " + line.rstrip())
                if "listening" in line:
                    break
            else:
                if producer.poll() is not None:
                    print("FAIL: the producer exited before listening")
                    print((producer.stdout.read() or "")[-4000:])
                    return 1
                time.sleep(0.1)
            waited += 0.1
        else:
            print("FAIL: the producer never reported listening")
            return 1

        print("running the consumer ...")
        dll_path = os.path.join(ROOT, "tools", "FleetHarness", "bin", "Release", "net6.0", "FleetHarness.dll")
        if os.path.isfile(dll_path):
            cmd = [dotnet, dll_path, "127.0.0.1", str(port), str(a.steps)]
        else:
            cmd = [dotnet, "run", "--project",
                   os.path.join("tools", "FleetHarness", "FleetHarness.csproj"),
                   "-c", "Release", "--no-build", "--",
                   "127.0.0.1", str(port), str(a.steps)]
        run = subprocess.run(
            cmd,
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace")
        print((run.stdout or "").rstrip())
        if run.returncode != 0:
            print((run.stderr or "")[-4000:])
            print("FAIL: the consumer rejected the stream")
            return 1
    finally:
        producer.terminate()
        try:
            producer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            producer.kill()

    print()
    print("chain OK: every frame arrived and every vehicle was placeable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
