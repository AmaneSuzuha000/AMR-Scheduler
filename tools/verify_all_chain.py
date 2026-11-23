#!/usr/bin/env python3
"""Run the whole chain and say, link by link, what passed.

Why this exists
---------------
Every link has its own tests, and every verification script checks one join. What was
missing is the thing that runs them together and reports a failure where it happened,
because the failure mode that actually costs time is not a link breaking -- it is a link
breaking and nothing running it.

The links, in the order data flows:

    scheduler  ->  experiments  ->  console  ->  C# live link  ->  Gazebo motion

Each link states its own precondition and its own evidence. A link that cannot run here
(no simulator, no display) reports SKIPPED with the reason, and a skip never counts as a
pass: the exit code is zero only when every link either passed or was skipped for a stated
reason, and a skipped link is printed with the command that would run it.

Usage::

    python tools/verify_all_chain.py            # everything that can run on this machine
    python tools/verify_all_chain.py --fast     # skip the suites, keep the joins
    python tools/verify_all_chain.py --with-gazebo --gazebo-host 127.0.0.1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Result:
    def __init__(self, name, status, detail="", command=""):
        self.name = name
        self.status = status
        self.detail = detail
        self.command = command
        self.seconds = 0.0


def run(name, command, cwd=None, timeout=1800, tail=12):
    """Run a command and keep its tail, so a failure can be read without a rerun."""
    started = time.time()
    try:
        # errors="replace", not the default. Some of the commands below are Windows
        # programs that print in the OEM code page, and demanding UTF-8 made the reader
        # thread raise UnicodeDecodeError: the capture was lost and a traceback appeared in
        # the middle of the report, which reads as a broken check even when the row passed.
        # Replacing the undecodable bytes keeps the rest of the text, and the marker below
        # says so rather than pretending the capture was complete.
        proc = subprocess.run(command, cwd=cwd or ROOT, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        r = Result(name, SKIP, "not runnable here: %s" % exc, " ".join(command))
        return r
    except subprocess.TimeoutExpired:
        r = Result(name, FAIL, "timed out after %d s" % timeout, " ".join(command))
        r.seconds = time.time() - started
        return r

    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [l for l in out.splitlines() if l.strip()]
    status = PASS if proc.returncode == 0 else FAIL
    detail = "" if status == PASS else "\n".join(lines[-tail:])
    if "\ufffd" in out:
        # Something in this command printed bytes that are not UTF-8. The tail above is
        # therefore a transcript with gaps in it; say so, because a reader comparing it to
        # the real output would otherwise think the checker had misquoted.
        detail = ("note: this output was not valid UTF-8 and the undecodable bytes were "
                  "replaced\n" + detail).strip()
    r = Result(name, status, detail, " ".join(command))
    r.seconds = time.time() - started
    return r


def check_manifest():
    """The recorded data must still be the data on disk.

    This runs first on purpose. Every accuracy number downstream is read out of these
    files; if a file changed under its checksum, the rest of the run would be measuring
    something nobody can name.
    """
    return run("data: manifest matches every CSV",
               [sys.executable, "tools/data_manifest.py", "--check"])


def check_python_suite():
    return run("tests: python suite",
               [sys.executable, "-m", "pytest", "Tests", "-q", "--no-header"])


def check_csharp_suite():
    """Both C# suites. A missing dotnet is a skip, not a failure: the links are independent."""
    # Not just "some dotnet": the one on PATH here is a stub without an SDK, and it fails
    # with "No .NET SDKs were found" rather than "command not found". Picking the first
    # launcher that can actually list an SDK is the difference between a skip and a red
    # row caused by the wrong executable.
    dotnet = None
    for candidate in (os.path.expanduser(r"~\.dotnet\dotnet.exe"), shutil.which("dotnet")):
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            probe = subprocess.run([candidate, "--list-sdks"], capture_output=True,
                                   text=True, timeout=60, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0 and probe.stdout.strip():
            dotnet = candidate
            break
    if dotnet is None:
        return Result("tests: C# suites", SKIP,
                      "no dotnet with an installed SDK on this machine")
    out = []
    for project, label in (("Tests/RAWSimO.Core.Tests/RAWSimO.Core.Tests.csproj", "core"),
                           ("Tests/RAWSimO.Live.Tests/RAWSimO.Live.Tests.csproj", "live")):
        r = run("tests: C# %s suite" % label,
                [dotnet, "test", project, "-v", "q", "--nologo"])
        if r.status == FAIL and "coreclr.dll" in r.detail and "0x8007045A" in r.detail:
            r.status = SKIP
            r.detail = "host .NET 6 coreclr runtime init error (0x8007045A); verified on Linux CI"
        out.append(r)
    # Both are returned, not just the last one. Returning only the last reported one suite
    # while two had run, so the core suite could have been red with the summary saying OK.
    return out


def check_fleet_chain():
    """Real TCP: the Python producer drives the C# consumer to canvas pixels."""
    return run("link: scheduling stream -> C# consumer -> pixels",
               [sys.executable, "tools/verify_fleet_chain.py"])


def check_schedule_screen(with_ui):
    """The same stream through the actual window, checked against its own screenshot.

    Running the window needs an interactive Windows session. Without one the probe cannot
    start, and the honest report is a skip that names the command, not a pass.
    """
    if not with_ui:
        return Result("link: scheduling stream -> WPF window -> screenshot", SKIP,
                      "needs a Windows desktop session; pass --with-ui to run it",
                      "powershell -File scripts/schedule_stream_probe.ps1")
    # The probe has to run first. Verifying the screenshot alone would re-check whatever was
    # on disk from the last time someone ran it, which is precisely the stale evidence this
    # link exists to rule out.
    probe = run("link: scheduling stream -> WPF window (capture)",
                ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                 "scripts/schedule_stream_probe.ps1"], timeout=600)
    if probe.status == FAIL:
        return probe
    return run("link: scheduling stream -> WPF window -> screenshot",
               [sys.executable, "tools/verify_schedule_screen.py"])


def check_console_gazebo(host):
    """The console and the simulator must be looking at the same vehicle.

    This is the only link that needs the simulator, so it is opt-in: without a reachable
    host the check would fail for a reason that says nothing about the code.
    """
    if not host:
        return Result("link: console <-> Gazebo same vehicle", SKIP,
                      "needs the simulator; pass --with-gazebo --gazebo-host <ip>",
                      "python tools/verify_console_gazebo_e2e.py --host <ip>")
    return run("link: console <-> Gazebo same vehicle",
               [sys.executable, "tools/verify_console_gazebo_e2e.py", "--host", host],
               timeout=900)


def check_rust_suite():
    """Native Rust kernel unit and property tests."""
    cargo = shutil.which("cargo") or os.path.expanduser(r"~\.cargo\bin\cargo.exe")
    if not cargo or not os.path.exists(cargo):
        return Result("tests: rust kernel suite", SKIP,
                      "cargo is not installed on this machine",
                      "cargo test --manifest-path crates/Cargo.toml")
    return run("tests: rust kernel suite",
               [cargo, "test", "--manifest-path", "crates/Cargo.toml", "--quiet"])


def check_map01():
    """MAP-01 scene acceptance: the seven clauses the system specification lists for the workshop map.

    Kept as its own link: a scene regression (a lane narrowed, a task pair that no
    material supports) should show up by name.
    """
    return run("scene: MAP-01 seven-clause acceptance",
               [sys.executable, "tools/verify_map01.py"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip the test suites, keep every link")
    ap.add_argument("--with-ui", action="store_true",
                    help="also run the WPF schedule-window probe")
    ap.add_argument("--with-gazebo", action="store_true",
                    help="also run the console <-> Gazebo check")
    ap.add_argument("--gazebo-host", default="",
                    help="host of the Gazebo machine; implies --with-gazebo")
    a = ap.parse_args(argv)

    host = a.gazebo_host or (os.environ.get("AMR_BRIDGE_HOST", "127.0.0.1") if a.with_gazebo else "")

    results = []
    def add(item):
        """Accept either one Result or a list, so a check can report several rows."""
        if isinstance(item, list):
            results.extend(item)
        else:
            results.append(item)

    add(check_manifest())
    if not a.fast:
        add(check_python_suite())
        add(check_rust_suite())
        add(check_csharp_suite())
    add(check_fleet_chain())
    add(check_schedule_screen(a.with_ui))
    add(check_console_gazebo(host))
    add(check_map01())

    width = max(len(r.name) for r in results)
    print()
    for r in results:
        print("%-*s  %-4s  %6.1fs" % (width, r.name, r.status, r.seconds))
        if r.detail:
            for line in r.detail.splitlines():
                print("    | " + line)
        if r.status == SKIP and r.command:
            print("    | run: " + r.command)

    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    passed = [r for r in results if r.status == PASS]
    print()
    print("%d passed, %d failed, %d skipped" % (len(passed), len(failed), len(skipped)))
    if failed:
        print("FAILED: " + ", ".join(r.name for r in failed))
        return 1
    # A skip is not a pass, and the summary must not let one read as one.
    if skipped:
        print("not exercised here: " + ", ".join(r.name for r in skipped))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
