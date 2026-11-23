"""Windows-side orchestrator for MAP-01 full-chain (WSL2 Gazebo+Nav2+V18+RViz2 + Windows WPF)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import resolve_policy_name
from tools.live_bridge.map01_live_helpers import assemble_three_panel_video


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "out" / "map01_full_chain_01")
    p.add_argument("--schedule-port", type=int, default=9500)
    p.add_argument("--max-wall-s", type=float, default=900.0)
    p.add_argument("--policy", type=str, default="V18_EA_JOINT")
    args = p.parse_args()
    policy_canon = resolve_policy_name(args.policy)

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    out_wsl = "/mnt/" + out.drive[0].lower() + out.as_posix()[2:]

    bash_cmd = (
        "set -eo pipefail; "
        "cd ${AMR_ROOT:-$(pwd)}; "
        "source /opt/ros/humble/setup.bash; "
        "source /root/map01_guard_install/setup.bash; "
        "CY=/opt/ros/humble; "
        "export AMENT_PREFIX_PATH=$HOME/amr_ws/install/scout_gazebo_description:$CY:$AMENT_PREFIX_PATH; "
        "export LD_LIBRARY_PATH=$CY/lib:$CY/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; "
        "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; "
        "export CYCLONEDDS_URI=file://$HOME/amr_ws/local69_cyclone_loopback.xml; "
        "unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE FASTDDS_BUILTIN_TRANSPORTS; "
        "export PATH=${AMR_ROOT:-$(pwd)}/tools/ros2/bin:$PATH; "
        "export DISPLAY=${DISPLAY:-:0}; "
        "export QT_QPA_PLATFORM=xcb; "
        "export ROS_DOMAIN_ID=44; "
        "export GAZEBO_MASTER_URI=http://127.0.0.1:11362; "
        "export PYTHONPATH=${AMR_ROOT:-$(pwd)}:${AMR_ROOT:-$(pwd)}/tools/live_bridge:${PYTHONPATH:-}; "
        f"python3 -u tools/live_bridge/run_map01_full_chain.py --out '{out_wsl}' "
        f"--schedule-port {int(args.schedule_port)} --max-wall-s {float(args.max_wall_s)} --wait-wpf-s 55 --policy '{policy_canon}'"
    )

    import os
    wsl_env = dict(os.environ)
    wsl_env["WSL_UTF8"] = "1"
    wsl_log = (out / "wsl_runner.log").open("w", encoding="utf-8")
    wsl_proc = subprocess.Popen(
        ["wsl", "-e", "bash", "-lc", bash_cmd],
        cwd=str(ROOT),
        env=wsl_env,
        stdout=wsl_log,
        stderr=subprocess.STDOUT,
    )

    ready_json = out / "control_ready.json"
    wpf_stop = out / "wpf_stop.flag"
    wpf_ready = out / "wpf_ready.json"
    wpf_thread = None
    started = time.monotonic()

    def _run_wpf_capture(sched_host: str, sched_port: int) -> None:
        import runpy
        saved_argv = list(sys.argv)
        try:
            sys.argv = [
                str(ROOT / "tools/live_bridge/capture_wpf_window.py"),
                "--exe", str(ROOT / r"RAWSimO.Visualization\bin\Release\net6.0-windows\RAWSimO.Visualization.dll"),
                "--map", str(ROOT / r"docs\证据\MAP-01G-双机演示\rail_freight_workshop_v1_map.xinst"),
                "--setting", str(ROOT / r"docs\证据\MAP-01G-双机演示\rail_freight_workshop_v1_map.xsett"),
                "--control", str(ROOT / r"docs\证据\MAP-01G-双机演示\rail_freight_workshop_v1_map.xconf"),
                "--schedule", f"{sched_host}:{sched_port}",
                "--out", str(out / "capture" / "wpf"),
                "--stop-file", str(wpf_stop),
                "--ready-file", str(wpf_ready),
                "--hz", "2.0",
                "--max-wall-s", str(args.max_wall_s + 60.0),
            ]
            runpy.run_path(str(ROOT / "tools/live_bridge/capture_wpf_window.py"), run_name="__main__")
        except SystemExit as exc:
            (out / "wpf_capture.log").write_text(f"wpf exit: {exc.code}\n", encoding="utf-8")
        except Exception as exc:
            (out / "wpf_capture.log").write_text(f"wpf error: {exc!r}\n", encoding="utf-8")
        finally:
            sys.argv = saved_argv

    import threading
    try:
        while wsl_proc.poll() is None and time.monotonic() - started < args.max_wall_s + 120.0:
            if ready_json.is_file() and wpf_thread is None:
                try:
                    raw = ready_json.read_text(encoding="utf-8").strip()
                    info = json.loads(raw) if raw else None
                except (json.JSONDecodeError, OSError):
                    info = None
                if info and "port" in info:
                    sched_host = info.get("host") or "127.0.0.1"
                    wpf_thread = threading.Thread(
                        target=_run_wpf_capture,
                        args=(sched_host, int(info["port"])),
                        daemon=True,
                    )
                    wpf_thread.start()
            time.sleep(0.5)
    finally:
        wpf_stop.write_text("stop\n", encoding="utf-8")
        if wpf_thread is not None:
            wpf_thread.join(timeout=20.0)
        if wsl_proc.poll() is None:
            try:
                rc = wsl_proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                wsl_proc.terminate()
                try:
                    rc = wsl_proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    wsl_proc.kill()
                    rc = wsl_proc.wait(timeout=5.0)
        else:
            rc = wsl_proc.returncode
        wsl_log.close()

    print("WSL runner exit code:", rc)
    report_path = out / "full_chain_report.json"
    if report_path.is_file():
        print("Report:", report_path.read_text(encoding="utf-8"))
    if any((out / "capture" / sub).glob("*.jpg") for sub in ("gazebo", "rviz", "wpf")):
        video_info = assemble_three_panel_video(out / "capture", out / "map01_full_chain_demo.mp4", fps=8)
        (out / "video_summary.json").write_text(json.dumps(video_info, indent=2), encoding="utf-8")
        print("Video:", json.dumps(video_info, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
