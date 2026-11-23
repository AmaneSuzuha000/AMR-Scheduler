"""Launch and capture the RAWSimO.Visualization WPF window cleanly using Win32 PrintWindow."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time

from PIL import Image


PW_RENDERFULLCONTENT = 2


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def _enable_dpi_awareness() -> None:
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def find_main_hwnd(pid: int) -> int | None:
    user32 = ctypes.windll.user32
    found: list[tuple[int, int]] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        proc_id = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
        if proc_id.value != pid:
            return True
        rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
        if area > 10000:
            found.append((area, hwnd))
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    if not found:
        return None
    found.sort(reverse=True)
    return found[0][1]


def capture_hwnd(hwnd: int) -> Image.Image | None:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 10 or h <= 10:
        return None

    hwnd_dc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    old_obj = gdi32.SelectObject(mem_dc, bmp)
    try:
        ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(hwnd, mem_dc, 0)
        if not ok:
            return None
        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        buf = (ctypes.c_ubyte * (w * h * 4))()
        bits = gdi32.GetDIBits(mem_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        if bits == 0:
            return None
        return Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1).convert("RGB")
    finally:
        gdi32.SelectObject(mem_dc, old_obj)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)


def main() -> int:
    _enable_dpi_awareness()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--setting", type=Path)
    p.add_argument("--control", type=Path)
    p.add_argument("--schedule", default="127.0.0.1:9500")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stop-file", type=Path, required=True)
    p.add_argument("--ready-file", type=Path)
    p.add_argument("--hz", type=float, default=4.0)
    p.add_argument("--max-wall-s", type=float, default=900.0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    exe_abs = args.exe.resolve()
    map_abs = args.map.resolve()
    if exe_abs.suffix.lower() == ".dll":
        dotnet_cli = Path.home() / ".dotnet" / "dotnet.exe"
        if not dotnet_cli.is_file():
            dotnet_cli = Path("dotnet")
        cmd = [str(dotnet_cli), str(exe_abs), "--map", str(map_abs)]
    else:
        cmd = [str(exe_abs), "--map", str(map_abs)]
    if args.setting is not None:
        cmd.extend(["--setting", str(args.setting.resolve())])
    if args.control is not None:
        cmd.extend(["--control", str(args.control.resolve())])
    cmd.extend(["--schedule", str(args.schedule)])

    import os
    env = os.environ.copy()
    env.setdefault("AMR_WPF_SOFTWARE", "1")
    if Path.home() / ".dotnet".is_dir():
        env.setdefault("DOTNET_ROOT", str(Path.home() / ".dotnet"))
    proc = subprocess.Popen(cmd, cwd=str(exe_abs.parent), env=env)
    manifest_f = (args.out / "wpf_frames.jsonl").open("w", encoding="utf-8")
    period = 1.0 / max(0.5, args.hz)
    started = time.monotonic()
    idx = 0
    hwnd = None
    try:
        while time.monotonic() - started < args.max_wall_s:
            if args.stop_file.exists() or proc.poll() is not None:
                break
            if hwnd is None or not ctypes.windll.user32.IsWindow(hwnd):
                hwnd = find_main_hwnd(proc.pid)
                if hwnd is not None and args.ready_file is not None and not args.ready_file.exists():
                    args.ready_file.write_text(json.dumps({"pid": proc.pid, "hwnd": hwnd}), encoding="utf-8")
            if hwnd is not None:
                img = capture_hwnd(hwnd)
                if img is not None:
                    fname = f"{idx:06d}.jpg"
                    img.save(args.out / fname, quality=88)
                    row = {
                        "idx": idx,
                        "file": fname,
                        "wall_s": time.monotonic(),
                        "utc_s": time.time(),
                        "width": img.width,
                        "height": img.height,
                    }
                    manifest_f.write(json.dumps(row) + "\n")
                    manifest_f.flush()
                    idx += 1
            time.sleep(period)
    finally:
        manifest_f.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
