"""Isolate the RAWSim-O load failure: .xinst or .xsett?"""
import glob, io, os, subprocess, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
AMR = "E:/Remote/项目/AMR"
CLI = AMR + "/RAWSimO.CLI/bin/Debug/net6.0/RAWSimO.CLI.dll"
DOTNET = os.path.join(os.environ["USERPROFILE"], ".dotnet", "dotnet.exe")
G = AMR + "/docs/证据/MAP-01G-双机演示"
T = AMR + "/docs/证据/T1.6-generated"
OUT = os.path.join(os.environ["TEMP"], "matrix2")
os.makedirs(OUT, exist_ok=True)
OURS = G + "/rail_freight_workshop_v1_map"
GOOD = T + "/generated-map"
combos = [
    ("A_ours",     OURS + ".xinst", OURS + ".xsett", OURS + ".xconf"),
    ("B_ourinst_goodsett", OURS + ".xinst", GOOD + ".xsett", GOOD + ".xconf"),
    ("C_goodinst_oursett", GOOD + ".xinst", OURS + ".xsett", OURS + ".xconf"),
    ("D_good",     GOOD + ".xinst", GOOD + ".xsett", GOOD + ".xconf"),
]
env = dict(os.environ, DOTNET_ROOT=os.path.dirname(DOTNET))
for name, inst, sett, conf in combos:
    missing = [f for f in (inst, sett, conf) if not os.path.exists(f)]
    if missing:
        print("==", name, "MISSING", missing); continue
    outdir = os.path.join(OUT, name); os.makedirs(outdir, exist_ok=True)
    cmd = [DOTNET, CLI, inst, sett, conf, outdir, "0"]
    code = None; log = ""
    try:
        p = subprocess.run(cmd, cwd=AMR, env=env, capture_output=True, timeout=300)
        code = p.returncode
        log = p.stdout.decode("utf-8","replace") + chr(10) + p.stderr.decode("utf-8","replace")
    except subprocess.TimeoutExpired:
        code = "TIMEOUT"
    show = hex(code) if isinstance(code, int) and code else code
    ls = [l.strip() for l in log.splitlines() if l.strip()]
    exc = [l for l in ls if "Unhandled" in l or "Caught an unhandled" in l]
    where = [l for l in ls if l.startswith("at RAWSimO")][:2]
    st = glob.glob(os.path.join(outdir, "*", "statistics.txt"))
    print("==", name, "exit", show, "stats", bool(st))
    print("    ", (exc[0][:150] if exc else (ls[-1][:150] if ls else "NO OUTPUT")))
    for w in where: print("     ->", w[:150])
