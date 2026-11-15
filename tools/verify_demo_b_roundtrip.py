#!/usr/bin/env python3
"""Demo B 完整往返：模拟 C# 客户端，把上行指令全部走一遍。

上一个脚本只验了服务端**主动推送**的三类（hello/pose/status）。
这里补上**上行指令 + 应答**：ping->pong、nav->nav_result、cancel->nav_result。
只有这些走通，"C# 能控制系统"才算真的验证过。

注意：本脚本按 C# LiveMessages.cs 的字段名发指令，
若服务端不认，就说明两边协议不一致。
"""
import sys, json, socket, subprocess, time
sys.path.insert(0, ".")

PORT = 39413


def start_bridge():
    p = subprocess.Popen(
        [sys.executable, "tools/live_bridge/live_bridge.py",
         "--mode", "sim", "--rate", "20", "--port", str(PORT),
         "--start", "0.9", "1.35", "0", "--watchdog", "0", "--proto", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=".")
    time.sleep(1.5)
    return p


def main():
    proc = start_bridge()
    got = []
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=8)
        s.settimeout(0.3)

        def pump(seconds):
            """收 seconds 秒内所有报文。"""
            buf = b""
            t_end = time.time() + seconds
            while time.time() < t_end:
                try:
                    data = s.recv(8192)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        m = json.loads(line.decode("utf-8"))
                        got.append(m)
                        if buf == b"" and b"\n" not in buf:
                            pass
            return buf

        def send(obj):
            s.sendall((json.dumps(obj) + "\n").encode())

        # --- 1. hello ---
        pump(0.8)
        hello = next((m for m in got if m["type"] == "hello"), None)
        assert hello, "没收到 hello"
        print("1) hello: proto=%s frame=%s caps=%s"
              % (hello["proto"], hello["frame"], hello.get("caps")))

        # --- 2. ping -> pong ---
        n0 = len(got)
        send({"type": "ping", "t1": time.time()})
        pump(0.8)
        pong = next((m for m in got[n0:] if m["type"] == "pong"), None)
        assert pong, "ping 没有回 pong"
        print("2) ping->pong: t1=%.6f t2=%.6f t3=%.6f (RTT %.2f ms)"
              % (pong["t1"], pong["t2"], pong["t3"],
                 (pong["t3"] - pong["t1"]) * 1000))

        # --- 3. nav -> nav_result ---
        n0 = len(got)
        send({"type": "nav", "goal": "py-e2e-1",
              "path": [[1.0, 1.35], [1.15, 1.35]], "max_speed": 0.3})
        deadline = time.time() + 20
        res = None
        while time.time() < deadline:
            pump(0.5)
            res = next((m for m in got[n0:]
                        if m["type"] == "nav_result" and m.get("goal") == "py-e2e-1"), None)
            if res:
                break
        assert res, "nav 没有回 nav_result"
        print("3) nav->nav_result: goal=%s status=%s" % (res["goal"], res["status"]))

        # --- 4. nav 长路径 + cancel ---
        n0 = len(got)
        send({"type": "nav", "goal": "py-e2e-2",
              "path": [[3.0, 3.0], [3.2, 3.2]], "max_speed": 0.15})
        pump(0.6)
        send({"type": "cancel", "goal": "py-e2e-2"})
        deadline = time.time() + 12
        res2 = None
        while time.time() < deadline:
            pump(0.4)
            res2 = next((m for m in got[n0:]
                         if m["type"] == "nav_result" and m.get("goal") == "py-e2e-2"), None)
            if res2:
                break
        assert res2, "cancel 没有回 nav_result"
        print("4) cancel->nav_result: goal=%s status=%s reason=%s"
              % (res2["goal"], res2["status"], res2.get("reason")))

        # --- 5. estop ---
        n0 = len(got)
        send({"type": "estop"})
        pump(0.8)
        st = [m for m in got[n0:] if m["type"] == "status" and m.get("error")]
        print("5) estop: 收到 %d 条带 error 的 status, error=%s"
              % (len(st), st[-1]["error"] if st else None))
        send({"type": "estop_release"})
        pump(0.8)

        s.close()
    finally:
        try:
            proc.kill(); proc.wait(timeout=5)
        except Exception:
            pass

    ok = (hello and pong and res and res["status"] == "succeeded"
          and res2 and res2["status"] == "canceled")
    print("")
    print("判定:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
