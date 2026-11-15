#!/usr/bin/env python3
"""Demo B 协议一致性：C# 侧声明的字段 <-> live_bridge 实际发出的字段。

为什么需要这个
--------------
本机没有 .NET SDK（只有运行时），C# 那 17 个测试项目跑不起来。
但「C# 能不能正确消费」这件事，有一半是**纯协议问题**：
服务端发的字段名/类型，和客户端 record 里声明的是否一致。

这一半可以在 Python 里严格验证：
  1. 从 LiveMessages.cs 里解析出每个消息类型声明的 JSON 字段；
  2. 真起一个 live_bridge，抓它实际发出的报文；
  3. 逐字段比对——声明了但没发 = 客户端会拿到 null；
     发了但没声明 = 客户端会丢字段。
"""
import sys, re, json, socket, subprocess, time, os
sys.path.insert(0, ".")

CS = "RAWSimO.Live/Protocol/LiveMessages.cs"

# 从 record 定义里提取 (消息类型 -> 字段名集合)
def parse_cs_messages(path):
    src = open(path, encoding="utf-8").read()
    out = {}
    # 形如：public sealed record HelloMessage(string Vehicle, int Proto, ...)
    #       : LiveMessage("hello");
    pat = re.compile(
        r'public\s+sealed\s+record\s+(\w+)\s*\(([^;]*?)\)\s*\n?\s*:\s*LiveMessage\("(\w+)"\)',
        re.S)
    for m in pat.finditer(src):
        cls, params, typ = m.group(1), m.group(2), m.group(3)
        fields = []
        for p in params.split(','):
            p = p.strip()
            if not p:
                continue
            # 去掉 attribute 前缀 [property: JsonPropertyName("max_speed")]
            attr = re.search(r'JsonPropertyName\("([^"]+)"\)', p)
            name = attr.group(1) if attr else None
            if not name:
                # 取参数名：最后一个标识符，去掉默认值
                core = re.sub(r'\s*=.*$', '', p).strip()
                ids = re.findall(r'\b([A-Za-z_]\w*)\s*$', core)
                if not ids:
                    continue
                name = ids[0]
                # PascalCase -> 客户端通常靠 camelCase 策略映射
                name = name[0].lower() + name[1:]
            fields.append((name, p))
        out[typ] = {"class": cls, "fields": [f[0] for f in fields]}
    return out


# 起一个 live_bridge（sim 模式），抓 2 秒报文
def capture_bridge(port=39412, seconds=4.0):
    proc = subprocess.Popen(
        [sys.executable, "tools/live_bridge/live_bridge.py",
         "--mode", "sim", "--rate", "20", "--port", str(port),
         "--start", "0.9", "1.35", "0", "--watchdog", "0", "--proto", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=".")
    time.sleep(1.5)
    msgs = []
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=8)
        s.settimeout(2.0)
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
                    msgs.append(json.loads(line.decode("utf-8")))
        s.close()
    finally:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
    return msgs


def main():
    declared = parse_cs_messages(CS)
    print("=== C# 声明的消息类型 ===")
    for t, v in sorted(declared.items()):
        print("  %-12s %-22s %s" % (t, v["class"], v["fields"]))

    print("\n=== 抓 live_bridge 实际报文 ===")
    msgs = capture_bridge()
    by_type = {}
    for m in msgs:
        by_type.setdefault(m.get("type"), []).append(m)
    print("  共 %d 条，类型: %s" % (len(msgs), sorted(by_type)))

    print("\n=== 逐字段比对 ===")
    bad = 0
    for t in sorted(by_type):
        if t not in declared:
            print("  [!!] %-12s 服务端发了但 C# 未声明 -> 客户端会丢弃" % t)
            bad += 1
            continue
        sent = set(by_type[t][0].keys())
        want = set(declared[t]["fields"]) | {"type"}
        missing = sent - want      # 服务端多发的（客户端忽略）
        absent = want - sent        # 客户端期望但服务端没发
        status = "OK" if not absent else "缺字段"
        if absent:
            bad += 1
        print("  %-4s %-12s 发出=%s" % ("[%s]" % status, t, sorted(sent)))
        if absent:
            print("       客户端期望但服务端未发: %s" % sorted(absent))
        if missing:
            print("       服务端多发（客户端忽略）: %s" % sorted(missing))

    print("\n=== 未被触发的类型（只验证了声明）===")
    for t in sorted(declared):
        if t not in by_type:
            print("  %-12s 声明字段: %s" % (t, declared[t]["fields"]))

    print("")
    print("判定:", "PASS" if bad == 0 else "有 %d 处不一致" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
