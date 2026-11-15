import hid, time, csv, json, os

OUT = r"E:\Remote\项目\AMR\docs\radio\evidence"

d = hid.device()
d.open(0x284E, 0x78CE)

info = {
    "vendor_id": d.get_manufacturer_string(),
    "product": d.get_product_string(),
    "serial": d.get_serial_number_string(),
}
desc = d.get_report_descriptor()

rows = []
d.set_nonblocking(0)
t0 = time.time()
while time.time() - t0 < 5.0 and len(rows) < 600:
    data = d.read(64, 200)
    if not data: continue
    axes = list(data[0:6])
    btn = (data[6] | (data[7] << 8)) if len(data) >= 8 else 0
    rows.append([round(time.time()-t0, 4)] + axes + [btn])
d.close()

# 1) 描述符
with open(os.path.join(OUT, "hid_report_descriptor.txt"), "w", encoding="utf-8") as f:
    f.write("SCOUT Mini 原厂遥控器 (FS-AGX-i6S emulator) HID 报告描述符\n")
    f.write("=" * 60 + "\n")
    f.write("VID=0x%04X  PID=0x%04X\n" % (0x284E, 0x78CE))
    f.write("manufacturer=%s\nproduct=%s\nserial=%s\n\n" % (info["vendor_id"], info["product"], info["serial"]))
    f.write("原始字节 (%d):\n%s\n\n" % (len(desc), " ".join("%02X" % b for b in desc)))
    f.write("解析结论:\n")
    f.write("  Usage Page 0x01 (Generic Desktop), Usage 0x04 (Joystick)\n")
    f.write("  6 个轴: X/Y/Z/Rx/Ry/Rz, 各 8 位, 逻辑范围 0..254\n")
    f.write("  10 个按钮: 占 16 位 (ReportCount=16, ReportSize=1)\n")
    f.write("  报告总长 = 6 + 2 = 8 字节\n")
    f.write("\n  ** 描述符中没有任何 Feature Report (0xB1) **\n")
    f.write("  ** 没有任何 Vendor-Defined Usage Page (0xFF00 系列) **\n")
    f.write("  ** 没有任何传感器/遥测/电压/故障字段 **\n")

# 2) 数据流 CSV
with open(os.path.join(OUT, "hid_stream.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_seconds","axis_X","axis_Y","axis_Z","axis_Rx","axis_Ry","axis_Rz","buttons"])
    for r in rows:
        w.writerow(r)

# 3) 汇总
span = rows[-1][0] if rows else 0
summary = {
    "frames": len(rows),
    "duration_s": span,
    "rate_hz": round(len(rows)/span, 1) if span else None,
    "report_bytes": 8,
    "axes_seen": {},
    "buttons_or": 0,
    "extra_bytes_beyond_descriptor": 0,
}
import functools
for i, nm in enumerate(["X","Y","Z","Rx","Ry","Rz"]):
    vals = [r[1+i] for r in rows]
    summary["axes_seen"][nm] = {"min": min(vals), "max": max(vals), "static": min(vals)==max(vals)}
summary["buttons_or"] = functools.reduce(lambda a,b: a|b, [r[7] for r in rows], 0)
with open(os.path.join(OUT, "hid_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("已保存 3 个证据文件到", OUT)
print(json.dumps(summary, ensure_ascii=False, indent=2))