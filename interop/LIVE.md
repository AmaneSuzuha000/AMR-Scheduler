# Live state channel (Demo B)

> This describes the fleet stream and the `interop/csharp` probe that reads it.
> For which definition is authoritative when documents disagree, see
> [`PROTOCOLS.md`](../PROTOCOLS.md).

## What problem it solves

Before this, the C# side could only call `StreamDocument.Load(path)` on a **finished JSON
file**. That showed the format could be consumed by C#, but not that "the marker on the
map follows the vehicle", because the vehicle had long since stopped and the file would
never change again.

This module pushes the same `Frame` **while the run is going**.

## Protocol

TCP, newline-delimited JSON (JSONL). Receive on connect; there is no handshake.

```
server -> client:
  {"type":"hello","schema":"1.0","network":{...},"meta":{...}}   once, on connect
  {"type":"frame","step":N,"t":...,"agvs":[...],...}             one line per frame
  {"type":"done","step":N,"summary":{...}}                       once, at the end
```

Why not WebSocket: the C# side targets .NET Framework 4.0, and WebSocket would add a
dependency. `TcpClient` plus `StreamReader.ReadLine` needs nothing beyond the framework,
and it can be inspected by eye with telnet.

## Running it

```bash
python -m amr_sched.live --port 8770 --n-tasks 50 --n-agvs 4 --rate 20
```

| argument | default | meaning |
|---|---|---|
| `--port` | 8770 | listening port |
| `--n-tasks` | 50 | number of tasks |
| `--n-agvs` | 4 | number of vehicles |
| `--seed` | 0 | random seed |
| `--rate` | 20 | frames pushed per second |
| `--steps-per-frame` | 5 | simulation steps advanced per frame |
| `--max-steps` | none | stop after step N |

**The server waits for the first client before advancing.** Otherwise 30 tasks finish in
about eight seconds at 60 frames per second, and the run is over before anyone has opened
the window.

## C# side

```csharp
var client = new LiveClient("127.0.0.1", 8770);
NetworkSpec net = null;

client.Message += msg =>
{
    if (msg.Type == "hello") { net = msg.Net; return; }
    if (msg.Type == "done")  { Console.WriteLine("finished"); return; }

    foreach (var agv in msg.Agvs)
    {
        double x, y;
        LiveClient.PoseToXY(net, agv.Pose, out x, out y);
        DrawPoint(agv.AgvId, x, y);      // the vehicle is here
    }
};
client.Start();    // receives on a background thread; does not block the UI
```

`PoseToXY` is the important part: when a vehicle is mid-edge it interpolates by
`progress`, so the marker moves smoothly between two nodes instead of jumping.

> `PoseToXY` does not implement the maths itself: it calls `PoseMath.cs` in this same
> folder, which `RAWSimO.Live` compiles as a linked file. One body of source, two
> binaries -- see `PROTOCOLS.md`.

Build with the compiler that ships with .NET Framework; no SDK needed:

```bat
cd interop\csharp
%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe ^
  /out:LiveProbe.exe /r:System.Web.Extensions.dll ^
  LiveClient.cs StreamReader.cs LiveProbe.cs PoseMath.cs
```

## Measured record

What the C# side actually received from the Python push:

```
[hello] schema=1.0 network=17 nodes / 36 edges
[frame 1]   step=0    AGV00(0,0)    AGV01(0,0)    AGV02(0,0)    AGV03(0,0)
[frame 2]   step=4    AGV00(40,70)  AGV01(40,70)  AGV02(95,55)  AGV03(95,55)
[frame 100] step=396  AGV00(40,40)  AGV01(0,0)    AGV02(80,40)  AGV03(80,40)
[frame 400] step=1596 AGV00(-20,55) AGV01(40,0)   AGV02(0,0)    AGV03(40,0)
[done] step=1903 completed=30/30 completion time=1860.0s

frames received   : 477
distinct layouts  : 55
step monotonic    : yes
PASS
```

## What this does and does not show

**It shows:**

- the Python side can push running vehicle state out in real time;
- the C# side can receive, parse, and compute continuous coordinates from it.

**It does not show:**

- that those coordinates are drawn inside RAWSim-O's own map control;
- that RAWSim-O's vehicle entities are driven by Python.

Both need changes to the RAWSim-O rendering layer and are separate work. What exists is a
**working data channel plus a C# client verified against it**.

## Tests

```bash
python -m pytest Tests/test_live_stream.py -v      # 7 tests, pure Python
python interop/verify_live.py                      # end to end, real socket
interop\csharp\LiveProbe.exe 127.0.0.1 8770        # C# side, measured
```
