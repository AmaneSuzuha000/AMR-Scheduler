# Wire protocols: what exists, and which definition is authoritative

There are two producers and two C# consumers in this repository. That is one more
than is comfortable, and this file exists to say so plainly rather than let a reader
discover it by accident.

## The two producers

| producer | entry point | message types | channel |
|---|---|---|---|
| single-vehicle bridge | `tools/live_bridge/live_bridge.py` | `hello` `pose` `status` `nav` `nav_result` `cancel` `ping` `pong` `estop` `estop_release` | TCP, newline JSON, default port **9300** |
| fleet scheduler | `amr_sched/live.py`, `tools/fleet_stream_server.py` | `hello` `frame` `done` | TCP, newline JSON, default port **8770** |

Both open with a message called `hello`, which is the single most confusing thing
about this repository. They are told apart by payload shape, not by type name:

- a **fleet** hello carries a `network` object (nodes + edges) and a `meta` object;
- a **single-vehicle** hello carries `vehicle`, `proto`, `frame`, `caps`.

`RAWSimO.Live/Protocol/LiveMessageParser.cs` implements exactly that discrimination, and
`Tests/test_fleet_stream_e2e.py` pins it from the Python side so neither producer can
grow the other's marker field unnoticed.

**Authoritative definition.** There is no single schema file. The authoritative
definition of each message is the pair:

1. the producer that emits it (`amr_sched/stream.py` for `hello`/`frame`/`done`,
   `tools/live_bridge/live_bridge.py` for the single-vehicle family), and
2. the parser that accepts it (`RAWSimO.Live/Protocol/LiveMessageParser.cs`),

with the tests on both sides as the contract. A prose document that drifts from these
is wrong by definition; `interop/LIVE.md` and `tools/live_bridge/GAZEBO.md` are
descriptions, not sources of truth.

## The two consumers

| consumer | language | target | status |
|---|---|---|---|
| `RAWSimO.Live/` | C# / net6.0 | the current library and its tests | **authoritative** |
| `interop/csharp/` | C# / .NET Framework 4.0, built with `csc.exe` | standalone probe binaries | legacy, still tested |

### Why both still exist

`interop/csharp` was written first, against the fleet stream, and it works: it is what
produced the 477-frame receive record in `interop/LIVE.md`. It targets .NET Framework
4.0 because the original goal was to drop a probe next to an unmodified RAWSim-O build
without installing an SDK.

`RAWSimO.Live` was written afterwards for the single-vehicle bridge and grew the fleet
messages later. It is the one under test in CI, the one with a maintained parser, and
the one anything new should target.

### The overlap, and how it was resolved

`interop/csharp/LiveClient.cs` used to have `PoseToXY`, which interpolated a pose along
an edge, and `RAWSimO.Live/Tracking/FleetTopology.cs` had `Place`, which did the same job
with typed results and explicit failure modes. **Two implementations of one rule is a
defect, not a feature** -- they agreed on the day they were written and nothing kept them
agreeing.

The intended fix used to be "delete `PoseToXY` and reference `RAWSimO.Live`". That was
not doable: `interop/csharp` is built with the .NET Framework 4.0 `csc.exe` and no SDK on
purpose, and it cannot reference a net6.0 library. Raising its target framework would
defeat the reason it exists.

So the rule moved into one file instead:

| file | compiled by | how |
|---|---|---|
| `interop/csharp/PoseMath.cs` | the interop probe | listed on the `csc` command line |
| the same file | `RAWSimO.Live` | `<Compile Include="..\interop\csharp\PoseMath.cs" Link="Tracking\PoseMath.cs" />` |

One body of source, two binaries. `PoseToXY` and `Place` both call `PoseMath`; neither
clamps, splits an edge id, or interpolates on its own. Drift is now impossible rather than
merely untested, and `Tests/test_protocol_authority.py` fails if a third copy reappears.

`PoseMath.cs` is written in C# 5 for the .NET Framework 4.0 compiler: no expression-bodied
members, no string interpolation, no out-var, no nullable reference annotations.

## Rules that follow from the above

1. A new consumer targets `RAWSimO.Live`, never `interop/csharp`.
2. A new message type is added to the producer *and* `LiveMessageParser` *and* a test
   in the same change. A message that parses but is never consumed is not integrated:
   that was exactly the state of the fleet messages before `FleetTopology` existed.
3. Do not add a third newline-JSON protocol. If a new transport is genuinely needed,
   extend one of the two above.
4. Any prose describing the wire format must name the file it describes. Prose that
   cannot name its source is a liability.

### Directed road centerlines

Fleet hello network edges may carry `centerline: [[x,y], ...]` in directed travel order. The Python network is authoritative; vertices must match the edge endpoints and declared arc length. Omission retains the legacy straight-road display. New WPF clients preserve all vertices for base roads and planned routes and use the shared `PoseMath.cs` arc-length interpolation for algorithm poses. Available external poses always use their reported coordinates; unavailable poses are hidden. Geometry is display data, not permission to plan or infer an undeclared road.

### Offline recording limits

`FleetReplay` accepts recorded UTF-8 JSONL lines up to 4 MiB, including any event
wrapper. Its indexed file reader supports at most 8 GiB and one million observed
frames; its `TextReader` interface retains the smaller in-memory limits. Both
readers call `LiveMessageParser.ParseRecording`, which uses the same field and
identity validation as live messages with the recording byte limit. This permits
complete recorded trails larger than a live frame without truncating observations.
The public live parser retains its 64 KiB message limit. TCP uses the negotiated
fleet observation budget below; offline replay has no command transport.

`RecordedFrameSizeTests` covers raw and wrapped long observations through both
recording readers, UTF-8 byte limits, invalid observed poses, and rejection of the
same long frame by the public live parser.

### Live fleet observation limits

`LiveTcpClient` reads at most 64 KiB of UTF-8 bytes for the initial hello. After
that hello has been validated as a fleet hello, only fleet `frame` observations
and repeated fleet `hello` messages may use up to 4 MiB per line. Single-vehicle
messages, commands, receipts and unknown message types retain the ordinary
64 KiB limit. Scene/run identity and all field validation still apply before
dispatch. The larger observation budget preserves complete six-vehicle trails;
it does not truncate or resample position, fault or resource records.

`LiveClientOptions` can lower either limit, but cannot raise these hard bounds.
`Utf8LineReader` enforces bytes incrementally before decoding, rejects malformed
UTF-8, and rejects over-limit input without waiting for a newline. Limits exclude
LF and include any preceding CR. `FleetWireBudgetTests` exercises fragmented TCP,
exact and excess byte limits, long observations, invalid identity/pose, and
ordinary messages on a fleet connection. The single-vehicle bridge's input
limit and Python's fleet encoder are unchanged.
