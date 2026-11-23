// Multi-AGV scheduling console — front end.
//
// Key design: this is not a player. After start, the server advances the
// Simulator step by step and pushes one frame per advance; this file only draws it.
// So "pause" really freezes the simulation and "insert urgent" really changes the running task set.

const $ = (id) => document.getElementById(id);
let RUN_ID = null;
let MAP = null;
let LAST = null;
let BOUND = null;
let WS = null;
let PAUSED = false;
//: Latest /api/live snapshot, or null when no bridge is attached.
let LIVE = null;

const COLORS = {
  IDLE: "#6e7d90", TO_PICKUP: "#58a6ff", LOADING: "#e3b341",
  TO_DROPOFF: "#7ee787", UNLOADING: "#a5d6ff", BLOCKED: "#ff7b72",
  PARKING: "#8b949e"
};

// The live stream sends state as an upper-case enum name; the replay tape sends the
// value of the same enum, which is lower-case. Looking up only one spelling made every
// replayed vehicle grey, i.e. the picture said "idle" for four vehicles that were
// driving. Accept either.
function stateColor(s) {
  if (!s) return "#8b949e";
  return COLORS[String(s).toUpperCase()] || "#8b949e";
}
const KIND = {
  AISLE: "#2a3546", DISASSEMBLY: "#8b5cf6", WORKSTATION: "#1a5fb4",
  ASSEMBLY: "#0f766e", BUFFER: "#7c5e10", DOCK: "#7a2a2a"
};

function log(msg, cls) {
  const d = document.createElement("div");
  if (cls) d.className = cls;
  d.textContent = new Date().toLocaleTimeString() + "  " + msg;
  const box = $("log");
  box.insertBefore(d, box.firstChild);
  while (box.childNodes.length > 40) box.removeChild(box.lastChild);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch (e) {}
    throw new Error(m);
  }
  return r.json();
}

function cfg() {
  return {
    layout: $("fLayout").value,
    n_agvs: +$("fAgvs").value,
    n_tasks: +$("fTasks").value,
    seed: +$("fSeed").value,
    aisle_capacity: +$("fCap").value,
    urgent_ratio: +$("fUrgent").value,
    planner: $("fPlanner").value,
    allocator: $("fAlloc").value,
    policy: $("fPolicy").value,
    speed: +$("fSpeed").value
  };
}
async function start() {
  $("btnStart").disabled = true;
  try {
    const res = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg())
    });
    RUN_ID = res.run_id;
    MAP = res.map;
    BOUND = bounds(MAP);
    LAST = res.frame;
    PAUSED = false;
    $("btnPause").disabled = false;
    $("btnStop").disabled = false;
    $("btnUrgent").disabled = false;
    // Enabled on start, not on finish: the tape grows as the run advances, so a run
    // still in progress can already be replayed up to its current step.
    $("btnReplay").disabled = false;
    $("btnPause").textContent = "Pause";
    setBadge("bState", "Running", "run");
    setBadge("bRun", RUN_ID.slice(-6), "");
    log("Started run " + RUN_ID, "log-ok");
    draw();
    poll();
  } catch (e) {
    log("Start failed: " + e.message, "log-e");
    $("btnStart").disabled = false;
  }
}

function setBadge(id, text, cls) {
  const b = $(id);
  b.textContent = text;
  b.className = "badge" + (cls ? " " + cls : "");
}

// Poll the current frame. Polling instead of a direct WebSocket push keeps
// the front end from being flooded when the server runs fast — polling rate-limits naturally.
let pollTimer = null;
function poll() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!RUN_ID) return;
    try {
      const d = await api("/api/runs/" + RUN_ID);
      if (d.frames && d.frames.length) LAST = d.frames[d.frames.length - 1];
      if (d.state) LAST.state = d.state;
      draw();
      if (d.state === "finished" || d.state === "stopped") {
        onFinish(d);
        return;
      }
    } catch (e) { /* silent: retry next tick */ }
    poll();
  }, 120);
}

function onFinish(d) {
  // 回放期间不改状态徽标：它描述的是"这一轮跑完了"，而画面上正在放的是历史，
  // 两者同时显示会让人以为画面是实时的。
  if (!REPLAY) setBadge("bState", d.state === "stopped" ? "Stopped" : "Completed", "ok");
  $("btnStart").disabled = false;
  $("btnPause").disabled = true;
  $("btnStop").disabled = true;
  $("btnUrgent").disabled = true;
  const s = LAST && LAST.summary;
  if (s) {
    log("Finished " + (s.tasks_completed || 0) + " tasks in " +
        Math.round(s.makespan_seconds || 0) + " s", "log-ok");
  }
}
function bounds(map) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const n of map.nodes) {
    if (n.x < x0) x0 = n.x;
    if (n.y < y0) y0 = n.y;
    if (n.x > x1) x1 = n.x;
    if (n.y > y1) y1 = n.y;
  }
  const pad = 40;
  return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
}

function project(p) {
  const cv = $("cv");
  const sw = (BOUND.x1 - BOUND.x0), sh = (BOUND.y1 - BOUND.y0);
  const k = Math.min(cv.width / sw, cv.height / sh);
  const ox = (cv.width - sw * k) / 2, oy = (cv.height - sh * k) / 2;
  return [ox + (p.x - BOUND.x0) * k, oy + (p.y - BOUND.y0) * k, k];
}

function nodeAt(id) {
  for (const n of MAP.nodes) if (n.id === id) return n;
  return null;
}

function fitCanvas() {
  const cv = $("cv"), stage = $("stage");
  const w = stage.clientWidth, h = stage.clientHeight;
  if (w > 0 && h > 0 && (cv.width !== w || cv.height !== h)) {
    cv.width = w;
    cv.height = h;
  }
}

// The static layer: background, roads, stations. Split out of draw() so replay can
// repaint the map without touching the live path -- replay and live must not be able
// to draw into each other.
function drawBase() {
  fitCanvas();
  const cv = $("cv"), ctx = cv.getContext("2d");
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (!MAP) return;

  // Edges
  for (const e of MAP.edges) {
    const a = nodeAt(e.src), b = nodeAt(e.dst);
    if (!a || !b) continue;
    const pa = project(a), pb = project(b);
    ctx.strokeStyle = "#1b2430";
    ctx.lineWidth = Math.max(1, e.length / 26);
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }

  // Nodes
  for (const n of MAP.nodes) {
    const p = project(n);
    const r = n.kind === "AISLE" ? 4 : 9;
    ctx.fillStyle = KIND[n.kind] || "#2a3546";
    ctx.beginPath();
    ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
    ctx.fill();
    if (n.kind !== "AISLE") {
      ctx.strokeStyle = "#3a4a5e";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    if (n.kind !== "AISLE") {
      ctx.fillStyle = "#7d8b9e";
      ctx.font = "11px Consolas,monospace";
      ctx.textAlign = "center";
      ctx.fillText(n.id, p[0], p[1] - 14);
    }
  }
}

function draw() {
  // While playing history, the live paint would fight the replay paint.
  if (REPLAY) return;
  drawBase();
  if (!MAP) return;

  // The bridge overlay is drawn whether or not a run exists: seeing the vehicle in
  // Gazebo move across the workshop is a check in its own right, and it must not require
  // launching a scheduling job first.
  drawLive();

  if (!LAST) return;

  // Remaining route per vehicle
  for (const a of LAST.agvs || []) {
    if (!a.path || !a.path.length) continue;
    const start = nodeAt(a.node);
    if (!start) continue;
    const ps = project(start);
    ctx.strokeStyle = "rgba(88,166,255,.45)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(ps[0], ps[1]);
    for (const nid of a.path) {
      const n = nodeAt(nid);
      if (!n) continue;
      const p = project(n);
      ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // AGVs
  // Position comes from the server as continuous map coordinates (a.xy), plus
  // a.progress along the current edge and a.yaw for the heading. Older frames
  // without those fields fall back to the edge midpoint, which is only correct
  // at progress = 0.5 but keeps the console usable against a stale server.
  for (const a of LAST.agvs || []) {
    let p = null;
    if (a.xy && a.xy.length === 2) {
      p = project({ x: a.xy[0], y: a.xy[1] });
    } else if (a.pos && a.pos.indexOf("->") >= 0) {
      const parts = a.pos.split("->");
      const n1 = nodeAt(parts[0]), n2 = nodeAt(parts[1]);
      if (n1 && n2) {
        const t = (typeof a.progress === "number") ? a.progress : 0.5;
        const pa = project(n1), pb = project(n2);
        p = [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t];
      }
    } else {
      const n = nodeAt(a.pos);
      if (n) p = project(n);
    }
    if (!p) continue;
    const col = COLORS[a.state] || "#8b949e";
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(p[0], p[1], 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#0b0f14";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#e6edf3";
    ctx.font = "bold 10px Consolas,monospace";
    ctx.textAlign = "center";
    ctx.fillText(a.id.replace("AGV", ""), p[0], p[1] + 3.5);
  }

  drawHud();
  drawStats();
}

// The bridge overlay is drawn with a deliberately different vocabulary from the simulated
// vehicles: a hollow ring, a dashed halo, and a text tag carrying the channel name. A pose
// that came in over a socket must never be mistakable at a glance for one the scheduler
// computed, and a stale one must never be mistakable for a moving one.
function drawLive() {
  const lv = LIVE;
  if (!lv || !lv.attached || lv.waiting) return;
  const p = project({ x: lv.x, y: lv.y });
  const stale = !!lv.stale;
  const col = stale ? "#f0883e" : "#39d0d8";

  const cv = $("cv"), c = cv.getContext("2d");
  // halo: dashed while live, solid-ish double ring when stale, so motion state is visible
  // without reading the label
  c.save();
  c.strokeStyle = col;
  c.lineWidth = 1.5;
  c.setLineDash(stale ? [2, 3] : [6, 5]);
  c.beginPath();
  c.arc(p[0], p[1], 14, 0, Math.PI * 2);
  c.stroke();
  c.setLineDash([]);
  c.beginPath();
  c.arc(p[0], p[1], 7, 0, Math.PI * 2);
  c.stroke();
  c.restore();

  // heading tick, so a wrong yaw sign is visible rather than hidden by a round marker
  if (typeof lv.yaw === "number") {
    c.save();
    c.strokeStyle = col;
    c.lineWidth = 2;
    c.beginPath();
    c.moveTo(p[0], p[1]);
    c.lineTo(p[0] + Math.cos(-lv.yaw) * 18, p[1] + Math.sin(-lv.yaw) * 18);
    c.stroke();
    c.restore();
  }

  c.save();
  c.fillStyle = col;
  c.font = "10px Consolas,monospace";
  c.textAlign = "left";
  const tag = "bridge:" + (lv.channel || "?") +
    (stale ? "  STALE " + (lv.age_s || 0).toFixed(1) + "s" : "");
  c.fillText(tag, p[0] + 16, p[1] - 10);
  c.restore();
}
function drawHud() {
  if (!LAST) return;
  const c = LAST.counts || {};
  const done = (c.completed || 0) + (c.failed || 0);
  const n = LAST.n_tasks || 1;
  const pct = Math.round(done / n * 100);
  $("hud").innerHTML =
    "<b>Step " + (LAST.step || 0) + "</b>  progress " + done + "/" + n +
    " (" + pct + "%)<br>" +
    "pending " + (c.pending || 0) + "  assigned " + (c.assigned || 0) +
    "  picked " + (c.picked || 0) + "<br>" +
    "deadlock breaks " + (LAST.deadlock_breaks || 0) +
    "  preemptions " + (LAST.preemptions || 0);
}

function drawStats() {
  if (!LAST) return;
  const c = LAST.counts || {};
  const s = LAST.summary;
  const rows = [
    ["Steps", LAST.step || 0],
    ["Completed", (c.completed || 0) + " / " + (LAST.n_tasks || 0)],
    ["Pending", c.pending || 0],
    ["Deadlock breaks", LAST.deadlock_breaks || 0],
    ["Preemptions", LAST.preemptions || 0]
  ];
  if (s) {
    rows.push(["Completion time", Math.round(s.makespan_seconds || 0) + " s"]);
    rows.push(["On-time rate", ((s.on_time_rate || 0) * 100).toFixed(1) + "%"]);
    rows.push(["Urgent on-time", ((s.urgent_on_time_rate || 0) * 100).toFixed(1) + "%"]);
    rows.push(["Utilization", ((s.agv_utilization || 0) * 100).toFixed(1) + "%"]);
  }
  $("stats").innerHTML = rows.map(r =>
    "<div class=\"stat\"><span>" + r[0] + "</span><span>" + r[1] + "</span></div>"
  ).join("");

  $("agvlist").innerHTML = (LAST.agvs || []).map(a => {
    const col = COLORS[a.state] || "#8b949e";
    return "<div class=\"agv\"><div class=\"dot\" style=\"background:" + col +
      "\"></div><div class=\"nm\">" + a.id + "</div><div class=\"st\">" +
      a.state + (a.task ? " · " + a.task : "") + "</div></div>";
  }).join("");
}

async function togglePause() {
  if (!RUN_ID) return;
  try {
    const r = await api("/api/pause?run_id=" + encodeURIComponent(RUN_ID), { method: "POST" });
    PAUSED = r.state === "paused";
    $("btnPause").textContent = PAUSED ? "Resume" : "Pause";
    setBadge("bState", PAUSED ? "Paused" : "Running", PAUSED ? "" : "run");
    log(PAUSED ? "Paused" : "Resumed");
  } catch (e) { log("Action failed: " + e.message, "log-e"); }
}

async function stopRun() {
  if (!RUN_ID) return;
  try {
    await api("/api/stop?run_id=" + encodeURIComponent(RUN_ID), { method: "POST" });
    log("Stopped", "log-e");
    setBadge("bState", "Stopped", "ok");
    $("btnStart").disabled = false;
    $("btnPause").disabled = true;
    $("btnStop").disabled = true;
    $("btnUrgent").disabled = true;
  } catch (e) { log("Stop failed: " + e.message, "log-e"); }
}

async function insertUrgent() {
  if (!RUN_ID) return;
  try {
    const r = await api("/api/urgent?run_id=" + encodeURIComponent(RUN_ID), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pickup: $("uPick").value,
        dropoff: $("uDrop").value,
        due_slack: +$("uSlack").value
      })
    });
    log("Inserted urgent " + r.task_id + " (step " + r.step + ")", "log-u");
  } catch (e) { log("Insert failed: " + e.message, "log-e"); }
}

$("btnStart").onclick = start;
$("btnLiveOn").onclick = attachLive;
$("btnLiveOff").onclick = detachLive;
$("btnPause").onclick = togglePause;
$("btnStop").onclick = stopRun;
$("btnUrgent").onclick = insertUrgent;
$("fSpeed").onchange = async () => {
  if (!RUN_ID) return;
  try {
    await api("/api/speed?run_id=" + encodeURIComponent(RUN_ID), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speed: +$("fSpeed").value })
    });
  } catch (e) {}
};

// ---------- live bridge overlay ----------
// Polls /api/live and draws whatever the Ubuntu bridge is reporting. Independent of the
// scheduling run: attaching a bridge is what makes the canvas show a vehicle that this
// process did not simulate.
let liveTimer = null;
async function livePoll() {
  if (liveTimer) clearTimeout(liveTimer);
  liveTimer = setTimeout(async () => {
    try {
      const d = await api("/api/live");
      LIVE = d.live || null;
      setBadge("bLive", LIVE && LIVE.attached
        ? (LIVE.waiting ? "Bridge: waiting" : (LIVE.stale ? "Bridge: STALE" : "Bridge: " + LIVE.channel))
        : "Bridge: off",
        LIVE && LIVE.attached ? (LIVE.stale ? "warn" : "ok") : "");
      draw();
    } catch (e) { /* server restarted: keep polling */ }
    livePoll();
  }, 200);
}

async function attachLive() {
  const host = $("lHost").value || "127.0.0.1";
  const port = +$("lPort").value || 9300;
  $("btnLiveOn").disabled = true;
  try {
    const r = await api("/api/live/attach", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host, port })
    });
    LIVE = r.live || null;
    log("Bridge attached " + host + ":" + port + " (frame " + (LIVE && LIVE.channel || "?") + ")", "log-ok");
    $("btnLiveOff").disabled = false;
    // The overlay needs map geometry even when no run is active.
    if (!MAP) await loadMap();
    draw();
  } catch (e) {
    log("Attach failed: " + e.message, "log-e");
    $("btnLiveOn").disabled = false;
  }
}

async function detachLive() {
  try {
    await api("/api/live", { method: "DELETE" });
  } catch (e) {}
  LIVE = null;
  $("btnLiveOn").disabled = false;
  $("btnLiveOff").disabled = true;
  setBadge("bLive", "Bridge: off", "");
  log("Bridge detached");
  draw();
}

async function loadMap() {
  try {
    const m = await api("/api/map?layout=" + encodeURIComponent($("fLayout").value));
    MAP = m;
    BOUND = bounds(MAP);
  } catch (e) {}
}

// Start on the bare map so the live overlay has geometry before any run exists.
loadMap().then(() => draw());
livePoll();

log("Console ready");

// Switching layout must reload the map. Without this the canvas keeps drawing the
// previous layout while the run uses the new one: measured, picking "Workshop" left
// the canvas showing the 17-node ring while a 62-node workshop run was in progress.
$("fLayout").addEventListener("change", () => { loadMap().then(() => draw()); });


// For automated screenshots/demos: ?auto=1 starts a run with the current form values
(function autoStart() {
  const q = new URLSearchParams(location.search);
  if (q.get("auto") !== "1") return;
  if (q.get("tasks")) $("fTasks").value = q.get("tasks");
  if (q.get("agvs")) $("fAgvs").value = q.get("agvs");
  if (q.get("speed")) $("fSpeed").value = q.get("speed");
  setTimeout(start, 250);
})();

// ---------------------------------------------------------------------------
// Click a station to order.
//
// The map is the only place where "which station" is unambiguous, so ordering
// starts there rather than from a dropdown of 62 node ids. A click resolves to
// the nearest non-junction node within a few pixels; junctions are excluded
// because they are not delivery points, and a miss says so instead of silently
// ordering something else.
// ---------------------------------------------------------------------------
const ORDER_HIT_RADIUS = 16;

function stationUnder(cx, cy) {
  if (!MAP) return null;
  let best = null, bestD = ORDER_HIT_RADIUS * ORDER_HIT_RADIUS;
  for (const n of MAP.nodes) {
    if (n.kind === "AISLE") continue;
    const p = project(n);
    const dx = p[0] - cx, dy = p[1] - cy;
    const d = dx * dx + dy * dy;
    if (d <= bestD) { bestD = d; best = n; }
  }
  return best;
}

$("cv").addEventListener("click", async (ev) => {
  const rect = ev.target.getBoundingClientRect();
  const node = stationUnder(ev.clientX - rect.left, ev.clientY - rect.top);
  if (!node) { log("这里不是工位（路口不能下单）", "log-u"); return; }
  if (!RUN_ID) { log("先启动一次运行，再点工位下单", "log-u"); return; }
  try {
    const r = await api("/api/order?run_id=" + encodeURIComponent(RUN_ID), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ station: node.id }),
    });
    const role = r.clicked_role === "dropoff" ? "点的是收货端" : "点的是取货端";
    log("下单 " + r.task_id + ": " + r.pickup + " → " + r.dropoff + "（" + role + "）", "log-ok");
    draw();
  } catch (e) {
    log("下单失败: " + (e && e.message ? e.message : e), "log-e");
  }
});


// ---------------------------------------------------------------------------
// Replay
//
// The live canvas draws LAST, a frame that arrives over a websocket. Replay draws
// from the tape instead: one record per step, so the movement is continuous. The two
// never run at once -- starting playback freezes the live view, and the badge says so,
// because a picture that might be live or might be history is worse than no picture.
//
// Route geometry is not part of the tape (it would triple the record size), so replay
// shows where the vehicles were, not where they were headed. That is a real limitation,
// stated in the panel rather than papered over.
// ---------------------------------------------------------------------------
let REPLAY = null;   // { runId, frames, i, timer }

async function startReplay() {
  if (!RUN_ID) { log("先跑完一轮，再回放", "log-u"); return; }
  try {
    const first = await api("/api/runs/" + encodeURIComponent(RUN_ID) + "/replay?start=0&limit=5000");
    if (!first.total) { log("这一轮没有回放记录", "log-u"); return; }
    const frames = first.frames.slice();
    for (let s = frames.length; s < first.total; s += 5000) {
      const page = await api("/api/runs/" + encodeURIComponent(RUN_ID) +
        "/replay?start=" + s + "&limit=5000");
      frames.push(...page.frames);
    }
    if (REPLAY && REPLAY.timer) clearInterval(REPLAY.timer);
    REPLAY = { runId: RUN_ID, frames: frames, i: 0, timer: null };
    $("btnReplay").textContent = "Stop replay";
    $("btnReplay").classList.add("danger");
    setBadge("bState", "回放 " + frames.length + " 帧", "run");
    log("回放 " + frames.length + " 帧，从第 " + frames[0].t + " 步开始", "log-ok");
    drawReplayFrame();
    REPLAY.timer = setInterval(() => {
      if (!REPLAY) return;
      REPLAY.i += 1;
      if (REPLAY.i >= REPLAY.frames.length) { stopReplay(); return; }
      drawReplayFrame();
    }, 40);
  } catch (e) {
    log("回放失败: " + (e && e.message ? e.message : e), "log-e");
  }
}

function stopReplay() {
  if (REPLAY && REPLAY.timer) clearInterval(REPLAY.timer);
  REPLAY = null;
  $("replayHud").hidden = true;
  $("hud").style.visibility = "visible";
  $("btnReplay").textContent = "Replay";
  $("btnReplay").classList.remove("danger");
  setBadge("bState", $("btnStart").disabled ? (PAUSED ? "Paused" : "Running") : "Completed", "");
  draw();
}

function drawReplayFrame() {
  if (!REPLAY) return;
  const f = REPLAY.frames[REPLAY.i];
  if (!f) return;
  drawBase();
  const cv = $("cv"), ctx = cv.getContext("2d");
  for (const a of f.a || []) {
    if (a.x === null || a.y === null) continue;
    const p = project({ x: a.x, y: a.y });
    ctx.fillStyle = stateColor(a.s);
    ctx.beginPath();
    ctx.arc(p[0], p[1], 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#0b0f14";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "#e6edf3";
    ctx.font = "bold 10px Consolas,monospace";
    ctx.textAlign = "center";
    ctx.fillText(a.i.replace("AGV", ""), p[0], p[1] + 3.5);
  }
  // 进度写进 HTML 覆盖层，不画在 canvas 上：左上角那块实时 HUD 是 HTML，画在
  // canvas 上的字会被它整个盖住（第一次截图就是这么丢掉标签的）。
  const hud = $("replayHud");
  hud.hidden = false;
  hud.textContent = "回放  step " + f.t + "  (" + (REPLAY.i + 1) + "/" + REPLAY.frames.length + ")";
  // 实时 HUD 和右侧车辆面板读的都是实时状态，回放时它们是陈旧的，必须换掉而不是
  // 留在那里误导人 —— 否则画面上四辆车在跑，右边的表还写着全部 idle。
  $("hud").style.visibility = "hidden";
  $("agvlist").innerHTML = (f.a || []).map(a =>
    "<div class=\"agv\"><div class=\"dot\" style=\"background:" + stateColor(a.s) +
    "\"></div><div class=\"nm\">" + a.i + "</div><div class=\"st\">" +
    a.s + (a.n ? " · " + a.n : "") + "</div></div>").join("");
}

$("btnReplay").addEventListener("click", () => {
  if (REPLAY) stopReplay(); else startReplay();
});

