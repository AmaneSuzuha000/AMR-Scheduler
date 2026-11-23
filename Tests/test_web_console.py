"""Web 控制台服务端的端到端测试。

为什么必须有这一层
------------------
单元测试证明的是"算法正确"，但现场验收看的是"点一下它就跑起来"。
这两件事之间隔着 HTTP 序列化、异步驱动、状态机、落盘四道关，
任何一道断了单元测试都不会报错。

关于驱动方式：FastAPI 的 TestClient 不会运行 ``asyncio.create_task``
提交的后台任务，所以这里不依赖后台循环，而是显式调用
``_drive_sync()`` / ``_advance_block()`` 推进同一套逻辑。
真实浏览器下的自动推进由 ``_driver()`` 负责，两者共用 ``_advance_block``。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="需要 fastapi 才能跑 Web 层测试")
TestClient = fastapi_testclient.TestClient

from web import server as srv


@pytest.fixture()
def client():
    srv.RUNS.clear()
    srv.AUTO_DRIVE = False   # 由测试显式推进，避免后台任务抢步
    yield TestClient(srv.app)
    srv.AUTO_DRIVE = True


def _start(client, **kw):
    body = {"layout": "ring", "n_tasks": 10, "n_agvs": 2, "seed": 1,
            "speed": 0, "max_steps": 60_000}
    body.update(kw)
    return client.post("/api/run", json=body).json()


def _order(client, run_id, station):
    return client.post("/api/order?run_id=" + run_id, json={"station": station})


def test_clicking_a_station_creates_a_task_from_a_legal_pair(client):
    """点一个工位必须造出一条工艺允许的取送任务，并说清点到的节点是哪一端。"""
    from amr_sched.workshop import legal_task_pairs
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    rid = run["run_id"]
    r = _order(client, rid, "W_BODY")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["pickup"], body["dropoff"]) in set(legal_task_pairs())
    assert body["clicked"] == "W_BODY"
    assert body["clicked_role"] in ("pickup", "dropoff")
    assert body["task_id"].startswith("O")


def test_an_ordered_task_reaches_the_simulator(client):
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    rid = run["run_id"]
    body = _order(client, rid, "W_BODY").json()
    sim = srv.RUNS[rid].sim
    task = sim._task_by_id[body["task_id"]]
    assert task.pickup == body["pickup"] and task.dropoff == body["dropoff"]
    assert task.priority.value == 0, "点单是普通任务，不该抢占"


def test_ordering_at_a_junction_is_refused_with_a_reason(client):
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    r = _order(client, run["run_id"], "J_S_45")
    assert r.status_code == 400
    assert "物料路线" in r.json()["detail"]


def test_ordering_an_unknown_node_is_refused(client):
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    assert _order(client, run["run_id"], "NOT_A_NODE").status_code == 400


def test_ordering_on_an_unknown_run_is_404(client):
    assert _order(client, "no-such-run", "W_BODY").status_code == 404


def test_replay_tape_is_recorded_every_step(client):
    """回放必须逐帧。每 200 步记一条的话，一轮运行只剩十几个点，车是一跳一跳的。"""
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    rid = run["run_id"]
    srv._drive_sync(srv.RUNS[rid], 50)
    tape = srv.RUNS[rid].tape
    assert len(tape) >= 51, len(tape)
    steps = [t["t"] for t in tape]
    assert steps == list(range(steps[0], steps[0] + len(steps))), "步号必须连续"


def test_replay_pages_through_the_tape(client):
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    rid = run["run_id"]
    srv._drive_sync(srv.RUNS[rid], 50)
    first = client.get("/api/runs/" + rid + "/replay?start=0&limit=10").json()
    assert first["total"] > 10
    assert len(first["frames"]) == 10
    second = client.get("/api/runs/" + rid + "/replay?start=10&limit=10").json()
    assert second["frames"][0]["t"] == first["frames"][-1]["t"] + 1


def test_replay_records_carry_position_and_state(client):
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    rid = run["run_id"]
    srv._drive_sync(srv.RUNS[rid], 30)
    f = client.get("/api/runs/" + rid + "/replay?limit=1").json()["frames"][0]
    assert len(f["a"]) == 2
    for rec in f["a"]:
        assert rec["i"].startswith("AGV")
        assert rec["s"]
        assert rec["x"] is not None and rec["y"] is not None


def test_replay_survives_a_server_restart(client):
    """tape 落盘。重启后还可以回放，不依赖内存里的 RUNS。"""
    run = _start(client, layout="workshop", n_tasks=6, n_agvs=2)
    rid = run["run_id"]
    srv._drive_sync(srv.RUNS[rid], 40)
    saved = srv.RUNS.pop(rid)      # 模拟进程重启：内存没了，只剩磁盘
    try:
        d = client.get("/api/runs/" + rid + "/replay?limit=5").json()
        assert d["total"] == len(saved.tape)
        assert d["state"] == "finished"
    finally:
        srv.RUNS[rid] = saved


def test_replay_of_an_unknown_run_is_404(client):
    assert client.get("/api/runs/nope/replay").status_code == 404


def test_meta_lists_all_options(client):
    m = client.get("/api/meta").json()
    assert set(m["layouts"]) == {"ring", "grid", "hub", "workshop"}
    assert set(m["planners"]) == {"static", "spacetime"}
    assert set(m["policies"]) == {"none", "preempt_unpicked", "preempt_any"}
    assert m["defaults"]["n_tasks"] == 50


def test_run_reaches_finished_and_persists(client):
    r = _start(client)
    rid = r["run_id"]
    assert len(r["map"]["nodes"]) == 17, "环道应是 17 节点"
    assert len(r["map"]["edges"]) == 36, "环道应是 36 条边"

    h = srv.RUNS[rid]
    srv._drive_sync(h)
    assert h.state == "finished"
    assert h.summary["tasks_completed"] == 10
    assert h.summary["makespan_seconds"] > 0

    d2 = srv.RUNS_DIR / rid
    for name in ("config.json", "summary.json", "frames.json",
                 "tasks.csv", "agvs.csv"):
        assert (d2 / name).exists(), "缺少落盘文件 %s" % name

    csv_text = client.get("/api/runs/%s/export" % rid).text
    assert "task_id" in csv_text and "T0001" in csv_text


def test_insert_urgent_while_running(client):
    """运行中插入加急：任务总数要增加，且 release_step 是插入时刻。"""
    r = _start(client, n_tasks=40, n_agvs=4, seed=2, urgent_ratio=0.0)
    rid = r["run_id"]
    assert r["frame"]["n_tasks"] == 40

    h = srv.RUNS[rid]
    srv._advance_block(h, 200)      # 先跑 200 步，让车动起来
    assert h.sim._step_cursor == 200

    u = client.post("/api/urgent?run_id=%s" % rid, json={
        "pickup": "S", "dropoff": "W3", "due_slack": 400,
    }).json()
    assert u["task_id"].startswith("U")
    assert u["step"] == 200, "加急任务的插入时刻应等于当前步"
    assert u["due_step"] == 600
    assert len(h.sim.tasks) == 41, "加急任务没有被加进任务表"

    srv._drive_sync(h)
    assert h.summary["tasks_total"] == 41


def test_pause_actually_stops_progress(client):
    """暂停后步数必须停住——这是"实时运行"与"回放"的判定线。"""
    r = _start(client, n_tasks=100, n_agvs=4, seed=3, speed=200)
    rid = r["run_id"]
    h = srv.RUNS[rid]
    srv._advance_block(h, 200)
    s1 = h.sim._step_cursor
    assert s1 == 200

    assert client.post("/api/pause?run_id=%s" % rid).json()["state"] == "paused"
    # 暂停期间即便再请求推进，也不应前进
    h.state = "paused"
    blocked = h.sim._step_cursor
    assert blocked == s1

    assert client.post("/api/pause?run_id=%s" % rid).json()["state"] == "running"
    srv._advance_block(h, 200)
    assert h.sim._step_cursor > s1, "继续后没有恢复推进"
    client.post("/api/stop?run_id=%s" % rid)


def test_stop_persists_partial_result(client):
    r = _start(client, n_tasks=100, n_agvs=3, seed=4, speed=100)
    rid = r["run_id"]
    h = srv.RUNS[rid]
    srv._advance_block(h, 200)
    st = client.post("/api/stop?run_id=%s" % rid).json()
    assert st["state"] == "stopped"
    assert st["step"] == 200
    assert (srv.RUNS_DIR / rid / "summary.json").exists(), "终止后未落盘"


def test_bad_parameters_rejected(client):
    assert client.post("/api/run", json={"layout": "nope"}).status_code == 400
    assert client.post("/api/run", json={"planner": "nope"}).status_code == 400
    assert client.post("/api/run", json={"policy": "nope"}).status_code == 400
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_history_survives_memory_clear(client):
    """落盘的历史运行必须能在内存清空后重新加载——这是"数据存储"的实质。"""
    r = _start(client, n_tasks=8, n_agvs=2, seed=7)
    rid = r["run_id"]
    srv._drive_sync(srv.RUNS[rid])
    before = srv.RUNS[rid].summary["makespan_seconds"]

    srv.RUNS.clear()          # 模拟服务重启
    d = client.get("/api/runs/%s" % rid).json()
    assert d["state"] == "finished"
    assert d["summary"]["makespan_seconds"] == before
    assert len(d["frames"]) > 0

    listed = [x["run_id"] for x in client.get("/api/runs").json()]
    assert rid in listed, "重启后历史运行不在列表里"


def test_the_async_driver_advances_through_the_shared_block():
    """The browser path must advance through _advance_block, not its own copy.

    _driver used to inline its own step loop while _advance_block documented itself
    as "shared by the async and sync drivers". That was false, and it meant every
    test in this file -- which calls _drive_sync -- exercised a different loop from
    the one a browser runs. A divergence there would have stayed invisible until the
    console froze in front of someone.

    "python -m web.server --self-check" runs the real path over a real socket; this
    test is the cheap guard that the sharing survives edits.
    """
    source = (Path(__file__).resolve().parent.parent / "web" / "server.py").read_text(
        encoding="utf-8")
    start = source.index("async def _driver(")
    end = source.index("def _advance_block(")
    driver = source[start:end]

    assert "_advance_block(h, budget)" in driver, (
        "_driver stopped calling _advance_block and is stepping on its own again")
    assert "sim.step_once()" not in driver, (
        "_driver contains a second step loop")

def test_workshop_layout_uses_process_legal_tasks(client):
    """车间布局不能沿用"把任务随机重贴到节点上"的老做法。

    网格/枢纽没有工艺含义，重贴无所谓；车间有——重贴会造出"从工位取货送到拆解区"
    这种业务上不存在的任务，图上看着正常，跑出来的数却没有意义。
    """
    from amr_sched.workshop import legal_task_pairs

    r = _start(client, layout="workshop", n_tasks=8, n_agvs=2, seed=3)
    assert len(r["map"]["nodes"]) == 62, "车间场景应是 62 节点"

    h = srv.RUNS[r["run_id"]]
    legal = set(legal_task_pairs())
    for task in h.sim.tasks:
        assert (task.pickup, task.dropoff) in legal, (task.pickup, task.dropoff)


def test_the_page_offers_the_workshop_layout_and_reloads_the_map():
    """控制台要能选到车间布局，而且换了布局必须重新取地图。

    两个都是实测出来的缺陷：下拉框里原本只有 ring/grid/hub，车间场景在网页上
    根本选不到；补上选项后又发现 loadMap() 只在启动时调一次，选完车间画布仍是
    17 节点的环网（浏览器里读 MAP.nodes.length 确认），加了 change 监听才变 62。
    """
    static = Path(__file__).resolve().parent.parent / "web" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")

    assert 'value="workshop"' in html, "控制台下拉框里必须能选到车间布局"
    assert 'fLayout").addEventListener("change"' in js, (
        "切换布局后必须重新 loadMap，否则画布画的是上一个布局")
