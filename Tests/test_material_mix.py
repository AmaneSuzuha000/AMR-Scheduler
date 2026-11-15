"""物料分类型的配对不变量。

experiments/run_material_mix_case.py 做的是 A/B 配对比较。
配对要成立，同一 seed 下两种口径必须产出**逐字段相同**的任务——
只有 material_type 不同。这条不变了一旦破坏，差值里就会混进任务差异，
而且不会报错，只会给出静悄悄的错误结论（第一版正是如此）。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amr_sched.entities import MaterialType                          # noqa: E402
from amr_sched.scenarios import ScenarioConfig, generate_scenario    # noqa: E402


def test_material_mix_does_not_perturb_task_sequence():
    for seed in range(6):
        a = generate_scenario(ScenarioConfig(n_tasks=20, n_agvs=3, seed=seed))
        b = generate_scenario(ScenarioConfig(n_tasks=20, n_agvs=3, seed=seed,
                                             material_mix=True))
        ta, tb = a[1], b[1]
        assert len(ta) == len(tb)
        for x, y in zip(ta, tb):
            assert x.task_id == y.task_id
            assert x.pickup == y.pickup
            assert x.dropoff == y.dropoff
            assert x.release_step == y.release_step
            assert x.due_step == y.due_step
            assert x.priority == y.priority
            # 只有物料不同
            assert x.material_type is None
            assert y.material_type is not None


def test_material_mix_default_is_off():
    """默认必须关闭：打开会改变装卸时长，历史 4540 行数据不再可比。"""
    cfg = ScenarioConfig()
    assert cfg.material_mix is False
    _net, tasks, _agvs = generate_scenario(ScenarioConfig(n_tasks=10, seed=0))
    assert all(t.material_type is None for t in tasks)


def test_material_weights_produce_expected_mix():
    _net, tasks, _agvs = generate_scenario(
        ScenarioConfig(n_tasks=300, n_agvs=2, seed=0, material_mix=True))
    c = Counter(t.material_type for t in tasks)
    assert set(c) == {MaterialType.BOGIE, MaterialType.WHEELSET, MaterialType.BRAKE_VALVE}
    # 权重 0.2/0.45/0.35，300 个样本容差 ±8 个百分点
    n = sum(c.values())
    assert abs(c[MaterialType.BOGIE] / n - 0.20) < 0.08
    assert abs(c[MaterialType.WHEELSET] / n - 0.45) < 0.08
    assert abs(c[MaterialType.BRAKE_VALVE] / n - 0.35) < 0.08


def test_typical_load_steps_ordering():
    """转向架最重、制动阀最轻——这是业务事实，不能被改反。"""
    b = MaterialType.BOGIE.typical_load_steps()
    w = MaterialType.WHEELSET.typical_load_steps()
    v = MaterialType.BRAKE_VALVE.typical_load_steps()
    assert b > w > v > 0
