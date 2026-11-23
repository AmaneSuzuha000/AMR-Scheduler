"""预约表测试。"""

import pytest

from amr_sched.reservation import Interval, ReservationTable


def test_interval_overlap():
    assert Interval(0, 5).overlaps(Interval(4, 8))
    assert not Interval(0, 5).overlaps(Interval(5, 8))   # 左闭右开，相邻不算冲突
    assert Interval(0, 5).overlaps(Interval(2, 3))


def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        Interval(5, 0)


def test_node_reservation_blocks_same_time():
    rt = ReservationTable()
    rt.reserve_node("A", Interval(0, 3), "agv1")
    assert rt.is_node_free("A", Interval(1, 2), capacity=1) is False
    assert rt.is_node_free("A", Interval(1, 2), capacity=2) is True
    assert rt.is_node_free("A", Interval(3, 4), capacity=1) is True


def test_reservation_ignores_self():
    rt = ReservationTable()
    rt.reserve_node("A", Interval(0, 3), "agv1")
    assert rt.is_node_free("A", Interval(0, 3), capacity=1, exclude_agv="agv1") is True


def test_edge_capacity():
    rt = ReservationTable()
    rt.reserve_edge("A", "B", Interval(0, 5), "agv1")
    assert rt.is_edge_free("A", "B", Interval(1, 2), capacity=1) is False
    assert rt.is_edge_free("A", "B", Interval(1, 2), capacity=2) is True


def test_release_all():
    rt = ReservationTable()
    rt.reserve_node("A", Interval(0, 3), "agv1")
    rt.reserve_edge("A", "B", Interval(0, 3), "agv1")
    rt.release_all("agv1")
    assert rt.total_reservations() == 0


def test_purge_before_frees_past():
    rt = ReservationTable()
    rt.reserve_node("A", Interval(0, 3), "agv1")
    rt.reserve_node("A", Interval(10, 12), "agv1")
    rt.purge_before(5)
    assert len(rt._node["A"]) == 1


def test_release_specific_edge():
    rt = ReservationTable()
    span = Interval(0, 5)
    rt.reserve_edge("A", "B", span, "agv1")
    rt.release_edge("A", "B", span, "agv1")
    assert rt.total_reservations() == 0
