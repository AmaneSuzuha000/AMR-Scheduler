"""Clock ordering, replay and receipt freshness across independent ROS sources."""
import pytest

from tools.wheelset_scene.navigation_samples import EquipmentSamples


def test_future_arrivals_do_not_hide_the_previous_causal_equipment_sample():
    samples = EquipmentSamples()
    old, future = object(), object()
    samples.add('joint', old, source_s=1., received_wall=10.)
    samples.add('joint', future, source_s=1.1, received_wall=10.01)
    assert samples.select('joint', source_now=1.05, wall_now=10.02) is old
    assert samples.select('joint', source_now=1.1, wall_now=10.03) is future
    assert samples.select('cargo', source_now=1.1, wall_now=10.03) is None


@pytest.mark.parametrize('source_s', [1., .9])
def test_duplicate_or_reordered_source_samples_never_renew_wall_credit(source_s):
    samples = EquipmentSamples()
    original = object()
    assert samples.add('contact', original, source_s=1., received_wall=10.)
    assert not samples.add('contact', object(), source_s=source_s, received_wall=10.29)
    assert samples.select('contact', source_now=1.1, wall_now=10.29) is original
    assert samples.select('contact', source_now=1.1, wall_now=10.31) is None


@pytest.mark.parametrize('source_now,wall_now', [(1.151, 10.1), (.99, 10.1), (1.1, 9.9), (1.1, 10.301)])
def test_source_or_receipt_age_and_clock_rollback_fail_closed(source_now, wall_now):
    samples = EquipmentSamples(); samples.add('cargo', object(), source_s=1., received_wall=10.)
    assert samples.select('cargo', source_now=source_now, wall_now=wall_now) is None


def test_finite_history_and_invalid_timestamps_preserve_freshness_contract():
    samples = EquipmentSamples(capacity=3)
    for index in range(10):
        samples.add('carrier', index, source_s=index/100, received_wall=10.+index/100)
    assert len(samples.rows['carrier']) == 3
    assert samples.select('carrier', source_now=.065, wall_now=10.1) is None
    assert not samples.add('carrier', 'late', source_s=.10, received_wall=10.)
    with pytest.raises(ValueError, match='finite'):
        samples.add('carrier', None, source_s=float('nan'), received_wall=11.)
    with pytest.raises(ValueError, match='finite'):
        samples.select('carrier', source_now=.1, wall_now=float('inf'))


def test_serialized_contact_history_preserves_full_fields_and_detaches_each_selection():
    from copy import deepcopy
    samples=EquipmentSamples(serialize_contacts=True)
    row={'header':{'stamp':{'sec':1,'nanosec':2}},'states':[{'normal':[1.,2.,3.],
        'positions':[[4.,5.,6.]],'wrenches':[],'info':'raw contact info'}],
        '_topic':'/scout_bin/task/bin_contacts','_received_wall':10.}
    expected=deepcopy(row)
    assert samples.add(row['_topic'],row,source_s=1.,received_wall=10.)
    row['states'][0]['normal'][0]=-1
    first=samples.select(expected['_topic'],source_now=1.1,wall_now=10.1)
    assert first==expected
    first['states'].clear()
    assert samples.select(expected['_topic'],source_now=1.1,wall_now=10.1)==expected
    assert samples.select(expected['_topic'],source_now=1.1,wall_now=10.31) is None


def test_serialized_contact_history_keeps_future_and_capacity_rules():
    samples=EquipmentSamples(capacity=3,serialize_contacts=True);topic='/scout_bin/task/bin_contacts'
    for index in range(5):
        samples.add(topic,{'value':index},source_s=index/100.,received_wall=10.+index/100.)
    assert len(samples.rows[topic])==3 and samples.select(topic,source_now=.015,wall_now=10.1) is None
    assert samples.select(topic,source_now=.035,wall_now=10.1)=={'value':3}
    assert not samples.add(topic,{'value':'duplicate'},source_s=.04,received_wall=10.2)
    assert samples.select(topic,source_now=.04,wall_now=10.1)=={'value':4}
