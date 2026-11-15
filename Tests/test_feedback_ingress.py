"""A receiver may retain evidence, never renew freshness or decide fleet state."""
from copy import deepcopy
import threading
from types import SimpleNamespace as NS

import pytest

from tools.live_bridge.feedback_ingress import InputWorker, ReceiptQueue, SensorIngress
from tools.wheelset_scene.navigation_equipment_ingress import EquipmentIngress


def fixture():
    ingress = SensorIngress.__new__(SensorIngress)
    worker = InputWorker.__new__(InputWorker)
    worker.lock = threading.RLock(); worker.owner_thread = threading.get_ident()
    worker.error = None; worker.thread = NS(is_alive=lambda: True)
    ingress.worker = worker; ingress.guard = ReceiptQueue(2)
    ingress.clock = ingress.regression = None; ingress.reported_error = False
    sample = dict(stamp=10., received=100., valid=True)
    ingress.state = NS(health=NS(samples={'imu': sample}), filtered=object(), filtered_received=100.)
    def lookup(*args):
        assert worker.lock._is_owned()
        return 'same-snapshot-transform'
    ingress.tf = NS(lookup_transform=lookup); ingress.time_zero = 0
    events = []; clocks = []; guards = []; failures = []
    ingress.owner = NS(health=NS(samples={}), sim=NS(execution=NS(fail=failures.append)),
        on_clock=clocks.append, on_guard=lambda m, **kw: guards.append((m, kw)),
        record=lambda *a, **k: events.append((a, k)))
    return ingress, events, clocks, guards, failures


def clock(value):
    return NS(clock=NS(sec=value, nanosec=0))


def transform(parent, child, sec, nanosec=0):
    return NS(header=NS(frame_id=parent, stamp=NS(sec=sec, nanosec=nanosec)),
              child_frame_id=child)


def test_tf_edge_observations_keep_original_source_and_receipt_without_retiming():
    ingress, _, _, _, _ = fixture()
    ingress.tf_edges = {}
    ingress.note_transforms(NS(transforms=[
        transform('map', 'odom', 12, 300000000),
        transform('odom', 'base_footprint', 12, 100000000),
        transform('base_footprint', 'laser', 0),
    ]), 102.)
    ingress.note_transforms(NS(transforms=[transform('odom', 'base_footprint', 13)]), 103.)
    assert ingress.tf_edges == {
        'map->odom': dict(source_s=12.3, received_wall_s=102.),
        'odom->base_footprint': dict(source_s=13., received_wall_s=103.)}
    assert ingress.owner.health.samples == {}


@pytest.mark.parametrize('latest,best_effort,expected',[
    (False,False,None),(True,False,{'depth':1,'reliability':'reliable'}),
    (False,True,{'depth':100,'reliability':'best_effort'}),
])
def test_declared_dynamic_qos_matches_traced_receiver_and_leaves_static_default(monkeypatch,latest,best_effort,expected):
    import sys
    from tools.live_bridge.feedback_ingress import dynamic_tf_qos
    monkeypatch.setitem(sys.modules,'rclpy.qos',NS(QoSProfile=lambda **kw:kw,
        ReliabilityPolicy=NS(RELIABLE='reliable',BEST_EFFORT='best_effort')))
    assert dynamic_tf_qos(latest_tf=latest,best_effort_tf=best_effort)==expected
    if expected is not None:
        assert set(expected)=={'depth','reliability'}  # No static_qos or lifetime override.


def test_conflicting_dynamic_qos_is_rejected_before_starting_any_receiver():
    from tools.live_bridge.feedback_ingress import dynamic_tf_qos
    with pytest.raises(ValueError,match='choose one'):
        dynamic_tf_qos(latest_tf=True,best_effort_tf=True)


def test_frozen_udp_profile_uses_only_the_declared_transport_without_global_changes():
    from pathlib import Path
    import xml.etree.ElementTree as ET
    path=Path(__file__).resolve().parents[1]/'tools/ros2/fastdds_udp_only.xml'
    root=ET.parse(path).getroot()
    namespace={'d':'http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'}
    assert [n.text for n in root.findall('.//d:transport_descriptor/d:type',namespace)]==['UDPv4']
    assert root.findtext('.//d:participant/d:rtps/d:useBuiltinTransports',namespaces=namespace)=='false'
    assert root.find('.//d:participant',namespace).get('is_default_profile')=='true'


def test_tf_diagnostic_detaches_observation_and_queries_each_edge_under_lock():
    ingress, _, _, _, _ = fixture()
    ingress.tf_edges = {'map->odom': dict(source_s=12.3, received_wall_s=102.)}
    queries = []
    def lookup(parent, child, when):
        assert ingress.worker.lock._is_owned() and when == ingress.time_zero
        queries.append((parent, child))
        if parent == 'map':
            return transform(parent, child, 12, 300000000)
        raise LookupError('edge missing')
    ingress.tf = NS(lookup_transform=lookup)
    before = deepcopy(ingress.tf_edges)
    row = ingress.transform_diagnostic()
    assert queries == [('map', 'odom'), ('odom', 'base_footprint')]
    assert row['map->odom']['buffer_source_s'] == 12.3
    assert row['odom->base_footprint']['buffer_error'] == 'edge missing'
    row['map->odom']['source_s'] = -1
    assert ingress.tf_edges == before


def test_delayed_consumption_preserves_receipt_age_and_all_guard_transitions():
    ingress, _, _, guards, failures = fixture()
    ingress.guard.append(('latched', 99.)); ingress.guard.append(('armed', 99.1))
    ingress.pull()
    assert guards == [('latched', {'received_at': 99.}), ('armed', {'received_at': 99.1})]
    assert ingress.owner.health.samples['imu']['received'] == 100.
    assert ingress.owner.filtered_received == 100. and not failures
    assert ingress.owner.input_transform == 'same-snapshot-transform'
    ingress.state.health.samples['imu']['received'] = 999.
    assert ingress.owner.health.samples['imu']['received'] == 100.


def test_overflow_stays_invalid_after_drain_and_does_not_apply_partial_guard_history():
    ingress, events, _, guards, failures = fixture()
    for value in range(3): ingress.guard.append((value, float(value)))
    ingress.pull(); ingress.pull()
    assert not guards and len(failures) == 2
    assert ingress.owner.health.samples == {} and ingress.owner.guard_status is None
    assert len(events) == 1 and 'overflow' in failures[0]


def test_clock_rollback_between_pulls_reaches_original_validator():
    ingress, _, clocks, _, _ = fixture()
    for value in (10, 9, 11): ingress.receive_clock(clock(value), 100.)
    ingress.pull()
    assert [m.clock.sec for m in clocks] == [10, 9, 11]


@pytest.mark.parametrize('mode', ['exception', 'stopped'])
def test_worker_loss_invalidates_health_and_guard_even_with_cached_good_data(mode):
    ingress, _, _, _, failures = fixture()
    if mode == 'exception': ingress.worker.error = 'bad input'
    else: ingress.worker.thread = NS(is_alive=lambda: False)
    ingress.pull()
    assert failures and ingress.owner.guard_received_at is None
    assert not ingress.owner.health.samples


def test_input_thread_cannot_consume_or_apply_observations():
    ingress, _, _, guards, failures = fixture()
    exceptions = []
    def attempt():
        try: ingress.pull()
        except RuntimeError as exc: exceptions.append(str(exc))
    thread = threading.Thread(target=attempt); thread.start(); thread.join(timeout=1)
    assert not thread.is_alive() and exceptions == ['only the decision thread may consume inputs']
    assert not guards and not failures


def test_equipment_history_preserves_fault_then_recovery_in_arrival_order():
    ingress = EquipmentIngress.__new__(EquipmentIngress)
    worker = InputWorker.__new__(InputWorker); worker.owner_thread = threading.get_ident()
    worker.lock = threading.RLock(); worker.error = None; worker.thread = NS(is_alive=lambda: True)
    ingress.worker = worker; ingress.queue = ReceiptQueue(3)
    values = [('actuator', 'a', 'topic', {'fault': v}, 100.+v) for v in (0, 1, 0)]
    for value in values: ingress.queue.append(deepcopy(value))
    assert ingress.pull() == values
    assert not ingress.pull()


def test_equipment_overflow_rejects_whole_snapshot():
    ingress = EquipmentIngress.__new__(EquipmentIngress)
    worker = InputWorker.__new__(InputWorker); worker.owner_thread = threading.get_ident()
    worker.lock = threading.RLock(); worker.error = None; worker.thread = NS(is_alive=lambda: True)
    ingress.worker = worker; ingress.queue = ReceiptQueue(1)
    ingress.queue.append('normal'); ingress.queue.append('fault')
    with pytest.raises(RuntimeError, match='overflow'): ingress.pull()


def test_tf_unavailable_remains_missing_and_never_reuses_last_transform():
    ingress, _, _, _, failures = fixture()
    ingress.owner.input_transform = 'previous'
    def missing(*a): raise LookupError('missing map')
    ingress.tf = NS(lookup_transform=missing); ingress.pull()
    assert ingress.owner.input_transform is None and ingress.owner.input_transform_error == 'missing map'
    assert not failures


def test_station_scan_snapshot_crosses_ingress_with_original_receipt_only():
    ingress, _, _, _, _ = fixture()
    sample = dict(frame_id='laser_link',source_s=10.,received_wall_s=99.9,ranges=(1.,2.))
    ingress.state.latest_scan=sample
    ingress.owner.station_refiner=object()
    ingress.static_frames=set()
    ingress.pull()
    assert ingress.owner.latest_scan is sample
    assert ingress.owner.latest_scan['received_wall_s']==99.9
    # A dynamic or missing extrinsic is never silently replaced by model constants.
    assert ingress.owner.laser_extrinsic is None


def test_brief_native_fault_is_not_hidden_by_later_healthy_equipment(monkeypatch):
    import sys
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py', NS())
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py.convert', NS(message_to_ordereddict=lambda m: m))
    adapter = ContactBinRuntime.__new__(ContactBinRuntime)
    row = dict(header={'stamp': {'sec': 1, 'nanosec': 0}}, name=['a_servo_fault'], position=[1.])
    values = [('actuator', 'a', 'topic', row, 100.),
              ('actuator', 'a', 'topic', dict(row, position=[0.]), 100.1)]
    adapter.ingress = NS(pull=lambda: values); adapter.actuators = {}; adapter.contacts = {}
    adapter.samples = NS(add=lambda *a, **kw: None); records = []
    adapter.record = lambda *a, **kw: records.append((a, kw))
    with pytest.raises(RuntimeError, match='lease fault: a'): adapter.consume_inputs()
    assert adapter.actuators['a']['position'] == [1.] and len(records) == 1


@pytest.mark.parametrize('process', [False, True])
def test_raw_writer_keeps_concurrent_order_source_receipts_and_immutable_values(tmp_path, process):
    from array import array
    import json
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer = EventWriter(tmp_path/'raw.jsonl', time.monotonic(), process=process)
    def send(source):
        for seq in range(200):
            data = {'source_s': seq*.01, 'received': 100.+seq*.01, 'values': array('d',[1.,2.])}
            writer.record('raw',source=source,seq=seq,data=data)
            data['received'] = -1; data['values'][0] = -1
    threads = [threading.Thread(target=send,args=(source,)) for source in ('a','b','c')]
    for thread in threads:thread.start()
    for thread in threads:thread.join(timeout=3.);assert not thread.is_alive()
    result = writer.close();rows = [json.loads(s) for s in writer.path.read_text().splitlines()]
    assert result['records'] == len(rows) == 600 and result['queue_empty']
    assert [r['wall_s'] for r in rows] == sorted(r['wall_s'] for r in rows)
    for source in ('a','b','c'):
        selected = [r for r in rows if r['source']==source]
        assert [r['seq'] for r in selected] == list(range(200))
        assert all(r['data']['received']==100.+r['seq']*.01 and r['data']['values']==[1.,2.] for r in selected)
    with pytest.raises(RuntimeError,match='closed'):writer.record('late')


def test_raw_writer_serialization_failure_invalidates_evidence(tmp_path):
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer = EventWriter(tmp_path/'raw.jsonl',time.monotonic())
    writer.record('malformed',source_s=float('nan'))
    with pytest.raises(RuntimeError,match='incomplete'):writer.close()
    assert not writer.thread.is_alive()


def test_raw_writer_final_flush_failure_cannot_report_complete_evidence(tmp_path, monkeypatch):
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer = EventWriter(tmp_path/'final_flush_failure.jsonl', time.monotonic())
    original_close = writer.stream.close
    def failed_close():
        original_close()
        raise OSError('injected final flush failure')
    monkeypatch.setattr(writer.stream, 'close', failed_close)
    writer.record('last_snapshot', sequence=1)
    with pytest.raises(RuntimeError, match='incomplete.*final flush failure'):
        writer.close()
    assert not writer.thread.is_alive() and writer.count == 1


def test_raw_writer_overflow_is_sticky_and_does_not_silently_drop():
    import queue
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer = EventWriter.__new__(EventWriter)
    writer.lock = threading.Lock();writer.closed = False;writer.error = None
    writer.queue = queue.Queue(maxsize=1);writer.started = time.monotonic()
    writer.record('first')
    with pytest.raises(RuntimeError,match='overflow'):writer.record('second')
    assert writer.queue.get()['kind']=='first'
    with pytest.raises(RuntimeError,match='failed'):writer.record('third')


def test_equipment_owner_turn_keeps_bounded_fifo_and_remaining_fault():
    ingress = EquipmentIngress.__new__(EquipmentIngress)
    worker = InputWorker.__new__(InputWorker); worker.owner_thread = threading.get_ident()
    worker.lock = threading.RLock(); worker.error = None; worker.thread = NS(is_alive=lambda: True)
    ingress.worker = worker; ingress.queue = ReceiptQueue(2048)
    for index in range(70): ingress.queue.append(('fault' if index == 40 else 'normal', index))
    first = ingress.pull(); second = ingress.pull(); third = ingress.pull()
    assert [len(first), len(second), len(third)] == [32, 32, 6]
    assert [index for _, index in first+second+third] == list(range(70))
    assert second[8] == ('fault', 40) and not ingress.pull()


def test_bounded_drain_does_not_clear_overflow_or_increase_storage():
    queue = ReceiptQueue(40)
    for index in range(41): queue.append(index)
    assert queue.drain(limit=32) == list(range(32))
    assert len(queue.pending) == 8 and queue.limit == 40 and queue.error == 'input queue overflow'
    assert queue.drain(limit=32) == list(range(32, 40))
    assert queue.error == 'input queue overflow'


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_bad_drain_budget_preserves_all_pending_inputs(limit):
    queue = ReceiptQueue(3); queue.append('fault')
    with pytest.raises(ValueError, match='positive drain'): queue.drain(limit=limit)
    assert list(queue.pending) == ['fault']


def test_direct_ros_snapshot_detaches_nested_message_arrays():
    from array import array
    from tools.wheelset_scene.navigation_records import ros_message_value
    class Message:
        @classmethod
        def get_fields_and_field_types(cls): return {'stamp': 'test', 'values': 'test'}
    message = Message(); message.stamp = {'sec': 12, 'nanosec': 3}
    child = Message(); child.stamp = None; child.values = array('d', [1., 2.])
    message.values = [child]
    row = ros_message_value(message)
    message.stamp['sec'] = -1; child.values[0] = -1
    assert row == {'stamp': {'sec': 12, 'nanosec': 3},
                   'values': [{'stamp': None, 'values': [1., 2.]}]}


def test_large_snapshot_does_not_hold_other_receivers_enqueue_lock(tmp_path, monkeypatch):
    import json,time
    from tools.wheelset_scene import navigation_records as module
    entered=threading.Event();release=threading.Event();fast_done=threading.Event()
    original=module.plain_ros_value
    def snapshot(value):
        if isinstance(value,dict) and value.get('slow'):
            entered.set();assert release.wait(timeout=2.)
        return original(value)
    monkeypatch.setattr(module,'plain_ros_value',snapshot)
    writer=module.EventWriter(tmp_path/'raw.jsonl',time.monotonic())
    slow=threading.Thread(target=lambda:writer.record('contact',slow=True,received=10.))
    def fast():writer.record('guard',received=11.);fast_done.set()
    slow.start();assert entered.wait(timeout=1.)
    sensor=threading.Thread(target=fast);sensor.start()
    try:assert fast_done.wait(timeout=1.)
    finally:release.set();slow.join(timeout=2.);sensor.join(timeout=2.)
    assert not slow.is_alive() and not sensor.is_alive();writer.close()
    rows=[json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [r['kind'] for r in rows]==['guard','contact']
    assert [r['received'] for r in rows]==[11.,10.]
    assert rows[0]['wall_s']<=rows[1]['wall_s']


def test_close_rejects_snapshot_still_in_progress_without_late_enqueue(tmp_path,monkeypatch):
    import time
    from tools.wheelset_scene import navigation_records as module
    entered=threading.Event();release=threading.Event();errors=[]
    original=module.plain_ros_value
    def snapshot(value):
        if isinstance(value,dict) and value.get('slow'):
            entered.set();assert release.wait(timeout=2.)
        return original(value)
    monkeypatch.setattr(module,'plain_ros_value',snapshot)
    writer=module.EventWriter(tmp_path/'raw.jsonl',time.monotonic())
    def send():
        try:writer.record('late',slow=True)
        except RuntimeError as exc:errors.append(str(exc))
    worker=threading.Thread(target=send);worker.start();assert entered.wait(timeout=1.)
    try:result=writer.close()
    finally:release.set();worker.join(timeout=2.)
    assert result['records']==0 and result['queue_empty'] and not worker.is_alive()
    assert len(errors)==1 and 'closed' in errors[0] and not writer.path.read_text()


def test_process_recorder_writes_every_accepted_snapshot_in_order(tmp_path):
    import json,time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer=EventWriter(tmp_path/'raw.jsonl',time.monotonic(),process=True)
    for index in range(300):
        values={'values':[index,2.], 'received_wall':100.+index}
        writer.record('equipment',seq=index,message=values);values['values'][0]=-1
    summary=writer.close()
    rows=[json.loads(line) for line in writer.path.read_text().splitlines()]
    assert summary['records']==summary['accepted_records']==300 and summary['writer_process']
    assert not writer.thread.is_alive() and [r['seq'] for r in rows]==list(range(300))
    assert all(r['message']['values']==[r['seq'],2.] for r in rows)


def test_process_recorder_serialization_failure_is_not_delivery_success(tmp_path):
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer=EventWriter(tmp_path/'bad.jsonl',time.monotonic(),process=True)
    writer.record('invalid',source_s=float('nan'))
    with pytest.raises(RuntimeError,match='incomplete'):writer.close()
    assert not writer.thread.is_alive()


def test_process_recorder_death_invalidates_cached_evidence(tmp_path):
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer=EventWriter(tmp_path/'dead.jsonl',time.monotonic(),process=True)
    writer.thread.terminate();writer.thread.join(timeout=2.)
    with pytest.raises(RuntimeError,match='failed'):writer.record('after_death')
    with pytest.raises(RuntimeError,match='incomplete'):writer.close()
    assert not writer.thread.is_alive()


def test_process_health_probe_does_not_hold_other_receivers_enqueue_lock():
    import queue,time
    from tools.wheelset_scene.navigation_records import EventWriter
    entered=threading.Event();release=threading.Event();fast_done=threading.Event();errors=[]
    def alive():
        if threading.current_thread().name=='slow_process_probe':
            entered.set();assert release.wait(timeout=2.)
        return True
    writer=EventWriter.__new__(EventWriter)
    writer.lock=threading.Lock();writer.closed=False;writer.error=None
    writer.queue=queue.Queue(maxsize=3);writer.started=time.monotonic()
    writer.process=True;writer.thread=NS(is_alive=alive);writer.accepted=0
    def send(kind):
        try:
            writer.record(kind)
            if kind=='guard':fast_done.set()
        except Exception as exc:errors.append(exc)
    slow=threading.Thread(target=send,args=('equipment',),name='slow_process_probe')
    fast=threading.Thread(target=send,args=('guard',))
    slow.start();assert entered.wait(timeout=1.);fast.start()
    try:assert fast_done.wait(timeout=1.)
    finally:release.set();slow.join(timeout=2.);fast.join(timeout=2.)
    assert not slow.is_alive() and not fast.is_alive() and not errors
    assert writer.accepted==2
    assert [writer.queue.get_nowait()['kind'] for _ in range(2)]==['guard','equipment']


def equipment_process_fixture(profile, pending, commands, stopped, failed, ready,
                              accepted, counts, topics, answer):
    """Spawned transport peer only; no ROS or production decisions."""
    import queue,time
    mode=profile.get('test_mode','normal');error=None
    def emit(kind,key,topic,message,request_id=None):
        sequence=accepted.value+1
        if mode=='sequence':sequence+=1
        try:pending.put_nowait((sequence,kind,key,topic,message,time.monotonic(),request_id))
        except queue.Full:failed.value=1;return
        accepted.value+=1
    ready.value=1
    if mode=='dead':return
    if mode=='hung':time.sleep(20.);return
    try:
        for index,fault in enumerate((0.,1.,0.)):
            topic='/scout_bin/a/joint_states';counts[topics.index(topic)]+=1
            emit('actuator','a',topic,dict(header={'stamp':{'sec':index+1,'nanosec':0}},
                name=['a_servo_fault'],position=[fault]))
        if mode=='overflow':failed.value=1
        end=False
        while not end:
            try:command=commands.get(timeout=.02)
            except queue.Empty:continue
            if command is None:end=True;continue
            identity,entity,fields=command
            if mode=='forged':identity='other-epoch:1'
            emit('state',entity,fields,{'requested_name':fields['name'],'success':True},identity)
    finally:
        pending.close()
        if failed.value:pending.cancel_join_thread()
        else:pending.join_thread()
        answer.send(dict(passed=not failed.value,error=error,accepted=accepted.value,
            callbacks=dict(zip(topics,counts[:])),unresolved_queries=0))
        answer.close()


def make_equipment_process(mode='normal'):
    import time
    from tools.wheelset_scene.navigation_equipment_ingress import EquipmentProcessIngress
    owner=NS(profile={'tasks':{'task':{}},'robots':{'a':{}},'test_mode':mode})
    ingress=EquipmentProcessIngress(owner,_target=equipment_process_fixture)
    until=time.monotonic()+5.
    while not ingress.ready.value and ingress.thread.is_alive() and time.monotonic()<until:time.sleep(.005)
    assert ingress.ready.value
    return ingress


def drain_equipment(ingress, rows, expected=None):
    import time
    until=time.monotonic()+3.
    while time.monotonic()<until:
        rows.extend(ingress.pull())
        if expected is None or len(rows)>=expected:return
        time.sleep(.002)
    raise AssertionError('input peer did not deliver expected rows')


def test_equipment_process_fifo_query_identity_immutable_request_and_complete_close():
    ingress=make_equipment_process();rows=[]
    try:
        drain_equipment(ingress,rows,3)
        assert [row[3]['position'][0] for row in rows]==[0.,1.,0.]
        request={'name':'cargo','reference_frame':'world'}
        future=ingress.query('cargo',request);request['name']='mutated'
        drain_equipment(ingress,rows,4)
        assert future.done() and future.result()=={'requested_name':'cargo','success':True}
        assert rows[3][0:3]==('state','cargo',{'name':'cargo','reference_frame':'world'})
        summary=ingress.close(consume=lambda:drain_equipment(ingress,rows))
        assert summary['passed'] and summary['accepted_samples']==summary['consumed_samples']==4
        assert summary['callbacks']['/scout_bin/a/joint_states']==3
        assert not ingress.thread.is_alive() and not ingress.transfer.is_alive() and ingress.close() is summary
    finally:
        if not ingress.closed:
            ingress.close(consume=lambda:drain_equipment(ingress,rows))


def test_equipment_process_close_consumes_inflight_query_before_final_count():
    ingress=make_equipment_process();rows=[]
    future=ingress.query('cargo',{'name':'cargo','reference_frame':'world'})
    summary=ingress.close(consume=lambda:rows.extend(ingress.pull()))
    assert summary['passed'] and len(rows)==4 and future.done() and not future.cancelled()
    assert rows[-1][0]=='state' and summary['unresolved_queries']==0


@pytest.mark.parametrize('operation',['pull','query','close'])
def test_equipment_process_owner_only_mutations(operation):
    ingress=make_equipment_process();errors=[];rows=[]
    def attempt():
        try:
            if operation=='query':ingress.query('cargo',{'name':'cargo'})
            else:getattr(ingress,operation)()
        except Exception as exc:errors.append(str(exc))
    worker=threading.Thread(target=attempt);worker.start();worker.join(timeout=1.)
    try:
        assert not worker.is_alive() and errors==['only the decision thread may consume inputs']
        assert not ingress.stopped.is_set() and not ingress.queries
    finally:ingress.close(consume=lambda:rows.extend(ingress.pull()))


@pytest.mark.parametrize('mode,match',[('sequence','sequence'),('overflow','overflow'),('dead','stopped')])
def test_equipment_process_faults_are_sticky_and_close_never_passes(mode,match):
    import time
    ingress=make_equipment_process(mode)
    if mode=='dead':ingress.thread.join(timeout=2.)
    else:
        until=time.monotonic()+2.
        while not ingress.accepted.value and time.monotonic()<until:time.sleep(.002)
    with pytest.raises(RuntimeError,match=match):
        until=time.monotonic()+2.
        while True:
            ingress.pull()
            assert time.monotonic()<until
            time.sleep(.002)
    with pytest.raises(RuntimeError):ingress.pull()
    with pytest.raises(RuntimeError,match='incomplete'):ingress.close(consume=lambda:ingress.pull())
    assert ingress.closed and not ingress.thread.is_alive() and not ingress.close_result['passed']
    with pytest.raises(RuntimeError,match='already failed'):ingress.close()


def test_equipment_process_rejects_foreign_query_reply_and_cancels_future():
    ingress=make_equipment_process('forged');rows=[]
    future=ingress.query('cargo',{'name':'cargo','reference_frame':'world'})
    try:
        with pytest.raises(RuntimeError,match='reply identity'):
            drain_equipment(ingress,rows,4)
    finally:
        with pytest.raises(RuntimeError,match='incomplete'):ingress.close(consume=lambda:ingress.pull())
    assert future.cancelled() and not ingress.thread.is_alive()


def test_equipment_process_query_timeout_cannot_refresh_or_complete_cached_state():
    ingress=make_equipment_process();rows=[]
    future=ingress.query('cargo',{'name':'cargo','reference_frame':'world'})
    identity=next(iter(ingress.queries));entity,fields,issued,stored=ingress.queries[identity]
    ingress.queries[identity]=(entity,fields,issued-3.,stored)
    with pytest.raises(RuntimeError,match='timed out'):ingress.pull()
    with pytest.raises(RuntimeError,match='incomplete'):
        ingress.close(consume=lambda:rows.extend(ingress.pull()))
    # Teardown records the real late reply, without clearing its runtime
    # timeout or giving the caller another opportunity to command motion.
    assert future.done() and not future.cancelled() and not ingress.thread.is_alive()
    assert rows[-1][0]=='state' and future.result()['success']
    result=ingress.close_result
    assert not result['passed'] and result['input_complete']
    assert result['original_input_error']=='contact equipment query timed out'
    assert result['accepted_samples']==result['consumed_samples']==4
    assert ingress.error=='contact equipment query timed out'


def test_equipment_close_preserves_prior_error_and_rejects_new_queries_while_draining():
    ingress=make_equipment_process();rows=[]
    future=ingress.query('cargo',{'name':'cargo','reference_frame':'world'})
    ingress.error='original runtime input fault'
    attempts=[]
    def consume():
        with pytest.raises(RuntimeError,match='closing'):
            ingress.query('new_cargo',{'name':'new_cargo'})
        attempts.append(True)
        rows.extend(ingress.pull(limit=1))
    with pytest.raises(RuntimeError,match='incomplete'):
        ingress.close(consume=consume)
    result=ingress.close_result
    assert attempts and result['input_complete'] and not result['passed']
    assert result['accepted_samples']==result['consumed_samples']==4
    assert result['original_input_error']==ingress.error=='original runtime input fault'
    assert future.done() and not future.cancelled()
    assert not ingress._shutdown_drain_active


def test_equipment_process_duplicate_and_mismatched_query_do_not_send_extra_commands():
    ingress=make_equipment_process();rows=[]
    try:
        with pytest.raises(ValueError,match='identity'):ingress.query('cargo',{'name':'other'})
        future=ingress.query('cargo',{'name':'cargo'})
        with pytest.raises(RuntimeError,match='duplicate'):ingress.query('cargo',{'name':'cargo'})
        drain_equipment(ingress,rows,4)
        assert future.result()['requested_name']=='cargo' and ingress.query_sequence==1
    finally:ingress.close(consume=lambda:rows.extend(ingress.pull()))


def test_equipment_process_hung_close_is_bounded_and_unaccepted():
    import time
    ingress=make_equipment_process('hung');began=time.monotonic()
    with pytest.raises(RuntimeError,match='incomplete'):ingress.close(consume=lambda:ingress.pull())
    assert time.monotonic()-began<6. and not ingress.thread.is_alive()
    assert not ingress.close_result['passed']


def test_equipment_process_unconsumed_fault_is_applied_by_owner_during_close():
    from tools.wheelset_scene.navigation_bin_runtime import ContactBinRuntime
    ingress=make_equipment_process()
    adapter=ContactBinRuntime.__new__(ContactBinRuntime)
    adapter.ingress=ingress;adapter.actuators={};adapter.contacts={};events=[]
    adapter.samples=NS(add=lambda *a,**k:None);adapter.record=lambda *a,**k:events.append((a,k))
    with pytest.raises(RuntimeError,match='incomplete'):
        ingress.close(consume=adapter.consume_inputs)
    # The intermediate fault remains a failed result even after a newer clear
    # sample arrives; accepted tail observations are still completely drained.
    assert adapter.actuators['a']['position']==[0.]
    assert len(events)==3 and not ingress.thread.is_alive() and not ingress.close_result['passed']
    assert 'lease fault: a' in ingress.close_result['error']
    assert ingress.close_result['accepted_samples']==ingress.close_result['consumed_samples']==3
    assert ingress.close_result['receiver_result']['passed']
    assert ingress.close_result['pending_samples']==0


def test_equipment_process_close_keeps_fault_and_finishes_later_inflight_query():
    ingress=make_equipment_process();rows=[]
    future=ingress.query('cargo',{'name':'cargo','reference_frame':'world'})
    def consume():
        for row in ingress.pull(limit=1):
            rows.append(row)
            if row[0]=='actuator' and row[3]['position']==[1.]:
                raise RuntimeError('native fault cannot be forgotten')
    with pytest.raises(RuntimeError,match='incomplete'):
        ingress.close(consume=consume)
    result=ingress.close_result
    assert len(rows)==4 and rows[-1][0]=='state' and future.done() and not future.cancelled()
    assert result['accepted_samples']==result['consumed_samples']==4
    assert result['pending_samples']==result['unresolved_queries']==0
    assert result['receiver_result']['passed'] and not result['passed']
    assert result['error']=='RuntimeError: native fault cannot be forgotten'


def test_equipment_owner_drain_rechecks_process_death_before_any_control_turn():
    import time
    ingress = make_equipment_process()
    try:
        with pytest.raises(RuntimeError, match='process stopped'):
            with ingress.owner_turn():
                ingress.thread.terminate(); ingress.thread.join(timeout=2.)
                # A row already transported can be consumed, but the context
                # must fail before returning to the caller's control path.
                ingress.pull(limit=1)
        assert not ingress._owner_turn_active
        with pytest.raises(RuntimeError, match='incomplete'):
            ingress.close(consume=lambda: ingress.pull(limit=1))
    finally:
        if not ingress.closed:
            with pytest.raises(RuntimeError):
                ingress.close(consume=lambda: ingress.pull(limit=1))
    assert not ingress.thread.is_alive() and not ingress.transfer.is_alive()


def test_process_event_snapshot_normalizes_arrays_off_owner_without_mutation(tmp_path):
    from array import array
    import json
    import time
    from tools.wheelset_scene.navigation_records import EventWriter
    writer = EventWriter(tmp_path/'native_arrays.jsonl', time.monotonic(), process=True)
    original = {'array': array('d', [1., 2.]), 'nested': [{'values': [3., 4.]}]}
    writer.record('sample', source_s=7., received_wall=9., message=original)
    original['array'][0] = -1.
    original['nested'][0]['values'][0] = -1.
    result = writer.close()
    row = json.loads(writer.path.read_text())
    assert result['records'] == result['accepted_records'] == 1
    assert row['message'] == {'array': [1., 2.], 'nested': [{'values': [3., 4.]}]}
    assert row['source_s'] == 7. and row['received_wall'] == 9.
    assert '_local_snapshot' not in row


def test_equipment_process_nonprogressing_close_consumer_failure_is_bounded():
    import time
    ingress=make_equipment_process();started=time.monotonic()
    def consume():raise RuntimeError('storage unavailable before consume')
    with pytest.raises(RuntimeError,match='incomplete'):
        ingress.close(consume=consume)
    assert time.monotonic()-started<3. and not ingress.thread.is_alive()
    assert not ingress.close_result['passed'] and 'storage unavailable' in ingress.close_result['error']


def test_equipment_owner_never_waits_for_a_partially_received_pipe_message():
    import queue
    import time
    from tools.wheelset_scene.navigation_equipment_ingress import EquipmentProcessIngress
    entered = threading.Event(); release = threading.Event(); released_slots = []
    class SlowPipe:
        delivered = False
        def get(self, **options):
            if self.delivered:
                time.sleep(.005)
                raise queue.Empty
            entered.set()
            assert release.wait(2.)
            self.delivered = True
            return (1, 'actuator', 'a', '/scout_bin/a/joint_states', {'position': [1.]}, 10., None)
        def release(self): released_slots.append(True)
    ingress = EquipmentProcessIngress.__new__(EquipmentProcessIngress)
    ingress.pending = SlowPipe(); ingress.inbox = queue.Queue(maxsize=32)
    ingress.transfer_stop = threading.Event(); ingress.transfer_error = None
    ingress.sequence = 0; ingress.profile = {'robots': ['a'], 'tasks': {}}
    ingress.check = lambda: None
    transport = threading.Thread(target=ingress._transfer_inputs)
    transport.start()
    try:
        assert entered.wait(1.)
        started = time.monotonic()
        assert ingress.pull() == []
        assert time.monotonic()-started < .1
        assert ingress.sequence == 0 and not released_slots
        release.set()
        rows = []
        until = time.monotonic()+1.
        while not rows and time.monotonic() < until:
            rows.extend(ingress.pull(limit=1)); time.sleep(.001)
        assert rows == [('actuator', 'a', '/scout_bin/a/joint_states', {'position': [1.]}, 10.)]
        assert ingress.sequence == 1 and released_slots == [True]
    finally:
        ingress.transfer_stop.set(); release.set(); transport.join(timeout=2.)
    assert not transport.is_alive()


def test_equipment_capacity_counts_rows_in_pipe_and_transport_inbox_together():
    import multiprocessing
    import queue
    from tools.wheelset_scene.navigation_equipment_ingress import _ReservedEquipmentQueue
    pending = _ReservedEquipmentQueue(multiprocessing.get_context('spawn'), 2)
    try:
        pending.put_nowait('first'); pending.put_nowait('second')
        # Reading the full pipe message into another buffer retains its slot.
        assert pending.get(timeout=1.) == 'first'
        with pytest.raises(queue.Full): pending.put_nowait('overflow')
        pending.release()
        pending.put_nowait('third')
        assert pending.get(timeout=1.) == 'second'
        assert pending.get(timeout=1.) == 'third'
        pending.release(); pending.release()
    finally:
        pending.close(); pending.join_thread()


def test_equipment_poll_uses_owner_consumed_clock_and_post_ingress_wall(monkeypatch):
    import sys
    from tools.wheelset_scene import navigation_bin_runtime as module
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    monkeypatch.setitem(sys.modules, 'rosidl_runtime_py.convert', NS(message_to_ordereddict=lambda m: m))
    adapter = module.ContactBinRuntime.__new__(module.ContactBinRuntime)
    adapter.closed = False; adapter.ingress = object(); adapter.samples = EquipmentSamples()
    node = NS(clock_s=1.)
    order = []
    def equipment():
        order.append('equipment')
        adapter.samples.add('fresh', 'valid', source_s=1.2, received_wall=10.1)
        adapter.samples.add('old', 'expired', source_s=1.2, received_wall=9.7)
        adapter.samples.add('future_source', 'invalid', source_s=1.3, received_wall=10.1)
    def sensor():
        order.append('sensor'); node.clock_s = 1.2
    def wall():
        assert order == ['equipment', 'sensor']
        return 10.2
    node.query_feedback = sensor
    adapter.runtime = NS(nodes={'a': node}, sim=NS(handlings={'a': object()}))
    adapter.consume_inputs = equipment; adapter.pending = {}; adapter.last_query = 10.2
    adapter.last_publish = 9.; adapter.getter = NS(service_is_ready=lambda: False)
    selected = []
    def poll_robot(robot, handling, now, source_now):
        selected.append((now, source_now, [adapter.samples.select(key, source_now=source_now, wall_now=now)
            for key in ('fresh', 'old', 'future_source')]))
    adapter.poll_robot = poll_robot
    monkeypatch.setattr(module.time, 'monotonic', wall)
    adapter.poll(9.9)
    assert selected == [(10.2, 1.2, ['valid', None, None])]
    assert adapter.source_now == 1.2
