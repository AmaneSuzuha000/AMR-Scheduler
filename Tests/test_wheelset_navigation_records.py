import gc
import json
import tracemalloc

from tools.wheelset_scene.navigation_records import Records


def test_reiterable_evidence_does_not_retain_scanning_arrays(tmp_path):
    path = tmp_path/'observations.jsonl'
    with path.open('w') as stream:
        for n in range(240):
            stream.write(json.dumps(dict(kind='scan', seq=n, ranges=[n+.01]*1080))+'\n')
        stream.write(json.dumps(dict(kind='fleet_enabled', wall_s=5.))+'\n')
    records = Records(path)
    gc.collect(); tracemalloc.start()
    try:
        count = 0
        for row in records:
            if row['kind'] == 'scan':
                assert len(row['ranges']) == 1080
                count += 1
        _, peak = tracemalloc.get_traced_memory()
    finally: tracemalloc.stop()
    assert count == 240 and peak < 1_000_000
    assert list(records.only('fleet_enabled')) == [dict(kind='fleet_enabled', wall_s=5.)]
    assert sum(1 for row in records if row['kind'] == 'scan') == count


def test_record_filter_still_decodes_malformed_unselected_records(tmp_path):
    import pytest
    path = tmp_path/'corrupt.jsonl'
    path.write_text('{"kind":"scan","ranges":broken}\n')
    with pytest.raises(json.JSONDecodeError): list(Records(path).only('fleet_enabled'))


def _delayed_text_peer(path, pending, answer):
    import time
    from tools.wheelset_scene.navigation_records import _write_text_rows
    time.sleep(.3)
    _write_text_rows(path, pending, answer)


def _failed_text_peer(path, pending, answer):
    answer.send(dict(passed=False, records=0, error='injected disk writer failure'))
    answer.close()


def test_process_text_log_preserves_exact_owner_bytes_order_and_complete_close(tmp_path):
    from tools.wheelset_scene.navigation_records import BufferedTextLog,ProcessTextLog
    path=tmp_path/'fleet.jsonl'
    writer=ProcessTextLog(path)
    rows=[json.dumps(dict(kind='feedback',seq=i,物料='轻料箱'),ensure_ascii=False)+'\n'
          for i in range(100)]
    for text in rows:
        assert writer.write(text)==len(text)
    result=writer.close()
    assert result['passed'] and result['writer_exited']
    assert result['accepted_records']==result['records']==len(rows)
    assert writer.close()==result
    assert json.loads(path.with_suffix('.jsonl.writer.json').read_text())==result
    reference=tmp_path/'synchronous.jsonl'
    with BufferedTextLog(reference) as stream:
        for text in rows:
            stream.write(text)
    assert path.read_bytes()==reference.read_bytes()


def test_process_text_log_owner_does_not_wait_for_slow_storage_peer(tmp_path):
    import time
    from tools.wheelset_scene.navigation_records import ProcessTextLog
    writer=ProcessTextLog(tmp_path/'delayed.jsonl',_target=_delayed_text_peer)
    started=time.monotonic()
    for i in range(40):
        writer.write(str(i)+'\n')
    assert time.monotonic()-started<.2
    assert writer.close()['records']==40


def test_process_text_log_overflow_is_sticky_and_never_a_successful_close(tmp_path):
    import pytest
    from tools.wheelset_scene.navigation_records import ProcessTextLog
    path=tmp_path/'overflow.jsonl'
    writer=ProcessTextLog(path,limit=1,_target=_delayed_text_peer)
    writer.write('first\n')
    with pytest.raises(RuntimeError,match='queue overflow'):
        writer.write('must not silently replace first\n')
    with pytest.raises(RuntimeError,match='text log incomplete'):
        writer.close()
    assert writer.close_result['writer_exited'] and not writer.close_result['passed']
    assert writer.close_result['accepted_records']==1
    assert path.read_text()=='first\n'


def test_process_text_log_failed_peer_never_grants_complete_evidence(tmp_path):
    import pytest
    from tools.wheelset_scene.navigation_records import ProcessTextLog
    writer=ProcessTextLog(tmp_path/'failed.jsonl',_target=_failed_text_peer)
    try:
        writer.write('uncommitted evidence\n')
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError,match='text log incomplete'):
        writer.close()
    assert not writer.close_result['passed'] and writer.close_result['writer_exited']


def test_process_text_log_receiver_cannot_write_owner_events(tmp_path):
    import threading
    from tools.wheelset_scene.navigation_records import ProcessTextLog
    writer=ProcessTextLog(tmp_path/'owner.jsonl')
    errors=[]
    def receive():
        try:
            writer.write('receiver forged owner event\n')
        except RuntimeError as exc:
            errors.append(str(exc))
    peer=threading.Thread(target=receive)
    peer.start();peer.join(timeout=1.)
    assert not peer.is_alive() and errors==['only the decision owner may submit text logs']
    assert writer.close()['accepted_records']==0


def test_process_text_log_rejects_existing_evidence_without_truncation(tmp_path):
    import pytest
    from tools.wheelset_scene.navigation_records import ProcessTextLog
    path=tmp_path/'old.jsonl'
    path.write_text('original\n')
    with pytest.raises(FileExistsError):
        ProcessTextLog(path)
    assert path.read_text()=='original\n'


def _raw_contact():
    return dict(header=dict(stamp=dict(sec=12,nanosec=345),frame_id='world'),
                states=[dict(collision1_name='bin::body::bin',
                    collision2_name='scout_a::handling_deck::support',
                    contact_normals=[dict(x=0.,y=0.,z=1.)],
                    total_wrench=dict(force=dict(x=0.,y=0.,z=19.62)),
                    info='time: [12.000000345]')])


def test_local_contact_packet_freezes_source_and_history_reuses_existing_bytes(monkeypatch):
    import pickle
    from tools.wheelset_scene.navigation_records import freeze_contact_message
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    source=_raw_contact()
    topic='/scout_bin/inspection_consumables/bin_contacts'
    packet=freeze_contact_message(source,topic,100.)
    source['states'][0]['info']='mutated after reception'
    expected=dict(_raw_contact(),_topic=topic,_received_wall=100.)
    samples=EquipmentSamples(serialize_contacts=True)
    def forbidden(*args,**kwargs):
        raise AssertionError('owner repacked a contact already frozen by input process')
    with monkeypatch.context() as patch:
        patch.setattr(pickle,'dumps',forbidden)
        assert samples.add(topic,packet,source_s=12.000000345,received_wall=100.)
        assert samples.select(topic,source_now=12.01,wall_now=100.01)==expected
    assert samples.rows[topic][-1][2].payload is packet['_local_contact_payload']


def test_frozen_contact_duplicate_cannot_renew_source_or_receipt_freshness():
    from tools.wheelset_scene.navigation_records import freeze_contact_message
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    topic='/scout_bin/inspection_consumables/bin_contacts'
    packet=freeze_contact_message(_raw_contact(),topic,100.)
    samples=EquipmentSamples()
    assert samples.add(topic,packet,source_s=12.000000345,received_wall=100.)
    assert not samples.add(topic,packet,source_s=12.000000345,received_wall=101.)
    assert samples.select(topic,source_now=12.01,wall_now=101.) is None


def test_frozen_contact_causal_metadata_mismatch_is_rejected_by_owner():
    import pytest
    from tools.wheelset_scene.navigation_records import freeze_contact_message
    from tools.wheelset_scene.navigation_samples import EquipmentSamples
    topic='/scout_bin/inspection_consumables/bin_contacts'
    packet=freeze_contact_message(_raw_contact(),topic,100.)
    packet['header']['stamp']['sec']=13
    samples=EquipmentSamples()
    samples.add(topic,packet,source_s=12.000000345,received_wall=100.)
    with pytest.raises(ValueError,match='metadata differs'):
        samples.select(topic,source_now=12.01,wall_now=100.01)


def test_raw_recorder_expands_full_contact_only_in_writer_peer(tmp_path):
    import time
    from tools.wheelset_scene.navigation_records import EventWriter,freeze_contact_message
    topic='/scout_bin/inspection_consumables/bin_contacts'
    writer=EventWriter(tmp_path/'frozen_contacts.jsonl',time.monotonic(),process=True)
    for index in range(20):
        raw=_raw_contact()
        raw['header']['stamp']['nanosec']=index
        packet=freeze_contact_message(raw,topic,100.+index)
        writer.record('bin_equipment_sample',topic=topic,message=packet)
        packet['header']['stamp']['sec']=-1
    summary=writer.close()
    assert summary['records']==summary['accepted_records']==20
    rows=[json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [row['message']['header']['stamp']['nanosec'] for row in rows]==list(range(20))
    assert all(row['message']['header']['stamp']['sec']==12 for row in rows)
    assert all(row['message']['states']==_raw_contact()['states'] for row in rows)
    assert all('_local_contact_payload' not in row['message'] for row in rows)
    assert [row['message']['_received_wall'] for row in rows]==[100.+i for i in range(20)]


def test_raw_recorder_bad_frozen_contact_cannot_close_successfully(tmp_path):
    import pytest,time
    from tools.wheelset_scene.navigation_records import EventWriter,freeze_contact_message
    topic='/scout_bin/inspection_consumables/bin_contacts'
    packet=freeze_contact_message(_raw_contact(),topic,100.)
    packet['_received_wall']=101.
    writer=EventWriter(tmp_path/'bad_frozen_contact.jsonl',time.monotonic(),process=True)
    writer.record('bin_equipment_sample',topic=topic,message=packet)
    with pytest.raises(RuntimeError,match='incomplete'):
        writer.close()
    assert writer.count==0 and not writer.thread.is_alive()
