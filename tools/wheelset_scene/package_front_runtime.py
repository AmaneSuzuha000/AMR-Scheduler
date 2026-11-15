"""Freeze front-line normal/recovery evidence alongside immutable runtime0.3."""
from datetime import datetime,timezone
import hashlib
import json
import zipfile

from .build import ROOT
from .evidence import build_fingerprint
from .verify_front_workshop import verify
from .probe_axis_servo import verify as verify_native


def main():
    target=ROOT/'out/wheelset_workshop_runtime_0.4.zip'
    if target.exists():raise SystemExit('Archive exists; preserve it and use a new version.')
    sim=ROOT/'simulation/wheelset_workshop';runs=sim/'process_runs'
    passed=['front_05_normal','front_05_recovery'];audits=[verify(runs/name) for name in passed]
    native_audit=verify_native(runs/'native_servo_02')
    for name in passed:
        assert json.loads((runs/name/'front_replay.json').read_text())['final_operations']==9
        negative=json.loads((ROOT/'out'/('runtime_0.4_causal_'+name+'.json')).read_text())
        assert negative['passed'] and all(c['rejected'] and c['hashes_recomputed'] for c in negative['cases'])
    baseline=ROOT/'out/wheelset_workshop_runtime_0.3.zip'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='89eaa9a07e0c70eab8ef4aec95017cac7a89064f8938839f50c3899de7a64e02'
    with zipfile.ZipFile(baseline) as archive:old=json.loads(archive.read('runtime_manifest.json'))
    for entry in old['files']:
        if entry['path'].startswith(('amr_sched/','assets/wheelset_workshop/models/','tools/ros2/')):
            assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    files={ROOT/entry['path'] for entry in old['files']}
    files.update((ROOT/'tools/wheelset_scene').glob('*.py'))
    files.update(p for p in (ROOT/'tools/ros2/wheelset_axis_servo').rglob('*') if p.is_file())
    files.update(sim/name for name in ['FRONT_RUNNING.md','PROJECT_GOAL.md','RUNNING.md','NEXT_IMPLEMENTATION.md','process_runs/README.md'])
    files.add(ROOT/'Tests/test_wheelset_front_flow.py')
    files.add(ROOT/'Tests/test_wheelset_press_action.py')
    for name in ['front_01_normal','front_02_normal','front_03_normal','front_04_normal',*passed,'native_servo_01','native_servo_02']:
        files.update(p for p in (runs/name).rglob('*') if p.is_file())
    for pattern in ['front_preflight_*','run_front_*.log','probe_native_servo_*.log','build_wheelset_axis_servo*.log',
        'native_axis_build_manifest.json','runtime_0.4_causal_*.json','verify_runtime_04_archive.py',
        'front_runtime_final_tests.log','replay_front_05_*.log','causal_front_05_*.log']:
        for path in (ROOT/'out').glob(pattern):
            if path.is_file():files.add(path)
            elif path.is_dir():files.update(p for p in path.rglob('*') if p.is_file())
    entries=[];expected_fingerprint=build_fingerprint()
    assert all(json.loads((runs/name/'result.json').read_text())['build_fingerprint']==expected_fingerprint for name in passed)
    manifest=dict(runtime_version='0.4',created_utc=datetime.now(timezone.utc).isoformat(),
        build_fingerprint=expected_fingerprint,validated_runs=old['validated_runs'],
        validated_joint_runs=old['validated_joint_runs'],validated_front_runs=passed,
        validated_native_runs=['native_servo_02'],audits=audits,native_audit=native_audit,
        unit_only_next_stage=['Dedicated linear PressAction; ram model/inventory/Gazebo integration pending.'],
        retained_failed_front_runs=dict(front_01_normal='Constraint drift and failed stopping.',
            front_02_normal='Runtime completed but independent cap endpoint and cargo-sweep audits failed.',
            front_03_normal='Observed negativeZ after control gap; deliberately stopped own Gazebo and retained evidence.',
            front_04_normal='Five front operations completed; callback source-time skew gate interrupted run; partial cargo sweeps passed.',
            native_servo_01='First near-zero velocity was a turning point; sustained stop audit failed.'),
        baseline_archive_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),files=entries,
        claim_boundary='One identified wheelset through five front operations and three inspections/outfeed; four separately measured removed parts and retained partial-disassembly recovery. Native per-step bounded effort and a tested local command-lease brake. Includes prior runtime0.3 same-world four-wheelset/two-SCOUT supply slice; new prefix is not yet integrated with that fleet/multiple jobs or downstream preparation/pressing/storage. Assumed no-contact rig, no true tags, contact force, localization/Nav2 or industrial acceptance.')
    archived_fingerprint=hashlib.sha256()
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files,key=lambda p:p.relative_to(ROOT).as_posix()):
            data=path.read_bytes();relative=path.relative_to(ROOT).as_posix()
            entries.append(dict(path=relative,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
            if relative=='scenes/wheelset_workshop_video_v1.json' or relative.startswith(('assets/wheelset_workshop/models/','simulation/wheelset_workshop/worlds/')):
                archived_fingerprint.update(relative.encode('utf-8'));archived_fingerprint.update(b'\0')
                archived_fingerprint.update(hashlib.sha256(data).digest())
            archive.writestr(relative,data)
        assert archived_fingerprint.hexdigest()==expected_fingerprint
        assert build_fingerprint()==expected_fingerprint
        manifest['build_fingerprint']=archived_fingerprint.hexdigest()
        archive.writestr('runtime_manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for entry in entries:assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    result=dict(passed=True,file=str(target),size_bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),file_count=len(entries)+1,
        validated_front_runs=passed,validated_native_runs=['native_servo_02'])
    target.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
