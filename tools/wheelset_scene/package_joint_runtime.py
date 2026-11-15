"""Create a standalone joint release using the immutable 0.2 baseline list."""
import hashlib
import json
from pathlib import Path
import zipfile
from datetime import datetime,timezone
from .build import ROOT
from .evidence import build_fingerprint
from .verify_joint_workshop import verify


def main():
    target=ROOT/'out/wheelset_workshop_runtime_0.3.zip'
    if target.exists():raise SystemExit('Archive exists; preserve it and choose a new version.')
    runs=ROOT/'simulation/wheelset_workshop/process_runs';names=['joint_03_normal','joint_05_recovery']
    audits=[verify(runs/name) for name in names]
    baseline=ROOT/'out/wheelset_workshop_runtime_0.2.zip'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='9b591b6f0f779ebdb11bcb6a794d2eead7ffd2e3d86862ff898d312f2d73eaee'
    with zipfile.ZipFile(baseline) as archive:old=json.loads(archive.read('runtime_manifest.json'))
    for entry in old['files']:
        if entry['path'].startswith(('amr_sched/','assets/wheelset_workshop/models/','tools/ros2/')):
            assert hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    files={ROOT/entry['path'] for entry in old['files']}
    files.update(p for p in (ROOT/'tools/wheelset_scene').glob('*.py'))
    files.update(ROOT/'simulation/wheelset_workshop'/name for name in ['README.md','PROJECT_GOAL.md','RUNNING.md','JOINT_RUNNING.md','NEXT_IMPLEMENTATION.md','process_runs/README.md'])
    files.add(ROOT/'Tests/test_wheelset_material_supply.py')
    for name in ['joint_01_normal','joint_02_recovery','joint_03_recovery','joint_04_recovery',*names]:
        files.update(p for p in (runs/name).rglob('*') if p.is_file())
    for name in ['run_joint_01.log','run_joint_02.log','run_joint_03.log','run_joint_04.log','run_joint_05.log','verify_runtime_03_archive.py',
        'runtime_0.3_causal_mutations.json','runtime_0.3_prior_archives.json','runtime_0.3_failed_package_01.json','check_joint_causal_mutations.py']:
        path=ROOT/'out'/name
        if path.is_file():files.add(path)
    entries=[]
    manifest=dict(runtime_version='0.3',created_utc=datetime.now(timezone.utc).isoformat(),
        build_fingerprint=build_fingerprint(),validated_runs=old['validated_runs'],validated_joint_runs=names,
        retained_failed_joint_runs={
            'joint_01_normal':'Shuttle velocity audit failed; only one device concurrently active.',
            'joint_02_recovery':'Unexpected receipt stall latched original 2s execution watchdog.',
            'joint_03_recovery':'Completed four wheelsets but failed required two-device concurrency.',
            'joint_04_recovery':'Unexpected 2.958s receipt stall latched original 2s execution watchdog.'},
        joint_recipe_assumptions=dict(initial_stock={'derust':2,'magnetic':0},
            station_capacity=4,nominal_spindle_rad_s=1.6,recovered_magnetic_spindle_rad_s=.8,
            recovery_output_block_until_sim_s=300),
        baseline_archive_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),audits=audits,files=entries,
        claim_boundary='Same-world three-station/four-wheelset actuation, two wheel-driven SCOUT surrogates and an assumed overhead shuttle. Measured handoffs commit finite stock and unblock device starts. No recovered factory mechanism, contact payload, localization/Nav2, remaining main-line process or industrial acceptance.')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        # Hash the very bytes being archived, not an earlier read of a live
        # checkout. Extracted audits still validate recorded source/data hashes.
        for path in sorted(files,key=lambda p:p.relative_to(ROOT).as_posix()):
            data=path.read_bytes();relative=path.relative_to(ROOT).as_posix()
            entries.append(dict(path=relative,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
            archive.writestr(relative,data)
        archive.writestr('runtime_manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for entry in entries:assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    result=dict(passed=True,file=str(target),size_bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),file_count=len(entries)+1,validated_joint_runs=names)
    target.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
