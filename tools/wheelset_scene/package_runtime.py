"""Bundle runtime sources and evidence without overwriting a visual release."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

from .build import ROOT,SCENE
from .evidence import build_fingerprint
from .verify_actuated_run import verify as verify_actuated
from .verify_resource_run import verify as verify_resources
from .verify_multi_actuated import verify as verify_multi
from .verify_light_material_run import verify as verify_fleet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',required=True);args=parser.parse_args()
    if not args.version or any(ch not in '0123456789abcdefghijklmnopqrstuvwxyz._-' for ch in args.version):
        raise SystemExit('Simple lowercase runtime version required.')
    target=ROOT/'out'/f'wheelset_workshop_runtime_{args.version}.zip'
    if target.exists():raise SystemExit('Runtime archive exists; use a new version.')
    sim=ROOT/'simulation/wheelset_workshop';runs=sim/'process_runs'
    required_actuated=['actuated_06_normal','actuated_06_stall','actuated_06_feedback_loss']
    required_resources=['resource_02_normal','resource_02_blocked','resource_02_device_recovery',
        'resource_02_transfer_recovery','resource_gz_02_normal','resource_gz_02_recovery']
    required_multi=['multi_actuated_05_normal','multi_actuated_05_recovery']
    required_fleet=['light_material_07_normal','light_material_07_recovery']
    audits=[verify_actuated(runs/name) for name in required_actuated]
    audits.extend(verify_resources(runs/name) for name in required_resources)
    audits.extend(verify_multi(runs/name) for name in required_multi)
    audits.extend(verify_fleet(runs/name) for name in required_fleet)
    fingerprint=build_fingerprint()
    for name in required_actuated+['resource_gz_02_normal','resource_gz_02_recovery']+required_multi:
        assert json.loads((runs/name/'result.json').read_text())['build_fingerprint']==fingerprint
    for name in required_fleet:
        report=json.loads((runs/name/'result.json').read_text())
        assert report['clearance']['build_fingerprint']==fingerprint
        for filename,digest in report['source_sha256'].items():
            assert hashlib.sha256((ROOT/'tools/wheelset_scene'/filename).read_bytes()).hexdigest()==digest
        for source,digest in report['model_source_sha256'].items():
            filename=Path(source).name
            if filename=='metric_model.py':path=ROOT/'tools/ros2/amr_gazebo_guard/launch'/filename
            else:
                folder='meshes' if filename.endswith('.dae') else 'urdf'
                path=ROOT/'tools/ros2/wheelset_dependencies/scout_gazebo_description'/folder/filename
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    scheduler_hashes=json.loads((ROOT/'out/runtime_0.2_remote_scheduler_hashes.json').read_text(encoding='utf-8-sig'))
    for name,digest in scheduler_hashes.items():
        assert hashlib.sha256((ROOT/'amr_sched'/name).read_bytes()).hexdigest()==digest
    files={SCENE,sim/'PROJECT_GOAL.md',sim/'RUNNING.md',sim/'process_runs/README.md',sim/'runtime/session.json',sim/'build_report.json'}
    for name in ['runtime_0.2_dependencies.log','check_runtime_02_dependencies.sh',
        'runtime_0.2_remote_scheduler_hashes.json','check_fleet_audit_mutations.py','runtime_0.2_audit_mutations.json',
        'verify_runtime_02_archive.py','runtime_0.2_existing_archives.json']:
        if (ROOT/'out'/name).is_file():files.add(ROOT/'out'/name)
    for directory in [ROOT/'tools/wheelset_scene',ROOT/'assets/wheelset_workshop/models',sim/'worlds',
        ROOT/'amr_sched',ROOT/'tools/ros2/wheelset_dependencies']:
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    files.add(ROOT/'tools/ros2/amr_gazebo_guard/launch/metric_model.py')
    for path in (ROOT/'tools/__init__.py',ROOT/'Tests/__init__.py'):
        if path.exists():files.add(path)
    files.update(ROOT/'Tests'/name for name in ['test_wheelset_actuated_cell.py',
        'test_wheelset_process_runtime.py','test_wheelset_resource_runtime.py',
        'test_wheelset_device_action.py','test_wheelset_light_material_network.py'])
    # Keep unsuccessful trials as historical evidence, without treating old
    # source fingerprints or failed outcomes as current successful checks.
    selected=[p for p in runs.iterdir() if p.is_dir() and
        (p.name.startswith('actuated_') or p.name.startswith('m2_') or p.name in required_resources
         or p.name.startswith('multi_actuated_') or p.name.startswith('light_material_')
         or p.name in ['resource_gz_01_normal'])]
    for directory in selected:files.update(p for p in directory.rglob('*') if p.is_file())
    for name in ['agv_network_01','agv_network_02','agv_reference_01','agv_reference_02','agv_reference_03']:
        files.update(p for p in (sim/name).rglob('*') if p.is_file())
    entries=[]
    for p in sorted(files,key=lambda p:p.relative_to(ROOT).as_posix()):
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(runtime_version=args.version,created_utc=datetime.now(timezone.utc).isoformat(),
        build_fingerprint=fingerprint,validated_runs=required_actuated+required_resources+required_multi+required_fleet,
        audits=audits,files=entries,claim_boundary='Runtime development release: constrained multi-workpiece actuation/resource ownership and two wheel-driven SCOUT surrogates with Gazebo truth feedback on a separate light-material road. Synthetic handling and missing downstream equipment handoff; no full-line, localization/Nav2, hardware or industrial acceptance.')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for entry in entries:archive.write(ROOT/entry['path'],entry['path'])
        archive.writestr('runtime_manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for entry in entries:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    result=dict(passed=True,file=str(target),size_bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),file_count=len(entries)+1,
        verified_run_count=len(audits),build_fingerprint=fingerprint)
    (target.with_suffix('.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
