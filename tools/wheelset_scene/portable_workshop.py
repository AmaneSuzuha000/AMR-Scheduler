"""Prepare and verify a relocatable development source bundle, without old builds.

This is a delivery prerequisite. Packaging and compilation do not establish a
successful joint simulation, recovery, or fresh-directory runtime acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'source_bundle_manifest.json'
ARTIFACTS = 'tools/ros2/wheelset_dependencies/navigation_debs'
FINGERPRINT = 'a5d904d45f00c611351ac24534889f55cdbdd6cc41cb23434a5c4cca90dbfb88'
SOURCE_DIRECTORIES = ['amr_sched', 'tools/wheelset_scene', 'tools/live_bridge',
    'tools/ros2/amr_gazebo_guard', 'tools/ros2/wheelset_axis_servo',
    'tools/ros2/wheelset_dependencies/scout_gazebo_description',
    'assets/wheelset_workshop/models', 'simulation/wheelset_workshop/worlds']


def included(path):
    return path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.log')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def relative_file(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '\\' in name or ':' in name or any(k in ('', '.', '..') for k in p.parts):
        raise ValueError('canonical relative bundle path required: '+name)
    if p.as_posix() != name: raise ValueError('noncanonical path: '+name)
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('bundle path leaves its root: '+name)
    return target


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/MANIFEST).read_text())
    assert manifest['schema'] == 'wheelset_source_bundle_v1'
    assert manifest['acceptance'] == 'source_snapshot_only_joint_and_reproduction_pending'
    entries = manifest['files']; names = [e['path'] for e in entries]
    assert len(names) == len(set(names)) and MANIFEST not in names
    for e in entries:
        path = relative_file(root, e['path'])
        assert path.is_file() and path.stat().st_size == e['size'] and sha256(path) == e['sha256'], e['path']
    for directory in SOURCE_DIRECTORIES:
        actual = {p.relative_to(root).as_posix() for p in (root/directory).rglob('*') if included(p)}
        expected = {n for n in names if n.startswith(directory+'/')}
        assert actual == expected, 'unexpected or missing source files: '+directory
    from .evidence import build_fingerprint
    # Run this entry point from the extracted tree, so imported build paths are
    # bound to that tree, rather than silently validating another checkout.
    assert root == ROOT.resolve(), 'execute the verifier from the extracted source tree'
    assert build_fingerprint() == manifest['build_fingerprint'] == FINGERPRINT
    return dict(passed=True, scope='source_snapshot_integrity', files=len(entries),
        build_fingerprint=manifest['build_fingerprint'], root=str(root))


def pack(artifacts, output):
    from .evidence import build_fingerprint
    assert build_fingerprint() == FINGERPRINT, 'frozen visual build changed'
    artifacts, output = Path(artifacts).resolve(), Path(output).resolve()
    assert not output.exists() and not output.with_suffix('.json').exists(), 'existing delivery preserved'
    metadata = json.loads((artifacts/'artifacts.json').read_text())
    origin = json.loads((artifacts/'origin_manifest.json').read_text())
    assert metadata['schema'] == 'wheelset_official_navigation_artifacts_v1' and origin['passed']
    assert sha256(artifacts/'origin_manifest.json') == metadata['provenance_sha256']
    assert metadata['packages'] == origin['packages'] and len(metadata['packages']) == 4
    files = {}
    def add(path, name=None):
        name = name or path.relative_to(ROOT).as_posix()
        relative_file(ROOT, name)
        assert path.is_file() and not path.is_symlink(), str(path)
        assert name not in files, 'duplicate package member: '+name
        files[name] = path
    for directory in SOURCE_DIRECTORIES:
        for path in sorted((ROOT/directory).rglob('*')):
            if included(path):
                add(path)
    for path in (ROOT/'Tests').glob('test_wheelset_*.py'): add(path)
    for name in ['scenes/wheelset_workshop_video_v1.json',
        'requirements.txt', 'tools/ros2/wheelset_dependencies/README.md',
        'tools/ros2/wheelset_dependencies/snapshot_manifest.json',
        'simulation/wheelset_workshop/build_report.json', 'simulation/wheelset_workshop/cameras.json',
        'simulation/wheelset_workshop/STATUS.md', 'simulation/wheelset_workshop/PROJECT_GOAL.md',
        'simulation/wheelset_workshop/DEMONSTRATION_GAPS.md',
        'simulation/wheelset_workshop/PORTABLE_RUNNING.md', 'simulation/wheelset_workshop/NAVIGATION_RUNNING.md']:
        path = ROOT/name
        if path.is_file(): add(path)
        else: raise FileNotFoundError(name)
    for name in ('tools/__init__.py', 'Tests/__init__.py', 'LICENSE', 'LICENSE.md', 'LICENSE.txt'):
        path = ROOT/name
        if path.is_file(): add(path)
    for name in ('artifacts.json', 'origin_manifest.json'):
        add(artifacts/name, ARTIFACTS+'/'+name)
    for p in metadata['packages']:
        path = relative_file(artifacts, p['file'])
        assert sha256(path) == p['sha256'] and p['architecture'] == 'amd64'
        add(path, ARTIFACTS+'/'+p['file'])
    entries = [dict(path=n, size=p.stat().st_size, sha256=sha256(p)) for n,p in sorted(files.items())]
    manifest = dict(schema='wheelset_source_bundle_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        acceptance='source_snapshot_only_joint_and_reproduction_pending', build_fingerprint=FINGERPRINT,
        files=entries, official_dependency_packages=metadata['packages'],
        claim_boundary='Internal development source snapshot with frozen visuals, scheduler, native '
            'sources, tests and hash-checked official dependency archives. No old builds, runtime '
            'results, final joint recovery acceptance or fresh-directory runtime acceptance is included. '
            'The SCOUT asset provenance limitation remains in wheelset_dependencies/README.md.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries: archive.write(files[entry['path']], entry['path'])
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for entry in entries:
            with archive.open(entry['path']) as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
            assert digest.hexdigest() == entry['sha256'], entry['path']
    result = dict(passed=True, scope='source_archive_integrity', file=str(output),
        sha256=sha256(output), bytes=output.stat().st_size, files=len(entries)+1,
        acceptance=manifest['acceptance'], build_fingerprint=FINGERPRINT)
    output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def prepare_dependencies(output):
    if os.name == 'nt': raise SystemExit('Extract the official packages on Ubuntu 22.04 amd64.')
    artifacts = ROOT/ARTIFACTS
    metadata = json.loads((artifacts/'artifacts.json').read_text())
    origin = json.loads((artifacts/'origin_manifest.json').read_text())
    assert sha256(artifacts/'origin_manifest.json') == metadata['provenance_sha256']
    assert metadata['packages'] == origin['packages'] and origin['passed']
    output = Path(output).resolve(); assert not output.exists(), 'existing prefix preserved'
    for package in metadata['packages']:
        source = relative_file(artifacts, package['file'])
        assert sha256(source) == package['sha256'], package['file']
        for field,key in (('Package','package'), ('Version','version'), ('Architecture','architecture')):
            value = subprocess.check_output(['dpkg-deb','-f',str(source),field], text=True).strip()
            assert value == package[key], (field,value,package[key])
    output.mkdir(parents=True); prefix=output/'prefix'; prefix.mkdir()
    for package in metadata['packages']:
        subprocess.run(['dpkg-deb','-x',str(artifacts/package['file']),str(prefix)], check=True)
    for name,digest in origin['extracted_sha256'].items():
        assert sha256(relative_file(prefix,name)) == digest, name
    # BASH_SOURCE resolves this newly extracted prefix each time it is sourced.
    setup = '''# Source after /opt/ros/humble/setup.bash; no system package installation.
wheelset_dependency_prefix="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/prefix"
export AMENT_PREFIX_PATH="$wheelset_dependency_prefix/opt/ros/humble${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export CMAKE_PREFIX_PATH="$wheelset_dependency_prefix/opt/ros/humble${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$wheelset_dependency_prefix/opt/ros/humble/lib:$wheelset_dependency_prefix/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$wheelset_dependency_prefix/opt/ros/humble/local/lib/python3.10/dist-packages:$wheelset_dependency_prefix/opt/ros/humble/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
unset wheelset_dependency_prefix
'''
    (output/'setup.bash').write_text(setup)
    result = dict(passed=True, packages=metadata['packages'], extracted_files=len(origin['extracted_sha256']),
        prefix=str(prefix), setup_sha256=sha256(output/'setup.bash'))
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(passed=True, prefix=str(prefix), extracted_files=result['extracted_files'])


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('pack'); p.add_argument('--dependency-artifacts',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    sub.add_parser('verify')
    p=sub.add_parser('dependencies'); p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action == 'pack': result=pack(args.dependency_artifacts,args.output)
    elif args.action == 'verify': result=verify()
    else: result=prepare_dependencies(args.output)
    print(json.dumps(result))


if __name__ == '__main__': main()
