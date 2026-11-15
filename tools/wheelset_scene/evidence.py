"""Bind runtime evidence to the generated geometry, worlds and layout together."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_fingerprint():
    paths = [ROOT / 'scenes/wheelset_workshop_video_v1.json']
    for directory in ['assets/wheelset_workshop/models', 'simulation/wheelset_workshop/worlds']:
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def require_running_build():
    expected = build_fingerprint()
    sim = ROOT / 'simulation/wheelset_workshop'
    build = json.loads((sim / 'build_report.json').read_text(encoding='utf-8'))
    session = json.loads((sim / 'runtime/session.json').read_text(encoding='utf-8'))
    if build.get('build_fingerprint') != expected or session.get('build_fingerprint') != expected:
        raise RuntimeError('Geometry or layout changed since launch. Rebuild and restart this scene before collecting evidence.')
    return expected
