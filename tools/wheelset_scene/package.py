"""Verify current delivery evidence and create a portable scene archive."""
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote
import zipfile
from .build import ROOT, SCENE
from .evidence import build_fingerprint


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    sim = ROOT / 'simulation/wheelset_workshop'
    comparison = ROOT / 'visual_comparison/wheelset_workshop_v1'
    scene = json.loads(SCENE.read_text(encoding='utf-8'))
    scene_hash = digest(SCENE)
    captures = json.loads((comparison / 'gazebo/capture_report.json').read_text(encoding='utf-8'))
    build = json.loads((sim / 'build_report.json').read_text(encoding='utf-8'))
    static = json.loads((sim / 'validation.json').read_text(encoding='utf-8'))
    runtime = json.loads((sim / 'runtime/runtime_audit.json').read_text(encoding='utf-8'))
    assert build['scene_sha256'] == static['scene_sha256'] == runtime['scene_sha256'] == scene_hash, 'Stale scene evidence'
    assert static['passed'] and runtime['passed'], 'Unsuccessful check'
    fingerprint = build_fingerprint()
    assert build['build_fingerprint'] == static['build_fingerprint'] == runtime['build_fingerprint'] == fingerprint, 'Stale geometry evidence'
    renders = json.loads((comparison / 'blender_previews/render_report.json').read_text(encoding='utf-8'))
    assert renders['build_fingerprint'] == fingerprint, 'Stale Blender scene'
    assert renders['scene_sha256']==scene_hash, 'Stale Blender render layout'
    saved_blender=json.loads((sim/'blender_check.json').read_text(encoding='utf-8'))
    assert saved_blender['passed'] and saved_blender['build_fingerprint']==fingerprint, 'Stale editable scene check'
    assert saved_blender['blend_sha256']==digest(ROOT/'assets/wheelset_workshop/source/wheelset_workshop.blend'), 'Changed editable scene'
    camera_fit = json.loads((comparison / 'camera_fit_press_front.json').read_text(encoding='utf-8'))
    assert camera_fit['applied'] and camera_fit['build_fingerprint'] == fingerprint, 'Stale camera-fit evidence'
    inspection_fit=json.loads((comparison/'camera_fit_inspection.json').read_text(encoding='utf-8'))
    assert inspection_fit['build_fingerprint']==fingerprint, 'Stale inspection pose evidence'
    for view in inspection_fit['views']:
        assert digest(comparison/'gazebo'/f"{view['camera']}.png")==view['gazebo_image_sha256']
        assert digest(comparison/'references'/f"frame_{view['reference_second']:02d}s.png")==view['reference_sha256']
    identification_fit=json.loads((comparison/'camera_fit_identification.json').read_text(encoding='utf-8'))
    assert identification_fit['build_fingerprint']==fingerprint, 'Stale identification evidence'
    assert identification_fit['static_robot_hood_height_separation']['passed'], 'Identification arm/hood visual meshes overlap in height'
    for view in identification_fit['views']:
        assert digest(comparison/'gazebo'/f"{view['camera']}.png")==view['gazebo_image_sha256']
        assert digest(comparison/'references'/f"frame_{view['reference_second']:02d}s.png")==view['reference_sha256']
    storage_fit=json.loads((comparison/'camera_fit_storage.json').read_text(encoding='utf-8'))
    assert storage_fit['build_fingerprint']==fingerprint, 'Stale storage pose evidence'
    for view in storage_fit['views']:
        assert digest(comparison/'gazebo'/f"{view['camera']}.png")==view['gazebo_image_sha256']
        assert digest(comparison/'references'/f"frame_{view['reference_second']:02d}s.png")==view['reference_sha256']
    unknown_fit=json.loads((comparison/'camera_fit_unknown.json').read_text(encoding='utf-8'))
    assert unknown_fit['build_fingerprint']==fingerprint, 'Stale unknown-zone pose evidence'
    for view in unknown_fit['views']:
        assert digest(comparison/'gazebo'/f"{view['camera']}.png")==view['gazebo_image_sha256']
        assert digest(comparison/'references'/f"frame_{view['reference_second']:02d}s.png")==view['reference_sha256']
    for name in ['opening','service','component','sorting','press_feed','finishing','metrology','front_detail','transfer_detail','preparation_geometry','preparation_details']:
        fit=json.loads((comparison/f'camera_fit_{name}.json').read_text(encoding='utf-8'))
        assert fit['build_fingerprint']==fingerprint, f'Stale {name} camera evidence'
        if name=='preparation_details':
            assert fit['static_hole_wall_visibility']['passed'], 'Wiping plate wall blocks the aperture centre ray'
        for view in fit['views']:
            assert digest(comparison/'gazebo'/f"{view['camera']}.png")==view['gazebo_image_sha256']
            assert digest(comparison/'references'/f"frame_{view['reference_second']:02d}s.png")==view['reference_sha256']
    assert {c['camera'] for c in captures} == {c['id'] for c in scene['cameras']}, 'Missing camera evidence'
    for capture in captures:
        assert capture['scene_sha256'] == scene_hash, 'Stale capture'
        assert capture['build_fingerprint'] == fingerprint, 'Stale capture geometry'
        assert digest(comparison / 'gazebo' / capture['file']) == capture['sha256'], 'Changed capture'
        assert capture['source'] == 'Gazebo ROS camera image'
        assert (comparison / 'blender_previews' / capture['file']).is_file(), 'Missing Blender preview'
        assert digest(comparison / 'blender_previews' / capture['file']) == renders['cameras'][capture['camera']]['sha256']
    reference = json.loads((comparison / 'reference_index.json').read_text(encoding='utf-8'))
    for view in reference['views']:
        assert (comparison / view['reference']).is_file()
        assert (comparison / view['gazebo']).is_file()
    assert len(reference['views']) == sum(c['reference_second'] is not None for c in scene['cameras'])
    coverage=json.loads((comparison/'source_coverage.json').read_text(encoding='utf-8'))
    assert coverage['build_fingerprint']==fingerprint and coverage['reference_pair_count']==len(reference['views'])
    assert {r['segment'] for r in coverage['segments']}=={s['id'] for s in scene['video_segments']}
    assert not coverage['independent_visual_acceptance']
    links = []

    class Links(HTMLParser):
        def handle_starttag(self, tag, attributes):
            for key, value in attributes:
                if key in ('href', 'src') and value:
                    links.append(value)

    Links().feed((comparison / 'review.html').read_text(encoding='utf-8'))
    for link in links:
        assert '://' not in link, 'Review must stay offline'
        assert (comparison / unquote(link)).resolve().exists(), f'Broken review link: {link}'
    page_check = json.loads((comparison / 'page_check.json').read_text(encoding='utf-8'))
    assert page_check['passed'] and page_check['page_sha256'] == digest(comparison / 'review.html'), 'Stale page check'
    report = {
        'passed': True, 'scene_sha256': scene_hash, 'build_fingerprint': fingerprint,
        'native_gazebo_cameras': len(captures), 'reference_pairs': len(reference['views']),
        'static_page_links_checked': len(links), 'capture_hashes_verified': True,
        'claim_boundary': 'Delivery integrity and runtime loading only; source camera registration, dynamics and scheduling remain unverified.'
    }
    (sim / 'delivery_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    sources = [ROOT / p for p in ['tools/wheelset_scene', 'assets/wheelset_workshop',
                                'simulation/wheelset_workshop', 'visual_comparison/wheelset_workshop_v1']]
    files = [SCENE, ROOT / 'output/playwright/wheelset_review_check.py',
             ROOT / 'Tests/test_wheelset_process_runtime.py']
    for directory in sources:
        for path in directory.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if path.name in ('delivery_manifest.json', 'session.json') or path.suffix == '.blend1':
                continue
            files.append(path)
    files.sort(key=lambda p: p.relative_to(ROOT).as_posix())
    manifest = {
        'release': 'wheelset_workshop_video_v1_visual_' + scene.get('visual_revision','0.1'), 'delivery_date': datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
        'timezone': 'Asia/Shanghai', 'scene_sha256': scene_hash,
        'excluded': ['original_video', 'Blender_installation', 'host_specific_process_session', 'Blender_backup'],
        'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]
    }
    manifest_path = sim / 'delivery_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    archive_path = ROOT / ('out/wheelset_workshop_video_v1_visual_' + scene.get('visual_revision','0.1') + '.zip')
    archive_path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files + [manifest_path]:
            archive.write(path, 'wheelset_workshop_video_v1/' + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None, 'Archive CRC verification failed'
    print(json.dumps({'passed': True, 'files': len(files) + 1,
                      'archive': str(archive_path), 'bytes': archive_path.stat().st_size,
                      'sha256': digest(archive_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
