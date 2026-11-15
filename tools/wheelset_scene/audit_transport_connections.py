"""Inventory geometric transport gaps; geometry never certifies physical support."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def point_distance(point, start, end):
    delta = [end[i]-start[i] for i in range(2)]
    length2 = sum(v*v for v in delta)
    if length2 <= 0:
        raise ValueError('zero-length rail')
    t = max(0., min(1., sum((point[i]-start[i])*delta[i] for i in range(2))/length2))
    return math.dist(point[:2], [start[i]+t*delta[i] for i in range(2)])


def segment_distance(a, b):
    p, q = a['start'], b['start']
    u = [a['end'][i]-p[i] for i in range(2)]
    v = [b['end'][i]-q[i] for i in range(2)]
    cross = lambda x, y: x[0]*y[1]-x[1]*y[0]
    denominator = cross(u, v)
    if abs(denominator) > 1e-12:
        w = [q[i]-p[i] for i in range(2)]
        if 0 <= cross(w, v)/denominator <= 1 and 0 <= cross(w, u)/denominator <= 1:
            return 0.
    return min(
        *(point_distance(a[k], b['start'], b['end']) for k in ('start', 'end')),
        *(point_distance(b[k], a['start'], a['end']) for k in ('start', 'end')))


def inventory(scene, stages, origin, tolerance=.25):
    """Keep rails, painted routes and actual physical validation distinct."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid diagnostic tolerance')
    rails = scene['rail_lines']
    if len({r['id'] for r in rails}) != len(rails):
        raise ValueError('duplicate rail identity')
    components = [{r['id']} for r in rails]
    for rail in rails:
        point_distance(rail['start'], rail['start'], rail['end'])
    for index, a in enumerate(rails):
        for b in rails[index+1:]:
            if segment_distance(a, b) <= 1e-6:
                left = next(c for c in components if a['id'] in c)
                right = next(c for c in components if b['id'] in c)
                if left is not right:
                    left.update(right); components.remove(right)
    def nearby(point):
        return [r['id'] for r in rails if point_distance(point, r['start'], r['end']) <= tolerance]
    transitions = []
    previous = dict(id='infeed', target=origin)
    for number, stage in enumerate(stages, 1):
        start, end = previous['target'], stage['target']
        common = sorted(set(nearby(start)) & set(nearby(end)))
        transitions.append(dict(index=number, source=previous['id'], target=stage['id'],
            start=list(start), end=list(end), source_rails=nearby(start), target_rails=nearby(end),
            common_rail_geometry=common, horizontal_displacement_m=math.dist(start[:2], end[:2]),
            height_change_m=end[2]-start[2],
            classification='same_rail_geometry' if common else 'transfer_or_external_interface_required',
            physical_support_verified=False, physical_handoff_verified=False))
        previous = stage
    return dict(rail_count=len(rails), visual_route_count=len(scene.get('agv_routes', [])),
        rail_geometric_components=[sorted(c) for c in components],
        diagnostic_rail_association_tolerance_m=tolerance,
        transitions=transitions, classification_counts=dict(Counter(t['classification'] for t in transitions)),
        physical_transport_accepted=False,
        boundary='Centreline proximity only. Painted routes do not join rails or certify carriers. '
                 'A required transfer is not proof that a direct connecting rail should be added.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit('Use a fresh output directory')
    from .front_flow import ORIGIN, TRAVEL_Z, transfer_points
    from .finishing_flow import FULL_STAGES
    scene_path = ROOT/'scenes/wheelset_workshop_video_v1.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    report = inventory(scene, FULL_STAGES, ORIGIN)
    hashes = {}; snapshots = {}
    def remember(path):
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes(); hashes[relative] = hashlib.sha256(raw).hexdigest(); snapshots[relative] = raw
    for path in [scene_path, Path(__file__), *(ROOT/'tools/wheelset_scene'/n for n in
            ['front_flow.py', 'assembly_flow.py', 'finishing_flow.py', 'front_world.py'])]:
        remember(path)
    asset_counts = Counter(i['asset'] for i in scene['instances'])
    asset_report = {}
    for asset, count in sorted(asset_counts.items()):
        path = ROOT/'assets/wheelset_workshop/models'/('wsv1_'+asset)/'model.sdf'
        remember(path)
        model = ET.fromstring(snapshots[path.relative_to(ROOT).as_posix()]).find('model')
        asset_report[asset] = dict(instances=count, collision_elements=len(model.findall('.//collision')),
            static=model.findtext('static'), physical_support_verified=False)
    current = ORIGIN
    for transition, stage in zip(report['transitions'], FULL_STAGES):
        points = transfer_points(current, stage)
        transition.update(existing_waypoints=[list(p) for p in points],
            existing_max_axis_height_m=max([current[2], *(p[2] for p in points)]))
        current = stage['target']
    report.update(scene_id=scene['scene_id'], visual_revision=scene['visual_revision'],
        assets=asset_report, source_sha256=hashes, existing_transfer_height_m=TRAVEL_Z,
        six_metre_recipe_transitions=sum(t['existing_max_axis_height_m'] >= TRAVEL_Z for t in report['transitions']),
        notes=['Video-derived dimensions remain estimates.',
            'The isolated two-metre contact conveyor is separate from these static asset definitions.',
            'Wheelset transport does not validate complete bogie transport.'])
    output.mkdir(parents=True)
    for relative, raw in snapshots.items():
        target = output/'sources'/relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    (output/'connection_inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if args.render:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(16, 6))
        for r in scene.get('agv_routes', []):
            ax.plot([r['start'][0], r['end'][0]], [r['start'][1], r['end'][1]],
                color='#c4c9cd', linestyle='--', linewidth=.8, zorder=0)
        for r in scene['rail_lines']:
            ax.plot([r['start'][0], r['end'][0]], [r['start'][1], r['end'][1]], color='#245a81', linewidth=3)
        for t in report['transitions']:
            x, y = t['end'][:2]
            ax.scatter(x, y, s=25, color='#245a81'); ax.annotate(str(t['index']), (x, y), xytext=(2, 7), textcoords='offset points')
            if not t['common_rail_geometry']:
                ax.annotate('', (x, y), t['start'][:2], arrowprops=dict(arrowstyle='->', color='#b44b37', linestyle=':', lw=1.8))
        ax.plot([], [], color='#245a81', lw=3, label='配置中的轮对轨道中心线')
        ax.plot([], [], color='#b44b37', ls=':', lw=2, label='须补转运/交接证据的工序接口（示意连线）')
        ax.plot([], [], color='#c4c9cd', ls='--', label='视觉道路/边界标线，不代表机械承载')
        ax.set(xlim=(0, 258), ylim=(0, 72), xlabel='场景 X / m（估计尺寸）', ylabel='场景 Y / m（估计尺寸）',
            title='现有生产线连接盘点：蓝色不代表接触输送已通过；编号对应18阶段配方')
        ax.set_aspect('equal'); ax.grid(alpha=.15); ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout(); fig.savefig(output/'connection_inventory.png', dpi=150); plt.close(fig)
    print(json.dumps({k:report[k] for k in ['rail_count','visual_route_count','classification_counts',
        'six_metre_recipe_transitions','physical_transport_accepted']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
