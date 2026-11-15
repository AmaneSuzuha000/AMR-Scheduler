"""Measured wheel increments must retain sign and reject missing evidence."""
import math

import pytest

from tools.live_bridge.position_encoder import PositionEncoder, WHEELS


def test_position_encoder_reconstructs_forward_reverse_and_pure_turn():
    encoder = PositionEncoder(radius_m=.1, separation_m=.4)
    assert encoder.observe(0., WHEELS, [0.]*4) is None
    assert encoder.observe(.1, WHEELS, [1.]*4)['linear_x'] == pytest.approx(1.)
    reverse = encoder.observe(.2, WHEELS, [0.]*4)
    assert reverse['linear_x'] == pytest.approx(-1.)
    assert reverse['pose'] == pytest.approx((0., 0., 0.))
    turn = encoder.observe(.3, WHEELS, [-.2, .2, -.2, .2])
    assert turn['linear_x'] == pytest.approx(0.)
    assert turn['angular_z'] == pytest.approx(1.)
    assert turn['pose'] == pytest.approx((0., 0., .1))


def test_position_encoder_signed_distance_conserves_cancelled_joint_increments():
    encoder = PositionEncoder()
    encoder.observe(0., WHEELS, [0.]*4)
    signed_distance = 0.
    for i in range(1, 101):
        positions = [(.001 if i % 2 else 0.)]*4
        result = encoder.observe(i*.02, WHEELS, positions)
        signed_distance += result['linear_x']*result['interval_s']
    assert signed_distance == pytest.approx(0., abs=1e-15)
    assert encoder.pose == pytest.approx((0., 0., 0.))


def test_position_encoder_duplicate_and_gap_do_not_invent_velocity():
    encoder = PositionEncoder()
    assert encoder.observe(1., WHEELS, [1.]*4) is None
    assert encoder.observe(1., WHEELS, [99.]*4) is None
    result = encoder.observe(1.1, WHEELS, [1.1]*4)
    assert result['linear_x'] == pytest.approx(.08)
    assert encoder.observe(2., WHEELS, [100.]*4) is None
    result = encoder.observe(2.1, WHEELS, [100.1]*4)
    assert result['linear_x'] == pytest.approx(.08)


def test_position_encoder_invalid_input_preserves_previous_measurement():
    encoder = PositionEncoder()
    encoder.observe(1., WHEELS, [0.]*4)
    bad_inputs = [
        (math.nan, WHEELS, [0.]*4),
        (.9, WHEELS, [0.]*4),
        (1.1, WHEELS, [math.nan]*4),
        (1.1, WHEELS[:-1], [0.]*3),
        (1.1, (*WHEELS[:-1], WHEELS[0]), [0.]*4),
    ]
    for source, names, positions in bad_inputs:
        with pytest.raises(ValueError):
            encoder.observe(source, names, positions)
        assert encoder.previous == (1., (0.,)*4)
    assert encoder.observe(1.1, list(reversed(WHEELS)), [1.]*4)['linear_x'] == pytest.approx(.8)


def test_position_encoder_rejects_ambiguous_interleaved_position_jump():
    encoder = PositionEncoder(max_wheel_speed_mps=5.)
    assert encoder.observe(0., WHEELS, [0.]*4) is None
    assert encoder.observe(.02, WHEELS, [.05]*4)['linear_x'] == pytest.approx(.2)
    # A near-simultaneous alternate branch would imply 50 m/s. It must not be
    # published as motion; the following physically plausible pair resumes.
    assert encoder.observe(.021, WHEELS, [1.]*4) is None
    assert encoder.observe(.041, WHEELS, [1.05]*4)['linear_x'] == pytest.approx(.2)
    assert encoder.pose[0] == pytest.approx(.008)


def test_position_encoder_replays_recorded_waiting_stream_without_invented_motion():
    """Recorded 24 waiting robots must stay near zero while stock signs drift."""
    import json
    from pathlib import Path
    rows = {}
    path = Path('out/scout_encoder_diagnosis25/contact_run24a_sensors.jsonl')
    if not path.is_file():
        pytest.skip('frozen 24 sensor subset is not present in this checkout')
    for line in path.open(encoding='utf-8'):
        row = json.loads(line)
        if row['kind'] == 'wheel_joints' and row['robot'] == 'scout_b' and 20. <= row['source_s'] < 245.:
            rows.setdefault('joints', []).append(row)
    encoder = PositionEncoder()
    measurements = []
    for row in rows['joints']:
        result = encoder.observe(row['source_s'], row['message']['name'], row['message']['position'])
        if result is not None:
            measurements.append(result)
    assert len(measurements) > 10000
    distance = sum((left['linear_x']+right['linear_x'])*.5*(right['source_s']-left['source_s'])
                   for left, right in zip(measurements, measurements[1:]))
    assert abs(distance) < 1e-3
    assert max(abs(item['linear_x']) for item in measurements) < 1e-5


def test_position_encoder_arc_uses_measurement_geometry():
    encoder = PositionEncoder(radius_m=1., separation_m=2., max_wheel_speed_mps=100.)
    encoder.observe(0., WHEELS, [0.]*4)
    result = encoder.observe(.1, WHEELS, [0., math.pi, 0., math.pi])
    assert result['pose'] == pytest.approx((1., 1., math.pi/2.))
    for radius, separation, interval in ((0., .4, .2), (.08, math.inf, .2), (.08, .4, -1.)):
        with pytest.raises(ValueError):
            PositionEncoder(radius, separation, interval)


def test_position_encoder_bundle_is_explicit_and_preserves_original_physics(tmp_path):
    from pathlib import Path
    import yaml
    from tools.wheelset_scene.workshop_navigation import build_bundle
    original = build_bundle(tmp_path/'original', review_camera=False)
    measured = build_bundle(tmp_path/'measured', review_camera=False, position_encoder=True)
    assert original['encoder_source'] == 'stock_velocity'
    assert measured['encoder_source'] == 'joint_positions'
    assert (tmp_path/'original/navigation.world').read_bytes() == (tmp_path/'measured/navigation.world').read_bytes()
    assert (tmp_path/'original/map.pgm').read_bytes() == (tmp_path/'measured/map.pgm').read_bytes()
    for robot in measured['robot_configs']:
        old = yaml.safe_load((tmp_path/'original'/(robot+'_localization.yaml')).read_text())
        new = yaml.safe_load((tmp_path/'measured'/(robot+'_localization.yaml')).read_text())
        assert old[robot]['ekf_filter_node']['ros__parameters']['odom0'] == '/'+robot+'/wheel/odometry'
        assert new[robot]['ekf_filter_node']['ros__parameters']['odom0'] == '/'+robot+'/wheel/position_odometry'
        assert [i for i, value in enumerate(new[robot]['ekf_filter_node']['ros__parameters']['odom0_config']) if value] == [6]
    assert any(Path(key).name == 'position_encoder.py' for key in measured['source_sha256'])


def test_bundle_raises_the_estimated_posterior_without_touching_the_prior(tmp_path):
    """A long flat corridor undersamples at 500/2000 particles.

    The readiness runner deliberately perturbs the spawn prior and expects the
    filter to correct it.  With the shared particle band the corrective pass
    stops up to ~3cm short, which is outside the 2.5cm contact capture gate.
    The bundle therefore ships a wider band and a finer update cadence, records
    them in the manifest, and leaves the prior and every gate unchanged.
    """
    from tools.wheelset_scene.workshop_navigation import (
        AMCL_MAX_PARTICLES, AMCL_MIN_PARTICLES, AMCL_UPDATE_MIN_ANGLE_RAD,
        AMCL_UPDATE_MIN_DISTANCE_M, build_bundle,
    )
    import yaml
    assert (AMCL_MIN_PARTICLES, AMCL_MAX_PARTICLES) == (5000, 20000)
    assert (AMCL_UPDATE_MIN_DISTANCE_M, AMCL_UPDATE_MIN_ANGLE_RAD) == (.005, .005)
    manifest = build_bundle(tmp_path/'bundle', review_camera=False)
    assert manifest['amcl_min_particles'] == AMCL_MIN_PARTICLES
    assert manifest['amcl_max_particles'] == AMCL_MAX_PARTICLES
    assert manifest['amcl_update_min_distance_m'] == AMCL_UPDATE_MIN_DISTANCE_M
    assert manifest['amcl_update_min_angle_rad'] == AMCL_UPDATE_MIN_ANGLE_RAD
    for robot, config in manifest['robot_configs'].items():
        localization = yaml.safe_load((tmp_path/'bundle'/config['localization']).read_text())
        parameters = localization[robot]['amcl']['ros__parameters']
        assert parameters['min_particles'] == AMCL_MIN_PARTICLES
        assert parameters['max_particles'] == AMCL_MAX_PARTICLES
        assert parameters['update_min_d'] == AMCL_UPDATE_MIN_DISTANCE_M
        assert parameters['update_min_a'] == AMCL_UPDATE_MIN_ANGLE_RAD
        assert parameters['max_beams'] == manifest['amcl_max_beams']
