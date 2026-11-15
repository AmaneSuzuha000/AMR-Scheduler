"""Owner-side terminal take-over: the regulator must close a declared tolerance."""
import math

import pytest

from tools.live_bridge.terminal_docking import DockRegulator, DockSpec


def spec(**overrides):
    values = dict(capture_radius_m=.35, position_tolerance_m=.015, yaw_tolerance_rad=.04)
    values.update(overrides)
    return DockSpec(**values)


def test_capture_window_must_be_wider_than_the_tolerance_it_closes():
    """A take-over that cannot reach its own gate is a configuration error."""
    with pytest.raises(ValueError):
        spec(capture_radius_m=.015, position_tolerance_m=.015)
    with pytest.raises(ValueError):
        spec(yaw_tolerance_rad=math.pi)
    with pytest.raises(ValueError):
        spec(max_linear=0.)
    with pytest.raises(ValueError):
        spec(reverse_linear=0.)
    with pytest.raises(ValueError):
        spec(max_correction_rad=math.pi)
    with pytest.raises(ValueError):
        spec(hold_s=-.1)
    with pytest.raises(ValueError):
        spec(capture_radius_m=float('nan'))
    with pytest.raises(ValueError):
        spec(hold_s=float('inf'))


def test_target_must_be_finite():
    with pytest.raises(ValueError):
        DockRegulator(spec(), (1., 2., 3.), 0.)
    with pytest.raises(ValueError):
        DockRegulator(spec(), (1., float('nan')), 0.)
    with pytest.raises(ValueError):
        DockRegulator(spec(), (1., 2.), float('nan'))


def test_a_bearing_to_the_endpoint_is_never_the_steering_objective():
    """A bearing flips by pi on overshoot and spun the earlier take-over around."""
    behind = DockRegulator(spec(), (1., 0.), 0.).update((.5, 0., 0.), 0., 0., 10.)
    ahead = DockRegulator(spec(), (1., 0.), 0.).update((1.5, 0., 0.), 0., 0., 10.)
    assert behind['heading_error'] == pytest.approx(0.) and ahead['heading_error'] == pytest.approx(math.pi)
    # The same 0.5 m error ahead and behind must not produce opposite turns.
    assert behind['wz'] == pytest.approx(0.) and ahead['wz'] == pytest.approx(0.)
    assert behind['vx'] > 0. and ahead['vx'] < 0.
    assert abs(ahead['vx']) <= .04


def test_overshoot_is_corrected_by_a_bounded_reverse_not_a_turn():
    command = DockRegulator(spec(), (1., 0.), 0.).update((1.06, 0., 0.), 0., 0., 10.)
    assert command['phase'] == 'approach'
    assert -.04 <= command['vx'] < 0.
    assert command['wz'] == pytest.approx(0.)


def test_lateral_error_turns_toward_the_dock_axis_only_within_a_declared_cap():
    command = DockRegulator(spec(), (1., .4), 0.).update((0., 0., 0.), 0., 0., 10.)
    assert command['phase'] == 'approach'
    assert 0. < command['wz'] <= .40
    assert abs(command['lateral_error'] - .4) < 1e-9
    capped = DockRegulator(spec(max_correction_rad=.2), (1., 5.), 0.).update((0., 0., 0.), 0., 0., 10.)
    assert capped['wz'] <= .40
    assert abs(capped['steer_error']) <= .2 + 1e-9
    assert capped['steer_error'] < capped['heading_error']


def test_approach_is_capped_and_creeps_only_when_the_target_is_worth_it():
    command = DockRegulator(spec(), (1., 0.), 0.).update((0., 0., 0.), 0., 0., 10.)
    assert command['phase'] == 'approach'
    assert command['vx'] == pytest.approx(.10)
    far = DockRegulator(spec(), (5., 0.), 0.).update((0., 0., 0.), 0., 0., 10.)
    assert far['vx'] == pytest.approx(.10)
    near = DockRegulator(spec(), (.03, 0.), 0.).update((0., 0., 0.), 0., 0., 10.)
    assert near['vx'] == pytest.approx(.045)
    creeping = DockRegulator(spec(), (.02, 0.), 0.).update((0., 0., 0.), 0., 0., 10.)
    assert .02 <= creeping['vx'] <= .03


@pytest.mark.parametrize('distance', [.016, .02, .05, .10, .34, 1.5, 12.])
@pytest.mark.parametrize('bearing', [-3., -1., -.5, -.2, 0., .2, .5, 1., 3.])
def test_every_command_stays_inside_the_declared_twist_ceiling(distance, bearing):
    """The guard rejects a command outside its own bounds; never approach it."""
    target = (distance * math.cos(bearing), distance * math.sin(bearing))
    regulator = DockRegulator(spec(), target, 0.)
    command = regulator.update((0., 0., 0.), 0., 0., 0.)
    assert -.04 <= command['vx'] <= .10
    assert abs(command['wz']) <= .40
    assert command['phase'] in DockRegulator.phases


def test_position_inside_its_window_still_turns_to_the_handoff_pose():
    """Being at the dock is not the same as facing the handoff pose."""
    command = DockRegulator(spec(), (1., 0.), .30).update((1., 0., 0.), 0., 0., 10.)
    assert command['phase'] == 'approach'
    assert command['vx'] == pytest.approx(0.)
    assert command['wz'] > 0.
    assert command['aligned'] is False


def test_done_requires_both_windows_and_a_measured_standstill():
    regulator = DockRegulator(spec(hold_s=.30), (1., 0.), 0.)
    assert regulator.update((1., 0., 0.), 0., 0., 10.)['phase'] == 'hold'
    assert regulator.update((1., 0., 0.), 0., 0., 10.15)['done'] is False
    assert regulator.update((1., 0., 0.), 0., 0., 10.31)['done'] is True
    # A residual body speed is not a standstill, however close the pose is.
    coasting = DockRegulator(spec(), (1., 0.), 0.).update((1., 0., 0.), .2, 0., 10.)
    assert coasting['phase'] == 'hold' and coasting['done'] is False
    assert coasting['vx'] == 0. and coasting['wz'] == 0.
    spinning = DockRegulator(spec(), (1., 0.), 0.).update((1., 0., 0.), 0., .5, 10.)
    assert spinning['phase'] == 'hold' and spinning['done'] is False


def test_hold_window_restarts_when_the_vehicle_moves_again():
    regulator = DockRegulator(spec(hold_s=.30), (1., 0.), 0.)
    regulator.update((1., 0., 0.), 0., 0., 10.)
    regulator.update((1., 0., 0.), 0., 0., 10.2)
    moved = regulator.update((1., 0., 0.), .05, 0., 10.25)
    assert moved['done'] is False and moved['settled_s'] < .25
    assert regulator.update((1., 0., 0.), 0., 0., 10.4)['phase'] == 'hold'
    assert regulator.update((1., 0., 0.), 0., 0., 10.8)['done'] is True


def test_one_noisy_sample_cannot_erase_a_real_quiet_period():
    """The fixture keeps settling after a stop; the window leaks, it does not reset."""
    regulator = DockRegulator(spec(hold_s=.30), (1., 0.), 0.)
    regulator.update((1., 0., 0.), 0., 0., 10.)
    settled = regulator.update((1., 0., 0.), 0., 0., 10.28)['settled_s']
    assert settled > 0.
    noisy = regulator.update((1., 0., 0.), .02, .03, 10.30)
    assert 0. < noisy['settled_s'] < settled
    assert regulator.update((1., 0., 0.), 0., 0., 10.62)['done'] is True


def test_yaw_error_is_measured_across_the_branch_cut():
    """A target at +pi and a pose at -pi are aligned, not two pi apart."""
    command = DockRegulator(spec(), (1., 0.), math.pi).update((1., 0., -3.13), 0., 0., 10.)
    assert command['aligned'] is True
    assert abs(command['yaw_error']) < .02


def test_measured_pitch_or_roll_is_not_accepted_as_a_pose():
    with pytest.raises(ValueError):
        DockRegulator(spec(), (1., 0.), 0.).update((1., 0., float('nan')), 0., 0., 10.)
    with pytest.raises(ValueError):
        DockRegulator(spec(), (1., 0.), 0.).update((1., 0., 0.), float('inf'), 0., 10.)


def test_axis_controller_has_a_real_lateral_equilibrium_outside_a_tight_dock():
    # Regression witness: zero forward error and a nonzero lateral error admit
    # vx=wz=0 in the old law. Testing command ceilings alone did not detect it.
    lateral=.02
    yaw=math.atan2(-1.5*lateral,.5)
    old=DockRegulator(spec(position_tolerance_m=.005),(0.,0.),0.)
    row=old.update((0.,lateral,yaw),0.,0.,10.)
    assert not row['done'] and row['distance']>.005
    assert abs(row['vx'])<1e-12 and abs(row['wz'])<1e-12
    corrected=DockRegulator(spec(position_tolerance_m=.005,steering_mode='polar'),(0.,0.),0.)
    row=corrected.update((0.,lateral,yaw),0.,0.,10.)
    assert abs(row['vx'])+abs(row['wz'])>.01


@pytest.mark.parametrize('forward',[-.10,0.,.05,.30])
@pytest.mark.parametrize('lateral',[-.05,0.,.05])
@pytest.mark.parametrize('heading',[-.15,0.,.15])
def test_tight_pose_controller_closes_both_errors_in_a_lagged_unicycle_model(forward,lateral,heading):
    """A kinematic/lag check, not Gazebo contact or safety acceptance."""
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='polar'),(0.,0.),0.)
    x,y,yaw=forward,lateral,heading
    vx=wz=0.
    dt=.02
    for index in range(int(regulator.spec.deadline_s/dt)):
        row=regulator.update((x,y,yaw),abs(vx),wz,index*dt)
        assert -.04<=row['vx']<=.10 and abs(row['wz'])<=.40
        assert math.hypot(x,y)<regulator.spec.capture_radius_m
        if row['done']:
            assert math.hypot(x,y)<=.005 and abs(yaw)<=.04
            break
        vx+=(row['vx']-vx)*dt/.10
        wz+=(row['wz']-wz)*dt/.10
        x+=vx*math.cos(yaw)*dt
        y+=vx*math.sin(yaw)*dt
        yaw=math.atan2(math.sin(yaw+wz*dt),math.cos(yaw+wz*dt))
    else:
        pytest.fail(f'pose did not converge: {(x,y,yaw)}')


def test_polar_overshoot_uses_bounded_reverse_and_configuration_is_explicit():
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='polar'),(0.,0.),0.)
    row=regulator.update((.02,0.,0.),0.,0.,10.)
    assert row['reverse'] and -.04<=row['vx']<0. and abs(row['wz'])<1e-12
    with pytest.raises(ValueError):spec(steering_mode='undeclared')
    with pytest.raises(ValueError):spec(steering_mode='polar',polar_turn_gain=1.)


@pytest.mark.parametrize('side',[-1,1])
def test_staged_dock_withdraws_without_turning_at_recorded_corner_contact(side):
    # 48A first platform/table boundary overlap: yaw about 16 degrees.
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.,approach_side=side)
    row=regulator.update((.0012,-.0175,.279897),0.,0.,10.)
    assert row['phase']=='withdraw' and row['vx']*side>0.
    assert row['wz']==0. and not row['done']


@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('clearance',[-.01,.05,.32])
@pytest.mark.parametrize('lateral',[-.04,.04])
@pytest.mark.parametrize('heading',[-.15,.15])
def test_staged_dock_closes_with_delay_without_large_turns_near_table(
        side,clearance,lateral,heading):
    """Lagged kinematic regression, independent of Gazebo acceptance."""
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.,approach_side=side)
    x,y,yaw=side*clearance,lateral,heading
    vx=wz=0.;history=[];dt=.02
    for index in range(int(45/dt)):
        history.append((x,y,yaw,vx,wz,index*dt))
        mx,my,mh,mv,mw,mt=history[max(0,index-5)]
        row=regulator.update((mx,my,mh),abs(mv),mw,mt)
        assert -.04<=row['vx']<=.10 and abs(row['wz'])<=.40
        if side*x<.14 and row['phase'] not in ('hold','done'):
            assert row['phase'] in ('withdraw','insert')
            if row['phase']=='withdraw':assert row['wz']==0.
        if row['done']:
            assert math.hypot(x,y)<.006 and abs(yaw)<.04
            break
        vx+=(row['vx']-vx)*dt/.10;wz+=(row['wz']-wz)*dt/.10
        x+=vx*math.cos(yaw)*dt;y+=vx*math.sin(yaw)*dt
        yaw=wrap(yaw+wz*dt)
    else:pytest.fail(f'staged docking did not converge: {(x,y,yaw)}')


def wrap(value):
    return math.atan2(math.sin(value),math.cos(value))


def test_staged_insertion_restarts_clearance_instead_of_chasing_sideways_at_table():
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.)
    regulator.stage='insert'
    row=regulator.update((0.,.02,0.),0.,0.,10.)
    assert row['phase']=='withdraw' and row['vx']<0. and row['wz']==0.
    assert not row['done'] and row['distance']>.005


@pytest.mark.parametrize('overrides',[
    {'staging_distance_m':.4},{'staging_distance_m':.01},
    {'staging_yaw_limit_rad':.3},
    {'approach_stations':({'target_xy':[0.,0.],'yaw_rad':0.,'approach_side':0},)},
])
def test_staged_profile_refuses_undefined_or_unbounded_approaches(overrides):
    with pytest.raises(ValueError):
        spec(position_tolerance_m=.005,steering_mode='staged',**overrides)


def test_staged_last_millimetres_finish_heading_before_requesting_lateral_recovery():
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.)
    regulator.stage='insert';regulator.course=(.22,.04)
    command=regulator.update((-.0005,.006,.07),0.,0.,10.)
    assert command['phase']=='insert' and command['vx']==0.
    assert command['wz']<0. and not command['done']
    assert regulator.stage=='insert'
    command=regulator.update((0.,.004,.03),0.,0.,10.1)
    assert command['phase']=='hold' and not command['done']
    assert regulator.update((0.,.004,.03),0.,0.,10.4)['done']


def test_staged_small_axis_errors_still_close_the_circular_position_gate():
    """A 4mm axial and 3.2mm lateral error must not become a stopped equilibrium."""
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.)
    regulator.stage='insert';regulator.course=(.22,.04)
    x,y,yaw=-.004,.0032,0.
    velocity=angular=0.;dt=.02
    assert math.hypot(x,y)>regulator.spec.position_tolerance_m
    for index in range(int(5/dt)):
        command=regulator.update((x,y,yaw),abs(velocity),angular,index*dt)
        if command['done']:
            assert math.hypot(x,y)<=regulator.spec.position_tolerance_m
            assert abs(yaw)<=regulator.spec.yaw_tolerance_rad
            assert abs(velocity)<=regulator.spec.stop_speed
            assert abs(angular)<=regulator.spec.stop_wz
            break
        velocity+=(command['vx']-velocity)*dt/.10
        angular+=(command['wz']-angular)*dt/.10
        x+=velocity*math.cos(yaw)*dt;y+=velocity*math.sin(yaw)*dt
        yaw=wrap(yaw+angular*dt)
    else:pytest.fail(f'circular docking gate remains open: {(x,y,yaw)}')


@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('lateral',[-.014,.014])
def test_staged_lateral_recovery_survives_small_axial_sample_variation(side,lateral):
    """The recorded lateral stall must recover with fresh, noisy axial samples."""
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.,approach_side=side)
    regulator.stage='insert';regulator.course=(.31,.04)
    x,y,yaw=0.,lateral,0.
    velocity=angular=0.;dt=.02;withdrew=False
    for index in range(int(regulator.spec.deadline_s/dt)):
        if index%5==0:
            measured=(x+.004*(1 if index//5%2==0 else -1),y,yaw)
            motion=(abs(velocity),angular);stamp=index*dt
        command=regulator.update(measured,*motion,stamp)
        assert -.04<=command['vx']<=.10 and abs(command['wz'])<=.40
        if command['phase']=='withdraw':
            withdrew=True
            assert command['wz']==0.
        if command['done']:
            assert withdrew
            assert command['distance']<=regulator.spec.position_tolerance_m
            assert abs(command['yaw_error'])<=regulator.spec.yaw_tolerance_rad
            assert math.hypot(x,y)<=.025 and abs(yaw)<=.05
            break
        velocity+=(command['vx']-velocity)*dt/.10
        angular+=(command['wz']-angular)*dt/.10
        x+=velocity*math.cos(yaw)*dt;y+=velocity*math.sin(yaw)*dt
        yaw=wrap(yaw+angular*dt)
    else:pytest.fail(f'lateral stall did not recover: {(x,y,yaw)}')


@pytest.mark.parametrize('side',[-1,1])
@pytest.mark.parametrize('clearance',[.05,.32])
@pytest.mark.parametrize('lateral',[-.04,.04])
@pytest.mark.parametrize('heading',[-.15,.15])
def test_staged_dock_closes_in_wall_budget_with_intermittent_measurements(
        side,clearance,lateral,heading):
    """Counterexample to testing only uninterrupted simulated-time motion."""
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.,approach_side=side)
    x,y,yaw=side*clearance,lateral,heading
    velocity=angular=0.;history=[];dt=.02;rtf=.8
    for index in range(int(regulator.spec.deadline_s/dt)):
        source=index*dt*rtf
        history.append((x,y,yaw,velocity,angular,source))
        slot=max(0,index-round(.10/(dt*rtf)))//6*6
        mx,my,mh,mv,mw,stamp=history[slot]
        if index%50>=33:
            requested_velocity=requested_angular=0.
            regulator.settled_s=0.;regulator.last_now=None
        else:
            measured=(mx+.003*math.sin(stamp*4),my+.003*math.sin(stamp*3+1),mh)
            command=regulator.update(measured,abs(mv),mw,stamp)
            requested_velocity,requested_angular=command['vx'],command['wz']
            assert -.04<=requested_velocity<=.10 and abs(requested_angular)<=.40
            if command['phase']=='align':assert requested_velocity==0.
            if command['done']:
                assert command['distance']<=.005 and abs(command['yaw_error'])<=.04
                assert math.hypot(x,y)<=.025 and abs(yaw)<=.05
                break
        velocity+=(requested_velocity-velocity)*dt/.10
        angular+=(.84*requested_angular-angular)*dt/.10
        x+=velocity*math.cos(yaw)*dt*rtf;y+=velocity*math.sin(yaw)*dt*rtf
        yaw=wrap(yaw+angular*dt*rtf)
    else:pytest.fail(f'wall-time docking budget exhausted: {(x,y,yaw)}')


def test_repeated_measurements_and_gaps_cannot_fabricate_terminal_stop_proof():
    regulator=DockRegulator(spec(position_tolerance_m=.005,steering_mode='staged'),
                            (0.,0.),0.)
    for stamp in (10.,10.,10.,10.1,10.1):
        assert not regulator.update((0.,0.,0.),0.,0.,stamp)['done']
    regulator.settled_s=0.;regulator.last_now=None
    assert not regulator.update((0.,0.,0.),0.,0.,11.)['done']
    assert not regulator.update((0.,0.,0.),0.,0.,11.1)['done']
