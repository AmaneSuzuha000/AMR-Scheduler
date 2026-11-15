"""Explicit input interruptions for this workshop's short recovery campaign.

Motion commands still originate in Nav2. This harness can interrupt their
delivery, the owner's native mission lease, or the adapter's scan-health input.
It never creates a pose or completion observation.
"""
import math


class FaultCampaign:
    channels = ('velocity_link', 'mission_lease', 'adapter_scan_input')
    reasons = ('execution guard: velocity command timeout',
               'execution guard: mission lease timeout', 'localization health: scan stale')

    def __init__(self, runtime, executor, record):
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        self.runtime = runtime; self.executor = executor; self.record = record
        self.robot = 'scout_a'; self.node = runtime.nodes[self.robot]
        self.index = 0; self.phase = 'waiting_motion'; self.origin = None; self.rows = []
        self.injected = None; self.latched = False; self.last_resume = -math.inf
        self.mute_commands = False; self.mute_lease = False; self.mute_scan = False
        campaign = self

        class Relay(Node):
            def __init__(self, robot):
                super().__init__('workshop_fault_relay', namespace=robot)
                self.robot = robot; self.sequence = 0
                self.publisher = self.create_publisher(Twist, 'guard_input_cmd_vel', 1)
                self.subscription = self.create_subscription(Twist, 'cmd_vel', self.on_command, 1)

            def on_command(self, message):
                self.sequence += 1
                muted = self.robot == campaign.robot and campaign.mute_commands
                if not muted:
                    self.publisher.publish(message)
                record('velocity_relay', robot=self.robot, seq=self.sequence, muted=muted,
                       vx=message.linear.x, wz=message.angular.z)

        self.relays = [Relay(robot) for robot in runtime.nodes]
        for relay in self.relays:
            executor.add_node(relay)
        self.lease_original = self.node.publish_lease
        self.health_original = self.node.health.observe

        def lease():
            if not self.mute_lease:
                self.lease_original()

        def observe(source, *args, **kwargs):
            if not (source == 'scan' and self.mute_scan):
                self.health_original(source, *args, **kwargs)

        self.node.publish_lease = lease; self.node.health.observe = observe

    def event(self, kind, now, **values):
        row = dict(event=kind, wall=now, robot=self.robot, index=self.index, **values)
        self.rows.append(row); self.record('fault_campaign', **row)

    def pins(self):
        return {group: {key: value for key, value in members.items() if self.robot in value}
                for group, members in self.runtime.sim._res.pins().items()}

    def poll(self, now):
        gate = self.runtime.sim.executions[self.robot]; node = self.node
        others = {r: g.fault for r, g in self.runtime.sim.executions.items() if r != self.robot and g.fault}
        if others:
            raise RuntimeError('unexpected other-robot fault: '+str(others))
        if self.phase == 'complete':
            if gate.fault: raise RuntimeError('unexpected post-campaign fault: '+gate.fault)
            return
        if gate.pose is not None and self.origin is None:
            self.origin = (gate.pose['x'], gate.pose['y'])
        if self.phase == 'waiting_motion':
            if gate.fault: raise RuntimeError('unexpected fault before injection: '+gate.fault)
            if (gate.command and node.active and node.guard_fresh() and gate.pose and self.origin
                    and node.guard_status.state == 'armed' and node.guard_status.actual_speed > .05
                    and math.dist((gate.pose['x'], gate.pose['y']), self.origin) >= .25
                    and now-self.last_resume >= .7):
                self.command = dict(gate.command); self.retained = self.pins()
                self.injected = now; self.latched = False; self.phase = 'interrupted'
                self.mute_commands = self.index == 0; self.mute_lease = self.index == 1; self.mute_scan = self.index == 2
                self.event('INPUT_INTERRUPTED', now, channel=self.channels[self.index],
                           command=self.command, pins=self.retained)
            return
        if self.pins() != self.retained or gate.command != self.command:
            raise RuntimeError('interrupted route or corridor ownership changed before recovery')
        if gate.fault:
            if gate.fault != self.reasons[self.index]:
                raise RuntimeError('unexpected injected-fault outcome: '+gate.fault)
            if not self.latched:
                self.event('FAULT_LATCHED', now, reason=gate.fault, command=dict(gate.command), pins=self.pins())
                self.latched = True
        if self.phase == 'interrupted' and now-self.injected >= .9 and self.latched:
            self.mute_commands = self.mute_lease = self.mute_scan = False
            self.phase = 'waiting_stopped'
            self.event('INPUT_RESTORED', now, channel=self.channels[self.index])
        if (self.phase == 'waiting_stopped' and gate.stopped and node.guard_fresh()
                and node.guard_status.state == 'latched' and node.guard_status.stopped
                and not node.native_pending and node.goal_handle is None and node.guard_pending is None
                and node.health.reason(node.clock_s, now) is None):
            rejected = self.runtime.resume_execution(self.robot, self.command['route_id']+':obsolete')
            if rejected['accepted']: raise RuntimeError('obsolete route unexpectedly resumed')
            self.event('OLD_ROUTE_REJECTED', now, result=rejected)
            answer = self.runtime.resume_execution(self.robot, self.command['route_id'])
            self.event('RECOVERY_ATTEMPT', now, result=answer)
            if answer['accepted']:
                new = gate.command
                if (new['resumes_route_id'] != self.command['route_id'] or new['revision'] <= self.command['revision']
                        or new['task_id'] != self.command['task_id'] or self.pins() != self.retained):
                    raise RuntimeError('recovery did not retain task and reserved route')
                self.event('RECOVERY_AUTHORIZED', now, command=dict(new), pins=self.pins())
                self.origin = (gate.pose['x'], gate.pose['y']); self.last_resume = now
                self.index += 1; self.phase = 'complete' if self.index == len(self.channels) else 'waiting_motion'
        if self.phase != 'waiting_motion' and self.phase != 'complete' and now-self.injected > 15.:
            raise RuntimeError('injected fault failed to latch, stop or recover in time')

    def summary(self):
        return dict(passed=self.phase == 'complete', phase=self.phase, recovered=self.index,
                    channels=list(self.channels), events=self.rows,
                    scope='Nav2-to-native command delivery, native mission lease and adapter-only scan-health input interruptions')

    def close(self):
        self.node.publish_lease = self.lease_original; self.node.health.observe = self.health_original
        for relay in self.relays:
            self.executor.remove_node(relay); relay.destroy_node()
