"""Sensor-authoritative adapter for the existing durable Python scheduler.

No GetEntityState client is created here. Map localization supplies progress;
the unchanged native motion guard independently gates physical wheel commands.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu,LaserScan
from std_srvs.srv import Empty
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from tf2_ros import Buffer,TransformListener
import rclpy

from gz_frame import GzFrame
from nav2_scheduler import SchedulerNode
from sensor_health import SensorHealth,pose_quality


class SensorSchedulerNode(SchedulerNode):
    def __init__(self,sim,log_path,port=0,*,latest_tf=False,best_effort_tf=False,input_thread=False,**kwargs):
        from feedback_ingress import dynamic_tf_qos
        dynamic_profile=dynamic_tf_qos(latest_tf=latest_tf,best_effort_tf=best_effort_tf)
        self.station_refiner=None
        self.station_fit_worker=None
        self.latest_scan=None
        self.laser_extrinsic=None
        self.fine_pose_source=None
        self.fine_pose_received=None
        takeover=kwargs.get('dock_takeover')
        descriptor=None
        if takeover is not None:
            takeover=dict(takeover)
            descriptor=takeover.pop('measurement',None)
            kwargs['dock_takeover']=takeover
        runtime_files=[Path(__file__),Path(__file__).with_name('sensor_health.py'),Path(__file__).with_name('sensor_localization.yaml'),
                       Path(__file__).with_name('nav2_metric.yaml'),Path(__file__).with_name('nav2_metric_dwb.yaml'),
                       Path(__file__).with_name('feedback_ingress.py')]
        if descriptor is not None:
            if not input_thread:
                raise ValueError('station refinement requires coherent input-thread snapshots')
            from station_refinement import StationRefiner
            self.station_refiner=StationRefiner(descriptor)
            runtime_files += [Path(__file__).with_name('station_refinement.py'),
                             Path(__file__).with_name('station_fit_worker.py'),Path(descriptor['map_image'])]
            kwargs['pose_measurement']=descriptor
        super().__init__(sim,log_path,port,coordinate_frame=GzFrame(scale=1.),frame_id='map',
                         feedback_source='localized',runtime_files=runtime_files,
                         dynamic_tf_depth=100 if best_effort_tf else 1 if latest_tf else None,
                         dynamic_tf_reliability='best_effort' if best_effort_tf else 'reliable',**kwargs)
        if descriptor is not None:
            from station_fit_worker import StationFitWorker
            self.station_fit_worker=StationFitWorker(descriptor)
            self.record('station_fit_worker_started',process_id=self.station_fit_worker.process.pid,
                scope='Numerical inputs/results only; original source and receipt age checked by owner.')
        self.sensor_ingress=None
        self.input_transform=None
        self.input_transform_error=None
        self.tf=Buffer()
        # A bounded dynamic queue can discard old delivery backlog without
        # changing source timestamps or any freshness requirement. Keep the
        # listener's durable static-transform subscription unchanged.
        self.listener_tf=(TransformListener(self.tf,self,qos=dynamic_profile)
                          if dynamic_profile is not None else TransformListener(self.tf,self))
        self.filtered=None
        self.filtered_received=None
        self.localization_ready=False
        self.health=SensorHealth()
        self.nomotion=self.create_client(Empty,'/request_nomotion_update')
        self.nomotion_pending=None
        self.next_nomotion=0.
        self.last_health=None
        self.last_feedback_unavailable=None
        self.next_feedback_diagnostic=0.
        if input_thread:
            self.listener_tf.unregister()
            for subscription in tuple(self.subscriptions):
                if subscription.topic_name in ('/clock', self.get_namespace().rstrip('/')+'/motion_guard/status'):
                    self.destroy_subscription(subscription)
            from feedback_ingress import SensorIngress
            self.sensor_ingress=SensorIngress(self,latest_tf=latest_tf,best_effort_tf=best_effort_tf)
            return
        self.create_subscription(Odometry,'/odometry/filtered',self.on_filtered,qos_profile_sensor_data)
        self.create_subscription(LaserScan,'/scan',self.on_scan,qos_profile_sensor_data)
        self.create_subscription(Imu,'/imu/data',self.on_imu,qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped,'/amcl_pose',self.on_amcl,qos_profile_sensor_data)

    def note_sensor(self,source,msg,valid,reason):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        received=getattr(self,'_input_received_at',None)
        self.health.observe(source,stamp,time.monotonic() if received is None else received,valid,reason)

    def on_scan(self,msg):
        if getattr(self,'station_refiner',None) is not None or getattr(self,'retain_station_scans',False):
            from station_refinement import scan_snapshot
            received=getattr(self,'_input_received_at',None)
            scan=scan_snapshot(msg,time.monotonic() if received is None else received)
            previous=getattr(self,'latest_scan',None)
            if previous is None or scan['source_s']>previous['source_s']:
                self.latest_scan=scan
        useful=sum(math.isfinite(v) and msg.range_min<=v<msg.range_max for v in msg.ranges)
        self.note_sensor('scan',msg,bool(msg.ranges) and useful>=max(30,.1*len(msg.ranges)),
                         'insufficient valid obstacle returns')

    def on_imu(self,msg):
        self.note_sensor('imu',msg,math.isfinite(msg.angular_velocity.z),'invalid yaw rate')

    def on_amcl(self,msg):
        p,q=msg.pose.pose.position,msg.pose.pose.orientation
        valid=all(math.isfinite(v) for v in (p.x,p.y,q.x,q.y,q.z,q.w)) and pose_quality(msg.pose.covariance,(p.x,p.y))
        self.note_sensor('amcl',msg,valid,'invalid map pose or covariance exceeds preset')

    def on_filtered(self,msg):
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        previous=self.health.samples.get('odom')
        if not math.isfinite(stamp) or previous and stamp<=previous['stamp']:
            return
        self.filtered=msg
        received=getattr(self,'_input_received_at',None)
        self.filtered_received=time.monotonic() if received is None else received
        twist=msg.twist.twist
        self.note_sensor('odom',msg,all(math.isfinite(v) for v in
            (twist.linear.x,twist.linear.y,twist.angular.z)),'invalid filtered velocity')

    def feedback_available(self):
        if self.sensor_ingress is not None:
            self.sensor_ingress.pull()
        return (super().feedback_available() and self.localization_ready
                and self.health.reason(self.clock_s,time.monotonic()) is None)

    def request_map_update(self,now):
        """A lost service response must not block all later laser updates."""
        if (self.nomotion_pending is not None and not self.nomotion_pending.done()
                and now>=self.next_nomotion):
            self.nomotion.remove_pending_request(self.nomotion_pending)
            self.nomotion_pending.cancel()
            self.nomotion_pending=None
            self.record('localization_update_timeout')
        if (now>=self.next_nomotion and self.nomotion.service_is_ready()
                and (self.nomotion_pending is None or self.nomotion_pending.done())):
            self.nomotion_pending=self.nomotion.call_async(Empty.Request())
            self.next_nomotion=now+1.

    def query_feedback(self):
        if self.sensor_ingress is not None:
            self.sensor_ingress.pull()
        # AMCL's normal motion threshold can leave stationary map localization
        # silent. Request periodic fresh laser updates without truth feedback.
        now=time.monotonic()
        self.request_map_update(now)
        if not self.localization_ready or self.filtered is None or self.clock_s is None:
            if self.localization_ready:
                self.note_feedback_unavailable(now,'waiting for clock or filtered odometry',
                    clock_s=self.clock_s,filtered_present=self.filtered is not None)
            return
        reason=self.health.reason(self.clock_s,now)
        if reason!=self.last_health:
            self.record('localization_health',reason=reason,sources=self.health.samples)
            self.last_health=reason
        if reason:
            self.note_feedback_unavailable(now,'sensor health',health=reason)
            if self.sim.execution.bound:
                self.sim.execution.fail('localization health: '+reason)
                self.stop_guard()
            return
        stamp_odom=self.filtered.header.stamp.sec+self.filtered.header.stamp.nanosec*1e-9
        if time.monotonic()-self.filtered_received>.2 or not -.1<=self.clock_s-stamp_odom<=.2:
            self.note_feedback_unavailable(now,'filtered odometry stale',clock_s=self.clock_s,
                source_s=stamp_odom,receive_age_s=now-self.filtered_received)
            return
        try:
            if self.sensor_ingress is not None:
                transform=self.input_transform
                if transform is None:
                    raise RuntimeError(self.input_transform_error)
            else:
                transform=self.tf.lookup_transform('map','base_footprint',rclpy.time.Time())
        except Exception as exc:
            self.note_feedback_unavailable(now,'map transform unavailable',error=str(exc),
                frames=self.tf.all_frames_as_yaml())
            return
        stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1e-9
        if not -.1<=self.clock_s-stamp<=.2:
            self.note_feedback_unavailable(now,'map transform stale',clock_s=self.clock_s,source_s=stamp)
            return
        if self.last_feedback_unavailable is not None:
            self.record('localized_feedback_ready',source_s=stamp)
            self.last_feedback_unavailable=None
        p,q=transform.transform.translation,transform.transform.rotation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        twist=self.filtered.twist.twist
        self.record('localization',x=p.x,y=p.y,yaw=yaw,stamp=stamp)
        refiner=getattr(self,'station_refiner',None)
        if refiner is not None:
            refiner.note_prior([p.x,p.y,yaw],stamp)
            station=refiner.station([p.x,p.y,yaw])
            if station is not None:
                scan=self.latest_scan
                # Body speed must belong to the same physical time neighbourhood.
                if scan is None or abs(stamp_odom-scan['source_s'])>.15:
                    if self._should_watchdog_fallback(now):
                        self._emit_watchdog_fallback(p.x,p.y,yaw,twist,stamp)
                    return
                fit=refiner.measure(scan,self.laser_extrinsic,self.clock_s,now,
                                    self.sim.execution.last_timestamp,
                                    fit_worker=getattr(self,'station_fit_worker',None))
                if fit['valid'] and abs(stamp_odom-fit['source_s'])>.15:
                    fit.update(valid=False,reason='completed station fit misaligned with measured velocity')
                if not fit.get('duplicate'):
                    self.record('station_measurement',station=station,**fit)
                if not fit['valid']:
                    if self._should_watchdog_fallback(now):
                        self._emit_watchdog_fallback(p.x,p.y,yaw,twist,stamp)
                    return
                x,y,angle=fit['pose']
                self.fine_pose_source=fit['source_s']
                self.fine_pose_received=fit['received_wall_s']
                self.process_observation(x,y,angle,twist.linear.x,twist.linear.y,twist.angular.z,
                                         fit['source_s'],self.sim.execution.envelope())
                return
            self.fine_pose_source=self.fine_pose_received=None
        self.process_observation(p.x,p.y,yaw,twist.linear.x,twist.linear.y,twist.angular.z,
                                 stamp,self.sim.execution.envelope())

    def _should_watchdog_fallback(self, now):
        """Maintain keepalive feedback when an idle vehicle or slow station fit risks watchdog timeout."""
        exec_view = getattr(self.sim, 'execution', None)
        if exec_view is None:
            return False
        if getattr(self, 'dock', None) is not None:
            return False
        state = getattr(exec_view, 'state', None)
        command = getattr(exec_view, 'command', None)
        active = getattr(self, 'active', None)
        if state in ('idle', 'arrived') and command is None and active is None:
            return True
        last_receipt = getattr(exec_view, 'last_receipt', None)
        if last_receipt is not None and now - last_receipt >= 1.0:
            return True
        return False

    def _emit_watchdog_fallback(self, px, py, yaw, twist, stamp):
        self.fine_pose_source = self.fine_pose_received = None
        self.process_observation(px, py, yaw, twist.linear.x, twist.linear.y, twist.angular.z,
                                 stamp, self.sim.execution.envelope())

    def docking_measurement_time(self,now):
        """Only a fresh accepted fine pose may regulate or accumulate settle time."""
        if getattr(self,'station_refiner',None) is None:
            return now
        gate=self.sim.execution
        if (not self.station_refiner.fresh(self.clock_s,now) or gate.pose is None
                or gate.pose['timestamp']!=self.fine_pose_source
                or self.fine_pose_source!=self.station_refiner.latest['source_s']):
            return None
        return self.fine_pose_source

    def close_station_refinement(self):
        """Close the numerical peer before closing the owner's evidence log."""
        if getattr(self,'station_fit_worker',None) is not None:
            summary=self.station_fit_worker.close()
            self.record('station_fit_worker_closed',**summary)
            if not summary['passed']:
                raise RuntimeError('station numerical peer did not close cleanly')

    def close(self):
        try:
            return super().close()
        finally:
            self.close_station_refinement()

    def note_feedback_unavailable(self,now,reason,**details):
        """Expose rejected inputs without changing freshness or motion gates."""
        if reason!=self.last_feedback_unavailable or now>=self.next_feedback_diagnostic:
            if reason=='map transform stale' and self.sensor_ingress is not None:
                details['tf_edges']=self.sensor_ingress.transform_diagnostic()
            self.record('localized_feedback_unavailable',reason=reason,**details)
            self.last_feedback_unavailable=reason
            self.next_feedback_diagnostic=now+1.
