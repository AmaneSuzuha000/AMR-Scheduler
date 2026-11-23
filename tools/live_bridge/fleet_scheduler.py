"""One durable Python fleet scheduler, with one sensor/native adapter per robot.

The views below never own jobs, reservations or a simulation clock. Only the
fleet owner steps the shared Simulator; adapters translate that owner's MOVE.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from amr_sched.checkpoint import CheckpointStore,capture,definition,restore
from amr_sched.fleet_control import FleetControlServer
from amr_sched.stream import FrameBuilder
from tools.wheelset_scene.navigation_records import BufferedTextLog
from sensor_scheduler import SensorSchedulerNode


class RobotExecutionView:
    """Delegate to the same simulator while selecting one robot's feedback gate."""

    def __init__(self,sim,robot_id):
        if sim.execution_for(robot_id) is None:
            raise ValueError('unknown physical robot')
        self._sim=sim
        self.robot_id=robot_id

    @property
    def execution(self):
        return self._sim.execution_for(self.robot_id)

    def __getattr__(self,name):
        return getattr(self._sim,name)

    def step_once(self):
        raise RuntimeError('only the fleet owner advances the shared scheduler')

    def reset(self):
        raise RuntimeError('robot adapters cannot reset fleet occupancy')

    def resume_execution(self,route_id,now):
        return self._sim.resume_execution(route_id,now,self.robot_id)

    def replan_execution(self,route_id,now):
        return self._sim.replan_execution(route_id,now,self.robot_id)

    def preempt_execution(self,route_id,now):
        return self._sim.preempt_execution(route_id,now,self.robot_id)


class FleetRobotNode(SensorSchedulerNode):
    """Reuse the tested single-robot action/guard state machine without stepping."""

    def __init__(self,owner,robot_id,namespace,log_path,*,defer_input_start=False):
        self.fleet_owner=owner
        super().__init__(RobotExecutionView(owner.sim,robot_id),log_path,ros_namespace=namespace,
                         tracking_offset_m=owner.tracking_offset_m,max_tracking_offset_m=owner.max_tracking_offset_m,
                         latest_tf=owner.latest_tf,
                         best_effort_tf=owner.best_effort_tf,
                         input_thread=owner.input_thread,dock_takeover=owner.dock_takeover,
                         async_logs=owner.async_logs,defer_input_start=defer_input_start)
        self.builder=owner.builder

    def persist(self):
        return self.fleet_owner.persist()

    def build_path(self, *args, **kwargs):
        msg = super().build_path(*args, **kwargs)
        if not self.fleet_owner.constructing:
            self.fleet_owner._refresh_leases()
        return msg

    def tick(self):
        if self.closed:
            return
        if self.sensor_ingress is not None:
            self.sensor_ingress.pull()
        if self.fleet_owner.closing:
            # Shutdown pumps observations to prove a real stop, never recovery,
            # motion authorization, task dispatch or lease renewal.
            self.stop_guard()
            self.query_feedback()
            return
        gate=self.sim.execution
        if gate.fault:
            self.stop_guard()
        if self.guard_authorized and not self.guard_fresh():
            gate.fail('execution guard feedback timeout')
            self.stop_guard()
        self.reconcile_native()
        if self.active is not None and not gate.fault:
            self.set_speed(self.active)
        self.query_feedback()
        self.dock_step(time.monotonic())
        if self.requested_recovery and gate.stopped:
            if self.resume_execution(self.requested_recovery)['accepted']:
                self.requested_recovery=None
        commands=gate.drain_commands()
        # The owner already checkpoints all robots every scheduling tick.
        # Keep the additional write before each command batch, without a
        # redundant whole-fleet fsync for an adapter with nothing to dispatch.
        if commands:
            durable=self.persist()
            for command in commands:
                if durable or command['kind']=='CANCEL':
                    self.dispatch(command)
            self.fleet_owner._refresh_leases()
        self.authorize_waiting_move()
        self.publish_lease()

    def close(self):
        """The owner already pumps every robot together until stopping is proved."""
        gate=self.sim.execution
        clean=self.clean_idle_shutdown_ready()
        physical_stop=(self.guard_fresh() and self.guard_status.state=='latched'
                       and self.guard_status.stopped and self.guard_status.actual_speed<.01
                       and abs(self.guard_status.actual_wz)<.02
                       and self.guard_status.output_vx==0 and self.guard_status.output_wz==0)
        self.record('shutdown',stopped=bool(physical_stop),clean_idle=clean,
                    execution=gate.snapshot(),pins=self.sim._res.pins())
        try:
            if self.sensor_ingress is not None:
                self.sensor_ingress.close()
        finally:
            self.close_station_refinement()
        self.timer.cancel()
        self.closed=True
        for connection in self.display_clients:
            connection.close()
        if self.listener:
            self.listener.close()
        self.log_file.close()
        self.store.close()


class FleetRuntime:
    """Single executor, shared occupancy and checkpoint, separate native UUIDs."""

    def __init__(self,sim,out,namespaces,executor,*,resume=False,runtime_files=(),tracking_offset_m=0.,max_tracking_offset_m=0.315,latest_tf=False,best_effort_tf=False,input_thread=False,dock_takeover=None,async_logs=False,cvcr_coordinator=None,costmap_projector=None,defer_ingress=False):
        if sim.execution is not None or len(sim.executions)<2:
            raise ValueError('a fleet mapping with at least two physical robots is required')
        if set(namespaces)!=set(sim.executions) or len(set(namespaces.values()))!=len(namespaces):
            raise ValueError('one unique ROS namespace per physical robot required')
        if (any(node.capacity != 1 for node in sim.network.nodes.values())
                or any(edge.capacity != 1 for edge in sim.network.edges.values())):
            raise ValueError('single-point native fleet execution requires capacity 1; '
                             'explicitly use point_execution_network for the physical configuration')
        self.sim=sim
        self.out=Path(out)
        self.out.mkdir(parents=True,exist_ok=True)
        self.executor=executor
        self.namespaces=dict(namespaces)
        self.tracking_offset_m=tracking_offset_m
        self.max_tracking_offset_m=max_tracking_offset_m
        self.latest_tf=latest_tf
        self.best_effort_tf=best_effort_tf
        self.input_thread=input_thread
        self.dock_takeover=dock_takeover
        if type(async_logs) is not bool:
            raise ValueError('explicit boolean asynchronous log mode required')
        self.async_logs=async_logs
        self.cvcr_coordinator=cvcr_coordinator
        self.costmap_projector=costmap_projector
        self._latest_obstacle_points=[]
        self._latest_occupancy_grid=None
        self._latest_laserscan=None
        self.nodes={}
        self.builder=FrameBuilder(sim)
        self.failed=False
        self.constructing=True
        self.ready=False
        self.closing=False
        self.done=False
        self.epoch=None
        self.control_server=None
        self.restored=resume
        files=[Path(__file__),*[Path(__file__).with_name(name) for name in
               ('sensor_scheduler.py','nav2_scheduler.py','nav2_route_follower.py','gz_frame.py',
                'sensor_health.py','sensor_localization.yaml','nav2_metric.yaml','nav2_metric_dwb.yaml','feedback_ingress.py',
                'fleet_config.py','nav2_path.py','terminal_docking.py')],
               *runtime_files]
        self.fingerprint=hashlib.sha256((definition(sim,files)+json.dumps(dict(namespaces=namespaces,
            tracking_offset_m=tracking_offset_m,latest_tf=latest_tf,best_effort_tf=best_effort_tf,input_thread=input_thread,
            dock_takeover=dock_takeover,async_logs=async_logs),sort_keys=True)).encode()).hexdigest()
        self.store=CheckpointStore(self.out/'fleet.checkpoint.json')
        try:
            if resume:
                self.previous=restore(sim,self.store.load(),self.fingerprint)
            else:
                if self.store.path.exists():
                    raise RuntimeError('checkpoint exists; explicit resume or a new output directory required')
                self.previous={}
            # New event log per restart, never truncate the previous one.
            attempt=len(list(self.out.glob('fleet-events-*.jsonl')))
            if async_logs:
                from tools.wheelset_scene.navigation_records import ProcessTextLog
                self.log_file=ProcessTextLog(self.out/f'fleet-events-{attempt:03d}.jsonl')
            else:
                self.log_file=BufferedTextLog(self.out/f'fleet-events-{attempt:03d}.jsonl')
            self.attempt=attempt
            for robot,namespace in namespaces.items():
                node=FleetRobotNode(self,robot,namespace,self.out/f'{robot}-{attempt:03d}.jsonl',defer_input_start=True)
                self.nodes[robot]=node
                for prev in self.nodes.values():
                    if prev.sensor_ingress is not None and prev.sensor_ingress.worker.thread is not None:
                        prev.sensor_ingress.pull()
                context=self.previous.get(robot)
                if context:
                    for key in ('guard_id','permit','native_goal_id','native_state','seq'):
                        setattr(node,key,context[key])
                    node.native_pending=node.native_state in {'prepared','active','unknown'}
                    node.reconciling=node.native_pending
                    if node.native_pending and not node.native_goal_id:
                        raise ValueError('each unknown native fleet action requires its exact UUID')
                    node.restart_clock=context['clock_s']
                    node.restart_first_clock=True
                executor.add_node(node)
            if self.cvcr_coordinator is not None and hasattr(self.cvcr_coordinator, 'set_recovery_backends'):
                self.cvcr_coordinator.set_recovery_backends(self.nodes)
            self.constructing=False
            self.timer=next(iter(self.nodes.values())).create_timer(.05,self.tick)
            self.persist()
            if not defer_ingress:
                self.start_ingress()
        except Exception:
            # No scheduling or authorization has started during construction.
            for node in self.nodes.values():
                if node.sensor_ingress is not None:
                    node.sensor_ingress.close()
                node.timer.cancel()
                executor.remove_node(node)
                node.log_file.close()
                node.store.close()
                node.destroy_node()
            if hasattr(self,'log_file'):
                self.log_file.close()
            self.store.close()
            raise

    def record(self,kind,**values):
        self.log_file.write(json.dumps(dict(kind=kind,wall=time.monotonic(),**values),allow_nan=False)+'\n')

    def persist(self):
        # The base adapter calls persist during its own construction. During a
        # fleet restart, overwriting the checkpoint before every saved native
        # UUID is restored would lose reconciliation data if startup crashes.
        if self.failed or self.constructing:
            return False
        contexts={robot:dict(guard_id=node.guard_id,permit=node.permit,
            native_goal_id=node.native_goal_id,native_state=node.native_state,
            clock_s=node.clock_s,seq=node.seq) for robot,node in self.nodes.items()}
        if not self.constructing:
            self._refresh_leases()
        try:
            self.store.save(capture(self.sim,self.fingerprint,contexts))
            if not self.constructing:
                self._refresh_leases()
            return True
        except Exception as exc:
            self.failed=True
            for gate in self.sim.executions.values():
                gate.fail('durable fleet checkpoint failed: '+str(exc))
            for node in self.nodes.values():
                node.persistence_failed=True
                node.stop_guard()
            if hasattr(self,'log_file'):
                self.record('checkpoint_failed',reason=str(exc))
            return False

    def enable(self):
        """Initialization is fixed by the harness; runtime never reads world truth."""
        self.ready=True
        for node in self.nodes.values():
            node.localization_ready=True

    def start_owner_pump(self):
        """Run fleet/adapter turns explicitly on the existing decision thread.

        A harness that spins one callback at a time can keep receiving input
        while timer callbacks wait behind unrelated ready subscriptions. The
        input-thread harness therefore owns these turns directly; it does not
        add a lease worker or give a receiver scheduling authority.
        """
        import threading
        if getattr(self, 'owner_pump_thread', None) is not None:
            raise RuntimeError('fleet owner pump already started')
        self.owner_pump_thread = threading.get_ident()
        self.owner_pump_due = 0.
        self.timer.cancel()
        for node in self.nodes.values():
            node.timer.cancel()
        self.record('owner_pump_started', period_s=.05)

    def _refresh_leases(self):
        if not self.failed and not self.closing:
            for node in self.nodes.values():
                pub = getattr(node, 'publish_lease', None)
                if pub is not None:
                    pub()

    def _drain_inputs(self, budget_s: float = 0.12, limit: int = 2048, max_passes: int = 10):
        if getattr(self, 'input_thread', False) or getattr(self, 'executor', None) is None:
            return
        from rclpy.executors import ShutdownException, TimeoutException
        dl = time.perf_counter() + budget_s
        total = 0
        for _ in range(max_passes):
            if total >= limit or time.perf_counter() >= dl:
                break
            if hasattr(self.executor, '_cb_iter'):
                self.executor._cb_iter = None
            yielded = 0
            while total < limit and time.perf_counter() < dl:
                try:
                    handler, _, _ = self.executor.wait_for_ready_callbacks(timeout_sec=0.0)
                except (TimeoutException, ShutdownException):
                    break
                yielded += 1
                total += 1
                handler()
                if handler.exception() is not None:
                    raise handler.exception()
            if yielded == 0:
                break

    def pump_owner(self):
        """One due owner turn, with durable commands and original safety gates."""
        import threading
        if threading.get_ident() != self.owner_pump_thread:
            raise RuntimeError('only the decision thread may pump the fleet')
        now = time.monotonic()
        if now < self.owner_pump_due:
            return False
        # No replay burst of old control turns after a delayed callback.
        self.owner_pump_due = now+.05
        t_start = time.perf_counter()
        t_cpu_start = time.thread_time()
        p_cpu_start = time.process_time()
        self.tick()
        t_after_tick = time.perf_counter()
        if t_after_tick - t_start >= 0.015:
            self._drain_inputs()
            self._refresh_leases()
        max_node_s = 0.0
        max_node_id = None
        for rid, node in self.nodes.items():
            tn0 = time.perf_counter()
            node.tick()
            dt_n = time.perf_counter() - tn0
            if dt_n >= 0.015:
                self._drain_inputs()
                self._refresh_leases()
            if dt_n > max_node_s:
                max_node_s = dt_n
                max_node_id = rid
        t_end = time.perf_counter()
        t_cpu_end = time.thread_time()
        p_cpu_end = time.process_time()
        total_s = t_end - t_start
        if total_s >= 0.04:
            self.record(
                'slow_pump',
                total_ms=round(total_s * 1000.0, 2),
                thread_cpu_ms=round((t_cpu_end - t_cpu_start) * 1000.0, 2),
                proc_cpu_ms=round((p_cpu_end - p_cpu_start) * 1000.0, 2),
                tick_ms=round((t_after_tick - t_start) * 1000.0, 2),
                nodes_ms=round((t_end - t_after_tick) * 1000.0, 2),
                max_node=max_node_id,
                max_node_ms=round(max_node_s * 1000.0, 2),
                phases=getattr(self, '_last_tick_phases', None),
            )
        return True

    def tick(self):
        t0 = time.perf_counter()
        if getattr(self,'input_thread',False):
            # Consume already received observations before the original timeout
            # check. Reception never advances task, inventory or reservations.
            for node in self.nodes.values():
                node.query_feedback()
        t_fb = time.perf_counter()
        now=time.monotonic()
        self.sim.poll_execution(now)
        if any(node.persistence_failed for node in self.nodes.values()) and not self.failed:
            self.failed=True
            for gate in self.sim.executions.values():
                gate.fail('fleet native identity or persistence invalidated')
        if self.failed:
            for node in self.nodes.values():
                node.persistence_failed=True
                node.stop_guard()
        self._refresh_leases()
        clocks=[node.clock_s for node in self.nodes.values()]
        clock=min(clocks) if all(t is not None for t in clocks) else None
        if getattr(self.sim,'_operator_paused',False) and self.epoch is not None and clock is not None:
            # Keep scheduling time frozen without accumulating a catch-up burst.
            self.epoch=clock-self.sim._step_cursor*self.sim.config.dt
        coord_wall_ms = 0.0
        if (self.ready and not self.closing and not self.failed and clock is not None
                and not getattr(self.sim,'_operator_paused',False)):
            if self.epoch is None and all(g.bound for g in self.sim.executions.values()):
                self.epoch=clock-(self.sim._step_cursor*self.sim.config.dt if self.restored else 0.)
            if self.epoch is not None:
                due=int((clock-self.epoch)/self.sim.config.dt)
                coord_ran=False
                for _ in range(min(5,max(0,due-self.sim._step_cursor+1))):
                    if self.cvcr_coordinator is not None and not coord_ran:
                        coord_ran=True
                        tc0 = time.perf_counter()
                        telem={}
                        for rid,node in self.nodes.items():
                            gate=self.sim.execution_for(rid)
                            age=(max(0., now-gate.last_receipt) if gate and gate.last_receipt is not None else 0.)
                            gs=getattr(node,'guard_status',None)
                            spd=float(getattr(gs,'actual_speed',0.))/max(1e-3,float(self.sim.planner.max_speed)) if gs else 1.
                            telem[rid]={
                                'localization_source':'amcl',
                                'speed_scale':max(0.05,min(2.0,spd)) if (gate and gate.command) else 1.0,
                                'receipt_age_s':age,
                                'local_planner_state':('faulted' if (gate and gate.fault) else ('detouring' if rid in self.sim._physical_detours else 'nominal')),
                            }
                        blocked_corridors=None
                        if self.costmap_projector is not None:
                            rpos={r:(float(n.sim.execution.pose['x']),float(n.sim.execution.pose['y']))
                                  for r,n in self.nodes.items() if n.sim.execution.pose is not None}
                            if self._latest_occupancy_grid is not None:
                                origin_xy, resolution, width, height, grid_data, occ_th, ch_steps = self._latest_occupancy_grid
                                blocked_corridors=self.costmap_projector.project_occupancy_grid(
                                    origin_xy, resolution, width, height, grid_data,
                                    occ_threshold=occ_th, robot_positions=rpos, clear_horizon_steps=ch_steps,
                                )
                            elif self._latest_laserscan is not None:
                                pose_xyyaw, ranges, ang_min, ang_inc, rmin, rmax, ch_steps = self._latest_laserscan
                                blocked_corridors=self.costmap_projector.project_laserscan(
                                    pose_xyyaw, ranges, ang_min, ang_inc,
                                    range_min=rmin, range_max=rmax, robot_positions=rpos, clear_horizon_steps=ch_steps,
                                )
                            else:
                                blocked_corridors=self.costmap_projector.project_points(
                                    self._latest_obstacle_points,robot_positions=rpos)
                        tc_cpu0 = time.thread_time()
                        coord_rep=self.cvcr_coordinator.step_coordination(
                            now,nav2_telemetry=telem,blocked_corridors=blocked_corridors)
                        coord_wall_ms = round((time.perf_counter() - tc0) * 1000.0, 2)
                        coord_cpu_ms = round((time.thread_time() - tc_cpu0) * 1000.0, 2)
                        if coord_rep.triggered or coord_rep.replanned_detours or coord_rep.closed_corridors or coord_rep.reopened_corridors:
                            self.record('cvcr_step', tick=coord_rep.tick, trigger=coord_rep.trigger_reason,
                                        accepted=coord_rep.receipt_accepted, reassigned=list(coord_rep.reassigned_tasks),
                                        detours=list(coord_rep.replanned_detours),
                                        closed=[list(e) for e in coord_rep.closed_corridors],
                                        reopened=[list(e) for e in coord_rep.reopened_corridors],
                                        propose_ms=coord_rep.propose_ms, coord_wall_ms=coord_wall_ms,
                                        coord_cpu_ms=coord_cpu_ms)
                            self._refresh_leases()
                    if not self.sim.step_once():
                        self.done=True
                        if not self.sim._all_done():
                            for gate in self.sim.executions.values():
                                gate.fail('scheduler horizon exhausted')
                        break
        t_sim = time.perf_counter()
        # Refresh active motion leases before synchronous checkpoint persistence so I/O stalls cannot starve the guard watchdog.
        self._refresh_leases()
        # Scheduler mutations are durable before any adapter sends a native goal.
        self.persist()
        t_persist = time.perf_counter()
        frame=self.builder.frame()
        for agv in frame['agvs']:
            agv['pose']['available']=self.nodes[agv['agv_id']].feedback_available()
        if getattr(self,'control_server',None) is not None:
            frame['control']=self.control_server.control.state()
        self.record('frame',sim=clock,value=frame)
        t_rec = time.perf_counter()
        if getattr(self,'control_server',None) is not None:
            self.control_server.poll(frame)
        self._refresh_leases()
        t_poll = time.perf_counter()
        self._last_tick_phases = dict(
            fb_ms=round((t_fb - t0) * 1000.0, 2),
            coord_ms=coord_wall_ms,
            sim_ms=round((t_sim - t_fb) * 1000.0, 2),
            persist_ms=round((t_persist - t_sim) * 1000.0, 2),
            frame_rec_ms=round((t_rec - t_persist) * 1000.0, 2),
            poll_ms=round((t_poll - t_rec) * 1000.0, 2),
        )

    def start_control(self,port=0,host='127.0.0.1'):
        """Explicit TCP endpoint; loopback by default unless an external bridge specifies host."""
        if self.control_server is not None or self.closing or self.failed:
            raise RuntimeError('one active control endpoint per durable owner required')
        self.control_server=FleetControlServer(self,port,host=host)
        self.record('operator_endpoint',port=self.control_server.port,
                    control=self.control_server.control.state(),
                    hello={**self.builder.hello(),'control':self.control_server.control.state()})
        return self.control_server.port

    def pause_fleet(self,routes):
        """Latch every robot and keep all tasks, handling steps and physical pins."""
        if (self.failed or self.closing or not self.ready
                or routes!={r:g.recovery_token() for r,g in self.sim.executions.items()}):
            return dict(accepted=False,reason='active owner and current fleet versions required')
        self.sim._operator_paused=True
        for gate in self.sim.executions.values():
            gate.fail('operator fleet pause')
        for node in self.nodes.values():
            node.stop_guard()
        durable=self.persist()
        answer=dict(accepted=durable,action='pause_requested',
                    reason=None if durable else 'durable pause failed; fleet locked')
        self.record('operator_pause',result=answer)
        return answer

    def resume_fleet(self,routes):
        """Preflight the complete fleet before committing any resumed movement."""
        if (self.failed or self.closing or not self.ready or not self.sim._operator_paused
                or routes!={r:g.recovery_token() for r,g in self.sim.executions.items()}):
            return dict(accepted=False,reason='paused owner and current fleet versions required')
        for robot,node in self.nodes.items():
            gate=self.sim.executions[robot]
            if (gate.fault not in {'operator fleet pause','runtime restarted; reconciliation required'}
                    or node.persistence_failed or node.native_pending or node.goal_handle is not None
                    or node.guard_pending is not None or not node.guard_fresh()
                    or node.guard_status.state!='latched' or not node.guard_status.stopped
                    or not node.feedback_available()):
                return dict(accepted=False,reason='every native action must terminate with fresh stopped feedback',
                            robot_id=robot)
        now=time.monotonic()
        # The real simulator checks retained geometry, roads, tasks and pins.
        # Trial mutations stay on this temporary copy; no second clock is run.
        trial=deepcopy(self.sim)
        for robot in self.nodes:
            answer=trial.resume_execution(routes[robot],now,robot)
            if not answer['accepted']:
                return dict(accepted=False,reason=answer['reason'],robot_id=robot)
        saved={r:deepcopy(vars(g)) for r,g in self.sim.executions.items()}
        states={a.agv_id:a.state for a in self.sim.agvs}
        for robot in self.nodes:
            answer=self.sim.resume_execution(routes[robot],now,robot)
            if not answer['accepted']:
                for r,gate in self.sim.executions.items():
                    gate.__dict__.clear()
                    gate.__dict__.update(saved[r])
                for agv in self.sim.agvs:
                    agv.state=states[agv.agv_id]
                return dict(accepted=False,reason=answer['reason'],robot_id=robot)
        self.sim._operator_paused=False
        for node in self.nodes.values():
            node.active=node.waiting_move=None
            node.result_status=None
            node.cancel_requested=False
            node.stop_since=None
        durable=self.persist()
        if not durable:
            self.sim._operator_paused=True
        answer=dict(accepted=durable,action='fleet_resumed',
                    reason=None if durable else 'durable resume failed; fleet locked')
        self.record('operator_resume',result=answer)
        return answer

    def resume_execution(self,robot,route_id):
        if robot not in self.nodes:
            return dict(accepted=False,reason='unknown physical robot')
        return self.nodes[robot].resume_execution(route_id)

    def request_physical_detour(self,robot,route_id,blocked_edge):
        answer=self.sim.request_physical_detour(robot,route_id,blocked_edge,time.monotonic())
        if answer['accepted'] and not self.persist():
            answer=dict(accepted=False,reason='durable road closure write failed; fleet locked')
        self.record('physical_detour_request',robot=robot,result=answer)
        return answer

    def replan_execution(self,robot,route_id):
        if robot not in self.nodes:
            return dict(accepted=False,reason='unknown physical robot')
        return self.nodes[robot].replan_execution(route_id)

    def reopen_physical_corridor(self,edge,revision):
        answer=self.sim.reopen_physical_corridor(edge,revision)
        if answer['accepted'] and not self.persist():
            answer=dict(accepted=False,reason='durable corridor clear write failed; fleet locked')
        self.record('physical_corridor_reopen',result=answer)
        return answer

    def submit_physical_task(self,task):
        if self.failed or self.closing or self.sim._step_cursor>=self.sim.config.max_steps:
            return dict(accepted=False,reason='active durable fleet owner and remaining horizon required')
        answer=self.sim.submit_physical_task(task)
        if answer['accepted'] and not self.persist():
            answer=dict(accepted=False,reason='durable dynamic task write failed; fleet locked')
        if answer['accepted']:
            self.done=False
        self.record('physical_task_submit',result=answer)
        return answer

    def submit_obstacle_points(self,points):
        """Update latest continuous-space obstacle `(x, y)` points for `costmap_projector`."""
        self._latest_occupancy_grid=None
        self._latest_laserscan=None
        self._latest_obstacle_points=[(float(p[0]),float(p[1])) for p in points]

    def submit_occupancy_grid(self, origin_xy, resolution, width, height, data, *, occ_threshold=65, clear_horizon_steps=None):
        """Register raw ROS 2 `nav_msgs/OccupancyGrid` data for zero-hardcoding `CostmapCorridorProjector`."""
        self._latest_occupancy_grid=(
            (float(origin_xy[0]), float(origin_xy[1])),
            float(resolution),
            int(width),
            int(height),
            list(data),
            int(occ_threshold),
            clear_horizon_steps,
        )

    def submit_laserscan(self, robot_pose_xyyaw, ranges, angle_min, angle_increment, *, range_min=0.12, range_max=12.0, clear_horizon_steps=None):
        """Register raw ROS 2 `sensor_msgs/LaserScan` polar ranges for zero-hardcoding `CostmapCorridorProjector`."""
        self._latest_laserscan=(
            (float(robot_pose_xyyaw[0]), float(robot_pose_xyyaw[1]), float(robot_pose_xyyaw[2])),
            list(ranges),
            float(angle_min),
            float(angle_increment),
            float(range_min),
            float(range_max),
            clear_horizon_steps,
        )

    def request_physical_preemption(self,urgent_task_id):
        answer=self.sim.request_physical_preemption(urgent_task_id,time.monotonic())
        if answer['accepted'] and not self.persist():
            answer=dict(accepted=False,reason='durable urgent request write failed; fleet locked')
        self.record('physical_preemption_request',result=answer)
        return answer

    def preempt_execution(self,robot,route_id):
        if robot not in self.nodes:
            return dict(accepted=False,reason='unknown physical robot')
        return self.nodes[robot].preempt_execution(route_id)

    def begin_close(self):
        """Latch motion before another owned input stream drains or waits."""
        if getattr(self,'_close_requested',False):
            return
        self.closing=True
        self._close_needs_stop=not all(node.clean_idle_shutdown_ready() for node in self.nodes.values())
        if self._close_needs_stop:
            for gate in self.sim.executions.values():
                gate.fail('fleet runtime shutdown')
            for node in self.nodes.values():
                node.suppress_feedback=False
                node.stop_guard()
        self._close_requested=True
        if self.control_server is not None:
            self.control_server.close()
            self.control_server=None
        for node in self.nodes.values():
            node.query_feedback()

    def start_ingress(self):
        for node in self.nodes.values():
            if node.sensor_ingress is not None:
                node.sensor_ingress.start()

    def close(self):
        self.begin_close()
        if self._close_needs_stop:
            deadline=time.monotonic()+4.
            while time.monotonic()<deadline and not all(g.stopped for g in self.sim.executions.values()):
                self.executor.spin_once(timeout_sec=.05)
                if getattr(self, 'owner_pump_thread', None) is not None:
                    self.pump_owner()
        self.persist()
        self.timer.cancel()
        for node in self.nodes.values():
            node.query_feedback()
            node.close()
            self.executor.remove_node(node)
            node.destroy_node()
        self.log_file.close()
        self.store.close()
