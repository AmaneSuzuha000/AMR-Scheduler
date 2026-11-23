"""Process boundary for independent fleet/process ROS execution contexts."""
import json
import math
import os
from pathlib import Path
import time
from multiprocessing.managers import BaseManager

from .material_supply import MaterialSupply


class SupplyManager(BaseManager):pass


SupplyManager.register('MaterialSupply',MaterialSupply)


class SharedSupply:
    def __init__(self,proxy):self.proxy=proxy

    def __getattr__(self,name):
        if name in {'initial','stock','capacity','events','deliveries','consumed','waiting','destinations'}:
            return self.proxy.shared_state()[name]
        return getattr(self.proxy,name)

    def __getstate__(self):return self.proxy

    def __setstate__(self,state):self.proxy=state


class ServerIdentity:
    """Read-only liveness of the exact server owned by the parent runner."""
    def __init__(self,pid):self.pid=pid

    def poll(self):
        try:
            os.kill(self.pid,0)
            status=Path('/proc')/str(self.pid)/'stat'
            if status.exists() and status.read_text().split(') ',1)[1].startswith('Z'):return 1
        except ProcessLookupError:return 1
        return None


def run_child(output,supply,case,server_pid,stop_event,metadata,result_queue):
    try:
        from .whole_line_process import run_worker
        run_worker(output,supply,case,ServerIdentity(server_pid),stop_event,metadata,
                   supply_mode=metadata.get('supply_mode','legacy_handoff'))
        # Keep the IPC message small; raw evidence and the exact report are
        # already durably written before the parent observes completion.
        result_queue.put(dict(report_file=str(Path(output)/'result.json')))
    except Exception as exc:result_queue.put(dict(error=repr(exc)))


def receive_results(result_queue,result):
    from queue import Empty
    while True:
        try:message=result_queue.get_nowait()
        except Empty:return
        if 'report_file' in message:result['report']=json.loads(Path(message['report_file']).read_text())
        else:result.update(message)


def settle_evidence_writer(process, output, *, timeout_s=120.):
    """After physical shutdown, finish or explicitly fail the owned evidence writer.

    This deadline is for file serialization, not physical stop confirmation.
    A failed child report remains failed even when all files are immutable.
    """
    if timeout_s < 0 or not math.isfinite(timeout_s):
        raise ValueError('invalid evidence writer timeout')
    began = time.monotonic()
    process.join(timeout=timeout_s)
    timed_out = process.is_alive()
    forced = []
    if timed_out:
        # Only the parent's exact child, after its physical world has ended.
        # Keep partial files; an interrupted writer never qualifies as complete.
        process.terminate(); forced.append('terminate'); process.join(timeout=5.)
        if process.is_alive():
            process.kill(); forced.append('kill'); process.join(timeout=5.)
    exited = not process.is_alive()
    path = Path(output)/'result.json'
    child = None; error = None
    if exited and not timed_out:
        try:
            child = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(child, dict) or not isinstance(child.get('passed'), bool):
                raise ValueError('invalid child result schema')
        except (OSError, ValueError) as exc:
            error = repr(exc); child = None
    return dict(evidence_complete=exited and not timed_out and process.exitcode == 0 and child is not None and error is None,
        writer_exited=exited, writer_exitcode=process.exitcode, timed_out=timed_out,
        forced_actions=forced, child_report_file=str(path), child_report_read_error=error,
        child_physical_passed=None if child is None else child.get('passed'),
        elapsed_wall_s=time.monotonic()-began,
        claim_boundary='Serialization closure only; does not confirm physical stop or repair a failed run.')
