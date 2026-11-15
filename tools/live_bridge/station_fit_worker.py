"""Bounded numerical peer; only the owning scheduler admits its measurements."""
from copy import deepcopy
import multiprocessing
import queue
import threading
import time


def _fit_peer(connection, descriptor):
    """Receive numerical inputs only; no ROS, motion, task or lease API."""
    try:
        try:
            from station_refinement import StationRefiner
        except ImportError:
            from tools.live_bridge.station_refinement import StationRefiner
        matcher = StationRefiner(descriptor).matcher
        connection.send(dict(kind='ready'))
        while True:
            request = connection.recv()
            if request is None:
                connection.send(dict(kind='closed'))
                return
            token, inputs = request
            fit = matcher.fit(*inputs)
            connection.send(dict(kind='fit', token=token, fit=fit))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(dict(kind='error', reason=type(exc).__name__+': '+str(exc)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class StationFitWorker:
    """One in-flight fit; the owner never waits on a process pipe or a solver."""

    def __init__(self, descriptor, *, _target=_fit_peer):
        self.owner_thread = threading.get_ident()
        self.pending = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.ready = False
        self.busy = False
        self.closed = False
        self.error = None
        self.sequence = 0
        context = multiprocessing.get_context('spawn')
        self.connection, peer = context.Pipe(duplex=True)
        self.process = context.Process(target=_target, args=(peer, deepcopy(descriptor)))
        self.process.start()
        peer.close()
        self.transport = threading.Thread(target=self._transfer, name='station_fit_transport')
        self.transport.start()

    def _transfer(self):
        """All potentially blocking serialization and pipe reads stay off owner."""
        try:
            self.results.put_nowait(self.connection.recv())
            while not self.stop.is_set():
                try:
                    request = self.pending.get(timeout=.02)
                except queue.Empty:
                    continue
                self.connection.send(request)
                answer = self.connection.recv()
                self.results.put_nowait(answer)
                if request is None:
                    break
        except (EOFError, BrokenPipeError, OSError, queue.Full) as exc:
            if not self.stop.is_set():
                self.error = type(exc).__name__+': '+str(exc)

    def _check_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('only the decision owner may consume station fits')
        if self.closed:
            raise RuntimeError('station fit worker already closed')
        if self.error or not self.process.is_alive() or not self.transport.is_alive():
            raise RuntimeError('station fit worker failed: '+str(self.error or 'peer stopped'))

    def take(self):
        self._check_owner()
        try:
            answer = self.results.get_nowait()
        except queue.Empty:
            return None
        if answer['kind'] == 'ready':
            self.ready = True
            return None
        if answer['kind'] != 'fit' or not self.busy or answer.get('token') != self.sequence:
            self.error = 'unexpected numerical reply: '+str(answer.get('kind'))
            raise RuntimeError(self.error)
        self.busy = False
        return answer

    def submit(self, inputs):
        self._check_owner()
        if not self.ready or self.busy:
            return False
        self.sequence += 1
        self.pending.put_nowait((self.sequence, deepcopy(inputs)))
        self.busy = True
        return True

    def close(self):
        """Bounded resource cleanup never upgrades a failed run to acceptance."""
        if self.closed:
            return self.close_result
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('only the decision owner may close station fits')
        # A busy fit has at most one result; the peer then waits for this sentinel.
        try:
            self.pending.put(None,timeout=.1)
        except queue.Full:
            self.error=self.error or 'numerical request queue did not drain during close'
        self.transport.join(timeout=2.)
        self.stop.set()
        self.process.join(timeout=.5)
        terminated = self.process.is_alive()
        if terminated:
            self.process.terminate(); self.process.join(timeout=1.)
        self.connection.close()
        self.transport.join(timeout=1.)
        self.closed = True
        self.close_result = dict(passed=not terminated and not self.error
            and not self.transport.is_alive() and self.process.exitcode == 0,
            process_id=self.process.pid, process_exitcode=self.process.exitcode,
            transport_stopped=not self.transport.is_alive(), terminated=terminated,
            error=self.error, fits_submitted=self.sequence)
        return self.close_result
