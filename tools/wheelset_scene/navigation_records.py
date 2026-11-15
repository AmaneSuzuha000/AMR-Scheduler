"""Re-iterable raw JSONL evidence without retaining full sensor arrays."""
from array import array
import json
from pathlib import Path
import queue
import threading
import time


class BufferedTextLog:
    """Batch ordinary event-log flushes; durable checkpoints use their own store.

    Flush on the next write after 50ms, 64 rows, 256KiB, or close. Accepted
    timestamps and ordering stay fixed at the caller. This buffer is not an
    acknowledgement of durable task state or contact evidence.
    """

    def __init__(self, path, mode='x'):
        self.stream = Path(path).open(mode, encoding='utf-8', buffering=256*1024)
        self.last_flush = time.monotonic()
        self.rows = 0; self.bytes = 0

    def write(self, text):
        result = self.stream.write(text)
        self.rows += 1; self.bytes += len(text.encode('utf-8'))
        if self.rows >= 64 or self.bytes >= 256*1024 or time.monotonic()-self.last_flush >= .05:
            self.flush()
        return result

    def flush(self):
        self.stream.flush()
        self.rows = self.bytes = 0; self.last_flush = time.monotonic()

    def close(self):
        self.stream.close()  # Includes the final flush; failure must propagate.

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def _write_text_rows(path, pending, answer):
    """Write immutable owner-formatted text; never execute fleet decisions."""
    count = 0
    try:
        with BufferedTextLog(path, 'a') as stream:
            while True:
                try:
                    row = pending.get(timeout=.05)
                except queue.Empty:
                    stream.flush()
                    continue
                if row is None:
                    break
                sequence, text = row
                if sequence != count+1 or type(text) is not str:
                    raise ValueError('text log sequence or payload mismatch')
                stream.write(text)
                count += 1
        answer.send(dict(passed=True, records=count))
    except Exception as exc:
        answer.send(dict(passed=False, records=count,
                         error=type(exc).__name__+': '+str(exc)))
    finally:
        answer.close()


class ProcessTextLog:
    """Bounded FIFO logging; durable task/command checkpoints remain synchronous."""

    def __init__(self, path, *, limit=256, _target=_write_text_rows):
        import multiprocessing
        if type(limit) is not int or limit < 1:
            raise ValueError('positive bounded text log capacity required')
        self.owner_thread = threading.get_ident()
        self.path = Path(path)
        # Preserve the same exclusive creation contract before starting a peer.
        with self.path.open('x', encoding='utf-8'):
            pass
        self.accepted = 0
        self.error = None
        self.closed = False
        self.close_result = None
        self.next_check_wall = 0.
        context = multiprocessing.get_context('spawn')
        self.pending = context.Queue(maxsize=limit)
        self.answer, child_answer = context.Pipe(duplex=False)
        self.process = context.Process(target=_target,
            args=(str(self.path), self.pending, child_answer),
            name='amr_text_log_writer', daemon=True)
        try:
            self.process.start()
        except BaseException:
            child_answer.close()
            self.answer.close()
            self.pending.close()
            raise
        child_answer.close()

    def _check_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('only the decision owner may submit text logs')

    def write(self, text):
        self._check_owner()
        if type(text) is not str:
            raise TypeError('immutable text log string required')
        now = time.monotonic()
        # Bounded liveness checks avoid one OS waitpid per sensor/pose record.
        # Queue overflow remains immediate; close proves the complete count.
        if not self.closed and now >= self.next_check_wall:
            self.next_check_wall = now+.05
            if not self.process.is_alive():
                self.error = self.error or 'text log process ended before close'
        if self.closed or self.error:
            raise RuntimeError('text log closed or failed: '+str(self.error))
        try:
            self.pending.put_nowait((self.accepted+1, text))
        except queue.Full:
            self.error = 'text log queue overflow'
            raise RuntimeError(self.error)
        self.accepted += 1
        return len(text)

    def close(self):
        self._check_owner()
        if self.closed:
            if self.close_result and self.close_result['passed']:
                return self.close_result
            raise RuntimeError('text log close already failed: '+str(self.close_result))
        self.closed = True
        try:
            self.pending.put(None, timeout=2.)
        except queue.Full:
            self.error = self.error or 'text log close queue full'
        self.process.join(timeout=3.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
            self.error = self.error or 'text log process did not close'
        try:
            result = self.answer.recv() if self.answer.poll() else None
        except (EOFError, OSError):
            result = None
        self.answer.close()
        self.pending.cancel_join_thread()
        self.pending.close()
        passed = (self.error is None and result is not None and result['passed']
                  and result['records'] == self.accepted
                  and self.process.exitcode == 0 and not self.process.is_alive())
        self.close_result = dict(passed=passed, accepted_records=self.accepted,
            records=None if result is None else result['records'],
            error=self.error or (None if result is None else result.get('error')),
            process_id=self.process.pid, process_exitcode=self.process.exitcode,
            writer_exited=not self.process.is_alive())
        report_path=self.path.with_suffix(self.path.suffix+'.writer.json')
        with report_path.open('x',encoding='utf-8') as stream:
            json.dump(self.close_result,stream,indent=2)
            stream.write('\n')
        if not passed:
            raise RuntimeError('text log incomplete: '+str(self.close_result))
        return self.close_result


def freeze_contact_message(row, topic, received):
    """Freeze an internal ROS contact once, retaining its original small header."""
    import pickle
    if not topic.endswith('/bin_contacts'):
        raise ValueError('contact topic required for local frozen representation')
    value=dict(row,_topic=topic,_received_wall=received)
    return dict(header=plain_ros_value(value['header']),_topic=topic,
                _received_wall=received,
                _local_contact_payload=pickle.dumps(value,protocol=5))


def restore_contact_message(packet):
    """Restore only the local input process's bytes, checking all causal metadata."""
    import pickle
    if (set(packet)!={'header','_topic','_received_wall','_local_contact_payload'}
            or type(packet['_local_contact_payload']) is not bytes
            or not packet['_topic'].endswith('/bin_contacts')):
        raise ValueError('invalid local contact packet')
    # This is not a file/network pickle reader. Only our input process creates
    # these bytes after converting the native ROS message to primitive values.
    row=pickle.loads(packet['_local_contact_payload'])
    if (type(row) is not dict or any(row.get(key)!=packet[key]
            for key in ('header','_topic','_received_wall')) or 'states' not in row):
        raise ValueError('local contact metadata differs from frozen source')
    return row


def expand_contact_event(row):
    """Expand a local contact inside the recorder, outside the control owner."""
    message=row.get('message')
    if isinstance(message,dict) and '_local_contact_payload' in message:
        row=dict(row,message=restore_contact_message(message))
    return row


def _write_event_rows(path, pending, answer):
    """Isolated JSON/flush worker; the parent retains all decisions and clocks."""
    count = 0
    try:
        with BufferedTextLog(path, 'a') as stream:
            while True:
                try: row = pending.get(timeout=.05)
                except queue.Empty:
                    stream.flush()
                    continue
                if row is None:
                    break
                if '_local_snapshot' in row:
                    # Bytes are created only by the local EventWriter. Never
                    # deserialize an external evidence file or network packet.
                    import pickle
                    values = pickle.loads(row.pop('_local_snapshot'))
                    row.update(plain_ros_value(values))
                stream.write(json.dumps(expand_contact_event(row), allow_nan=False)+'\n')
                count += 1
        answer.send(dict(passed=True, records=count))
    except Exception as exc:
        answer.send(dict(passed=False, records=count, error=type(exc).__name__+': '+str(exc)))
    finally:
        answer.close()


def plain_ros_value(value):
    """Preserve ROS primitive sequences as JSON lists, including nested arrays."""
    if isinstance(value, array):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain_ros_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_ros_value(item) for item in value]
    return value


def ros_message_value(value):
    """Detach a ROS message directly into JSON values, without OrderedDict passes."""
    if isinstance(value, array):
        return value.tolist()
    if isinstance(value, dict):
        return {key: ros_message_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ros_message_value(item) for item in value]
    fields = getattr(value, 'get_fields_and_field_types', None)
    if fields is not None:
        return {key: ros_message_value(getattr(value, key)) for key in fields()}
    return value


class Records:
    def __init__(self, path, *, kinds=None):
        self.path = Path(path)
        self.kinds = frozenset(kinds) if kinds is not None else None

    def __iter__(self):
        with self.path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if self.kinds is None or row['kind'] in self.kinds:
                    yield row

    def only(self, *kinds):
        selected = set(kinds)
        if self.kinds is not None: selected &= self.kinds
        return Records(self.path, kinds=selected)


class EventWriter:
    """Bounded lossless raw recorder, without blocking decisions on JSON/flush.

    The enqueue lock fixes event order and time; immutable copies retain source
    and actual receive times. Writer failure or overload invalidates the run.
    """

    def __init__(self, path, started, *, limit=4096, process=False):
        self.path = Path(path); self.started = started
        self.stream = BufferedTextLog(self.path)
        self.lock = threading.Lock()
        self.error = None; self.closed = False; self.count = 0
        self.process = process; self.accepted = 0
        if process:
            import multiprocessing
            context = multiprocessing.get_context('spawn')
            self.stream.close()
            self.queue = context.Queue(maxsize=limit)
            self.answer, child_answer = context.Pipe(duplex=False)
            self.thread = context.Process(target=_write_event_rows,
                args=(str(self.path),self.queue,child_answer),name='amr_raw_evidence_writer',daemon=True)
        else:
            self.queue = queue.Queue(maxsize=limit)
            self.thread = threading.Thread(target=self.run,name='amr_raw_evidence_writer',daemon=True)
        self.thread.start()
        if process: child_answer.close()

    def record(self, kind, **values):
        # Freeze before taking the shared enqueue lock. A large contact sample
        # must not make every sensor receiver wait while holding its own lock.
        if getattr(self, 'process', False):
            # Freeze nested ROS arrays with the local C serializer; perform
            # recursive JSON normalization in the recorder process. The owner
            # retains the original source/receipt times and enqueue order.
            import pickle
            values = dict(_local_snapshot=pickle.dumps(values, protocol=5))
        else:
            values = plain_ros_value(values)
        # Process polling can release the GIL. Keep it outside the event-order
        # lock so another sensor can enqueue while the OS check is scheduled.
        process_alive = not getattr(self, 'process', False) or self.thread.is_alive()
        with self.lock:
            if not process_alive:
                self.error = self.error or 'raw evidence process ended before close'
            if self.closed or self.error:
                raise RuntimeError('raw evidence writer closed or failed: '+str(self.error))
            # The lock orders accepted events and their timestamps; copying
            # above never changes their original sensor receipt/source times.
            row = dict(kind=kind,wall_s=time.monotonic()-self.started,**values)
            try:self.queue.put_nowait(row)
            except queue.Full:
                self.error = 'raw evidence queue overflow'
                raise RuntimeError(self.error)
            self.accepted = getattr(self, 'accepted', 0)+1

    def run(self):
        try:
            while True:
                try:row = self.queue.get(timeout=.05)
                except queue.Empty:
                    if self.closed:break
                    self.stream.flush()
                    continue
                self.stream.write(json.dumps(expand_contact_event(row),allow_nan=False)+'\n')
                self.count += 1
        except Exception as exc:
            self.error = type(exc).__name__+': '+str(exc)
        finally:
            try: self.stream.close()
            except Exception as exc:
                self.error = self.error or type(exc).__name__+': '+str(exc)

    def close(self):
        with self.lock:self.closed = True
        if self.process:
            return self.close_process()
        self.thread.join(timeout=5.)
        if self.thread.is_alive() or self.error:
            raise RuntimeError('raw evidence incomplete: '+str(self.error))
        return dict(passed=True,records=self.count,queue_empty=self.queue.empty(),writer_exited=True)

    def close_process(self):
        try:
            # FIFO sentinel follows every accepted row from the same producer.
            self.queue.put(None, timeout=5.)
        except queue.Full:
            self.error = self.error or 'raw evidence close queue full'
        self.thread.join(timeout=5.)
        if self.thread.is_alive():
            self.thread.terminate(); self.thread.join(timeout=1.)
            self.error = self.error or 'raw evidence process did not close'
        try: result = self.answer.recv() if self.answer.poll() else None
        except (EOFError, OSError): result = None
        self.answer.close(); self.queue.cancel_join_thread(); self.queue.close()
        if result is not None: self.count = result['records']
        if (self.error or result is None or not result['passed']
                or self.count != self.accepted or self.thread.exitcode != 0):
            raise RuntimeError('raw evidence incomplete: '+str(self.error or result))
        return dict(passed=True,records=self.count,queue_empty=True,writer_exited=True,
                    writer_process=True,accepted_records=self.accepted)
