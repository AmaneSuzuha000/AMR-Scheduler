"""Ask a live bridge for a route over the wire, exactly as the console does.

Used as an end-to-end check: it proves the bytes on the wire round-trip through the same
protocol the C# side reads, without needing the GUI to be running.
"""
import argparse
import json
import socket
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9300)
    ap.add_argument('--target', default='J_S_125')
    ap.add_argument('--from-x', type=float, default=None)
    ap.add_argument('--from-y', type=float, default=None)
    ap.add_argument('--direct', action='store_true')
    args = ap.parse_args(argv)

    with socket.create_connection((args.host, args.port), timeout=10) as c:
        f = c.makefile('rb')
        hello = json.loads(f.readline())
        print('hello:', json.dumps(hello, ensure_ascii=False))
        req = {'type': 'route_request', 'target': args.target, 'direct': bool(args.direct)}
        if args.from_x is not None and args.from_y is not None:
            req['from'] = [args.from_x, args.from_y]
        c.sendall((json.dumps(req) + '\n').encode('utf-8'))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            msg = json.loads(f.readline())
            if msg.get('type') == 'route_result':
                print('route_result:', json.dumps(msg, ensure_ascii=False))
                return 0 if not msg.get('reason') else 2
        print('timed out waiting for route_result')
        return 1


if __name__ == '__main__':
    sys.exit(main())
