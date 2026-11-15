import json, socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.create_connection((host, port), 5)
f = s.makefile('r')
for _ in range(6):
    line = f.readline()
    if not line:
        break
    m = json.loads(line)
    keep = {k: v for k, v in m.items() if k in ('type','frame','proto','x','y','yaw','caps','error','vehicle')}
    print(m.get('type'), keep)
s.close()
