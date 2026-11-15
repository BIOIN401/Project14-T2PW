import sys
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c
i = 128
P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
print('canvas', P['canvas'])
for m in P['membranes']:
    print('box', {k: round(v) for k, v in m.items()})
for l in P['labels']:
    print('label', l)
for k, v in P['cloc'].items():
    x, y = v['x'] + v['w'] / 2, v['y'] + v['h'] / 2
    if not any(m['x0'] <= x <= m['x1'] and m['y0'] <= y <= m['y1'] for m in P['membranes']):
        print('outside cloc', k, round(x), round(y), P['states'][v['state']]['name'])
for k, v in P['ploc'].items():
    x, y = v['x'] + v['w'] / 2, v['y'] + v['h'] / 2
    if not any(m['x0'] <= x <= m['x1'] and m['y0'] <= y <= m['y1'] for m in P['membranes']):
        print('outside ploc', k, round(x), round(y), P['states'][v['state']]['name'])
