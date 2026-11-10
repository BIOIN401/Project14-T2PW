import json, os, collections, sys
import xml.etree.ElementTree as ET

L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
QIR = [x for x in L if x['q'] is True and x['ir'] and x['pwml']]

paths = collections.defaultdict(lambda: [0, set(), collections.Counter()])  # count, runs, attrs/nil


def walk(e, pre, run):
    tag = e.tag
    p = pre + '/' + tag
    r = paths[p]
    r[0] += 1; r[1].add(run)
    for a, v in e.attrib.items():
        r[2][a + '=' + (v if a in ('type', 'nil') else '*')] += 1
    if e.text and e.text.strip():
        r[2]['_text'] += 1
    for c in e:
        walk(c, p, run)


n = 0
for x in QIR:
    f = os.path.join(x['dir'], x['pwml'][0])
    try:
        t = ET.parse(f).getroot()
    except Exception as ex:
        print('parse fail', f, ex); continue
    n += 1
    walk(t, '', x['dir'])
print('runs parsed', n)
out = {p: {'n': v[0], 'runs': len(v[1]), 'attrs': dict(v[2])} for p, v in sorted(paths.items())}
json.dump(out, open('C:/t/ev-oa/payload/pwml_paths.json', 'w'), indent=0)
# collapse repeated lists: print only distinct leaf-level tag names (path last 2)
names = collections.defaultdict(lambda: [0, 0])
for p, v in out.items():
    k = p.split('/')[-1]
    names[k][0] += v['n']; names[k][1] = max(names[k][1], v['runs'])
print(len(out), 'distinct paths', len(names), 'distinct tag names')
