import sys
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c
i = sys.argv[1]
P = c.parse_pwml('%s/%s.pwml' % (c.PWML, i))
nm = P['names']
for r in P['reactions']:
    v = P['rviz'].get(r['id'], {})
    print(r['id'], [nm.get((e[0], e[1])) for e in r['left']], '->', [nm.get((e[0], e[1])) for e in r['right']], r['direction'],
          [nm.get(('ProteinComplex', e)) for e in r['enz']], P['states'].get(v.get('state'), {}).get('name'))
