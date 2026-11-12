import json
per={f:set(g) for f,g in json.load(open('s14_cover_per_file.json')).items()}
steps=[('BOUND',),('SPONT',),('ONTO','SENT'),('STALEINT',),('TSTATE',),('TCVLOC',),('TTRANS',),('NUCLEO','ZEROCX','OTHER')]
fixed=set()
for s in steps:
    fixed|=set(s); print('+',s,'-> clean files:',sum(1 for g in per.values() if g<=fixed))
# files whose only blockers are in a set
print('files with BOUND only:',sum(1 for g in per.values() if g<={'BOUND'}))
print('files that have transports (TSTATE)',sum(1 for g in per.values() if 'TSTATE' in g))
