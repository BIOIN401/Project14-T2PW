import json, sys
s = json.load(open(sys.argv[1]))
names = sys.argv[2:]
rts = ["compound-location","protein-location","nucleic-acid-location","protein-complex-visualization","reaction-visualization",
       "transport-visualization","interaction-visualization","edge","membrane-visualization","drawable-element-location",
       "zoom-visualization","label-location","sub-pathway-visualization"]
for rt in rts:
    print("#####", rt)
    fields = sorted({f for n in names if rt in s[n] for f in s[n][rt]["fields"]})
    for f in fields:
        print("  ", f)
        for n in names:
            if rt not in s[n]: continue
            r = s[n][rt]
            print(f"      {n:10s} {r['fields'].get(f,0):3d}/{r['n']:3d}  {r['vals'].get(f, '')}"[:230])
