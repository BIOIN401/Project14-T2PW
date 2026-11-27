import sys, glob, os
sys.path.insert(0, "C:/t/vis-pw7-work/scripts")
from ehit import edges_of
W="C:/t/vis-pw7-work"
labs=sys.argv[1:]
names=[os.path.basename(p) for p in sorted(glob.glob(f"{W}/cc_show/{labs[0]}/*.pwml"))]+["ent-item6.pwml"]
for n in names:
    row=[]
    for l in labs:
        p=f"{W}/cc_show/{l}/{n}" if n!="ent-item6.pwml" else f"{W}/cc_item6/{l}/ent-item6.pwml"
        es=[t for t in edges_of(p,12) if t[1] in ("148","195","149","219","218")]
        row.append(f"{l}: n={len(es)} len={sum(t[2] for t in es)} multi={sum(t[4]>1 for t in es)} hit={sum(bool(t[5]) for t in es)}")
    print(f"{n:28s}", " | ".join(row))
