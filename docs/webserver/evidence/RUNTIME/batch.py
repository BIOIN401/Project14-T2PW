import json,glob,re,os,collections,statistics as st,datetime as dt
base="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers"
rows=[]
for nl in sorted(glob.glob(base+"/*/night.log")):
    orch=nl.replace("\\","/").split("/")[-2]
    start=None
    for line in open(nl,encoding="utf-8",errors="replace"):
        m=re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) .* START (\S+)/(\S+) ",line)
        if m: start=dt.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S"); continue
        if line.startswith('{"cond"'):
            try: j=json.loads(line)
            except Exception: continue
            ld=j.get("leg_dir","")
            src=None
            pdir=os.path.dirname(ld.replace("\\","/"))
            p=pdir+"/01_source_text.txt"
            size=os.path.getsize(p) if os.path.exists(p) else None
            rows.append(dict(orch=orch,cond=j["cond"],mode=j["mode"],exit=j["exit"],min=j["minutes"],rx=j.get("reactions"),ix=j.get("interactions"),start=start,size=size))
print(len(rows))
rows=[r for r in rows if r["start"]]
by=collections.defaultdict(list)
for r in rows: by[(r["cond"],r["mode"])].append(r)
print("\nper condition (strict+research): n, min/median/max minutes, CV, paper chars")
for k,v in sorted(by.items(), key=lambda kv:-len(kv[1])):
    m=[r["min"] for r in v]
    if len(m)<2: continue
    sz=[r["size"] for r in v if r["size"]]
    print(f"{k[0]:20s}{k[1]:9s} n={len(m):2d} min={min(m):5.1f} med={st.median(m):5.1f} max={max(m):5.1f} cv={st.pstdev(m)/st.mean(m):.2f} chars={sz[0] if sz else None}")
def rank(a):
    s=sorted(range(len(a)),key=lambda i:a[i]); r=[0]*len(a)
    for i,j in enumerate(s): r[j]=i
    return r
def spear(x,y):
    rx,ry=rank(x),rank(y); n=len(x); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**.5
    return num/den if den else float('nan')
strict=[r for r in rows if r["mode"]=="strict"]
print("\nstrict legs n=",len(strict))
print("spearman minutes vs paper chars:", end=" ")
sub=[r for r in strict if r["size"]]; print(len(sub), round(spear([r["size"] for r in sub],[r["min"] for r in sub]),2))
sub=[r for r in strict if isinstance(r["rx"],int)]
print("spearman minutes vs reactions:", len(sub), round(spear([r["rx"] for r in sub],[r["min"] for r in sub]),2))
print("spearman minutes vs reactions+interactions:", round(spear([r["rx"]+(r["ix"] or 0) for r in sub],[r["min"] for r in sub]),2))
# within-paper residual vs hour
res=[]
for k,v in by.items():
    if k[1]!="strict" or len(v)<3: continue
    med=st.median([r["min"] for r in v])
    for r in v: res.append((r["start"].hour+r["start"].minute/60, r["min"]/med, k[0]))
print("spearman (minutes/paper median) vs start hour:", len(res), round(spear([a for a,_,_ in res],[b for _,b,_ in res]),2))
buck=collections.defaultdict(list)
for h,x,_ in res: buck[("00-06" if h<6 else "06-12" if h<12 else "12-18" if h<18 else "18-24")].append(x)
for k in sorted(buck): print(" start",k,"n=",len(buck[k]),"median ratio to paper median=",round(st.median(buck[k]),2))
# exit-code / over 60
print("legs over 60 min:",sum(1 for r in rows if r['min']>60),"of",len(rows)," median",st.median([r['min'] for r in rows]))
# by date
bd=collections.defaultdict(list)
for r in strict: bd[r["start"].date()].append(r["min"])
print("\nby date (strict): ")
for d in sorted(bd): print(d,len(bd[d]),round(st.median(bd[d]),1))
