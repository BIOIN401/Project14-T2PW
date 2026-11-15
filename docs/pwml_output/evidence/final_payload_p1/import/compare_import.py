import json,re,statistics as st,collections
B='C:/t/ev-fp/import/'
sel=json.load(open(B+'selected.json')); grp={i:g for g,l in sel.items() for i in l}
def load(s):
    d={}
    for l in open(B+s+'/results.jsonl'):
        r=json.loads(l); d[int(re.search(r'(\d+)\.pwml$',r['file']).group(1))]=r
    return d
def msgs(r):
    return [f"{k.rsplit('_',1)[0] if re.search(r'_\d+$',k) else k}: {m}" for k,v in r['errors'].items() for m in v]
def msgs_full(r): return [(k,m) for k,v in r['errors'].items() for m in v]
O,N=load('off'),load('on'); ids=sorted(O)
print('pairs',len(ids),len(N))
for g in ('all','P1_pathwhiz_pwml','P0_pathwhiz_nopwml'):
    sub=[i for i in ids if g=='all' or grp[i]==g]
    for nm,D in (('off',O),('on',N)):
        ec=[len(msgs_full(D[i])) for i in sub]
        print(g,nm,'success',sum(D[i]['success'] for i in sub),'/',len(sub),'zero-err',sum(c==0 for c in ec),'err mean %.1f median %.1f total %d'%(st.mean(ec),st.median(ec),sum(ec)),'secs %.0f'%sum(D[i]['seconds'] for i in sub))
print('per pair idx: off_err on_err off_ok on_ok')
for i in ids: print(i,grp[i][:2],len(msgs_full(O[i])),len(msgs_full(N[i])),O[i]['success'],N[i]['success'])
# categories (key stripped of number)
def cat(r): return collections.Counter(msgs(r))
co,cn=collections.Counter(),collections.Counter()
new=[]
for i in ids:
    a,b=cat(O[i]),cat(N[i]); co+=a; cn+=b
    for m in b:
        if m not in a: new.append((i,m))
print('CATEGORY off / on'); 
for m in sorted(set(co)|set(cn)): print(co[m],cn[m],m)
print('NEW with flag on (category not in same pair off):',len(new))
for i,m in new: print(' ',i,m,[k for k,v in N[i]['errors'].items() if any(m==f"{k.rsplit('_',1)[0]}: {x}" for x in v)][:3])
# count increases per pair
inc=[(i,len(msgs_full(N[i]))-len(msgs_full(O[i]))) for i in ids]
print('pairs on>off:',[x for x in inc if x[1]>0],'on<off:',[x for x in inc if x[1]<0])
# created vs expected
for sec in ('compounds','proteins','reactions','transports','compound_locations'):
    for nm,D in (('off',O),('on',N)):
        e=sum(D[i]['expected'].get(sec,0) for i in ids); c=sum(D[i]['created_main'].get(sec,0) for i in ids)
        print(sec,nm,'created',c,'expected',e)
for nm,D in (('off',O),('on',N)):
    print(nm,'created keys with reference/title:',sorted({k for i in ids for k in D[i]['created'] if 'refer' in k.lower()}), 'expected keys',sorted({k for i in ids for k in D[i]['expected']}))
