import json,collections,random,re
import netc
from exp_b import us2,P,metrics,target_pool,pick_seed
random.seed(7)
E=[json.loads(l) for l in open('entities.jsonl',encoding='utf-8')]
tax=json.load(open('org_tax.json',encoding='utf-8'))
SEED_TAX={'Homo sapiens':[10090,10116],'Mus musculus':[9606,10116],'Escherichia coli':[28901,573],'Arabidopsis thaliana':[4530,4577],
          'Bacillus subtilis':[1386],'Pseudomonas aeruginosa':[303,294]}
cands=collections.OrderedDict()
for e in E:
    if e['bucket']!='proteins' or e['name']=='Unknown' or e['organism'] not in SEED_TAX: continue
    acc=e['ids'].get('uniprot') or e['ids'].get('uniprot_id')
    if not acc or not re.match(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$|^[A-Z][0-9][A-Z0-9]{3}[0-9]$',str(acc)): continue
    if not e['gene'] or not re.match(r'^[A-Za-z0-9]{3,7}$',e['gene']): continue
    if e['step'] not in ('pathbank_protein_id','api_uniprot','uniprot'): continue
    cands[(e['organism'],acc)]=dict(org=e['organism'],acc=str(acc),gene=e['gene'],name=e['name'])
by=collections.defaultdict(list)
for k,v in cands.items(): by[v['org']].append(v)
quota={'Homo sapiens':7,'Escherichia coli':7,'Mus musculus':3,'Arabidopsis thaliana':3,'Bacillus subtilis':2,'Pseudomonas aeruginosa':2}
res=[]
for org,q in quota.items():
    L=by[org]; random.shuffle(L); got=0
    for c in L:
        if got>=q: break
        T=tax.get(org)
        seed=None
        for st in SEED_TAX[org]:
            rows=us2('gene_exact:%s AND taxonomy_id:%s AND reviewed:true'%(c['gene'],st),size=3)
            seed=pick_seed(rows,c['gene'])
            if seed: break
        if not seed: continue
        pool=target_pool(seed,T)
        m=[metrics(seed,x) for x in pool if x['seq']]
        m.sort(key=lambda x:-x['ident']*min(1,x['cov_seed']))
        top=m[0] if m else None
        truth_in=[x for x in m if x['acc']==c['acc']]
        res.append(dict(org=org,gene=c['gene'],truth=c['acc'],seed=dict(acc=seed['acc'],org=seed['org'],pn=seed['pn']),pool_n=len(pool),
            top=top,second=(m[1] if len(m)>1 else None),truth_hit=(truth_in[0] if truth_in else None),top1_correct=bool(top and top['acc']==c['acc'])))
        got+=1
json.dump(res,open('results_ctrl.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
print(len(res),sum(r['top1_correct'] for r in res),netc.ncached('uniprot'))
