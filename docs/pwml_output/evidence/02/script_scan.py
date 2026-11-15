import json,re,os,collections,sys
rows=[json.loads(l) for l in open('C:/t/ev-dedup/out/runs.jsonl')]
R='C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/'
canon=[r for r in rows if r['quarantine_ok']]
IDF=['pathbank_compound_id','chebi_id','pubchem_cid','kegg_id','hmdb_id','cas','chemspider_id','biocyc_id','db_id','pathwhiz_id']
def ids_of(e):
    out={}
    mi=e.get('mapped_ids') or {}
    for src in (e,mi):
        for k in IDF+['uniprot','uniprot_id','drugbank','drugbank_id','pathbank_protein_id','pathbank_complex_id','pathbank_protein_complex_id']:
            v=src.get(k)
            if v not in (None,'',0,'0',[],{}): out[k]=v
    return out
def jl(p):
    try: return json.load(open(p,encoding='utf-8'))
    except Exception: return None
out=open('C:/t/ev-oa/mapping/entities.jsonl','w',encoding='utf-8')
runs=[]
for r in canon:
    p=r['path']; base=p if re.match(r'^[A-Za-z]:',p) else R+p
    folder=os.path.dirname(base)
    d=jl(base)
    m=re.search(r'PMC\d+',p); paper=m.group(0) if m else ('e2e:'+p.split('/')[2])
    q=jl(folder+'/quarantine_report.json') or {}
    rem=jl(folder+'/removed_entity_report.json') or {}
    mode=q.get('export_mode')
    quar=q.get('quarantined',[])
    qpart=collections.Counter()
    for x in quar:
        qpart[(x.get('detail',{}).get('participant') or x.get('essential_participant') or '').strip().lower()]+=1
    remnames=set()
    for x in rem.get('removed_entities',[]):
        remnames.add((str(x.get('name') or x.get('entity') or '')).strip().lower())
    runs.append(dict(path=p,paper=paper,mode=mode,n_quar=len(quar),quar_reasons=dict(collections.Counter(x.get('reason') for x in quar)),
      quar_states=dict(collections.Counter(x.get('state') for x in quar)),n_removed=len(rem.get('removed_entities',[])),
      removed=[ {k:x.get(k) for k in ('name','kind','bucket','reason','entity','type')} for x in rem.get('removed_entities',[])][:50],
      ent_adm_removed=len((d.get('entity_admission_report') or {}).get('removed',[])),
      ent_adm_demoted=len((d.get('entity_admission_report') or {}).get('demoted',[]))))
    # species of the run
    sp=[s.get('name') for s in d['entities'].get('species',[])]
    org_cnt=collections.Counter((e.get('organism') or e.get('species')) for e in d['entities'].get('proteins',[]) if e.get('name')!='Unknown' and (e.get('organism') or e.get('species')))
    main_org=org_cnt.most_common(1)[0][0] if org_cnt else (sp[0] if sp else None)
    runs[-1]['main_org']=main_org; runs[-1]['species']=sp
    for b,lst in d['entities'].items():
        if b in ('subcellular_locations','cell_types','tissues','compartments'): continue
        for e in lst:
            mm=e.get('mapping_meta') or {}; res=mm.get('resolution') or {}
            amb=mm.get('ambiguity') or {}
            rec=dict(path=p,paper=paper,bucket=b,name=e.get('name'),cls=e.get('class'),organism=e.get('organism') or e.get('species') or (sp[0] if len(sp)==1 else None),
              taxid=e.get('taxonomy_id'),status=res.get('status'),issue=res.get('issue'),step=res.get('order_step'),failed=res.get('failed_check'),
              chosen=mm.get('chosen_rule'),provider=mm.get('provider'),source=mm.get('source'),ncand=len(mm.get('candidates') or []),
              amb_n=amb.get('candidate_count'),amb_rej=[x.get('reason') for x in (amb.get('rejections') or [])][:12],
              amb_acc=[x.get('accession') for x in (amb.get('rejections') or [])][:12],
              ids=ids_of(e),idstatus=e.get('identity_status') or mm.get('identity_status'),
              verdict=((mm.get('identity_verdict') or {}).get('reason')),
              raw_name=e.get('raw_name'),syn=(e.get('synonyms') or [])[:6],
              ec=e.get('ec_number'),gene=e.get('gene_name'),
              in_quarantine=(str(e.get('name') or '').strip().lower() in qpart),
              removed=(str(e.get('name') or '').strip().lower() in remnames),
              ncomp=len(e.get('components') or []) if b=='protein_complexes' else None,
              gen=e.get('generated'),genr=e.get('generation_reason'),
              rmfr=mm.get('real_mapping_failure_reason'),tgt=mm.get('target_organism'),fen=mm.get('functional_enzyme_name'),fbr=mm.get('fallback_reason'),
              main_org=main_org,
              jc=[dict(acc=(c.get('accession') or c.get('uniprot')),pn=c.get('protein_name') or c.get('name'),genes=(c.get('gene_names') or c.get('gene_name')),org=c.get('organism'),rev=c.get('reviewed'),score=c.get('score')) for c in ((mm.get('identity_verdict') or {}).get('judged_candidates') or [])][:8],
              ngate=((mm.get('identity_verdict') or {}).get('name_gate') or {}).get('reason'),
              vreason=((mm.get('identity_verdict') or {}).get('reason')),
              rejected_ids=mm.get('rejected_mapped_ids'),
              queries=(mm.get('queries_tried') or [])[:3],
              lit_alias=[x.get('alias') for x in (mm.get('literature_aliases') or [])][:4],
              query=mm.get('query'))
            out.write(json.dumps(rec,ensure_ascii=False)+'\n')
json.dump(runs,open('C:/t/ev-oa/mapping/runs.json','w'),indent=0)
print(len(runs),'runs;', len({r['paper'] for r in runs}),'papers')
print(collections.Counter(r['mode'] for r in runs))
