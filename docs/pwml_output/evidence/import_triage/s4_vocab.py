"""Compare vocabulary T2PW writes (locations, cell types, tissues, species) with rows in the PathBank dump
(read-only grep extract vocab_inserts.sql), simulating the importer lookup  klass.find_by(attribute_hash)."""
import re, json, collections, sys
VOCAB=sys.argv[1]  # path to the grep extract, see IMPORT-ERROR-TRIAGE.md section 9
import xml.etree.ElementTree as ET
def tuples(line):
    body=line[line.index('VALUES')+7:].rstrip().rstrip(';')
    out=[];i=0;n=len(body)
    while i<n:
        if body[i]=='(':
            i+=1;row=[];
            while True:
                if body[i]=="'":
                    j=i+1;s=[]
                    while True:
                        c=body[j]
                        if c==chr(92): s.append(body[j+1]); j+=2; continue
                        if c=="'" and body[j+1]=="'": s.append("'"); j+=2; continue
                        if c=="'": break
                        s.append(c); j+=1
                    row.append(''.join(s)); i=j+1
                else:
                    j=i
                    while body[j] not in ',)': j+=1
                    v=body[i:j]; row.append(None if v=='NULL' else v); i=j
                if body[i]==',': i+=1; continue
                if body[i]==')': i+=1; break
            out.append(row)
        else: i+=1
    return out
db={}
for line in open(VOCAB,encoding='utf8',errors='replace'):
    t=re.match(r'INSERT INTO `(\w+)`',line).group(1); db[t]=tuples(line)
print({k:len(v) for k,v in db.items()})
norm=lambda s:(s or '').rstrip().casefold()
# column positions
V={'subcellular-location':{norm(r[1]):r[4] for r in db['subcellular_locations']},
   'cell-type':{norm(r[1]):r[2] for r in db['cell_types']},
   'tissue':{norm(r[1]):r[2] for r in db['tissues']}}
SP_TAX={r[2]:r for r in db['species']}
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter(); miss=collections.Counter(); files_all_hit=0; files_with_miss=set()
for r in rows:
    t=ET.parse(r['file']).getroot(); fm=False
    for tag in V:
        for e in t.iter(tag):
            if e.find('name') is None: continue
            nm=e.findtext('name'); oid=(e.findtext('ontology-id') or '').strip()
            hit=norm(nm) in V[tag]
            # importer hash has ontology_id only if non-blank
            hit_full = hit and (not oid or V[tag][norm(nm)]==oid)
            C[(tag,'nil_onto' if not oid else 'has_onto', 'hit' if hit_full else 'miss')]+=1
            if not hit_full: miss[(tag,nm,oid)]+=1; fm=True
    for s in t.iter('species'):
        if s.find('taxonomy-id') is None: continue
        tax=s.findtext('taxonomy-id') or ''
        C[('species','tax_in_db' if tax in SP_TAX else 'tax_not_in_db', 'class_blank' if not (s.findtext('classification') or '') else 'class_set')]+=1
        if tax not in SP_TAX: miss[('species',s.findtext('name'),tax)]+=1; fm=True
    if fm: files_with_miss.add(r['file'])
for k,v in sorted(C.items()): print(k,v)
print('files with >=1 vocabulary miss in production:',len(files_with_miss),'of',len(rows))
print('--- top misses (tag,name,onto) count')
for k,v in miss.most_common(40): print(v,k)
json.dump({'files_with_miss':sorted(files_with_miss)},open('s4_vocab_files_with_miss.json','w'),indent=1)

# --- residual ontology failures in production: nil-ontology names with no row of that name (case-insensitive)
res_files=collections.defaultdict(list)
for r in rows:
    t=ET.parse(r['file']).getroot()
    for tag in V:
        for e in t.iter(tag):
            if e.find('name') is None: continue
            nm=e.findtext('name'); oid=(e.findtext('ontology-id') or '').strip()
            if not oid and norm(nm) not in V[tag]: res_files[r['file']].append((tag,nm))
print('files with >=1 nil-ontology name absent from production vocabulary:',len(res_files),'of',len(rows))
allv=collections.Counter(x for v in res_files.values() for x in v)
print('distinct absent names:',len(allv))
# locations used by those files: how many files also depend on them for biological states
json.dump({f:v for f,v in res_files.items()},open('s4_residual_ontology_misses.json','w'),indent=1)
