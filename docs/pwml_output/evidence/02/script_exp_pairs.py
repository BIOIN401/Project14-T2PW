import json
import netc
from exp_b import us2,P,align,metrics
PAIRS=[('ORMDL1 vs ORMDL3 (human paralogs)','Q9P0S3','Q8N138'),('SPHK1 vs SPHK2 (human paralogs)','Q9NYA1','Q9NRA0'),('KCNJ2 vs KCNJ4 (Kir2.1 vs Kir2.3)','P63252','P48050'),
 ('SHMT1 vs SHMT2 (human paralogs)','P34896','P34897'),('SMURF1 vs SMURF2 (human paralogs)','Q9HCE7','Q9HAU4'),('E. coli htrB(LpxL) vs msbB(LpxM)','P0ACV0','P24205'),
 ('human POR vs E. coli CysJ (shared FAD/FMN domains)','P16435','P38038'),('human KCNJ2 vs mouse Kcnj2 (ortholog)','P63252','P35561')]
out=[]
for lab,a,b in PAIRS:
    rows=us2('accession:%s OR accession:%s'%(a,b),size=5)
    d={P(r)['acc']:P(r) for r in rows}
    if a in d and b in d:
        idt,cols,mt=align(d[a]['seq'],d[b]['seq'],3,-1,-5); m=dict(acc=b,ident=round(idt,3),cov_seed=round(cols/len(d[a]['seq']),3),cov_cand=round(cols/len(d[b]['seq']),3)); m['pair']=lab; m['len_a']=d[a]['len']; m['len_b']=d[b]['len']; m['gene_a']=d[a]['gene']; m['gene_b']=d[b]['gene']
        out.append(m); print(lab,m['ident'],m['cov_seed'],m['len_a'],m['len_b'])
    else: print('missing',lab,list(d))
json.dump(out,open('results_pairs.json','w',encoding='utf-8'),indent=0)
