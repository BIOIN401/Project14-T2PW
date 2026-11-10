import urllib.request,urllib.parse,json,hashlib,os,time,sys
CACHE='C:/t/ev-oa/mapping/cache/'
_last={}
CNT={}
def get(url,src,accept='application/json',maxreq=300):
    h=hashlib.sha1((url+accept).encode()).hexdigest()
    fp=CACHE+src+'_'+h+'.txt'
    if os.path.exists(fp):
        return open(fp,encoding='utf-8').read()
    n=CNT.get(src,0)
    # count previous cached files as spent budget
    if n>=maxreq: raise RuntimeError('cap')
    dt=time.time()-_last.get(src,0)
    if dt<0.4: time.sleep(0.4-dt)
    req=urllib.request.Request(url,headers={'User-Agent':'research-client/1.0','Accept':accept})
    try:
        with urllib.request.urlopen(req,timeout=40) as r: txt=r.read().decode('utf-8','replace')
    except urllib.error.HTTPError as e:
        txt='HTTPERR %d %s'%(e.code,e.read().decode('utf-8','replace')[:300])
    except Exception as e:
        txt='NETERR %s'%e
    _last[src]=time.time(); CNT[src]=n+1
    open(fp,'w',encoding='utf-8').write(txt)
    return txt
def ncached(src):
    return len([f for f in os.listdir(CACHE) if f.startswith(src+'_')])
if __name__=='__main__':
    print(get('https://rest.uniprot.org/uniprotkb/search?query=gene_exact:fur+AND+organism_id:562&fields=accession,gene_names,protein_name,reviewed&size=5','uniprot')[:800])
    print(get('https://www.ebi.ac.uk/ols4/api/search?q=SEPHCHC&ontology=chebi&rows=3','ols')[:600])
