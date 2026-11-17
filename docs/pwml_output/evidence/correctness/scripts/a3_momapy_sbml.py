import json, momapy.io
B="C:/t/corr-assess"
res={}
for i in [s["idx"] for s in json.load(open(f"{B}/sample.json",encoding="utf-8"))]:
    for tag,p in (("ours",f"{B}/a3/ours/{i}.sbml"),("pw",f"{B}/pathwhiz_exports/{i}.sbml")):
        try:
            r=momapy.io.read(p,reader="sbml"); m=r.obj
            res[f"{i}_{tag}"]=dict(ok=True,layout=None if m.layout is None else "present")
        except Exception as e:
            res[f"{i}_{tag}"]=dict(ok=False,err=repr(e)[:150])
json.dump(res,open(f"{B}/a3/momapy_sbml_load.json","w"),indent=1)
import collections; print(collections.Counter((k.split('_')[1],v['ok'],v.get('layout')) for k,v in res.items()))
