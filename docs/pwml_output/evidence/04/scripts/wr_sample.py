import json, random
w=json.load(open("id_wrong.json"))
u={}
for r in w: u.setdefault((r["paper"],r["bucket"],r["name"].casefold()),r)
U=list(u.values()); random.seed(20261110); random.shuffle(U)
for r in U[:36]:
    print(r["bucket"][:4],"|",r["name"],"|indep",r["independent"],"|",[(a,b,c[:3]) for a,b,c in r["per_id"]][:2])
