import csv, json, collections
r=list(csv.DictReader(open("C:/t/ev-oa/accvar/precision_runs.csv")))
sp=collections.Counter()
for x in r:
    if x["qok"]=="True" and x["complete"]=="True": sp.update(json.loads(x["species"]))
print("species",dict(sp))
sa=[float(x["sub_acc"]) for x in r if x["qok"]=="True" and x["complete"]=="True" and x["sub_acc"]]
pa=[float(x["prod_acc"]) for x in r if x["qok"]=="True" and x["complete"]=="True" and x["prod_acc"]]
print(sum(sa)/len(sa),sum(pa)/len(pa))
t=list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
cases={"PMC12444477","PMC13231680","PMC12657337","PMC12421875","PMC12312563","PMC12856317","PMC12180156","PMC12096016","PMC12452463","PMC12782028"}
print("pinned-case canonical runs",sum(1 for x in t if x["qok"]=="True" and x["paper"] in cases), "papers",len({x["paper"] for x in t if x["qok"]=="True" and x["paper"] in cases}))
