import sys
sys.path.insert(0, "C:/t/ws-E2E3/src")
from t2pw.service import organisms
from t2pw.web.views import search_organisms
orgs = organisms.load_organisms()
for q in ["human", "mouse", "yeast", "E. coli", "e coli", "coli", "escherichia coli", "homo", "arabidopsis", "rat", "cerevisiae", "baker"]:
    r = search_organisms(orgs, q)
    print(q, "->", len(r), [o.label for o in r[:4]])
print([ (o.label,o.common_name) for o in orgs if o.common_name][:6], sum(1 for o in orgs if o.common_name))
print([o.label for o in orgs if o.label.startswith("Escherichia coli")][:8])
