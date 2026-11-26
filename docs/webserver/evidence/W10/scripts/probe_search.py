import sys, time
sys.path.insert(0, "C:/t/ws-search/src")
from t2pw.web.pathbank import PathBankIndex
from t2pw.web.smart_search import classify
from t2pw.service.organisms import load_organisms
idx = PathBankIndex("C:/t/ws-search-work/index/pathbank_index.sqlite")
orgs = load_organisms()
for q in ["TCA cycle", "glycolysis in Saccharomyces cerevisiae", "enterobactin biosynthesis", "enterobactin biosynthesis E. coli",
          "krebs cycle", "glycolisis", "ergosterol biosynthesis in yeast", "how does yeast make ergosterol?", "alanine metabolism human", "PMID 12345678", "10.1038/nature12373"]:
    t = time.time()
    r = classify(q, orgs)
    ids = None
    if r.organism is not None:
        stem = " ".join(r.organism.label.split()[:2]).casefold()
        ids = [o.id for o in orgs if o.label.casefold().startswith(stem)]
    hits = idx.search(r.pathway_name, ids) if r.kind == "pathway" else []
    print(f"{q!r}: {r.kind} name={r.pathway_name!r} org={getattr(r.organism,'label',None)} ({time.time()-t:.2f}s)")
    for h in hits:
        print("    ", h.pw_id, h.smp_id, h.name, h.species_id, h.score, h.organisms)
