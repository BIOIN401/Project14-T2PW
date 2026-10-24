import json,sys,os
R=r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
legs={
 "759_fail":R+"runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/papers/PMC12096016/strict/",
 "759p_pass":R+"runs_papers/ORCH-759p/legs/enterobactin-strict/2026-10-22_2101/papers/PMC12096016/strict/",
 "767_fail":R+".claude/worktrees/orch767/runs_papers/ORCH-767/legs/enterobactin-strict/2026-10-23_2224/papers/PMC12096016/strict/",
}
for k,L in legs.items():
    print("=====",k, os.path.exists(L))
    for fn in ("merged_payload.json","final_mapped.json"):
        p=L+fn
        if not os.path.exists(p): print(" missing",fn); continue
        d=json.load(open(p,encoding="utf-8"))
        E=d["entities"]
        print(" --",fn,"proteins:",[r.get("name") for r in E.get("proteins",[])])
        for c in E.get("protein_complexes",[]) or []:
            comps=[{kk:c2.get(kk) for kk in ("name","stoichiometry","uniprot","mapped_ids")} if isinstance(c2,dict) else c2 for c2 in c.get("components",[]) or []]
            print("   PC",c.get("name"),"|gen:",c.get("generation_reason"),"|",json.dumps(comps,ensure_ascii=False)[:400])
        # enzyme references to enterobactin synthase complex
        for pk,pl in (d.get("processes") or {}).items():
            for r in pl or []:
                for e in (r.get("enzymes") or []):
                    nm=e.get("entity") or e.get("protein_complex") or e.get("protein")
                    if nm and "synthase complex" in str(nm):
                        print("   ref",pk,r.get("name"),"->",nm,e.get("provenance"),e.get("confidence"),"|",str(e.get("evidence"))[:150])
    s1=L+"stage1_payload.json"
    if os.path.exists(s1):
        print("  stage1 has 'enterobactin synthase complex':", "enterobactin synthase complex" in open(s1,encoding="utf-8").read())
    src=os.path.dirname(os.path.dirname(L.rstrip("/")))+"/01_source_text.txt"
    if os.path.exists(src):
        t=open(src,encoding="utf-8").read()
        print("  source has 'enzyme complex':", "EntD/EntE/EntF enzyme complex" in t, "| 'enterobactin synthase':", t.lower().count("enterobactin synthase"), "| 'cyclic trilactone':", "cyclic trilactone" in t)
