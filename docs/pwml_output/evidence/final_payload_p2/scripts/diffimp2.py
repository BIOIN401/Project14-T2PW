import json,re,collections
exec(open("C:/t/fp/docs/pwml_output/evidence/final_payload_p2/scripts/diffimp.py",encoding="utf-8").read().split("for nm,D in")[0])
for i in sorted(O):
    a,b=msgs(O[i]),msgs(N[i])
    nb=b-a
    key=[m for m in nb if m.startswith(("compound_location:","pathway:","reaction_left_element","protein_complex: Species","species:","transport_element: Element"))]
    if key or (not N[i]["success"]) and O[i]["success"]:
        oe,ne=O[i]["expected"],N[i]["expected"]
        print(i,"keys",{m:nb[m] for m in key},"| compounds created off/on",O[i]["created_main"].get("compounds"),N[i]["created_main"].get("compounds"),"expected",oe.get("compounds"),ne.get("compounds"),"| proteins",O[i]["created_main"].get("proteins"),N[i]["created_main"].get("proteins"),"| species created",O[i]["created_main"].get("species"),N[i]["created_main"].get("species"))
