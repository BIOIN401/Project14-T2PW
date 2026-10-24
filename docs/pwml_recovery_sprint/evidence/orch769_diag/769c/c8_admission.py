"""Where do c-Jun/JNK/MRLC appear in final_mapped.json outside entities? Read-only, stdlib."""
import json
D = r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict"
f = json.load(open(f"{D}/final_mapped.json", encoding="utf-8"))
targets = ["c-Jun", "MRLC", "JNK", "Profilin", "DIA1"]


def walk(o, path):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk(v, f"{path}/{i}")
    elif isinstance(o, str):
        yield path, o


hits = {}
for p, s in walk({k: v for k, v in f.items() if k not in ("processes", "source_text_index")}, ""):
    for t in targets:
        if s == t:
            hits.setdefault(t, []).append(p)
for t, ps in hits.items():
    print(t, len(ps), ps[:6])
ear = f.get("entity_admission_report")
print("entity_admission_report", json.dumps(ear, ensure_ascii=False)[:700] if not isinstance(ear, list) else len(ear))
