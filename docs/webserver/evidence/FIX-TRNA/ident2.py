import json
wt = json.load(open("C:/t/vis-runs/scratch/fixtrna/replay_wt.json", encoding="utf-8"))
mn = json.load(open("C:/t/vis-runs/scratch/fixtrna/replay_main.json", encoding="utf-8"))
def strip(s):
    d = json.loads(s); d["report"].pop("llm", None); return d
print("patch+report minus report.llm (provider string from env) identical, wt flag-off vs original code:", sum(strip(wt[p]["off"]) == strip(mn[p]["off"]) for p in wt), "of", len(wt))
print("patch lists identical:", sum(strip(wt[p]["off"])["patch"] == strip(mn[p]["off"])["patch"] for p in wt))
