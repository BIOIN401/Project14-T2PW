"""ORCH-762: per-protein identity table of every replay scenario, offline (recorded UniProt answers).

Usage: python replay_table.py --src <tree>/src --label base|tip  (base = 493c1a10 src extracted to scratch)
Prints, per scenario and stage, each protein row's shipped UniProt, identity status, verdict reason and
judged record; each borrowed-identity flag; and the eval-v2 gold score of the menaquinone replays."""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--label", required=True)
args = ap.parse_args()
sys.path.insert(0, args.src)
import t2pw  # noqa

WT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WT / "tests"))
import orch762_harness as H  # noqa

for name, run in H.SCENARIOS.items():
    http = H.ReplayHttp()
    out, doors = run(http, Path(f"C:/t/bt-orch762/table-{args.label}-{name}"))
    print(f"== {args.label} {name}: unrecorded={len(http.unrecorded)} ai={len(doors.ai_calls)} lit={len(doors.literature_calls)}")
    for stage in ("stage2", "stage6"):
        if stage not in out:
            continue
        pl = out[stage]["payload"]
        ents = pl["entities"]
        for row in ents.get("proteins", []):
            m = row.get("mapping_meta", {})
            v = m.get("identity_verdict") or {}
            jc = v.get("judged_candidate") or {}
            print(f"  {stage} P {row['name']:10s} uni={(row.get('mapped_ids') or {}).get('uniprot','')!s:8s} status={m.get('identity_status')} reason={v.get('reason')} judged={jc.get('accession') or jc.get('uniprot')}|{jc.get('organism')} rej={m.get('rejected_mapped_ids')}")
        for row in ents.get("compounds", []):
            m = row.get("mapping_meta", {})
            if m.get("route") == "protein" or "EntB" in row.get("name", ""):
                print(f"  {stage} C {row['name']:10s} route={m.get('route')} {m.get('route_reason')} ids={row.get('mapped_ids')} rej={m.get('rejected_mapped_ids')}")
        print("  ", stage, "borrowed:", [(r["name"], r["flag"], r["route"]) for r in pl.get("borrowed_protein_identities", [])])


# gold scoring (eval-v2 score_leg, no paper text) of the replays' Stage-2 and Stage-6 payloads
from t2pw.bench.eval_v2 import reactions as R  # noqa: E402
from t2pw.bench.eval_v2.gold import load_supplement  # noqa: E402

cases = load_supplement()
for name, paper in (("kf147", "PMC12421875"), ("lactis_species", "PMC12421875"), ("enterobactin_holo", "PMC12096016")):
    case = cases[paper]
    out, _ = H.SCENARIOS[name](H.ReplayHttp(), Path(f"C:/t/bt-orch762/table-gold-{args.label}-{name}"))
    for stage in ("stage2", "stage6"):
        sc = R.score_leg(case, out[stage]["payload"], [], None, expected_species=case.species)
        keys = {k: v for blk in sc.values() if isinstance(blk, dict) for k, v in blk.items() if k in ("gold_reactions", "gold_reactions_matched", "recall", "precision", "enzyme_verdicts")}
        print("GOLD", args.label, name, paper, stage, keys)
