"""ORCH verification of REV-121 FINDING 1, independent of both prior censuses.

Claim under test: the guard fires on 7 archived legs at the PRODUCTION CALL SITE,
not the 5 that both the orchestrator's pre-dispatch census and the implementer's
census reported by evaluating the predicate on the RAW committed final_mapped.json.

Method: drive quarantine_and_close itself on each archived payload in the base tree
and in the tip tree, and compare the resulting payloads. Nothing is inferred from a
predicate call; the difference between the two trees IS the answer.
"""
import json, sys, subprocess
from pathlib import Path

TREE = Path(sys.argv[1])
ARCHIVE = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
sys.path.insert(0, str(TREE / "src"))
import t2pw
print("TREE UNDER MEASUREMENT:", t2pw.__file__)
assert str(TREE).lower() in t2pw.__file__.lower(), "wrong tree resolved"

from t2pw.pipeline.strict_quarantine import quarantine_and_close

legs = sorted(
    fm for fm in ARCHIVE.rglob("final_mapped.json")
    if ".claude" not in fm.parts and ".git" not in fm.parts
    and ".pytest_tmp_baseline" not in fm.parts and "papers" in fm.parts
)
print("legs:", len(legs))

out = {}
for fm in legs:
    rel = fm.relative_to(ARCHIVE).as_posix()
    payload = json.loads(fm.read_text(encoding="utf-8"))
    mode = "research" if "/research/" in rel else "pathwhiz"
    try:
        res = quarantine_and_close(payload, strict_db=True, mode=mode)
        p = res.payload if hasattr(res, "payload") else {}
        states = sorted(
            str(r.get("name")) for r in (p.get("biological_states") or [])
            if isinstance(r, dict)
        )
        el = p.get("element_locations") or {}
        missing = sum(
            1 for b, rows in el.items() if isinstance(rows, list)
            for r in rows if isinstance(r, dict)
            and not str(r.get("biological_state") or "").strip()
        )
        pr = p.get("processes") or {}
        rx = sorted(str(r.get("name")) for r in (pr.get("reactions") or []) if isinstance(r, dict))
        out[rel] = {
            "ok": bool(getattr(res, "ok", None) if hasattr(res, "ok") else None),
            "states": states,
            "rows_missing_state": missing,
            "n_reactions": len(rx),
            "reactions": rx,
            "payload_digest": json.dumps(p, sort_keys=True, ensure_ascii=False),
        }
    except Exception as exc:
        out[rel] = {"error": f"{type(exc).__name__}: {exc}"[:200]}

dest = Path(sys.argv[2])
dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", dest, "legs recorded:", len(out))
errs = {k: v["error"] for k, v in out.items() if "error" in v}
print("errors:", len(errs))
for k, v in list(errs.items())[:5]:
    print("   ", k, v)
