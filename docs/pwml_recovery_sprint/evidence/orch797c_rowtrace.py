"""ORCH-797c: where do compound rows of one c011 research leg disappear (run per tree, compare).

Usage: python orch797c_rowtrace.py --src <tree>/src
Prints, for the leg's raw payload, after ``replay._normalized`` and after the c011 seam, the compound rows whose
names match WATCH, and every compound name present in the raw payload but absent after normalization.
"""

import argparse
import sys
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--leg", default="2026-07-28_2122/papers/PMC12657337")
args = ap.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402

print("t2pw resolved:", Path(t2pw.__file__).resolve().parent)
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "tests"))
import test_c011_freeze_seam_golden_equivalence as C  # noqa: E402
from t2pw.pipeline.name_keys import strict_key  # noqa: E402

WATCH = ("ipp", "nadh", "nadph", "glucose", "dhap", "dihydroxyacetone")


def names(payload):
    return [str(r.get("name")) for r in (payload.get("entities") or {}).get("compounds") or [] if isinstance(r, dict)]


leg = next(p for p in C.replay._legs() if args.leg in p.as_posix() and p.name == "research")
raw = C.replay._payload_for(leg)[1]
norm = C.replay._normalized(raw)
print("raw compounds:", len(names(raw)), "normalized:", len(names(norm)))
print("raw watched:", [n for n in names(raw) if any(w in n.casefold() for w in WATCH)])
print("normalized watched:", [n for n in names(norm) if any(w in n.casefold() for w in WATCH)])
gone = sorted(set(names(raw)) - set(names(norm)))
print("raw names absent after normalization:", gone)
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    for d in (work / "outputs", work / "tmp"):
        d.mkdir(parents=True, exist_ok=True)
    result = C._seam(None)(norm, strict_db=True, outputs_dir=work / "outputs", pathway_context=None,
                           export_mode="research", research_mode=True, tmp=work / "tmp")
    frozen = result["payload"] or norm
print("frozen compounds:", len(names(frozen)))
print("frozen watched:", [n for n in names(frozen) if any(w in n.casefold() for w in WATCH)])
print("normalized names absent after the seam:", sorted(set(names(norm)) - set(names(frozen))))
_ = strict_key

# Which normalizer pass drops them: patch the two row-removing helpers and re-run normalization (tip only).
import traceback  # noqa: E402
from t2pw.pipeline import process_normalizer as PN  # noqa: E402

_orig_remove, _orig_dedupe = PN._remove_entity, PN._dedupe_named_rows


def _remove(rows, name):
    if any(w in str(name).casefold() for w in WATCH):
        print("REMOVE", name, "via", [f.name for f in traceback.extract_stack()[-6:-1]])
    return _orig_remove(rows, name)


def _dedupe(rows):
    before = [str(r.get("name")) for r in rows if isinstance(r, dict)]
    _orig_dedupe(rows)
    after = {str(r.get("name")) for r in rows if isinstance(r, dict)}
    lost = [n for n in before if n not in after and any(w in n.casefold() for w in WATCH)]
    if lost:
        for n in lost:
            print("  PARTNERS of", repr(n), [repr(b) for b in before if PN._normalize(b) == PN._normalize(n)], "key", repr(PN._normalize(n)))
        print("DEDUPE lost", lost, "kept", sorted(a for a in after if any(w in a.casefold() for w in WATCH)),
              "via", [f.name for f in traceback.extract_stack()[-6:-1]])


PN._remove_entity, PN._dedupe_named_rows = _remove, _dedupe
C.replay._normalized(C.replay._payload_for(leg)[1])
