"""C-121 IMPLEMENTER end-to-end replay -- G3, and F-179's unmoved-verdict arm.

Mirrors ``c121_replay_e2e.py`` (the orchestrator's verified pre-dispatch harness)
but drives the CANDIDATE build's own production seam instead of a bare
``ensure_autostates`` call, and adds two things the pre-dispatch replay did not
need:

* an ``evaluate_reaction_support`` verdict per leg, base arm and repaired arm, so
  **G4** is a measurement rather than a byte-comparison of one file;
* the ``quarantine_and_close`` arm -- the real seam -- beside the direct helper
  arm, so "the fixed path" means the shipped path and not a stand-in.

Read-only on the repository. PWML bytes go to a scratch directory OUTSIDE it:
these are archived-payload replays, not production deliverables, and must never
enter the PathWhiz import set (D-099 section 7).

    <py> c121_impl_replay.py <scratch-out-dir> [--tree <import root>] [--repo <archive>]
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
DEFAULT_TREE = HERE.parents[3]
DEFAULT_REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tree", default=str(DEFAULT_TREE))
ap.add_argument("--repo", default=str(DEFAULT_REPO))
args = ap.parse_args()

TREE = Path(args.tree).resolve()
REPO = Path(args.repo).resolve()
OUT = Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TREE / "src"))
import t2pw  # noqa: E402

print("EXPECTED tree :", TREE.as_posix())
print("RESOLVED t2pw :", Path(t2pw.__file__).as_posix())
if Path(t2pw.__file__).resolve().parent != (TREE / "src" / "t2pw").resolve():
    print("MEASUREMENT_TREE_REFUSED")
    raise SystemExit(98)
print("ARCHIVE root  :", REPO.as_posix())
print("SCRATCH out   :", OUT.as_posix(), "(outside the repository)")
assert REPO not in OUT.parents and OUT != REPO, "scratch must live outside the repo"

from lxml import etree  # noqa: E402
from t2pw.pwml.ir import (  # noqa: E402
    build_pwml_ir,
    validate_pwml_ir,
    validate_required_pwml_contract,
)
from t2pw.pwml.writer import DeterministicPwmlBuilder, blocking_pwml_ir_errors  # noqa: E402
from t2pw.pwml.validate import (  # noqa: E402
    discover_structure_signature,
    repair_tree,
    validate_generated_tree,
)
from t2pw.pipeline.process_normalizer import restore_autostates_if_required  # noqa: E402
from t2pw.pipeline import reaction_support as RS  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402

REF = REPO / "reference" / "PW000001.pwml"

LEGS = {
    "PMC11961743": "runs_smoke/2026-09-08_1528/papers/PMC11961743/strict/final_mapped.json",
    "PMC4471609": "runs_smoke/2026-09-08_1528/papers/PMC4471609/strict/final_mapped.json",
    "PMC9544450_c120": "runs_validation/c120/2026-09-08_1240/papers/PMC9544450/strict/final_mapped.json",
    "PMC12312563": "runs_verify/2026-08-21_2014/papers/PMC12312563/strict/final_mapped.json",
    "PMC13231680": "runs_verify/2026-08-24_1203/papers/PMC13231680/strict/final_mapped.json",
    "CTRL_PMC9544450_734": "runs_smoke/2026-09-08_1528/papers/PMC9544450/strict/final_mapped.json",
    "CTRL_PMC10269868": "runs_smoke/2026-09-08_1528/papers/PMC10269868/strict/final_mapped.json",
}

#: The orchestrator's pre-dispatch reference sizes (D-099 section 7), for
#: comparison. A mismatch is REPORTED, never silently accepted.
REFERENCE_BYTES = {
    "PMC11961743": 101095,
    "PMC4471609": 80048,
    "PMC9544450_c120": 43794,
    "PMC12312563": 23098,
    "PMC13231680": 17295,
    "CTRL_PMC9544450_734": 49464,
    "CTRL_PMC10269868": 48458,
}


def codes(rep):
    return sorted({e.get("code") for e in (rep.get("errors") or []) if isinstance(e, dict)})


def inject_metadata(p, name):
    md = p.setdefault("metadata", {})
    md.setdefault("pathway_name", "C121 replay " + name)
    md.setdefault("name", "C121 replay " + name)
    md.setdefault("pathway_subject", "Metabolic")
    md.setdefault("subject", "Metabolic")
    md.setdefault("width", 3200)
    md.setdefault("height", 1400)
    return p


_SUPPORT_FN = None
for _cand in ("evaluate_reaction_support", "annotate_reaction_support",
              "evaluate_reaction_support_report", "score_reaction_support"):
    if hasattr(RS, _cand):
        _SUPPORT_FN = getattr(RS, _cand)
        print("F-179 verdict function:", _cand)
        break
if _SUPPORT_FN is None:
    print("F-179 verdict function: NONE FOUND -- available:",
          sorted(n for n in dir(RS) if "support" in n.lower()))


def support_verdict(payload):
    """The reaction-support verdict, as a comparable digest."""
    if _SUPPORT_FN is None:
        return "NO_FUNCTION"
    try:
        rep = _SUPPORT_FN(deepcopy(payload))
    except TypeError:
        try:
            rep = _SUPPORT_FN(deepcopy(payload), strict=True)
        except Exception as exc:
            return "RAISED " + type(exc).__name__ + ": " + str(exc)[:160]
    except Exception as exc:
        return "RAISED " + type(exc).__name__ + ": " + str(exc)[:160]
    try:
        return json.dumps(rep, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(rep)[:2000]


def rx_counts(p):
    pr = p.get("processes") or {}
    return {k: len(pr.get(k) or []) for k in ("reactions", "transports", "interactions")}


signature = discover_structure_signature(REF)
summary = []

for name, rel in LEGS.items():
    path = REPO / rel
    print("\n" + "=" * 74 + "\n### " + name + "  (" + rel + ")")
    if not path.exists():
        print("  MISSING")
        continue
    original = json.loads(path.read_text(encoding="utf-8"))
    arch_counts = rx_counts(original)
    print("  archived counts:", arch_counts)
    base_support = support_verdict(original)

    row = {"leg": name, "archived_counts": arch_counts}

    for arm in ("base", "repaired", "seam"):
        p = deepcopy(original)
        if arm == "repaired":
            acted = restore_autostates_if_required(p)
            print("  [repaired] guard acted:", acted)
        elif arm == "seam":
            result = quarantine_and_close(p, strict_db=True)
            print("  [seam] quarantine_and_close ok=" + str(result.ok)
                  + " refusals=" + str(result.refusal_reasons))
            if not result.ok:
                row["seam"] = "quarantine refused: " + str(result.refusal_reasons)
                continue
            p = deepcopy(dict(result.payload))
        counts = rx_counts(p)
        support = support_verdict(p)
        same_support = support == base_support
        inject_metadata(p, name)
        gate = validate_required_pwml_contract(deepcopy(p), strict_db=True)
        line = ("  [" + arm + "] gate ok=" + str(gate.get("ok"))
                + " codes=" + str(codes(gate) or "none")
                + " counts=" + str(counts)
                + " support_identical_to_base=" + str(same_support))
        row[arm] = {"gate_ok": gate.get("ok"), "codes": codes(gate), "counts": counts,
                    "support_identical_to_base": same_support}
        if not gate.get("ok"):
            print(line)
            continue
        try:
            ir, irrep = build_pwml_ir(p, pathway_name="C121 replay " + name,
                                      pathway_subject="Metabolic", strict_db=True,
                                      width=3200, height=1400)
            blocking = blocking_pwml_ir_errors(irrep)
            irval = validate_pwml_ir(ir)
            if blocking or irval.get("errors"):
                print(line + " | IR BLOCKED blocking=" + str(len(blocking))
                      + " val_errors=" + str(len(irval.get("errors") or [])))
                print("      blocking codes:", sorted({e.get("code") for e in blocking}))
                row[arm]["ir"] = "blocked"
                continue
            bargs = SimpleNamespace(name="C121 replay " + name, description="",
                                    subject="Metabolic", pw_id="PW000000",
                                    height=1400, width=3200,
                                    background_color="#FFFFFF", ref=str(REF))
            builder = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=bargs)
            build_result = builder.build()
            tree = etree.ElementTree(build_result.root)
            repaired_tree = repair_tree(tree, signature)
            vreport = validate_generated_tree(repaired_tree, signature)
            xml = etree.tostring(repaired_tree.getroot(), encoding="utf-8",
                                 xml_declaration=True, pretty_print=True)
            (OUT / (name + "." + arm + ".pwml")).write_bytes(xml)
            tree_errors = len(vreport.get("errors") or [])
            ref = REFERENCE_BYTES.get(name)
            refnote = ""
            if arm in ("repaired", "seam") and ref is not None:
                refnote = (" | MATCHES pre-dispatch reference" if len(xml) == ref
                           else " | DIFFERS from pre-dispatch reference " + str(ref))
            print(line + " | PWML " + str(len(xml)) + " bytes | tree_errors="
                  + str(tree_errors) + " | ir_reactions="
                  + str(len(ir.get("reactions") or [])) + refnote)
            row[arm].update(bytes=len(xml), tree_errors=tree_errors,
                            ir_reactions=len(ir.get("reactions") or []))
        except Exception as exc:
            print(line + " | RAISED " + type(exc).__name__ + ": " + str(exc)[:200])
            row[arm]["raised"] = type(exc).__name__ + ": " + str(exc)[:200]
    summary.append(row)

print("\n" + "=" * 74 + "\nSUMMARY")
for row in summary:
    rep = row.get("repaired") or {}
    seam = row.get("seam") if isinstance(row.get("seam"), dict) else {}
    base = row.get("base") or {}
    print("  " + row["leg"].ljust(22)
          + " archived_rx=" + str(row["archived_counts"]["reactions"])
          + " | base gate=" + str(base.get("gate_ok"))
          + " | repaired gate=" + str(rep.get("gate_ok"))
          + " bytes=" + str(rep.get("bytes"))
          + " tree_err=" + str(rep.get("tree_errors"))
          + " rx=" + str((rep.get("counts") or {}).get("reactions"))
          + " support_same=" + str(rep.get("support_identical_to_base"))
          + " | seam gate=" + str(seam.get("gate_ok"))
          + " bytes=" + str(seam.get("bytes"))
          + " rx=" + str((seam.get("counts") or {}).get("reactions"))
          + " support_same=" + str(seam.get("support_identical_to_base")))
print("\nREPLAY DONE")
