"""ORCH-772 -- keep the trimmed Reaction Evidence fixtures rendering like their committed legs.

``tests/test_reaction_evidence.py`` tests 2 and 13 require each trimmed bundle under
``tests/fixtures/reaction_evidence/`` to render byte-identically to the committed leg it was cut from
("unused keys dropped, every rendered value kept"). The D-114 section reads keys the earlier trim
dropped: the IR's reaction and transport names (to read "drawn"), a transport's cargo and states,
and an omitted interaction's endpoints. This copies exactly those keys from the committed leg into
the bundle, VERBATIM, and nothing else; then it proves the bundle renders like the leg, and prints
the keys it added. Run with ``PYTHONPATH=<tree>/src`` through ``bounded_run.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "reaction_evidence"
LEGS = {
    "enterobactin_strict": "runs_verify/2026-09-01_1612/papers/PMC12096016/strict",
    "lipid_a_strict": "runs_verify/2026-08-24_1428/papers/PMC12444477/strict",
    "enterobactin_research": "runs_verify/2026-08-21_2057/papers/PMC12096016/research",
    "enterobactin_refused_strict": "runs_verify/2026-08-18_1328/papers/PMC12452463/strict",
}
TRANSPORT_KEYS = ("transport_elements", "cargo", "cargo_complex", "from_biological_state", "to_biological_state")
ENDPOINT_KEYS = ("entity_1", "entity_2", "left", "right", "source", "target", "relationship", "interaction_type")


def _render(bundle: dict) -> str:
    from t2pw.app.reaction_evidence import build_report, load_run_artifacts, render_text

    with tempfile.TemporaryDirectory() as tmp:
        for name, content in bundle.items():  # as tests/test_reaction_evidence.py::write_leg does
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        return render_text(build_report(load_run_artifacts(tmp)))


def _copy(dst: dict, src: dict, keys: tuple, where: str, added: list) -> None:
    for key in keys:
        if key in src and key not in dst:
            dst[key] = src[key]
            added.append(f"{where}.{key}")


def main() -> int:
    from t2pw.app.reaction_evidence import load_run_artifacts, render_text, build_report

    bad = 0
    for name, leg in LEGS.items():
        path = FIXTURES / f"{name}.bundle.json"
        raw = path.read_text(encoding="utf-8")
        bundle = json.loads(raw)
        assert json.dumps(bundle, indent=1, ensure_ascii=False) == raw.rstrip("\n"), f"{name}: not a round-trip format"
        full = load_run_artifacts(REPO / leg)
        added: list = []
        fm_b, fm_f = bundle.get("final_mapped.json"), full.get("final_mapped.json")
        if isinstance(fm_b, dict) and isinstance(fm_f, dict):
            rows_b = fm_b.get("processes", {}).get("transports") or []
            rows_f = fm_f.get("processes", {}).get("transports") or []
            assert len(rows_b) == len(rows_f), name
            for i, (rb, rf) in enumerate(zip(rows_b, rows_f)):
                _copy(rb, rf, TRANSPORT_KEYS, f"final_mapped.json/processes/transports/{i}", added)
        q_b, q_f = bundle.get("quarantine_report.json"), full.get("quarantine_report.json")
        if isinstance(q_b, dict) and isinstance(q_f, dict):
            rows_b, rows_f = q_b.get("quarantined") or [], q_f.get("quarantined") or []
            assert len(rows_b) == len(rows_f), name
            for i, (rb, rf) in enumerate(zip(rows_b, rows_f)):
                if rf.get("process_kind") in ("interaction", "transport"):
                    _copy(rb.setdefault("original_process", {}), rf.get("original_process") or {},
                          ENDPOINT_KEYS + TRANSPORT_KEYS, f"quarantine_report.json/quarantined/{i}/original_process", added)
        ir_b, ir_f = bundle.get("pwml_ir.json"), full.get("pwml_ir.json")
        if isinstance(ir_b, dict) and isinstance(ir_f, dict):
            for bucket in ("reactions", "transports"):
                if bucket not in ir_b.setdefault("processes", {}):
                    ir_b["processes"][bucket] = [{"name": r.get("name")} for r in ir_f.get("processes", {}).get(bucket) or []]
                    added.append(f"pwml_ir.json/processes/{bucket}[*].name ({len(ir_b['processes'][bucket])})")
        same = _render(bundle) == render_text(build_report(full))
        bad += not same
        print(f"{name}: renders like {leg}: {same}; added {len(added)} key(s)")
        for key in added:
            print(f"  + {key}")
        path.write_bytes((json.dumps(bundle, indent=1, ensure_ascii=False) + "\n").replace("\n", "\r\n").encode("utf-8"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
