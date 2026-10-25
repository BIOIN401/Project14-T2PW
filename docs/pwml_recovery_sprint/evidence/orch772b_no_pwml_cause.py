"""ORCH-772b (D-114 § 3; ORCH-772 review r0 note 1): why a leg wrote no PWML, in the leg's own recorded words.

Read from the leg directory only. The first record, in pipeline order, that says the leg stopped:

1. ``quarantine_report.json`` with ``ok: false``: its ``refusal_reasons`` (the boundary refused; no gate ran after it);
2. ``final_stage3_gate_report.json`` with ``ok: false``: the ``reason`` of each error (the final pre-export gate);
3. ``RESULT.txt`` ``issue codes`` line (the driver's codes for the step that failed, e.g. the required-field gate);
4. ``RESULT.txt`` ``message`` line.

Report items are joined with ``"; "``, a ``RESULT.txt`` line is taken as written; nothing is reworded or inferred.
``""`` when the leg records none, and ``reaction_evidence.build_report`` then says no PWML file is present. The caller
passes it as ``build_report(..., no_pwml_cause=...)``, which uses it only when the leg holds no PWML file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _result_line(leg: Path, key: str) -> str:
    try:
        text = (leg / "RESULT.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    found = re.search(rf"^{key}\s*:\s*(.*)$", text, re.M)
    value = found.group(1).strip() if found else ""
    return "" if value in ("", "(none reported)", "(not recorded)") else value


def recorded_no_pwml_cause(leg) -> str:
    leg = Path(leg)
    quarantine = _load(leg / "quarantine_report.json")
    if isinstance(quarantine, dict) and quarantine.get("ok") is False and quarantine.get("refusal_reasons"):
        return "; ".join(dict.fromkeys(str(r) for r in quarantine["refusal_reasons"]))
    gate = _load(leg / "final_stage3_gate_report.json")
    if isinstance(gate, dict) and gate.get("ok") is False:
        reasons = [str(e.get("reason")) for e in gate.get("errors") or [] if isinstance(e, dict) and e.get("reason")]
        if reasons:
            return "; ".join(dict.fromkeys(reasons))
    return _result_line(leg, "issue codes") or _result_line(leg, "message")
