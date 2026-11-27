"""VIS-9: a compound written under its mapped PathBank id gets that compound's pwc-id (PathWhiz's structure key) only
when a full PathBank name held for that id agrees with the written name."""
from __future__ import annotations

from test_pwml_writer import _build_pwml_for_ir, _minimal_pwml_ir_with_compounds


def _legacy(name: str, pid: int, cand_name: str, **extra) -> dict:
    record = {
        "key": "cmp_1", "name": name, "pathwhiz_id": pid, "pathbank_compound_id": pid,
        "db_status": "legacy_id_unverified",
        "mapping_meta": {"pathbank_compound_id": pid, "resolved_name": cand_name,
                         "candidates": [{"pathbank_compound_id": pid, "name": cand_name, "short_name": name}]},
    }
    record.update(extra)
    return record


def _pwc(records: list[dict]) -> list:
    builder, root = _build_pwml_for_ir(_minimal_pwml_ir_with_compounds(records))
    assert [c.findtext("pwc-id") for c in root.findall(".//compounds/compound")] == [
        c.get("pwc-id") for c in builder.section_items["compounds"]]
    return [c.get("pwc-id") for c in builder.section_items["compounds"]]


def test_name_agreement_writes_derived_pwc_id() -> None:
    assert _pwc([_legacy("chorismate", 7748, "Chorismate")]) == ["PW_C007748"]
    assert _pwc([_legacy("serine", 120, "L-Serine")]) == ["PW_C000120"]
    assert _pwc([_legacy("Pyruvic acid", 108870, "Pyruvate")]) == ["PW_C108870"]


def test_short_name_or_other_name_never_counts() -> None:
    assert _pwc([_legacy("MK", 15980, "Melleolide K")]) == [None]  # short name "MK" is on the candidate
    assert _pwc([_legacy("noreugenin", 21430, "Isobiflorin")]) == [None]


def test_known_derivation_clash_and_repeat_are_skipped() -> None:
    assert _pwc([_legacy("UDP-N-acetylbacillosamine", 268987, "UDP-N-acetylbacillosamine")]) == [None]
    assert _pwc([_legacy("1,4-dihydroxy-2-naphthoic acid", 40747, "1,4-dihydroxy-2-naphthoate")]) == [None]
    second = dict(_legacy("L-serine", 120, "L-Serine"), key="cmp_2")
    assert _pwc([_legacy("serine", 120, "L-Serine"), second]) == ["PW_C000120", None]


def test_no_pathbank_id_writes_none() -> None:
    assert _pwc([{"key": "cmp_1", "name": "Chorismate", "pathwhiz_id": 7748}]) == [None]
