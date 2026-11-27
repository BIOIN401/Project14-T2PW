"""VIS-9: a compound written under its mapped PathBank id gets that compound's pwc-id (PathWhiz's structure key) only
when a full PathBank name held for that id agrees with the written name, and the id is not on the skip list."""
from __future__ import annotations

import pytest

from test_pwml_writer import _build_pwml_for_ir, _minimal_pwml_ir_with_compounds
from t2pw.pwml import writer


def _legacy(name: str, pid: int, cand_name: str, key: str = "cmp_1", **extra) -> dict:
    record = {
        "key": key, "name": name, "pathwhiz_id": pid, "pathbank_compound_id": pid,
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


@pytest.mark.parametrize("ours, theirs", [
    ("chorismate", "Chorismate"), ("serine", "L-Serine"), ("Pyruvic acid", "Pyruvate"), ("NAD+", "NAD"),
    ("pantoate", "(R)-pantoate"), ("β-alanine", "beta-Alanine"), ("L-serine", "L-Serine"),
])
def test_names_that_agree(ours, theirs) -> None:
    assert writer._same_compound_name(ours, theirs)


@pytest.mark.parametrize("ours, theirs", [
    ("D-serine", "L-Serine"), ("(S)-pantoate", "(R)-pantoate"), ("β-alanine", "L-Alanine"),
    ("α-alanine", "L-Alanine"), ("α-D-glucose", "β-D-glucose"), ("MK", "Melleolide K"), ("noreugenin", "Isobiflorin"),
])
def test_names_that_do_not_agree(ours, theirs) -> None:
    assert not writer._same_compound_name(ours, theirs)


def test_name_agreement_writes_derived_pwc_id() -> None:
    assert _pwc([_legacy("chorismate", 7748, "Chorismate")]) == ["PW_C007748"]
    assert _pwc([_legacy("serine", 120, "L-Serine")]) == ["PW_C000120"]
    assert _pwc([_legacy("D-serine", 120, "L-Serine")]) == [None]


def test_short_name_or_other_name_never_counts() -> None:
    assert _pwc([_legacy("MK", 15980, "Melleolide K")]) == [None]  # short name "MK" is on the candidate
    assert _pwc([_legacy("noreugenin", 21430, "Isobiflorin")]) == [None]


@pytest.mark.parametrize("pid, name", [
    (268987, "UDP-N-acetylbacillosamine"),        # derived pwc_id belongs to another compound
    (247818, "Ceramide"),                         # pwc_id is not PW_C%06d(id)
    (40747, "1,4-dihydroxy-2-naphthoate"),        # wrong PathBank structure
    (40974, "4'-phosphopantetheine"), (2580, "Geraniol"), (442, "Cinnamic acid"),
])
def test_skip_list(pid, name) -> None:
    assert _pwc([_legacy(name, pid, name)]) == [None]


def test_missing_skip_list_derives_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(writer, "_PWC_ID_SKIP_PATH", tmp_path / "absent.json")
    writer._pwc_id_skip.cache_clear()
    try:
        assert _pwc([_legacy("chorismate", 7748, "Chorismate")]) == [None]
    finally:
        writer._pwc_id_skip.cache_clear()


def test_repeat_pwc_ids_are_skipped() -> None:
    assert _pwc([_legacy("serine", 120, "L-Serine"), _legacy("L-serine", 120, "L-Serine", key="cmp_2")]) == [
        "PW_C000120", None]
    trusted = {"key": "cmp_2", "name": "L-Serine", "pathwhiz_id": 120, "db_status": "matched",
               "db_row": {"id": 120, "name": "L-Serine", "pwc_id": "PW_C000120"}}
    assert _pwc([_legacy("serine", 120, "L-Serine"), trusted]) == [None, "PW_C000120"]


def test_no_pathbank_id_writes_none() -> None:
    assert _pwc([{"key": "cmp_1", "name": "Chorismate", "pathwhiz_id": 7748}]) == [None]
