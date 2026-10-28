"""ORCH-772 (D-114 §§ 1, 3) -- PWML coverage and the omission record in the Reaction Evidence .txt.

**G9 LABEL: NEW ACCEPTANCE TESTS FOR A NEW CAPABILITY.** Neither the coverage figure, the section
"OMITTED FROM THE PWML", the known-issue table nor the corpus coverage exist at base ``3ef3fa9c``; no base
failure is claimed or fabricated. Nothing here changes a verdict: the module only reads artifacts a run
already wrote (``t2pw.app.reaction_evidence``), and the corpus harness only adds a record to each replay.

* test 1 is the alarm for the code table and ``PWML_KNOWN_ISSUES.md`` drifting apart;
* tests 2-8 pin the coverage arithmetic and the section on synthetic artifact sets;
* test 9 is the committed sample leg; tests 10-11 the ORCH-768 harness on its three fixture legs.
"""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from t2pw.app.reaction_evidence import (
    KNOWN_ISSUES,
    build_report,
    known_issue_id,
    load_run_artifacts,
    render_text,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "docs" / "pwml_recovery_sprint" / "PWML_KNOWN_ISSUES.md"
SAMPLE_LEG = ROOT / "runs_verify" / "2026-09-01_1612" / "papers" / "PMC12096016" / "strict"
HARNESS = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "orch768_replay_corpus.py"
FIXTURE = ROOT / "tests" / "fixtures" / "orch768"


# 1 -------------------------------------------------------------------------
def _catalogue():
    text = CATALOGUE.read_text(encoding="utf-8")
    appendix = text[text.index("## Appendix — Reason code → KI id"):]
    codes, messages, table = [], [], None
    for line in appendix.splitlines():
        if line.startswith("| reason code |"):
            table = codes
        elif line.startswith("| message |"):
            table = messages
        elif line.startswith("| ") and table is not None:
            table.append([c.strip() for c in line.strip().strip("|").split("|")])
    parsed_codes = []
    for cells in codes:
        code = re.search(r"`([^`]+)`", cells[0]).group(1)
        found = re.findall(r"(prefix|suffix) `([^`]+)`", cells[1])
        if found:
            match, patterns = "+".join(k for k, _ in found), tuple(p for _, p in found)
        else:
            assert cells[1].startswith("exact"), cells
            match, patterns = "exact", (code,)
        parsed_codes.append((code, match, patterns, re.search(r"KI-\d{3}", cells[2]).group(0), "then label by" in cells[1]))
    parsed_messages = [
        (tuple(p.replace("`", "") for p in re.findall(r'"([^"]+)"', cells[0])), re.search(r"KI-\d{3}", cells[1]).group(0))
        for cells in messages
    ]
    workarounds, current = {}, None
    for line in text.splitlines():
        heading = re.match(r"### (KI-\d{3}) ", line)
        current = heading.group(1) if heading else current
        if line.startswith("| **User workaround** |"):
            workarounds[current] = line[len("| **User workaround** |"):].strip().rstrip("|").strip()
    return parsed_codes, parsed_messages, workarounds


def test_the_code_table_is_the_catalogue_appendix():
    codes, messages, workarounds = _catalogue()
    # The appendix as ORCH-771 wrote it (82, 19), plus ORCH-773's two rows: the
    # ``protein_complex_component_omitted:`` code and the "Located protein is isolated" message.
    assert len(codes) == 84 and len(messages) == 20  # ORCH-788 (89038702): + biological_state_missing_subcellular_location
    assert list(KNOWN_ISSUES["codes"]) == codes
    assert list(KNOWN_ISSUES["messages"]) == messages
    labels = {row[3] for row in codes} | {ki for _patterns, ki in messages}
    assert set(KNOWN_ISSUES["workarounds"]) == labels
    for ki, text in KNOWN_ISSUES["workarounds"].items():
        assert text == workarounds[ki], ki


# 2 -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reason, ki",
    [
        ("protein_missing_external_identity", "KI-001"),
        ("ambiguous_entity_type:nucleic_acids+proteins", "KI-011"),
        ("protein_missing_name", "KI-015"),  # suffix
        ("missing_inputs", "KI-015"),  # prefix
        ("gate.protein_menx_is_missing_a_uniprot_or_drugbank_identifier", "KI-001"),  # prefix and suffix
        ("gate.protein_has_degree_after_normalization", "KI-013"),
        ("gate.generated_protein_complex_enterobactin_synthase_complex_componen", "KI-012"),
        ("protein_complex_component_unrepresentable:EntB:protein_missing_external_identity", "KI-001"),  # inner reason
        ("protein_complex_component_unrepresentable:EntB:no_such_reason", "KI-015"),
        # ORCH-773 rows (NEW acceptance): labelled by the omitted component's own cause, else KI-015.
        ("protein_complex_component_omitted:EntB:protein_missing_external_identity", "KI-001"),
        ("protein_complex_component_omitted:EntB:no_such_reason", "KI-015"),
        ("Located protein is isolated in connectivity graph: EntX", "KI-013"),
        ("untyped_interaction", "KI-008"),
        ("duplicate_locked_reaction", "KI-014"),
        ("Generated protein complex 'enterobactin synthase complex' component 'enterobactin synthase' is missing "
         "positive stoichiometry.", "KI-012"),
        ("Generated protein complex 'X complex' component protein 'X' is missing a UniProt or DrugBank identifier.", "KI-001"),
        ("Protein 'MenD' is missing a UniProt or DrugBank identifier.", "KI-001"),
        ("Unknown protein/modifier reference: PIF1", "KI-004"),
        # The narrower message wins over the broad "PWML export ..." of KI-028.
        ("PWML export stopped by the quarantine boundary: unexportable_entity:1", "KI-016"),
        ("PWML export did not finish inside the time budget", "KI-028"),
        ("coarse_grained_same_entity_transformation", "unclassified"),
        ("", "unclassified"),
    ],
)
def test_a_reason_is_labelled_as_the_appendix_says(reason, ki):
    assert known_issue_id(reason) == ki


# synthetic artifact sets ---------------------------------------------------
WORKAROUND_001 = KNOWN_ISSUES["workarounds"]["KI-001"]


def _artifacts(pwml: bool = True) -> dict:
    payload = {
        "entities": {"compounds": [{"name": "X"}, {"name": "Y"}, {"name": "A"}, {"name": "B"}]},
        "processes": {
            "reactions": [{"name": "R1", "inputs": ["A"], "outputs": ["B"]}],
            "transports": [{"name": "T1", "transport_elements": [{"element": "A"}],
                            "from_biological_state": "outside", "to_biological_state": "cytosol"}],
            "interactions": [
                {"name": "X competes with Y", "entity_1": "X", "entity_2": "Y", "relationship": "competes with"},
                {"name": "X activates Y", "entity_1": "X", "entity_2": "Y", "relationship": "activates"},
            ],
        },
        "quarantined_locked_reactions": [
            {"locked_reaction_id": "rxn_lock_009", "reaction_name": "R1 twin", "reason": "duplicate_locked_reaction",
             "json_pointer": "/processes/reactions/3", "original_reaction": {"name": "R1 twin", "inputs": ["A"], "outputs": ["B"]}},
            {"locked_reaction_id": "rxn_lock_010", "reaction_name": "R gone", "reason": "missing_inputs",
             "original_reaction": {"name": "R gone", "outputs": ["B"]}},
        ],
    }
    quarantine = {
        "ok": True,
        "quarantined": [
            {"process_kind": "reaction", "name": "R quarantined", "pointer": "/processes/reactions/1",
             "state": "quarantined_unmapped_entity", "reason": "protein_missing_external_identity",
             "original_process": {"name": "R quarantined", "inputs": ["A"], "outputs": ["Y"], "locked_reaction_id": "rxn_lock_011"},
             "detail": {"omission": {"gate": "final_pre_export_stage3_gates", "cause": "process", "slot": "enzymes",
                                     "name": "EntX", "failing_entity": {"bucket": "proteins", "name": "EntX"}}}},
            {"process_kind": "reaction_coupled_transport", "name": "RCT", "pointer": "/processes/reaction_coupled_transports/0",
             "reason": "reaction_coupled_transport_has_no_exportable_ir_representation", "original_process": {"name": "RCT"}},
        ],
    }
    ir = {
        "entities": {"compounds": [{"key": "c_x", "name": "X"}, {"key": "c_y", "name": "Y"}]},
        "processes": {
            "reactions": [{"key": "rxn_1", "name": "R1"}],
            "transports": [{"key": "tr_1", "name": "T1"}],
            "interactions": [
                {"key": "int_1", "name": "X competes with Y", "interaction_type": "competes with", "export_status": "withheld",
                 "withheld": {"reason": "untyped_interaction", "raw_interaction_type": "competes with", "left": "X", "right": "Y"}},
                {"key": "int_2", "name": "X activates Y", "interaction_type": "Activation", "export_status": "exportable",
                 "left": {"key": "int_2_left", "entity_key": "c_x"}, "right": {"key": "int_2_right", "entity_key": "c_y"}},
            ],
        },
    }
    removed = {"removed_entities": [
        {"bucket": "proteins", "name": "EntX", "reason": "protein_missing_external_identity",
         "detail": {"gate": "final_pre_export_stage3_gates", "omission": "entity_omitted_per_row"}},
    ]}
    arts = {"final_mapped.json": payload, "quarantine_report.json": quarantine, "pwml_ir.json": ir,
            "removed_entity_report.json": removed}
    if pwml:
        arts["pathway.review_required.pwml"] = True
    return arts


def _section(text: str) -> str:
    return text[text.index("\nOMITTED FROM THE PWML ("): text.index("\nCORE COVERAGE\n")]


def _block(text: str, name: str) -> str:
    at = text.index(f"\nName: {name}\n", text.index("\nOMITTED FROM THE PWML ("))
    return text[text.rindex("\nOMITTED ", 0, at) + 1: text.index("\n\n", at)]


# 3 -------------------------------------------------------------------------
def test_coverage_is_printed_right_under_the_title_with_the_by_design_line():
    report = build_report(_artifacts())
    lines = render_text(report).splitlines()
    # ORCH-772b (D-114 AMENDMENT 2 § 1; merge rule 4): the headline leaves the KI-008 interaction out of the
    # denominator, the full count follows it. Before: "3 of 6 processes (50.0%)" headline, "3 of 5" secondary.
    assert lines[:3] == ["PATHWAY EVIDENCE REPORT", "=======================",
                         "PWML coverage: 3 of 5 processes (60.0%) -- target 75% (D-114 AMENDMENT 2: the 1 interactions "
                         "PathWhiz cannot carry, KI-008, are not counted)"]
    assert lines[3] == "  Full count, the 1 KI-008 interactions included: 3 of 6 (50.0%)"
    assert lines[4] == ("  By kind, drawn of full count (total = final payload + omitted before the freeze): "
                        "reactions 1 of 3; transports 1 of 1; interactions 1 of 2")
    assert lines[5] == "  Listed but not counted: 2 (merged duplicates: 1)"
    assert "Core-term coverage: Not recorded" in lines  # the core-term line stays, relabelled
    cov = report["pwml_coverage"]
    assert (cov["drawn"], cov["total"], cov["by_design_withheld"], cov["pwml_written"]) == (3, 6, 1, True)


# 4 -------------------------------------------------------------------------
def test_every_omitted_process_is_listed_with_its_verbatim_reason_ki_and_workaround():
    text = render_text(build_report(_artifacts()))
    assert "OMITTED FROM THE PWML (3)\n" in text
    quarantined = _block(text, "R quarantined")
    assert quarantined.splitlines() == [
        "OMITTED 1  [quarantine_report.json quarantined; pre-quarantine /processes/reactions/1; lock rxn_lock_011]",
        "Kind: reaction",
        "Name: R quarantined",
        "Equation: A -> Y",
        "Reason: protein_missing_external_identity",
        "Omission: gate=final_pre_export_stage3_gates; cause=process; slot=enzymes; name=EntX; failing entity=proteins EntX",
        "Known issue: KI-001",
        f"Add by hand: {WORKAROUND_001}",
    ]
    gone = _block(text, "R gone")
    assert "Reason: missing_inputs" in gone and "Known issue: KI-015" in gone
    assert "[final payload quarantined_locked_reactions; lock rxn_lock_010]" in gone
    withheld = _block(text, "X competes with Y")
    assert "Endpoints: X -[competes with]-> Y" in withheld and "Reason: untyped_interaction" in withheld
    assert "Known issue: KI-008" in withheld and "[pwml_ir.json; final payload /processes/interactions/0]" in withheld
    assert "Name: X activates Y" not in _section(text)  # drawn: not omitted
    assert "Name: R1\n" not in _section(text)
    assert "- proteins EntX: protein_missing_external_identity (KI-001); gate final_pre_export_stage3_gates" in text


# 5 -------------------------------------------------------------------------
def test_a_merged_duplicate_and_an_uncounted_kind_are_listed_but_not_counted():
    text = render_text(build_report(_artifacts()))
    section = _section(text)
    assert ("- merged duplicate, not counted: R1 twin  [final payload quarantined_locked_reactions; "
            "json_pointer /processes/reactions/3; lock rxn_lock_009]: duplicate_locked_reaction (KI-014)") in section
    assert ("- reaction_coupled_transport, not a counted kind: RCT  [quarantine_report.json quarantined; "
            "pre-quarantine /processes/reaction_coupled_transports/0]: "
            "reaction_coupled_transport_has_no_exportable_ir_representation (KI-015)") in section
    assert "OMITTED 4" not in section


# 6 -------------------------------------------------------------------------
def test_no_pwml_written_draws_nothing_and_lists_every_payload_process():
    arts = _artifacts(pwml=False)
    report = build_report(arts)
    text = render_text(report)
    assert text.splitlines()[2] == ("PWML coverage: 0 of 6 processes (0.0%) -- target 75% (D-114 AMENDMENT 2: the 0 "
                                    "interactions PathWhiz cannot carry, KI-008, are not counted)")
    # The IR's verdicts are not read without a PWML, so nothing is "withheld by design" either.
    assert "  Full count, the 0 KI-008 interactions included: 0 of 6 (0.0%)" in text
    assert "OMITTED FROM THE PWML (6)\n" in text
    assert "Not drawn because no PWML was written (4):\n" in text
    for line in ("- reaction R1: A -> B  [final payload /processes/reactions/0]",
                 "- transport T1: A: outside -> cytosol  [final payload /processes/transports/0]",
                 "- interaction X competes with Y: X -[competes with]-> Y  [final payload /processes/interactions/0]",
                 "- interaction X activates Y: X -[activates]-> Y  [final payload /processes/interactions/1]"):
        assert line in text
    # The pre-freeze removals keep their full blocks and reasons.
    assert "Reason: protein_missing_external_identity" in _block(text, "R quarantined")
    assert report["pwml_coverage"]["drawn"] == 0


# 7 -------------------------------------------------------------------------
@pytest.mark.parametrize("edit", ["refused", "research"])
def test_quarantine_rows_still_in_the_payload_are_not_counted_twice(edit):
    arts = _artifacts(pwml=False)
    if edit == "refused":
        arts["quarantine_report.json"]["ok"] = False
    else:
        arts["quarantine_report.json"]["research_mode"] = {"applied": False}
    cov = build_report(arts)["pwml_coverage"]
    # payload 4 + "R gone" (the merged twin is not counted); the quarantine rows are flags, not removals.
    assert (cov["drawn"], cov["total"]) == (0, 5)


# 8 -------------------------------------------------------------------------
def test_an_ir_that_does_not_line_up_is_not_recorded_never_guessed():
    arts = _artifacts()
    arts["pwml_ir.json"]["processes"]["reactions"][0]["name"] = "another payload's reaction"
    report = build_report(arts)
    text = render_text(report)
    assert report["pwml_coverage"]["drawn"] is None
    assert text.splitlines()[2].startswith("PWML coverage: Not recorded (the PWML verdict of some process is not recorded")
    assert "reactions Not recorded of 3" in text
    # Not called omitted: it may be drawn. Listed apart, with why its verdict is not recorded.
    assert ("Whether the PWML draws these is not recorded (1):\n- reaction R1: A -> B  [final payload /processes/reactions/0]: "
            "Not recorded (pwml_ir.json does not line up with the reactions of the final payload)\n") in _section(text)
    assert "OMITTED FROM THE PWML (3)\n" in text
    # A missing IR beside a written PWML is not recorded either.
    del arts["pwml_ir.json"]
    assert build_report(arts)["pwml_coverage"]["drawn"] is None


# 9 -------------------------------------------------------------------------
def test_the_committed_sample_leg_reads_its_old_ir_as_not_recorded():
    """The leg's IR predates per-row verdicts: its interactions' drawn state is not recorded."""

    report = build_report(load_run_artifacts(SAMPLE_LEG))
    text = render_text(report)
    assert text.splitlines()[2].startswith("PWML coverage: Not recorded")
    assert "reactions 5 of 6; transports 1 of 1; interactions Not recorded of 3" in text
    block = _block(text, "EntD-catalyzed phosphopantetheinylation of EntB")
    assert "Reason: protein_missing_external_identity" in block and "Known issue: KI-001" in block
    assert f"Add by hand: {WORKAROUND_001}" in block
    assert "OMITTED FROM THE PWML (1)\n" in text and "Whether the PWML draws these is not recorded (3):\n" in text


# 10-11: the ORCH-768 corpus harness on its three fixture legs --------------------------------
def _load_harness():
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus_772", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    H = _load_harness()
    legs = json.loads((FIXTURE / "legs.json").read_text(encoding="utf-8"))["legs"]
    inputs = tmp_path_factory.mktemp("orch772_inputs")
    for leg in legs:
        for meta in leg["inputs"].values():
            if meta["path"]:
                raw = meta["path"][: -len(".gz")]
                (inputs / raw).parent.mkdir(parents=True, exist_ok=True)
                (inputs / raw).write_bytes(gzip.decompress((FIXTURE / meta["path"]).read_bytes()))
                meta["path"] = raw
    out = tmp_path_factory.mktemp("orch772_replay")
    try:
        T = H.load_tree(None)
        records = {leg["id"]: H.replay_leg(T, leg, out / leg["id"].replace("/", "_"), primary=inputs) for leg in legs}
    finally:
        H.Offline.uninstall()
    return SimpleNamespace(H=H, T=T, records=records, legs=legs)


def test_each_replayed_leg_carries_the_measured_trees_coverage(replayed):
    assert Path(replayed.T.evidence.__file__).resolve() == (ROOT / "src" / "t2pw" / "app" / "reaction_evidence.py").resolve()
    for leg_id, record in replayed.records.items():
        cov = record["pwml_coverage"]
        assert cov["pwml_written"] is record["pwml_written"], leg_id
        assert cov["drawn"] is not None and 0 <= cov["drawn"] <= cov["total"], leg_id
        if not record["pwml_written"]:
            assert cov["drawn"] == 0, leg_id
        assert sum(n for n, *_ in record["pwml_coverage"]["omitted_by_reason"]) == cov["total"] - cov["drawn"], leg_id
        assert record["network_refused"] == [], leg_id
    got = {i: (r["pwml_coverage"]["drawn"], r["pwml_coverage"]["total"], r["pwml_coverage"]["by_design_withheld"])
           for i, r in replayed.records.items()}
    # Sour taste equals what its own leg directory gives (7/12): on a canonical checkpoint the leg's recorded
    # pre-freeze removals are carried (5 rows). WNT PCP writes a PWML at this tree (ORCH-769c) and draws 1 of
    # 66; 17 of its interactions are untyped (KI-008). Enterobactin was 0/21 (no PWML, as its leg directory
    # gives) until ORCH-773: the inferred wrapper leaves its reaction (D-113 s1), a PWML is written, and it
    # draws 10 of 21; 7 of its interactions are untyped (KI-008).
    assert got == {
        "ORCH-767/enterobactin-strict/2026-10-23_2224": (10, 21, 7),
        "ORCH-767/sour_taste_v2-strict/2026-10-24_0042": (7, 12, 0),
        "ORCH-767/wnt_pcp-strict/2026-10-24_0131": (1, 66, 17),
    }


def test_a_tree_without_coverage_reports_na_and_the_aggregates_say_so(replayed):
    H = replayed.H
    bare = SimpleNamespace(**{**vars(replayed.T), "evidence": None})
    assert H.pwml_coverage(bare, {}, {"processes": {}}, None, None) is None
    rows = [{"id": i, "recorded": "pwml", "replay": copy.deepcopy(r)} for i, r in replayed.records.items()]
    summary = H.coverage_summary(rows)
    assert summary["replayed.headline"]["legs"] == 3 and summary["replayed.headline"]["measured"] == 3
    for row in rows:
        row["class"] = H.classify({"recorded": None}, row["replay"])
    printed: list = []
    H.print_coverage(summary, rows, printed.append)
    # ORCH-772b (D-114 AMENDMENT 2 § 2; merge rule 4): the target line on every strict leg comes first, with the PWML
    # pass rate; then 91 / replayed / PWML-writing legs, headline and full count. Before: the stats lines came first.
    assert printed[1].lstrip().startswith("TARGET (D-114 AMENDMENT 2, every strict leg, no PWML = 0%): legs >= 75%: ")
    assert "PWML pass rate 3/3" in printed[1]  # round 1: ORCH-773 makes the enterobactin fixture leg export (was 2/3)
    assert [p.split()[0] for p in printed[2:8]] == [f"{d}.{v}" for d in ("strict_legs_with_an_outcome", "replayed", "pwml_written")
                                                     for v in ("headline", "full_count")]
    assert "omitted processes by known issue" in printed[8]
    rows[0]["replay"]["pwml_coverage"] = None
    assert "n/a" in H.coverage_summary(rows)
    # An LLM-originating leg (no replay) is 0% on the strict denominator and absent from the replayed one.
    rows[0] = {"id": "llm", "recorded": "other:extraction", "replay": None}
    summary = H.coverage_summary(rows)
    assert summary["strict_legs_with_an_outcome.headline"]["legs"] == 3
    assert summary["replayed.headline"]["legs"] == 2
    assert H.leg_percent(rows[0]) == 0.0
