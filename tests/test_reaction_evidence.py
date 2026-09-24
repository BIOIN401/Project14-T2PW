"""ORCH-742b -- reaction-evidence review model and .txt export.

**G9 LABEL: NEW ACCEPTANCE TESTS FOR A NEW CAPABILITY.** ``t2pw.app.reaction_evidence``
and its Streamlit view do not exist at the base SHA ``9f13188e``; no base failure is
claimed or fabricated. Nothing here touches extraction, RAG, mapping, quarantine or
export: the module only reads artifacts a run already wrote.

The fixture ``tests/fixtures/reaction_evidence/enterobactin_strict.bundle.json`` is a
TRIMMED copy of the committed leg ``runs_verify/2026-09-01_1612/papers/PMC12096016/strict``
(unused keys dropped, every rendered value kept); test 2 proves the trim renders
identically to the full leg. It is ONE ``{filename: content}`` bundle written out into
``tmp_path`` per test, never a committed ``quarantine_report.json`` / ``final_mapped.json``:
those names are counted repo-wide by pinned censuses (``git ls-files
"*quarantine_report.json"`` in c102/c106, ``"*final_mapped.json"`` in c030/c073/c075),
and a committed fixture under either name turned c106 test 08 red. No Streamlit import.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from t2pw.app.reaction_evidence import (
    DUPLICATE_WARNING,
    FINAL_MAPPED_FILENAME,
    NOT_RECORDED,
    build_report,
    load_run_artifacts,
    render_text,
)

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "tests" / "fixtures" / "reaction_evidence" / "enterobactin_strict.bundle.json"
COMMITTED_LEG = REPO / "runs_verify" / "2026-09-01_1612" / "papers" / "PMC12096016" / "strict"
SAMPLE = REPO / "docs" / "reaction_evidence_sample.txt"


def write_leg(target: Path, bundle: Path = BUNDLE) -> Path:
    """Write the bundle out as a leg directory: one file per artifact, real names."""

    target.mkdir(parents=True, exist_ok=True)
    for name, content in json.loads(bundle.read_text(encoding="utf-8")).items():
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        (target / name).write_text(text, encoding="utf-8")
    return target


@pytest.fixture()
def leg(tmp_path: Path) -> Path:
    return write_leg(tmp_path / "leg")


def _report(directory: Path):
    return build_report(load_run_artifacts(directory))


def _edit_payload(directory: Path, edit) -> None:
    path = directory / FINAL_MAPPED_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _all_strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _all_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _all_strings(value)
    elif isinstance(node, str):
        yield node


# 1 -------------------------------------------------------------------------
def test_report_is_extracted_from_existing_run_artifacts(leg):
    report = _report(leg)

    assert report["paper_id"] == "PMC12096016"
    assert report["pathway"] == "enterobactin biosynthesis"
    assert report["requested_organism"] == "Escherichia coli"
    assert report["species"] == ["Escherichia coli (taxonomy 562)"]
    # Taken from coverage_summary.json as written, never recomputed.
    assert report["coverage_ratio"] == "0.75"
    assert report["minimum_core_satisfied"] == "true"
    assert (report["surviving_processes"], report["quarantined_processes"]) == ("9", "1")
    assert report["unmatched_terms"] == ["NADH", "ATP", "EntD", "Fur"]
    assert len(report["matched_terms"]) == 12
    assert report["release_status"] == "review_required"

    first = report["reactions"][0]
    assert len(report["reactions"]) == 5
    assert (first["pointer"], first["lock_id"]) == ("/processes/reactions/0", "rxn_lock_003")
    assert first["equation"] == "chorismate -> isochorismate"
    assert first["enzymes"] == ["Isochorismate synthase (protein_complex; components: EntC)"]
    assert first["species"] == ["Escherichia coli"]
    assert first["admission"] == "core_accepted"
    assert first["mapping"][0].startswith("chorismate: matched; db_status=legacy_id_unverified;")
    assert "chebi_id=17333" in first["mapping"][0]
    assert first["sources"] == ["supplied paper PMC12096016 via paper_extraction"]

    # Every quotation shown is a verbatim string already present in the artifacts.
    on_disk = set(_all_strings(json.loads(BUNDLE.read_text(encoding="utf-8"))))
    shown = [q for r in report["reactions"] for q in r["evidence"]]
    shown += [q for r in report["quarantined"] for q in r["evidence"]]
    assert shown and set(shown) <= on_disk


# 2 -------------------------------------------------------------------------
def test_committed_sample_is_this_module_run_over_the_committed_leg(leg):
    full = render_text(_report(COMMITTED_LEG))
    assert full == render_text(_report(leg)), "the trimmed fixture must render like the full leg"
    # read_text normalizes CRLF, so this holds on an autocrlf checkout too.
    assert full == SAMPLE.read_text(encoding="utf-8")


# 3 -------------------------------------------------------------------------
def test_missing_optional_artifacts_read_not_recorded(tmp_path):
    empty = build_report(load_run_artifacts(tmp_path))
    text = render_text(empty)
    for line in ("Paper: Not recorded", "Coverage: Not recorded", "Minimum core satisfied: Not recorded",
                 "QUARANTINED / REJECTED (Not recorded)", "REMOVED ENTITIES (Not recorded)",
                 "Matched: Not recorded", "Surviving reactions read from: Not recorded"):
        assert line in text
    assert empty["reactions"] == [] and empty["artifacts"] == []

    # Only the coverage summary: a recorded False is a value, never "Not recorded".
    coverage = json.loads(BUNDLE.read_text(encoding="utf-8"))["coverage_summary.json"]
    coverage["minimum_core_satisfied"] = False
    coverage["unmatched_terms"] = []
    (tmp_path / "coverage_summary.json").write_text(json.dumps(coverage), encoding="utf-8")
    (tmp_path / "quarantine_report.json").write_text(json.dumps({"quarantined": []}), encoding="utf-8")
    text = render_text(build_report(load_run_artifacts(tmp_path)))
    assert "Minimum core satisfied: false" in text
    assert "Unmatched (0): none" in text
    title = "SURVIVING REACTIONS (0)"
    assert f"{title}\n{'-' * len(title)}\n{NOT_RECORDED}\n" in text
    title = "QUARANTINED / REJECTED (0)"
    assert (
        f"{title}\n{'-' * len(title)}\nFrom quarantine_report.json quarantined: 0\n"
        f"From final payload quarantined_locked_reactions: {NOT_RECORDED}\n\nNone recorded\n"
    ) in text

    # Unreadable JSON is treated as absent, not as a crash.
    (tmp_path / FINAL_MAPPED_FILENAME).write_text("{not json", encoding="utf-8")
    assert FINAL_MAPPED_FILENAME not in load_run_artifacts(tmp_path)


# 4 -------------------------------------------------------------------------
def test_multiple_evidence_quotations_are_all_kept_in_order(leg):
    reactions = _report(leg)["reactions"]
    assert reactions[4]["evidence"] == [
        "Enterobactin production from activated 2,3-DHB then involves three EntF-catalyzed cycles of condensation with L-serine",
        "three EntF-catalyzed cycles of condensation with L-serine",
        "EntE (2,3-dihydroxybenzoyl-AMP ligase; EC 6.2.1.71)",
        "catalyzed by the enterobactin synthase complex, which includes EntD, EntE, and EntF",
    ]
    # Reaction 1's enzyme evidence repeats the reaction's own: shown once, not twice.
    assert reactions[0]["evidence"] == ["EntC (isochorismate synthase; EC 5.4.4.2)"]
    text = render_text(_report(leg))
    assert '  4. "catalyzed by the enterobactin synthase complex, which includes EntD, EntE, and EntF"' in text


# 5 -------------------------------------------------------------------------
def test_quarantined_reaction_is_listed_with_reason_and_verbatim_evidence(leg):
    report = _report(leg)
    (row,) = report["quarantined"]
    assert row["name"] == "EntD-catalyzed phosphopantetheinylation of EntB"
    assert row["equation"] == "apo-EntB -> holo-EntB"
    assert (row["state"], row["reason"]) == ("quarantined_unmapped_entity", "protein_missing_external_identity")
    assert (row["essential_participant"], row["lock_id"]) == ("apo-EntB", "rxn_lock_004")
    assert row["evidence"][0].startswith("the ArCP domain of apo-EntB itself must be phosphopantetheinylated")
    assert row["name"] not in {r["name"] for r in report["reactions"]}
    text = render_text(report)
    assert "QUARANTINED / REJECTED (1)" in text and "Reason: protein_missing_external_identity" in text
    # The report's elided ``detail.row`` copy is never quoted.
    assert "chars elided" not in text
    assert "- proteins apo-EntB: degree_zero_after_quarantine" in text


# 6 -------------------------------------------------------------------------
def test_txt_is_byte_deterministic(leg, tmp_path):
    first = render_text(_report(leg)).encode("utf-8")
    second = render_text(_report(leg)).encode("utf-8")
    moved = render_text(_report(write_leg(tmp_path / "elsewhere" / "copy"))).encode("utf-8")
    assert first == second == moved
    assert str(leg) not in first.decode("utf-8") and b"\r" not in first


# 7 -------------------------------------------------------------------------
def test_unicode_biochemical_names_survive_and_still_map(leg):
    def edit(payload):
        payload["entities"]["compounds"][0]["name"] = "α-D-glucose 6-phosphate"
        reaction = payload["processes"]["reactions"][0]
        reaction["inputs"] = ["α-D-Glucose  6-phosphate", {"name": "Ca²⁺", "stoichiometry": 2}]
        reaction["outputs"] = ["β-D-fructose 6-phosphate", "NAD⁺"]
        reaction["evidence"] = "α-D-glucose 6-phosphate → β-D-fructose 6-phosphate (Km ≈ 2 µM)"

    _edit_payload(leg, edit)
    report = _report(leg)
    first = report["reactions"][0]
    assert first["equation"] == "α-D-Glucose  6-phosphate + 2 Ca²⁺ -> β-D-fructose 6-phosphate + NAD⁺"
    # NFKC + casefold + whitespace: the entity is found despite case and spacing.
    assert first["mapping"][0].startswith("α-D-Glucose  6-phosphate: matched;")
    assert first["mapping"][1] == f"Ca²⁺: {NOT_RECORDED}"
    data = render_text(report).encode("utf-8")
    assert "α-D-glucose 6-phosphate → β-D-fructose 6-phosphate (Km ≈ 2 µM)" in data.decode("utf-8")


# 8 -------------------------------------------------------------------------
def test_duplicate_chemistry_is_flagged_and_nothing_is_removed(leg):
    def edit(payload):
        reactions = payload["processes"]["reactions"]
        twin = copy.deepcopy(reactions[0])
        twin["name"] = "second extraction of the EntC step"
        twin["inputs"], twin["outputs"] = [" Chorismate "], ["ISOCHORISMATE"]
        reactions.append(twin)
        near = copy.deepcopy(reactions[0])
        near["outputs"] = ["isochorismate", "pyruvate"]  # different chemistry: not a duplicate
        reactions.append(near)

    _edit_payload(leg, edit)
    artifacts = load_run_artifacts(leg)
    before = copy.deepcopy(artifacts)
    report = build_report(artifacts)

    assert artifacts == before, "the review must not modify the pathway it reads"
    assert len(report["reactions"]) == 7
    assert [r["duplicate_of"] for r in report["reactions"]] == [[6], [], [], [], [], [1], []]
    text = render_text(report)
    assert text.count(f"WARNING: {DUPLICATE_WARNING}") == 2
    assert f"WARNING: {DUPLICATE_WARNING} (same substrates and products as reaction 6)" in text


# 9 -------------------------------------------------------------------------
def test_rag_source_section_and_chunk_are_shown_when_recorded(leg):
    """Field values copied from the committed leg runs_verify/2026-08-04_1207/.../PMC12452463/strict."""

    def edit(payload):
        payload["processes"]["reactions"][0].update(
            rag_provenance={"source_id": "seed_paper", "section": "introduction", "chunk_id": "cbea424a0ce3ebb83a7dacb98484ae4c"},
            source_papers=[{"source_id": "PMC12452463", "title": "Enterobactin: A key player", "uri": "https://doi.org/10.1080/21505594.2025.2563018"}],
            provenance_lineage=[{"stage": "rag_admission", "origin": "rag_literature", "paper_explicit": "not_evaluated",
                                 "review_required": True,
                                 "sources": [{"source_id": "PMC12452463", "source_type": "paper", "locator": "results"}]}],
        )

    _edit_payload(leg, edit)
    first = _report(leg)["reactions"][0]
    assert first["sources"] == [
        "RAG seed_paper; section=introduction; chunk=cbea424a0ce3ebb83a7dacb98484ae4c",
        "PMC12452463 Enterobactin: A key player https://doi.org/10.1080/21505594.2025.2563018",
        "paper PMC12452463 (results)",
    ]
    # not_evaluated is carried as written, never collapsed into false.
    assert first["lineage"] == ["rag_admission/rag_literature (paper_explicit=not_evaluated, review_required)"]


# 10 ------------------------------------------------------------------------
def test_report_builds_with_every_socket_blocked(leg, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("reaction evidence attempted a network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert "REACTION 5" in render_text(_report(leg))


# 11 ------------------------------------------------------------------------
def test_new_modules_import_no_llm_or_http_client():
    probe = (
        "import sys\n"
        "import t2pw.app.reaction_evidence, t2pw.app.reaction_evidence_view\n"
        "roots = {'openai', 'anthropic', 'httpx', 'requests', 'urllib3', 'aiohttp'}\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in roots"
        " or m.startswith(('t2pw.llm', 't2pw.rag')))\n"
        "print(t2pw.app.reaction_evidence.__file__)\n"
        "print(bad)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    module_file, bad = result.stdout.strip().splitlines()[-2:]
    assert Path(module_file).resolve().is_relative_to(REPO / "src")
    assert bad == "[]"


# 12 ------------------------------------------------------------------------
def test_streamlit_app_hook_is_one_module_level_call_at_the_end():
    tree = ast.parse((REPO / "src" / "t2pw" / "app" / "streamlit_app.py").read_text(encoding="utf-8"))
    *_, imported, called = tree.body
    assert isinstance(imported, ast.ImportFrom) and imported.module == "t2pw.app.reaction_evidence_view"
    assert isinstance(called, ast.Expr) and ast.unparse(called) == "render_reaction_evidence()"


# REPAIR ROUND 1 (B1/B2) -- NEW ACCEPTANCE TESTS, same G9 label as the module docstring.
# Each bundle is a trimmed copy of the committed leg named beside it; test 13 proves the
# trim renders byte-identically to that leg, so the assertions below hold for the leg.
FIXTURES = REPO / "tests" / "fixtures" / "reaction_evidence"
TRIMMED_LEGS = {
    "lipid_a_strict": "runs_verify/2026-08-24_1428/papers/PMC12444477/strict",
    "enterobactin_research": "runs_verify/2026-08-21_2057/papers/PMC12096016/research",
    "enterobactin_refused_strict": "runs_verify/2026-08-18_1328/papers/PMC12452463/strict",
}


def _bundle_leg(tmp_path: Path, name: str) -> Path:
    return write_leg(tmp_path / name, FIXTURES / f"{name}.bundle.json")


def _quarantined_block(text: str, name: str) -> str:
    at = text.index(f"\nName: {name}\n")
    return text[text.rindex("\nQUARANTINED ", 0, at) + 1 : text.index("\n\n", at)]


# 13 ------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(TRIMMED_LEGS))
def test_trimmed_bundle_renders_like_its_committed_leg(tmp_path, name):
    assert render_text(_report(_bundle_leg(tmp_path, name))) == render_text(_report(REPO / TRIMMED_LEGS[name]))


# 14 ------------------------------------------------------------------------
def test_payload_quarantined_locked_reactions_are_listed_with_reason_and_evidence(tmp_path):
    """B1: final_mapped.json quarantined_locked_reactions is a removal record too.

    RV742b-B1: rows are labelled by that record name, never by a writer stage the
    entry does not record; these four carry the writer's recorded action.
    """

    leg = _bundle_leg(tmp_path, "lipid_a_strict")
    report = _report(leg)
    text = render_text(report)
    on_disk = set(_all_strings(json.loads((FIXTURES / "lipid_a_strict.bundle.json").read_text(encoding="utf-8"))))
    locked = {row["name"]: row for row in report["quarantined"] if row["source"] == "final payload quarantined_locked_reactions"}

    expected = {
        "LpxC reaction": ("rxn_lock_003", "LpxC is a deacetylase that removes the acetyl group from the GlcNAc moiety"),
        "LpxD reaction": ("rxn_lock_004", "addition of the next acyl group by LpxD"),
        "LpxH reaction": ("rxn_lock_005", "removal of UMP by LpxH"),
        "LpxK reaction": ("rxn_lock_006", "phosphorylated by LpxK to produce lipid IVA"),
    }
    assert sorted(locked) == sorted(expected)
    for name, (lock, first_quote) in expected.items():
        row = locked[name]
        assert (row["lock_id"], row["reason"]) == (lock, "coarse_grained_same_entity_transformation")
        assert row["evidence"][0] == first_quote and set(row["evidence"]) <= on_disk
        block = _quarantined_block(text, name)
        assert "[final payload quarantined_locked_reactions; json_pointer " in block and f"lock {lock}]" in block
        assert "Reason: coarse_grained_same_entity_transformation" in block
        assert "Action: quarantined_from_active_reactions" in block and f'1. "{first_quote}"' in block

    assert "locked-reaction filter" not in text

    # LpxL / LpxM are in BOTH records: listed once, from strict quarantine, and the
    # payload-record count says so.
    assert [r["name"] for r in report["quarantined"]].count("LpxL reaction") == 1
    assert [r["name"] for r in report["quarantined"]].count("LpxM reaction") == 1
    assert "QUARANTINED / REJECTED (15)" in text
    assert (
        "From final payload quarantined_locked_reactions: 6 "
        "(2 already listed from quarantine_report.json by lock id: rxn_lock_007, rxn_lock_008)"
    ) in text


# 15 ------------------------------------------------------------------------
def test_removals_recorded_in_the_payload_are_not_called_not_recorded(tmp_path):
    """B1: no quarantine_report.json, but the payload records removed locked reactions."""

    leg = _bundle_leg(tmp_path, "lipid_a_strict")
    (leg / "quarantine_report.json").unlink()
    text = render_text(_report(leg))
    assert "QUARANTINED / REJECTED (6)" in text and "QUARANTINED / REJECTED (Not recorded)" not in text
    assert "From quarantine_report.json quarantined: Not recorded" in text
    for name in ("LpxC", "LpxD", "LpxH", "LpxK", "LpxL", "LpxM"):
        assert f"Name: {name} reaction" in text


# 16 ------------------------------------------------------------------------
def test_research_mode_flags_are_shown_as_not_applied(tmp_path):
    """B2: research_mode.applied=false -- the flagged reaction is still in the payload."""

    report = _report(_bundle_leg(tmp_path, "enterobactin_research"))
    text = render_text(report)
    note = (
        "research mode: quarantine ran diagnostically. Nothing was removed and the "
        "candidate pathway is unchanged; the decisions below are review flags."
    )
    for line in ("Export mode: research", "Quarantine ok: true", "Research mode applied: false",
                 f"Research mode note: {note}"):
        assert f"\n{line}\n" in text
    assert (
        "\nSURVIVING REACTIONS (6) -- research_mode.applied=false: quarantine removed nothing; "
        "reactions it flagged are still listed here\n"
    ) in text
    assert (
        "\nQUARANTINED / REJECTED (1) -- research_mode.applied=false: "
        "quarantine_report.json rows are review flags, not removals\n"
    ) in text

    entc = report["reactions"][0]
    assert (entc["name"], entc["lock_id"]) == ("EntC reaction", "rxn_lock_003")
    assert (
        "Status: listed as QUARANTINED 1: state=quarantined_unmapped_entity, "
        "reason=protein_missing_external_identity; scope=core\n"
    ) in text
    assert text.count("admission=Not recorded") == 0


# 17 ------------------------------------------------------------------------
def test_refused_quarantine_is_shown_and_per_reaction_status_is_filled(tmp_path):
    """B2: ok=false -- the payload read is not the released graph; say so, per reaction too."""

    report = _report(_bundle_leg(tmp_path, "enterobactin_refused_strict"))
    text = render_text(report)
    assert "\nQuarantine ok: false\n" in text and "\nRelease status: diagnostic_only\n" in text
    refused = "quarantine_report.json ok=false: payload not released (release status diagnostic_only)"
    assert f"\nSURVIVING REACTIONS (6) -- {refused}\n" in text
    assert f"\nQUARANTINED / REJECTED (5) -- {refused}\n" in text
    status = {r["name"]: r["quarantine"] for r in report["reactions"]}
    reason = "state=quarantined_broken_reference, reason=ambiguous_entity_type:protein_complexes+proteins"
    assert status["DHB activation to DHB-AMP"] == [f"listed as QUARANTINED 1: {reason}"]
    assert status["enterobactin assembly"] == [f"listed as QUARANTINED 2: {reason}"]
    assert status["chorismate to isochorismate"] == []
    assert f"Status: listed as QUARANTINED 1: {reason}; scope=core\n" in text
