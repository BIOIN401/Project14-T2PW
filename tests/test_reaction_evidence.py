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


def write_leg(target: Path) -> Path:
    """Write the bundle out as a leg directory: one file per artifact, real names."""

    target.mkdir(parents=True, exist_ok=True)
    for name, content in json.loads(BUNDLE.read_text(encoding="utf-8")).items():
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
    for title, body in (("QUARANTINED / REJECTED (0)", "None recorded"), ("SURVIVING REACTIONS (0)", NOT_RECORDED)):
        assert f"{title}\n{'-' * len(title)}\n{body}\n" in text

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
