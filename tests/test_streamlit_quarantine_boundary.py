"""The quarantine boundary, driven through the real Streamlit app.

Everything else about quarantine can be unit-tested; *where it runs* cannot. The
first version of this feature put quarantine inside ``run_pwml_export`` and was
never reached in production, because the app stops one step earlier: the
post-mapping Stage 3 revalidation sets ``refinement_gate_errors`` and never
initializes refinement review, so ``_generate_pwml_from_refinement_working_json``
returns before the exporter is called. A payload that failed Stage 3 -- the exact
case quarantine exists for -- reached the exporter never.

So these tests click the real widgets, in the real app script, and read the real
``st.session_state``, the same way ``t2pw.batch.driver`` drives it. The app is
never imported or duplicated here: ``AppTest`` execs it, so the boundary being
asserted is the one that ships.

Offline: ``map_payload`` is patched at its source module before the app execs
(the app binds it with ``from ... import``, so patching the module first is
enough), and the LLM audit and gap resolver are left switched off.
"""

from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pwml.writer import blocking_pwml_ir_errors  # noqa: E402
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    CLOSURE_REPORT_FILENAME,
    COVERAGE_REPORT_FILENAME,
    QUARANTINE_HISTORY_DIRNAME,
    QUARANTINE_REPORT_FILENAME,
    QUARANTINED_UNMAPPED_ENTITY,
    REMOVED_ENTITY_REPORT_FILENAME,
)

APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
OUTPUTS = ROOT / "outputs"

#: One app interaction here is pure Python -- no LLM, no DB -- but the app's own
#: import graph is large, so the default 3s is not enough on a cold run.
APP_TIMEOUT = 120.0

# Widget keys, all verified against the app and shared with t2pw.batch.driver.
KEY_POST_PIPELINE = "pwml_generate_btn"

#: A pathway name the stubbed mapping refuses, to force a run that ends before the
#: quarantine boundary. See ``offline_mapping``.
FAIL_MAPPING_SENTINEL = "force a mapping failure before the boundary"


def _core_payload() -> Dict[str, Any]:
    """Two fully mapped core reactions, ready for strict export."""

    payload: Dict[str, Any] = {
        "metadata": {
            "pw_id": "PW000000",
            "pathway_name": "Glutathione biosynthesis",
            "name": "Glutathione biosynthesis",
            "pathway_subject": "Metabolic",
            "subject": "Metabolic",
            "organism": "Homo sapiens",
            "width": 3200,
            "height": 1400,
        },
        "entities": {
            "species": [
                {"name": "Homo sapiens", "taxonomy_id": "9606", "classification": "Eukaryote"}
            ],
            "subcellular_locations": [{"name": "cytosol"}],
            "compounds": [
                {"name": "L-glutamate"},
                {"name": "L-cysteine"},
                {"name": "gamma-glutamylcysteine"},
                {"name": "glycine"},
                {"name": "glutathione"},
            ],
            "proteins": [
                {
                    "name": "glutamate-cysteine ligase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48506"},
                },
                {
                    "name": "glutathione synthetase",
                    "species": "Homo sapiens",
                    "mapped_ids": {"uniprot": "P48637"},
                },
            ],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "biological_states": [
            {"name": "cytosol", "species": "Homo sapiens", "subcellular_location": "cytosol"}
        ],
        "element_locations": {
            "compound_locations": [
                {"compound": name, "biological_state": "cytosol"}
                for name in (
                    "L-glutamate",
                    "L-cysteine",
                    "gamma-glutamylcysteine",
                    "glycine",
                    "glutathione",
                )
            ],
            "protein_locations": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "gamma-glutamylcysteine synthesis",
                    "inputs": ["L-glutamate", "L-cysteine"],
                    "outputs": ["gamma-glutamylcysteine"],
                    "enzymes": [],
                    "biological_state": "cytosol",
                },
                {
                    "name": "glutathione synthesis",
                    "inputs": ["gamma-glutamylcysteine", "glycine"],
                    "outputs": ["glutathione"],
                    "enzymes": [],
                    "biological_state": "cytosol",
                },
            ],
            "transports": [],
            "interactions": [],
        },
    }
    for reaction, protein in zip(
        payload["processes"]["reactions"],
        ["glutamate-cysteine ligase", "glutathione synthetase"],
    ):
        wrapper = f"{protein} complex"
        payload["entities"]["protein_complexes"].append(
            {
                "name": wrapper,
                "class": "protein_complex",
                "generated": True,
                "generation_reason": "single_protein_pathwhiz_wrapper",
                "species": "Homo sapiens",
                "components": [{"name": protein, "stoichiometry": 1}],
            }
        )
        reaction["enzymes"] = [
            {"entity": wrapper, "entity_type": "protein_complex", "role": "catalyst"}
        ]
    return payload


def _add_unmapped_modifier(payload: Dict[str, Any], reaction: Dict[str, Any], protein: str) -> None:
    """Attach a declared-but-unidentified protein as a modifier of ``reaction``.

    The choice of defect matters and both obvious alternatives are wrong here,
    which driving the real app is what established:

    * **An undeclared reaction participant** never reaches the boundary.
      ``normalize_process_payload`` declares missing participants as compounds on
      its way past, so the reaction arrives clean and quarantine correctly does
      nothing. The first version of this test asserted a quarantine production
      had already made unnecessary.
    * **An unidentified enzyme wrapped in a generated protein_complex** never
      reaches it either: ``validate_post_remap`` rejects the wrapper at
      ``generated_wrapper_component_missing_external_identity``, an earlier
      contract entirely.

    A bare unidentified *modifier* passes ``validate_post_remap`` -- which only
    audits generated wrappers -- and fails the post-mapping Stage 3 gate on
    ``Protein 'X' is missing a UniProt or DrugBank identifier``. That is the gate
    that stops the run before refinement review opens, so it is the one a
    boundary test has to reach.
    """

    payload["entities"]["proteins"].append({"name": protein, "species": "Homo sapiens"})
    reaction.setdefault("modifiers", []).append(
        {"entity": protein, "entity_type": "protein", "role": "inhibitor"}
    )


def _payload_with_one_bad_peripheral() -> Dict[str, Any]:
    """Two exportable core reactions plus one peripheral reaction that is not."""

    payload = _core_payload()
    payload["entities"]["compounds"].extend([{"name": "OPDA"}, {"name": "OPC-8:0"}])
    payload["element_locations"]["compound_locations"].extend(
        [
            {"compound": "OPDA", "biological_state": "cytosol"},
            {"compound": "OPC-8:0", "biological_state": "cytosol"},
        ]
    )
    peripheral = {
        "name": "peripheral OPDA reduction",
        "inputs": ["OPDA"],
        "outputs": ["OPC-8:0"],
        "enzymes": [],
        "biological_state": "cytosol",
    }
    payload["processes"]["reactions"].append(peripheral)
    _add_unmapped_modifier(payload, peripheral, "OPR3")
    return payload


def _payload_with_no_viable_core() -> Dict[str, Any]:
    """Every requested-core reaction unexportable; one unrelated survivor.

    The survivor is complete and gate-clean and has nothing to do with the
    request, which is precisely the graph a bare "is anything left?" check would
    call a success.
    """

    payload = _core_payload()
    payload["entities"]["compounds"].extend([{"name": "citrate"}, {"name": "isocitrate"}])
    payload["element_locations"]["compound_locations"].extend(
        [
            {"compound": "citrate", "biological_state": "cytosol"},
            {"compound": "isocitrate", "biological_state": "cytosol"},
        ]
    )
    payload["processes"]["reactions"].append(
        {
            "name": "citrate isomerisation",
            "inputs": ["citrate"],
            "outputs": ["isocitrate"],
            "enzymes": [],
            "biological_state": "cytosol",
        }
    )
    for index, reaction in enumerate(payload["processes"]["reactions"][:2]):
        _add_unmapped_modifier(payload, reaction, f"unmappable regulator {index}")
    return payload


STAGE_ZERO_CONTEXT = {
    "pathway_name": "Glutathione biosynthesis",
    "likely_organism": "Homo sapiens",
    "key_compounds": ["L-glutamate", "glutathione"],
    "key_proteins": ["glutamate-cysteine ligase"],
}


# ---------------------------------------------------------------------------
# Real streamlit, whatever the rest of the suite did to sys.modules.
# ---------------------------------------------------------------------------
def _is_real_streamlit(module: object) -> bool:
    """A ``MagicMock`` is not a module; a bare ``ModuleType`` stub has no path."""

    return isinstance(module, types.ModuleType) and hasattr(module, "__path__")


@pytest.fixture(autouse=True)
def real_streamlit():
    """Guarantee the genuine ``streamlit`` package for the duration of each test.

    Several ``test_rag_*`` and ``test_stage0_*`` modules install a ``MagicMock``
    as ``sys.modules["streamlit"]`` at *import* time -- they must, because
    ``t2pw.app.streamlit_app`` executes a Streamlit script on import. Pytest
    imports every test module during collection, so in a full-suite run the stub
    is already in place before anything here runs, and ``AppTest`` would be a
    MagicMock returning MagicMocks. This module additionally cannot import
    ``AppTest`` at module scope for the same reason -- collection order decides
    whether that import even resolves -- so it is imported inside :func:`_app_test`.

    Identical in shape to ``tests/test_batch_driver.py``'s fixture, and restores
    whatever it found, so those modules keep the stub they captured.
    """

    def _streamlit_keys():
        return [n for n in list(sys.modules) if n == "streamlit" or n.startswith("streamlit.")]

    saved = {name: sys.modules[name] for name in _streamlit_keys()}
    stubbed = not _is_real_streamlit(sys.modules.get("streamlit"))
    if stubbed:
        for name in saved:
            del sys.modules[name]
    import streamlit  # noqa: F401,PLC0415
    import streamlit.testing.v1  # noqa: F401,PLC0415

    assert _is_real_streamlit(sys.modules["streamlit"]), "the real streamlit did not load"
    try:
        yield
    finally:
        if stubbed:
            for name in _streamlit_keys():
                del sys.modules[name]
            sys.modules.update(saved)


def _app_test():
    """The real ``AppTest``, imported only once the fixture above has run."""

    from streamlit.testing.v1 import AppTest  # noqa: PLC0415

    return AppTest


@pytest.fixture()
def offline_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make Stage 2/6 mapping a no-op, at the module the app imports it from.

    The app binds ``map_payload`` with ``from t2pw.mapping.map_ids import ...`` at
    exec time, which AppTest performs *after* this fixture runs -- so patching the
    source module is enough and no wrapper script is needed. The payloads here are
    already fully mapped, so an identity mapping is also the honest stand-in.
    """

    import t2pw.mapping.enrich_entities as enrich_entities
    import t2pw.mapping.map_ids as map_ids

    def _no_enrichment(
        input_path: Any,
        output_path: Any,
        report_path: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Stage 7 enrichment, stubbed. It reaches UniProt over the network.

        Not hypothetically: an unstubbed run of this file fetched P48506 and wrote
        it into the committed ``data/enrichment_cache.json``, so the test both made
        a live request and dirtied a tracked file. Enrichment adds optional xrefs
        and changes no gate outcome, so removing it costs the boundary nothing --
        but the app reads the *enriched* file afterwards, so a no-op enrichment
        still has to produce one.
        """

        report = {"summary": {"enrichment_skipped_for_test": True}}
        Path(output_path).write_text(
            Path(input_path).read_text(encoding="utf-8"), encoding="utf-8"
        )
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    monkeypatch.setattr(enrich_entities, "run_enrichment", _no_enrichment)

    def _identity(payload: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
        # A pipeline run that dies BEFORE the quarantine boundary, on demand. The
        # lifecycle test needs one: a reset that only happens when the boundary is
        # reached is not a reset, and mapping raising is the ordinary way a real
        # run ends early here. Keyed off the payload so nothing else is affected.
        if str((payload.get("metadata") or {}).get("pathway_name") or "") == FAIL_MAPPING_SENTINEL:
            raise RuntimeError("mapping failed before the quarantine boundary (test)")
        mapped = deepcopy(payload)
        # ``validate_post_mapping`` requires a ``mapping_meta`` on every named
        # entity -- that stamp IS mapping's output contract, so a stub that skips
        # it is not an identity mapping, it is an incomplete one. The identifiers
        # in these payloads are already resolved; only the stamp is missing.
        for rows in (mapped.get("entities") or {}).values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict) and row.get("name"):
                    row.setdefault(
                        "mapping_meta",
                        {"source": "test_fixture", "resolution": {"status": "preresolved"}},
                    )
        return {"payload": mapped, "report": {"summary": {}, "unmapped": []}}

    monkeypatch.setattr(map_ids, "map_payload", _identity)


def _clear_artifacts() -> None:
    for name in (
        QUARANTINE_REPORT_FILENAME,
        REMOVED_ENTITY_REPORT_FILENAME,
        CLOSURE_REPORT_FILENAME,
        COVERAGE_REPORT_FILENAME,
    ):
        path = OUTPUTS / name
        if path.exists():
            path.unlink()


def _seed_finished_pipeline(at, payload: Dict[str, Any]) -> None:
    """Stand in for a Stage 1/2 run that already happened."""

    # ``pipeline_ready`` is what unhides the whole post-pipeline section, including
    # the button below. Seeding it stands in for a Stage 1/2 run that already
    # happened; every other key here is one that section reads on the way past.
    at.session_state["pipeline_ready"] = True
    at.session_state["stage_one"] = deepcopy(payload)
    at.session_state["stage_two"] = None
    at.session_state["stage_two_chunks"] = []
    at.session_state["stage_two_rounds"] = []
    at.session_state["chunk_details"] = []
    at.session_state["qa_hints"] = None
    at.session_state["final_payload"] = deepcopy(payload)
    at.session_state["final_payload_snapshot"] = deepcopy(payload)
    at.session_state["pipeline_source_text"] = "seeded by the boundary integration test"
    at.session_state["pathway_context"] = deepcopy(STAGE_ZERO_CONTEXT)
    at.session_state["pwml_strict_db"] = True
    # Everything that would reach the network or MySQL, switched off at the same
    # widgets an operator uses. The payloads here are already fully mapped and
    # fully identified, so nothing these would add is missing -- and leaving the
    # LLM audit on (its shipped default) makes this test a live API call.
    at.session_state["post_use_llm_audit"] = False
    at.session_state["post_use_gap_resolver"] = False
    at.session_state["post_use_llm_gap_resolver"] = False
    at.session_state["post_use_example_retrieval"] = False
    at.session_state["post_use_stoich_agent"] = False
    at.session_state["post_use_sbml_overwatch"] = False
    at.session_state["post_db_host"] = ""


def _run_post_pipeline(payload: Dict[str, Any]):
    """Load the real app, seed a finished pipeline, click "Run audit and DB mapping"."""

    _clear_artifacts()
    at = _app_test().from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    at.run()
    _seed_finished_pipeline(at, payload)
    at.run()
    at.button(key=KEY_POST_PIPELINE).click().run()
    return at


#: A Streamlit harness fault, not an app fault. Running several ``AppTest``
#: instances in one process eventually leaves a worker thread without a
#: ``ScriptRunContext``; the message appears on the run itself and no app code is
#: in the traceback. It is filtered by name rather than by ignoring
#: ``at.exception`` wholesale, so a real app exception still fails the test.
_HARNESS_FAULT = "FragmentThreadState not initialized"


def _app_exceptions(at) -> list:
    """Exceptions the app raised, with the known harness fault removed."""

    return [str(e.value) for e in at.exception if _HARNESS_FAULT not in str(e.value)]


def _artifact(name: str) -> Dict[str, Any]:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


# ── The recovery ────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("offline_mapping")
def test_stage3_fails_on_the_raw_payload_so_the_recovery_is_real() -> None:
    """(a) The payload really does fail the gate that stops the run."""

    from t2pw.pipeline.process_normalizer import (
        GateValidationError,
        normalize_process_payload,
        run_strict_post_normalization_gates,
    )

    # Normalized first, because that is the form the boundary actually sees --
    # asserting on the raw payload would let a defect normalization repairs pass
    # for one it cannot.
    normalized = normalize_process_payload(deepcopy(_payload_with_one_bad_peripheral()))
    payload = normalized[0] if isinstance(normalized, tuple) else normalized
    with pytest.raises(GateValidationError) as failure:
        run_strict_post_normalization_gates(payload)

    reasons = [str(row.get("reason", "")) for row in failure.value.details.get("errors", [])]
    assert any("OPR3" in reason and "UniProt" in reason for reason in reasons), reasons


@pytest.mark.usefixtures("offline_mapping")
def test_the_app_reaches_quarantine_and_the_valid_core_reaches_review() -> None:
    """(b) and (c): the boundary runs, and what survives clears Stage 3."""

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())

    assert _app_exceptions(at) == []
    report = at.session_state["quarantine_report"]
    assert report["ok"] is True
    states = {row["name"]: row["state"] for row in report["admissions"]}
    assert states["peripheral OPDA reduction"] == QUARANTINED_UNMAPPED_ENTITY
    assert states["gamma-glutamylcysteine synthesis"].endswith("_accepted")

    # Stage 3 -- the check that used to stop this run -- now passes, and refinement
    # review opened. That is the whole recovery, measured at the production seam.
    assert at.session_state["refinement_gate_errors"] == []
    working = at.session_state["refinement_working_json"]
    assert [row["name"] for row in working["processes"]["reactions"]] == [
        "gamma-glutamylcysteine synthesis",
        "glutathione synthesis",
    ]


@pytest.mark.usefixtures("offline_mapping")
def test_the_reviewer_sees_exactly_the_payload_that_will_be_exported() -> None:
    """One boundary, one payload. Two quarantine passes would diverge here."""

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    working = at.session_state["refinement_working_json"]

    from t2pw.pipeline.strict_quarantine import quarantine_and_close

    again = quarantine_and_close(
        deepcopy(working), strict_db=True, pathway_context=STAGE_ZERO_CONTEXT
    )
    # Re-quarantining the reviewed payload is a no-op: it is already the reduced
    # graph, so the exporter cannot be handed anything the reviewer did not see.
    assert again.payload == working
    assert again.quarantine_report["counts"][QUARANTINED_UNMAPPED_ENTITY] == 0


@pytest.mark.usefixtures("offline_mapping")
def test_one_canonical_payload_is_gated_serialized_and_exported() -> None:
    """There used to be THREE candidate "final" payloads, and no two consumers
    agreed on which one they meant:

    ``final_mapped_db``          post-remap, PRE-enrichment  -> serialized by the
                                                                batch driver as
                                                                final_mapped.json
    ``final_export_payload``     post-enrichment             -> what SBML was
                                                                built from
    ``final_mapped_quarantined`` post-quarantine, derived    -> what the
                                 from final_mapped_db           pre-export Stage 3
                                                                gate validated

    So the gate ran on an object that was neither serialized nor exported, and no
    passing report could honestly be made authoritative -- it would have asserted
    a pass on something that did not ship. One object now: enriched, then
    quarantined, then gated, then serialized and handed to the exporter. This is
    the assertion that says so at the production seam.
    """

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    artifacts = at.session_state["post_pipeline_artifacts"]

    from t2pw.pipeline.gate_reports import (
        ARTIFACT_SET_VERSION_KEY,
        CANONICAL_PAYLOAD_KEY,
        FINAL_GATE_REPORT_KEY,
        PAYLOAD_SHA256_KEY,
        PHASE_FINAL_PRE_EXPORT,
        PHASE_KEY,
        payload_sha256,
    )

    canonical = artifacts[CANONICAL_PAYLOAD_KEY]
    assert canonical, "the run produced no canonical payload"

    # 1. The gate report is BOUND to it, by hash, not by proximity.
    final_gate = artifacts[FINAL_GATE_REPORT_KEY]
    assert final_gate[PHASE_KEY] == PHASE_FINAL_PRE_EXPORT
    assert final_gate[PAYLOAD_SHA256_KEY] == payload_sha256(canonical)
    assert artifacts[ARTIFACT_SET_VERSION_KEY] >= 1

    # 2. The SBML build reads it -- recorded rather than inferred, because "the
    #    exporter consumed the payload the gate validated" is exactly the claim
    #    that used to be false.
    assert artifacts["sbml_input_source"] == CANONICAL_PAYLOAD_KEY

    # 3. The key every downstream reader reaches for names the same object, and
    #    it is NOT the pre-enrichment one the driver used to serialize.
    assert artifacts["final_mapped"] == canonical
    assert artifacts["final_mapped_quarantined"] == canonical

    # 4. And the reviewer -- whose working JSON is what run_pwml_export builds
    #    from -- was handed that same object.
    assert at.session_state["refinement_working_json"] == canonical


@pytest.mark.usefixtures("offline_mapping")
def test_a_passing_final_gate_is_recorded_and_not_only_a_failing_one() -> None:
    """A gate that only ever records failure cannot report success.

    This payload clears Stage 3 after quarantine. Before the report lifecycle the
    passing branch wrote nothing at all, so the only gate artifact a consumer
    could find was the stale pre-audit failure -- which is how a repaired leg
    stayed FAIL with no artifact able to countermand it.
    """

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    artifacts = at.session_state["post_pipeline_artifacts"]

    from t2pw.pipeline.gate_reports import FINAL_GATE_REPORT_KEY

    final_gate = artifacts[FINAL_GATE_REPORT_KEY]
    assert final_gate.get("ok") is True
    assert final_gate.get("errors", []) == []
    assert at.session_state["refinement_gate_errors"] == []


@pytest.mark.usefixtures("offline_mapping")
def test_all_four_artifacts_are_written_and_retained() -> None:
    """(d) The four artifacts, on disk, and referenced from session state."""

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())

    artifacts = at.session_state["quarantine_artifacts"]
    assert set(artifacts) == {
        QUARANTINE_REPORT_FILENAME,
        REMOVED_ENTITY_REPORT_FILENAME,
        CLOSURE_REPORT_FILENAME,
        COVERAGE_REPORT_FILENAME,
    }
    for name, path in artifacts.items():
        assert Path(path).exists(), name
        assert _artifact(name)["schema_version"] >= 1

    quarantined = _artifact(QUARANTINE_REPORT_FILENAME)["quarantined"]
    assert [row["name"] for row in quarantined] == ["peripheral OPDA reduction"]
    # The original row, kept whole, is what makes the decision reviewable.
    assert quarantined[0]["original_process"]["inputs"] == ["OPDA"]


@pytest.mark.usefixtures("offline_mapping")
def test_the_stage_zero_context_reaches_the_coverage_artifact_unchanged() -> None:
    """The requested core comes from Stage 0, not from what happened to survive."""

    _run_post_pipeline(_payload_with_one_bad_peripheral())
    coverage = _artifact(COVERAGE_REPORT_FILENAME)

    assert coverage["requested_core_declared"] is True
    assert coverage["requested_core_source"] == "pathway_context"
    assert coverage["requested_context"] == STAGE_ZERO_CONTEXT
    assert coverage["requested_core_terms"] == [
        "L-glutamate",
        "glutathione",
        "glutamate-cysteine ligase",
    ]


# ── The refusal ─────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("offline_mapping")
def test_a_run_with_no_viable_core_is_refused_at_the_boundary() -> None:
    """An unrelated survivor must not carry the run to SUCCESS.

    **MOVED BY C-041a (D-002 / PRODUCT_CONTRACT 7, both LOCKED).** The test name
    is a pre-D-002 misnomer and is kept only so the Chunk D node id does not
    move: what ``_payload_with_no_viable_core`` lacks is a viable *requested*
    core, not a viable core. Measured on the payload rather than read off the
    name -- ``citrate isomerisation`` survives as ``auxiliary_accepted``, the
    reduced graph clears the strict Stage 3 gates AND the required-PWML
    contract, and every strict invariant is clean. That is precisely D-002's
    "biologically correct, internally connected fragment representable without
    guessing", so the outcome is ``review_required`` PWML.

    What the test was actually protecting is unchanged and is now asserted
    directly instead of through a boolean: the unrelated survivor still cannot
    produce success. ``strict_acceptance_eligible`` is False, so it can never
    enter the strict denominator, and the shortfall is still named. At the base
    SHA this was ``ok is False`` and the run ended with no PWML at all, which is
    the "valid pathway core suppressed" that PRODUCT_CONTRACT 1 forbids.
    """

    at = _run_post_pipeline(_payload_with_no_viable_core())

    assert _app_exceptions(at) == []
    report = at.session_state["quarantine_report"]
    assert report["ok"] is True
    assert report["release"]["status"] == "review_required"
    assert report["release"]["strict_acceptance_eligible"] is False
    # The shortfall is still raised and still named -- it moved out of the
    # blocking list, it did not disappear.
    assert report["refusal_reasons"] == []
    assert any(
        reason.startswith("minimum_core:core_process_count_below_minimum")
        for reason in report["review_reasons"]
    )
    # And the surviving fragment really is the unrelated one: nothing was
    # admitted to make coverage pass.
    assert report["coverage"]["core_accepted_processes"] == 0
    assert report["coverage"]["surviving_processes"] == 1

    # Refinement review now OPENS on the fragment, because there is a fragment to
    # review. That is the behaviour change: the reviewer gets the graph plus the
    # flag, instead of nothing.
    assert not at.session_state["refinement_gate_errors"]
    assert at.session_state["refinement_working_json"] is not None
    assert not any("Quarantine could not produce a viable" in str(e.value) for e in at.error)


@pytest.mark.usefixtures("offline_mapping")
def test_the_refusal_still_writes_every_artifact() -> None:
    """A decision is recorded, whatever it decided.

    **MOVED BY C-041a (D-002, LOCKED).** The claim -- the artifact set is written
    and carries the coverage verdict -- is untouched, and every coverage
    assertion below is byte for byte what it was. Only ``ok`` moved, because this
    payload's shortfall is a shortfall in SIZE against the request and no longer
    blocks: see ``test_a_run_with_no_viable_core_is_refused_at_the_boundary``
    for the payload-level proof. Base SHA: ``ok is False``.
    """

    _run_post_pipeline(_payload_with_no_viable_core())

    coverage = _artifact(COVERAGE_REPORT_FILENAME)
    assert coverage["minimum_core_satisfied"] is False
    assert coverage["core_accepted_processes"] == 0
    report = _artifact(QUARANTINE_REPORT_FILENAME)
    assert report["ok"] is True
    assert report["release"]["status"] == "review_required"
    assert report["release"]["strict_acceptance_eligible"] is False
    assert any(r.startswith("minimum_core:") for r in report["review_reasons"])
    assert {row["name"] for row in report["quarantined"]} == {
        "gamma-glutamylcysteine synthesis",
        "glutathione synthesis",
    }


# ── One boundary, not two ───────────────────────────────────────────────────


def test_the_exporter_only_quarantines_when_no_decision_reached_it() -> None:
    """Static proof that the exporter cannot silently make a second decision.

    Two independent passes would judge two different payloads -- the boundary's
    input and its output -- and the second would overwrite the four artifacts
    describing the graph the reviewer actually approved. The call inside
    ``run_pwml_export`` therefore has to sit under ``if not quarantine_report``,
    and this asserts that rather than trusting the comment above it.
    """

    import ast

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    exporter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_pwml_export"
    )

    guarded: list = []
    for node in ast.walk(exporter):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)):
            continue
        # The guard is the payload-version check, not the mere presence of a
        # report: a stored decision about a payload the reviewer has since edited
        # must NOT authorize an export, so reuse is gated on `quarantine_reused`.
        if not (isinstance(node.test.operand, ast.Name) and node.test.operand.id == "quarantine_reused"):
            continue
        guarded.extend(
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "quarantine_and_close"
        )

    all_calls = [
        call
        for call in ast.walk(exporter)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "quarantine_and_close"
    ]

    assert len(all_calls) == 1
    assert len(guarded) == 1, "the exporter's quarantine call is not guarded by the payload-version check"


def test_the_boundary_is_the_only_unguarded_quarantine_call_in_the_app() -> None:
    """Exactly one function in the app decides; everything else carries it."""

    import ast

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    callers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "quarantine_and_close"
            for call in ast.walk(node)
        )
    }

    assert callers == {"run_quarantine_boundary", "run_pwml_export"}


# ── The actual export ───────────────────────────────────────────────────────
#
# A test that stops at refinement review proves the run was not blocked. It does
# NOT prove a file comes out: between review and the file sit the Stage 3
# revalidation, the required-field contract, build_pwml_ir, validate_pwml_ir and
# the writer, and quarantine's whole claim is that a reduced graph clears all of
# them. These click "Generate PWML" and check the bytes.
#
# The output directory is the app's own `outputs/`, hard-coded to PROJECT_ROOT
# and not reachable from any widget, so it cannot be isolated through the UI --
# isolating it would mean adding a parameter to the app for the benefit of a
# test. `outputs/pathway.pwml` is a TRACKED file, so this test overwrites one:
# the fixture below restores it, or a green run would leave the repo dirty and
# the next reader could not tell the test's output from a real export.

KEY_GENERATE_PWML = "refinement_generate_pwml"
PWML_OUTPUT = ROOT / "outputs" / "pathway.pwml"


@pytest.fixture(autouse=True)
def restore_tracked_pwml_output():
    """Put ``outputs/pathway.pwml`` back the way it was found.

    It is checked in, and the export test necessarily overwrites it. Leaving it
    modified would make every green run dirty the working tree and would blur a
    test artifact into a real export.
    """

    original = PWML_OUTPUT.read_bytes() if PWML_OUTPUT.exists() else None
    try:
        yield
    finally:
        if original is None:
            if PWML_OUTPUT.exists():
                PWML_OUTPUT.unlink()
        else:
            PWML_OUTPUT.write_bytes(original)


def _generate_pwml(at) -> Dict[str, Any]:
    """Click Generate PWML on a run that has reached refinement review."""

    at.button(key=KEY_GENERATE_PWML).click().run()
    assert "pwml_export_result" in at.session_state, "the export button did not run"
    return at.session_state["pwml_export_result"]


@pytest.mark.usefixtures("offline_mapping")
def test_the_quarantined_payload_exports_a_non_empty_pwml_end_to_end() -> None:
    """(a)-(g): from a Stage-3-failing payload to bytes on disk."""

    if PWML_OUTPUT.exists():
        PWML_OUTPUT.unlink()

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    assert at.session_state["refinement_gate_errors"] == [], "never reached review"

    result = _generate_pwml(at)

    assert result["ok"] is True, result.get("error")
    # (d) Stage 3, on the reduced payload.
    assert result["stage3_contract_report"]["ok"] is True
    assert result["stage3_gate_report"]["ok"] is True
    # (e) the required-field gate.
    assert result["required_gate_report"]["ok"] is True, result["required_gate_report"].get("errors")
    # (f) IR build and IR validation.
    assert result["pwml_ir_validation"]["ok"] is True
    assert blocking_pwml_ir_errors(result["pwml_ir_report"]) == []
    # (g) a real file with real content.
    assert result["output_path"]
    written = Path(result["output_path"])
    assert written.exists()
    assert written.stat().st_size > 0
    assert result["xml_bytes"].startswith(b"<?xml")
    assert b"<pathway" in result["xml_bytes"] or b"<Pathway" in result["xml_bytes"]

    # The exported graph is the quarantined one: the bad process is not in it.
    assert b"peripheral OPDA reduction" not in result["xml_bytes"]
    assert result["counts"]


@pytest.mark.usefixtures("offline_mapping")
def test_the_export_reuses_the_boundary_decision_when_the_payload_did_not_move() -> None:
    """No second quarantine pass, and the reviewer's graph is what shipped."""

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    boundary_hash = at.session_state["quarantine_report"]["admitted_payload_hash"]

    result = _generate_pwml(at)

    assert result["ok"] is True
    assert result["quarantine_reused"] is True
    assert result["quarantine_report"]["admitted_payload_hash"] == boundary_hash


@pytest.mark.usefixtures("offline_mapping")
def test_a_refinement_that_deletes_the_core_is_not_authorized_by_the_stale_decision() -> None:
    """The stale-decision regression, driven through the app.

    The boundary admitted a valid pathway and stored an ``ok`` report. The
    reviewer then deletes both requested-core reactions, leaving one unrelated
    survivor. The stored report is still ``ok`` -- it was true of the graph it
    judged -- so an export that trusted its existence would ship a pathway that
    is not the one requested.
    """

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    admitted_hash = at.session_state["quarantine_report"]["admitted_payload_hash"]
    assert at.session_state["quarantine_report"]["ok"] is True

    working = deepcopy(at.session_state["refinement_working_json"])
    working["entities"]["compounds"].extend([{"name": "citrate"}, {"name": "isocitrate"}])
    working["element_locations"]["compound_locations"].extend(
        [
            {"compound": "citrate", "biological_state": "cytosol"},
            {"compound": "isocitrate", "biological_state": "cytosol"},
        ]
    )
    working["processes"]["reactions"] = [
        {
            "name": "citrate isomerisation",
            "inputs": ["citrate"],
            "outputs": ["isocitrate"],
            "enzymes": [],
            "biological_state": "cytosol",
        }
    ]
    at.session_state["refinement_working_json"] = working
    at.run()

    result = _generate_pwml(at)

    # **MOVED BY C-041a (D-002, LOCKED) -- and the invariant under test did NOT
    # move.** What this test protects is that the STORED decision does not
    # authorize a payload it never judged, and that is carried entirely by
    # ``quarantine_reused is False`` plus the ``admitted_payload_hash``
    # difference below. Both are byte for byte what they were: the stale report
    # is still refused as an authorization and the payload is still re-judged
    # from scratch.
    #
    # What moved is the VERDICT of that fresh re-judgement. The refined payload
    # is one surviving ``citrate isomerisation`` that clears the Stage 3 gates
    # and the required-PWML contract with every strict invariant clean, so
    # D-002 makes it review_required rather than a refusal. Base SHA: ``ok is
    # False``. Coverage relief cannot reach the authorization question at all --
    # that decision is taken before any coverage verdict exists.
    assert result["quarantine_reused"] is False
    assert result["quarantine_report"]["admitted_payload_hash"] != admitted_hash
    assert result["ok"] is True
    assert result["quarantine_report"]["release"]["status"] == "review_required"
    assert result["quarantine_report"]["release"]["strict_acceptance_eligible"] is False
    assert any(
        reason.startswith("minimum_core:")
        for reason in result["quarantine_report"]["review_reasons"]
    )


@pytest.mark.usefixtures("offline_mapping")
def test_a_refinement_that_adds_an_unexportable_process_still_exports_the_core() -> None:
    """The other half: re-judging must recover, not just refuse."""

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())

    working = deepcopy(at.session_state["refinement_working_json"])
    working["entities"]["compounds"].extend([{"name": "OPDA"}, {"name": "OPC-8:0"}])
    working["element_locations"]["compound_locations"].extend(
        [
            {"compound": "OPDA", "biological_state": "cytosol"},
            {"compound": "OPC-8:0", "biological_state": "cytosol"},
        ]
    )
    _add_unmapped_modifier(
        working,
        working["processes"]["reactions"][-1],
        "regulator added during review",
    )
    working["processes"]["reactions"].append(
        {
            "name": "review-added peripheral",
            "inputs": ["OPDA"],
            "outputs": ["OPC-8:0"],
            "modifiers": [{"entity": "regulator added during review"}],
            "biological_state": "cytosol",
        }
    )
    at.session_state["refinement_working_json"] = working
    at.run()

    result = _generate_pwml(at)

    assert result["quarantine_reused"] is False
    assert result["ok"] is True, result.get("error")
    states = {
        row["name"]: row["state"] for row in result["quarantine_report"]["admissions"]
    }
    assert states["review-added peripheral"] == QUARANTINED_UNMAPPED_ENTITY
    assert b"review-added peripheral" not in result["xml_bytes"]
    assert Path(result["output_path"]).stat().st_size > 0


# ── Research mode fails open ────────────────────────────────────────────────


KEY_EXPORT_MODE = "export_mode_radio"
RESEARCH_LABEL = "Research (relaxed)"


def _run_post_pipeline_research(payload: Dict[str, Any]):
    """The same run, with the export-mode radio set to research.

    Set through the widget the operator uses, not by writing the derived
    ``export_mode`` key, so the test exercises the same path a human does.
    """

    _clear_artifacts()
    at = _app_test().from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    at.run()
    # Driven through the radio, not by writing the derived `export_mode` key:
    # the widget is what an operator touches, and the app derives the mode from
    # it on every rerun, so a written key would be overwritten anyway.
    at.radio(key=KEY_EXPORT_MODE).set_value(RESEARCH_LABEL).run()
    _seed_finished_pipeline(at, payload)
    at.run()
    at.button(key=KEY_POST_PIPELINE).click().run()
    return at


@pytest.mark.usefixtures("offline_mapping")
def test_research_mode_keeps_the_unmapped_candidate_and_does_not_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A research run must come out with the pathway it went in with.

    The peripheral process here is unmapped, which is the normal state of a
    genuinely novel enzyme rather than a defect. A strict run quarantines it --
    ``test_the_app_reaches_quarantine_and_the_valid_core_reaches_review`` pins
    that on the identical payload -- and a research run must not, because
    research mode's whole contract is that findings annotate rather than block.
    """

    # Enrichment runs BETWEEN mapping and the quarantine boundary, and the shared
    # ``offline_mapping`` stub makes it a byte-for-byte copy. That collapses the
    # pre-enrichment artifact (``final_mapped_db``) onto the post-enrichment one
    # (``final_mapped_enriched``), so a comparison against either looks correct
    # and the boundary assertion below would pin nothing about quarantine -- it
    # would silently span every stage from mapping onwards. This override is
    # local to this test, so no other node in this file sees it, and it does what
    # the real Stage 7 does to an already-identified protein: attach the optional
    # UniProt annotation under ``enrichment`` (see
    # ``enrich_entities._enrich_payload_inner``) and nothing else. Fixed content,
    # no clock and no network, so it is deterministic; it adds no reaction,
    # removes none, renames nothing and touches no unidentified protein, so it is
    # biologically inert -- and the two artifacts are now genuinely different
    # objects, which is what makes the assertion below non-vacuous.
    import t2pw.mapping.enrich_entities as enrich_entities  # noqa: PLC0415

    def _enrichment_that_annotates(
        input_path: Any,
        output_path: Any,
        report_path: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        enriched = json.loads(Path(input_path).read_text(encoding="utf-8"))
        for row in (enriched.get("entities") or {}).get("proteins") or []:
            if not isinstance(row, dict):
                continue
            # Exactly the real stage's own guard: a protein with no accession is
            # not enriched, so the unidentified modifier this payload exists to
            # exercise is left untouched.
            accession = str((row.get("mapped_ids") or {}).get("uniprot") or "")
            if not accession:
                continue
            row.setdefault("enrichment", {}).setdefault("protein", {})["uniprot"] = {
                "status": "ok",
                "uniprot_id": accession,
                "provenance": {"source": "test_fixture", "query": accession},
            }
        report = {"summary": {"enrichment_annotated_for_test": True}}
        Path(output_path).write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    monkeypatch.setattr(enrich_entities, "run_enrichment", _enrichment_that_annotates)

    at = _run_post_pipeline_research(_payload_with_one_bad_peripheral())

    assert _app_exceptions(at) == []
    report = at.session_state["quarantine_report"]
    assert report["export_mode"] == "research"
    assert report["ok"] is True
    assert report["research_mode"]["applied"] is False

    # The candidate the boundary passed on still holds every process, including
    # the unmapped one. This is what quarantine controls and what it must not
    # take away.
    artifacts = at.session_state["post_pipeline_artifacts"]
    candidate = artifacts["final_mapped_quarantined"]
    assert [row["name"] for row in candidate["processes"]["reactions"]] == [
        "gamma-glutamylcysteine synthesis",
        "glutathione synthesis",
        "peripheral OPDA reduction",
    ]
    # The two artifacts really are different objects here, so the equality below
    # is a statement about quarantine and not an accident of a no-op stage. If
    # enrichment ever reverts to a pass-through, this fails loudly rather than
    # letting the boundary assertion go quietly vacuous again.
    assert artifacts["final_mapped_db"] != artifacts["final_mapped_enriched"], (
        "enrichment produced no difference, so the boundary assertion below would "
        "hold for the pre-enrichment artifact too and pin nothing"
    )
    # ...and the difference is annotation only: strip the enrichment block the
    # stage added and the two artifacts are identical again. This is the proof
    # that the arranged difference changes no biology; it is deliberately NOT
    # part of the boundary comparison, which stays whole-object below.
    _without_annotation = deepcopy(artifacts["final_mapped_enriched"])
    for _row in _without_annotation["entities"]["proteins"]:
        _row.pop("enrichment", None)
    assert _without_annotation == artifacts["final_mapped_db"]

    # Byte for byte, not merely "the same processes": research quarantine forwards
    # the candidate it was handed unchanged, so the payload the boundary passed on
    # must be equal to the one it received. This is the assertion that fails if
    # quarantine ever edits a research payload -- reorders a list, drops an empty
    # bucket, normalizes a name -- while leaving the process count intact.
    #
    # The comparand is ``final_mapped_enriched`` because that key IS
    # ``final_export_payload``, the exact object handed to
    # ``freeze_canonical_payload`` (streamlit_app.py: the freeze call, and the
    # artifact published under this key). ``final_mapped_db`` is the PRE-enrichment
    # remap output, one stage too early, so comparing against it silently covers
    # enrichment as well as quarantine; ``final_mapped`` and ``final_export_input``
    # are both ``canonical_export_payload or final_export_payload`` and collapse to
    # the post-quarantine object itself, which would compare the payload to itself.
    # Indexed, not ``.get``: a missing key must fail loudly rather than fall back
    # to a comparand that pins less.
    mapped_in = artifacts["final_mapped_enriched"]
    assert candidate == mapped_in

    # And the run is not blocked one step later either. The post-mapping Stage 3
    # revalidation fails on this payload -- the unmapped modifier is exactly what
    # it rejects -- and in research mode it annotates instead of stopping, so
    # refinement review opens on the untouched candidate.
    assert at.session_state["refinement_gate_errors"] == []
    assert at.session_state["refinement_working_json"] == candidate


@pytest.mark.usefixtures("offline_mapping")
def test_research_mode_surfaces_the_flags_it_did_not_act_on() -> None:
    """Fail-open must not mean fail-silent."""

    at = _run_post_pipeline_research(_payload_with_one_bad_peripheral())

    flags = at.session_state["quarantine_report"]["research_mode"]["review_flags"]
    assert [flag["name"] for flag in flags] == ["peripheral OPDA reduction"]
    assert flags[0]["state"] == QUARANTINED_UNMAPPED_ENTITY
    assert any("Quarantine review flags" in str(w.value) for w in at.warning)


@pytest.mark.usefixtures("offline_mapping")
def test_research_mode_is_not_blocked_by_a_missing_requested_core() -> None:
    """The payload a strict run refuses outright still yields a candidate here."""

    at = _run_post_pipeline_research(_payload_with_no_viable_core())

    report = at.session_state["quarantine_report"]
    assert report["ok"] is True
    assert report["research_mode"]["would_have_refused"], (
        "the strict verdict is still recorded, it is just not acted on"
    )
    # Strict refuses this payload outright and hands nothing on. Research keeps
    # every process, so the candidate is intact for the reviewer and the
    # citation report even though a strict export could never be built from it.
    candidate = at.session_state["post_pipeline_artifacts"]["final_mapped_quarantined"]
    assert len(candidate["processes"]["reactions"]) == 3
    assert not any(
        "Quarantine could not produce a viable" in str(e.value) for e in at.error
    )
    # The strict counterpart of this payload is refused outright and never opens
    # review -- test_a_run_with_no_viable_core_is_refused_at_the_boundary pins
    # that. Research reaches review on the same bytes.
    assert at.session_state["refinement_working_json"] == candidate


@pytest.mark.usefixtures("offline_mapping")
def test_research_keeps_the_stage3_findings_without_claiming_the_gates_passed() -> None:
    """Fail-open, not fail-silent, at the gate as well as at quarantine.

    Research mode must not block on Stage 3, and it must not pretend Stage 3 was
    clean. Both halves are asserted here because either one alone is a bug: a run
    that blocks delivers nothing for a novel pathway, and a run that says "ready
    for review" over a failed gate tells the reviewer this candidate is
    PathWhiz-exportable when it is not.
    """

    at = _run_post_pipeline_research(_payload_with_one_bad_peripheral())
    artifacts = at.session_state["post_pipeline_artifacts"]

    # The strict verdict, retained in full rather than discarded on the way past.
    gate = artifacts["final_stage3_gate_report"]
    assert gate["ok"] is False
    reasons = [str(row.get("reason", "")) for row in gate["errors"]]
    assert any("OPR3" in reason and "UniProt" in reason for reason in reasons), reasons

    assert artifacts["research_stage3_failed_open"] is True
    assert artifacts["research_stage3_review_flags"] == gate["errors"]

    # Said out loud, and never as a pass.
    assert any("did NOT pass" in str(w.value) for w in at.warning)
    assert not any("ready for review" in str(i.value) for i in at.info)


@pytest.mark.usefixtures("offline_mapping")
def test_one_unmapped_enzyme_reaches_review_in_research_and_is_quarantined_in_strict() -> None:
    """The same payload, both modes, end to end through the real app.

    This is the pair that makes "fail-open" mean something: identical input,
    identical seeding, and the only difference is the export-mode radio. Research
    carries the unmapped process into refinement review; strict removes it and
    exports the rest.
    """

    payload = _payload_with_one_bad_peripheral()
    research = _run_post_pipeline_research(deepcopy(payload))
    strict = _run_post_pipeline(deepcopy(payload))

    def _reactions(at) -> list:
        working = at.session_state["refinement_working_json"]
        assert working is not None, "refinement review never opened"
        return [row["name"] for row in working["processes"]["reactions"]]

    assert "peripheral OPDA reduction" in _reactions(research)
    assert "peripheral OPDA reduction" not in _reactions(strict)

    assert strict.session_state["quarantine_report"]["counts"][QUARANTINED_UNMAPPED_ENTITY] == 1
    research_report = research.session_state["quarantine_report"]
    assert research_report["counts"][QUARANTINED_UNMAPPED_ENTITY] == 1, (
        "research must still MAKE the decision; it just does not apply it"
    )
    assert research_report["research_mode"]["would_have_quarantined"] == 1


# ── Lifecycle: a second run must not inherit the first run's decision ───────


def _second_post_pipeline_run(at, payload: Dict[str, Any]):
    """Begin a SECOND audit/mapping run in the same session, through the button.

    Deliberately not ``_run_post_pipeline``: that builds a fresh ``AppTest`` and
    wipes the artifacts itself, which would test the fixture rather than the app.
    Same session, same widget, new payload -- the reset under test is whatever the
    app does on its own between the two clicks.
    """

    _seed_finished_pipeline(at, payload)
    at.run()
    at.button(key=KEY_POST_PIPELINE).click().run()
    return at


@pytest.mark.usefixtures("offline_mapping")
def test_a_second_run_in_one_session_clears_the_previous_decision_before_deciding() -> None:
    """The real reset, driven by the real caller, in one session.

    Three things have to hold at once and each fails differently:

    * The previous decision must be gone from ``st.session_state`` and from disk
      **before the second boundary runs**, not merely overwritten when it does.
      Proven by making the second run die before the boundary -- mapping raises --
      so anything still present afterwards can only be the first run's.
    * ``quarantine_history/`` must survive. It is the archive; a reset that took
      it would destroy the only record of why the earlier payload was admitted.
    * The first run's decision must not authorize the second run's payload, which
      is asserted directly against the matcher rather than inferred from hashes.
    """

    import shutil

    from t2pw.pipeline.strict_quarantine import decision_matches

    history = OUTPUTS / QUARANTINE_HISTORY_DIRNAME
    if history.exists():
        shutil.rmtree(history)

    # ── First run: a decision, its artifacts, and a superseded one in history ──
    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    first_report = deepcopy(at.session_state["quarantine_report"])
    assert (OUTPUTS / QUARANTINE_REPORT_FILENAME).exists()

    # Edit the reviewed graph and export: a second payload version, so the first
    # decision is archived rather than overwritten. This is what puts something in
    # quarantine_history for the reset to leave alone.
    working = deepcopy(at.session_state["refinement_working_json"])
    working["entities"]["compounds"].append({"name": "spare compound"})
    at.session_state["refinement_working_json"] = working
    at.run()
    _generate_pwml(at)
    archived = sorted(p.name for p in history.iterdir()) if history.exists() else []
    assert archived == [first_report["decision_id"].replace(":", "_")], archived

    # ── Second run, same session, and it dies before the boundary ─────────────
    second_payload = _payload_with_no_viable_core()
    second_payload["metadata"]["pathway_name"] = FAIL_MAPPING_SENTINEL
    second_payload["metadata"]["name"] = FAIL_MAPPING_SENTINEL
    _second_post_pipeline_run(at, second_payload)

    assert any("Post-pipeline conversion failed" in str(e.value) for e in at.error), (
        "the second run was supposed to die before the boundary"
    )
    # Nothing from the first run survived into it.
    for key in (
        "quarantine_report",
        "quarantine_artifacts",
        "quarantine_coverage",
        "quarantine_ok",
        "quarantine_decision_controls",
    ):
        assert key not in at.session_state, key
    for name in (
        QUARANTINE_REPORT_FILENAME,
        REMOVED_ENTITY_REPORT_FILENAME,
        CLOSURE_REPORT_FILENAME,
        COVERAGE_REPORT_FILENAME,
    ):
        assert not (OUTPUTS / name).exists(), name

    # The archive is untouched.
    assert sorted(p.name for p in history.iterdir()) == archived

    # And even had it survived, it could not have authorized this payload.
    assert (
        decision_matches(
            first_report,
            second_payload,
            mode="pathwhiz",
            strict_db=True,
            pathway_context=STAGE_ZERO_CONTEXT,
        )
        is False
    )


@pytest.mark.usefixtures("offline_mapping")
def test_a_second_run_that_reaches_the_boundary_decides_for_itself() -> None:
    """The same reset, on the path where the second run does get that far."""

    at = _run_post_pipeline(_payload_with_one_bad_peripheral())
    first_hash = at.session_state["quarantine_report"]["admitted_payload_hash"]

    _second_post_pipeline_run(at, _payload_with_no_viable_core())
    second_report = at.session_state["quarantine_report"]

    # **MOVED BY C-041a (D-002, LOCKED) -- and the invariant under test did NOT
    # move.** "The second run decides for itself" is carried by the two hash
    # assertions, which are byte for byte what they were: the second run judged
    # a different payload and the on-disk artifact is that second decision, not
    # the first. Only the second decision's VERDICT moved, for the same
    # payload-level reason as above. Base SHA: ``ok is False``.
    assert second_report["admitted_payload_hash"] != first_hash
    assert _artifact(QUARANTINE_REPORT_FILENAME)["admitted_payload_hash"] == (
        second_report["admitted_payload_hash"]
    )
    assert second_report["ok"] is True
    assert second_report["release"]["status"] == "review_required"
    assert second_report["release"]["strict_acceptance_eligible"] is False


def test_the_reset_is_wired_into_the_start_of_a_pipeline_run() -> None:
    """Static: the app calls it, at both seams where a new run begins.

    Two, not one. The Stage 1/2 pipeline is one entry point; clicking "Run audit
    and DB mapping" again on an already-loaded pathway is the other, and it is the
    one the lifecycle test drives. A reset wired only into the first would leave a
    second mapping run inheriting the first's decision.
    """

    import ast

    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reset_quarantine_state"
    ]

    assert len(calls) >= 2, "both run entry points must void the previous decision"
    # It clears the on-disk artifacts too, not only the session keys.
    assert all(call.args for call in calls), "reset must be given the outputs directory"
