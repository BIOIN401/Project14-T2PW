"""The THIRD export entry point, and the ``db_resolution.available`` carrier.

``DECISIONS.md`` **D-033 (LOCKED)** amends D-032 clause 1: there are three
production export entry points, not two. The third is
``streamlit_app.run_pwml_export``, and -- measured on that module -- it is the
*only* one that builds a PWML IR and writes ``outputs/pathway.pwml``.

Both halves are **behavioural guards on pre-existing observable behaviour**, not
new capability, and both fail at base ``2b3de80a``: the seam tests because
``run_pwml_export`` runs no canonicalizer, so ``build_pwml_ir`` refuses every
raw compound row (``PWML_IR_COMPOUND_VERDICT_MISSING``) that the base exporter
used to repair silently *after* the freeze; the carrier tests because
``report["db_resolution"]`` carries neither ``available`` nor ``reason``, so the
preflight fires unconditionally and states, in product-visible content (D-032
clause 6), that a DB it in fact consulted was "unavailable (db_not_configured)".

Nothing here monkeypatches a module attribute to detect the seam:
``run_prefreeze_resolution``'s ``canonicalizers`` default binds
``PREFREEZE_CANONICALIZERS`` at definition time, so an attribute patch is
invisible to the call and reports a false zero. ``sys.setprofile`` is not.
"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ``t2pw.app.streamlit_app`` executes a Streamlit script on import -- it opens
# ``with st.form("pwml_pipeline")`` at module scope (:4310). Under the REAL
# streamlit, in a plain pytest process with no ``ScriptRunContext``, that form
# stays open on the global DeltaGenerator and every later ``AppTest`` in the
# process dies with "Forms cannot be nested in other forms" -- measured against
# ``tests/test_batch_driver.py``. The stub below is the established convention,
# copied from ``tests/test_rag_admission_production_path.py``; that file's
# ``real_streamlit`` autouse fixture (``test_batch_driver.py:56-90``) is written
# to cooperate with exactly this and documents it at ``:62``. Sound here because
# ``run_pwml_export`` makes no Streamlit call at all -- the stub neutralizes the
# top-level script and nothing else. The guard matters: if the real streamlit is
# already loaded, stubbing it would break whatever loaded it.
if "streamlit" not in sys.modules:
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st

from t2pw.pwml.ir import (  # noqa: E402
    COMPOUND_RESOLUTION_VERDICT_FIELD,
    PREFREEZE_DB_RESOLUTION_FIELD,
    build_pwml_ir,
)
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution  # noqa: E402


# Fixtures


class _CannedDb:
    """A REACHABLE PathBank DB that answers with a Glycine row."""

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        return [{
            "id": 78, "name": "Glycine", "short_name": "Gly", "hmdb_id": "HMDB0000123",
            "kegg_id": "C00037", "chebi_id": "15428", "pubchem_cid": "750",
            "cas": "56-40-6", "biocyc_id": "GLY", "chemspider_id": "730",
            "drugbank_id": "DB00145", "pwc_id": "PW_C000123", "description": "canned",
            "synonyms": "Glycine; Gly",
        }]


class _DownDb:  # CONFIGURED and unreachable; ``last_error`` is the true reason
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


class _EmptyReachableDb:
    """Reachable, and it has no row for the query -- the A4 population.

    P4-01: no worktree carries a ``.env``, so ``PathBankDbResolver.from_env()``
    answers ``None`` and every PathBank-dependent assertion would be vacuously
    clean. The reachable direction is therefore exercised with this canned
    double rather than with a live connection.
    """

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        return []


#: ``all_legacy`` is the discriminating population. Every row carries a
#: ``pathbank_compound_id``, so compound resolution takes its legacy-id branch
#: and ``continue``s **before** the resolver is consulted: the resolved rows are
#: byte-identical whether or not the DB was reachable. Any exporter that tried
#: to *infer* availability from the rows would have to guess here, which is why
#: the value has to be carried.
CARRIER_POPULATIONS: Dict[str, List[Dict[str, Any]]] = {
    "all_legacy": [
        {"name": "Glycine", "pathbank_compound_id": 78},
        {"name": "Pyruvic acid", "pathbank_compound_id": 91},
    ],
    "mixed": [
        {"name": "Glycine", "pathbank_compound_id": 78},
        {"name": "gly", "kegg_id": "C00037"},
    ],
}


def _shell_payload(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": deepcopy(compounds),
        },
        "biological_states": [
            {"name": "cyto_state", "species": "Homo sapiens",
             "subcellular_location": "cytosol"}
        ],
        "element_locations": {"compound_locations": []},
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _raw_extraction_payload() -> Dict[str, Any]:
    """A payload no canonicalizer has run over -- what the third seam receives.

    Structurally the ``_stage6_extraction_payload`` of
    ``tests/test_streamlit_stage8_export_contract.py``, which is already pinned
    as a payload the Stage-8 gate, the pre-export contract and the IR build all
    accept. A refusal in these tests is therefore the seam, never the fixture.
    """

    return {
        "metadata": {
            "name": "Caffeine demethylation",
            "pathway_name": "Caffeine demethylation",
            "subject": "Metabolic",
            "pathway_subject": "Metabolic",
        },
        "entities": {
            "species": [{"name": "Pseudomonas putida", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "caffeine", "pathbank_compound_id": 101},
                {"name": "theobromine", "pathbank_compound_id": 102},
            ],
            "proteins": [
                {"name": "NdmA", "species": "Pseudomonas putida",
                 "mapped_ids": {"uniprot": "Q9I147"}}
            ],
            "protein_complexes": [
                {
                    "name": "NdmA complex",
                    "species": "Pseudomonas putida",
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                    "components": [{"name": "NdmA", "stoichiometry": 1}],
                }
            ],
        },
        "biological_states": [
            {"name": "Pseudomonas putida cytosol", "species": "Pseudomonas putida",
             "subcellular_location": "cytosol"}
        ],
        "processes": {
            "reactions": [
                {
                    "name": "caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["theobromine"],
                    "biological_state": "Pseudomonas putida cytosol",
                    "enzymes": [{"entity": "NdmA complex",
                                 "entity_type": "protein_complex", "role": "catalyst"}],
                    "modifiers": [{"entity": "NdmA complex",
                                   "entity_type": "protein_complex", "role": "catalyst"}],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


#: Every frame whose ``call`` event answers "did the pre-freeze sequence run,
#: and did the exporter resolve anything of its own". ``build_pwml_ir`` is in
#: the set so a trace with no pre-freeze call can be told apart from a run that
#: never reached the exporter at all.
_TRACED = (
    "run_prefreeze_resolution",
    "resolve_compounds_prefreeze",
    "resolve_species_prefreeze",
    "build_pwml_ir",
    "_resolve_compound_rows",
)


def _export_with_trace(project_root: Path) -> tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    from t2pw.app.streamlit_app import run_pwml_export

    payload = _raw_extraction_payload()
    trace: List[str] = []

    def _profiler(frame: Any, event: str, arg: Any) -> None:  # noqa: ARG001
        if event == "call" and frame.f_code.co_name in _TRACED:
            trace.append(frame.f_code.co_name)

    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    sys.setprofile(_profiler)
    try:
        result = run_pwml_export(
            payload,
            pathway_name="Caffeine demethylation",
            pathway_description="",
            pathway_subject="Metabolic",
            project_root=project_root,
            ref_path=ROOT / "reference" / "PW000001.pwml",
            strict_db=False,
        )
    finally:
        sys.setprofile(None)
    return result, trace, payload


# Part 1 -- the seam


def test_run_pwml_export_runs_the_prefreeze_sequence_once_before_build_pwml_ir(
    tmp_path: Path,
) -> None:
    """D-033 clause 2. Measured by frame trace, not by reading the source.

    Fails at base: the trace is ``['build_pwml_ir']`` -- zero pre-freeze calls.
    """

    result, trace, _payload = _export_with_trace(tmp_path)

    assert trace.count("run_prefreeze_resolution") == 1, trace
    assert "build_pwml_ir" in trace, trace
    assert trace.index("run_prefreeze_resolution") < trace.index("build_pwml_ir"), trace
    # Merge rule 8 / PRODUCT_CONTRACT §5: every compound resolution happened
    # BEFORE the exporter was entered. Counted by position in the trace, so a
    # resolution that moved back inside ``build_pwml_ir`` fails here.
    assert trace[trace.index("build_pwml_ir"):].count("_resolve_compound_rows") == 0, trace
    assert result["ok"] is True, result.get("error")


def test_the_third_entry_point_exports_a_payload_no_canonicalizer_had_touched(
    tmp_path: Path,
) -> None:
    """The seam is wired for its effect, not merely called.

    Fails at base with ``PWML_IR_COMPOUND_VERDICT_MISSING`` and no file on disk.
    """

    result, _trace, _payload = _export_with_trace(tmp_path)

    assert result["ok"] is True, result.get("error")
    out_path = tmp_path / "outputs" / "pathway.pwml"
    assert out_path.is_file()
    assert out_path.read_bytes().startswith(b"<?xml")
    rows = result["pwml_ir"]["entities"]["compounds"]
    assert rows, "the fixture must carry compounds or this asserts nothing"
    assert all(
        str(row.get(COMPOUND_RESOLUTION_VERDICT_FIELD) or "").strip() for row in rows
    ), rows


def test_the_seam_is_on_the_whole_canonicalizer_tuple_not_on_one_stage(
    tmp_path: Path,
) -> None:
    """D-032 clause 1: wired once, for the tuple, so future stages are covered.

    Measured by trace -- every registered canonicalizer is observed executing on
    this entry point -- rather than by reading the seam's argument list. D-033
    clause 3 is explicit that an enumeration inherited from a charter or a prior
    card is not acceptable evidence, so the registry is re-derived here and every
    entry of it must appear in the trace.
    """

    _result, trace, payload = _export_with_trace(tmp_path)

    from t2pw.pwml.prefreeze_resolution import PREFREEZE_CANONICALIZERS

    registered = {canonicalizer.__name__ for _name, canonicalizer in PREFREEZE_CANONICALIZERS}
    assert registered, "an empty registry would make this assertion vacuous"
    assert registered <= set(trace), (sorted(registered), trace)
    # ``run_pwml_export`` deep-copies its input, so the caller's payload must be
    # untouched -- the seam mutates the export copy, never the reviewer's object.
    assert PREFREEZE_DB_RESOLUTION_FIELD not in payload


# Part 2 -- the carrier


@pytest.mark.parametrize("population", sorted(CARRIER_POPULATIONS))
@pytest.mark.parametrize("reachable", [True, False])
def test_db_reachability_reaches_the_exporter_in_both_directions(
    population: str, reachable: bool
) -> None:
    """A3. The exporter's report equals the pre-freeze stage's own verdict.

    Fails at base in **both** directions: ``db_resolution`` carries no
    ``available`` key at all, so the report cannot state either answer.
    """

    payload = _shell_payload(CARRIER_POPULATIONS[population])
    resolver = _CannedDb() if reachable else None
    prefreeze = run_prefreeze_resolution(payload, strict_db=False, db_resolver=resolver)
    prefreeze_available = (
        prefreeze["compounds"]["resolution_report"]["db_resolution"]["available"]
    )
    assert prefreeze_available is reachable, prefreeze

    _ir, ir_report = build_pwml_ir(payload, strict_db=False)
    assert ir_report["db_resolution"]["available"] is reachable
    assert payload[PREFREEZE_DB_RESOLUTION_FIELD]["available"] is reachable


def test_an_all_legacy_population_proves_the_value_is_carried_not_inferred() -> None:
    """The discriminating test C-051 identified, turned into a guard.

    Every row carries a ``pathbank_compound_id``, so the resolver is never
    consulted and the two legs' rows are **byte-identical** -- yet the exporter
    must still report the two different answers. Only a carried value can do
    that; anything derived from the rows is a guess.
    """

    blobs: Dict[bool, str] = {}
    reported: Dict[bool, Any] = {}
    for reachable in (True, False):
        payload = _shell_payload(CARRIER_POPULATIONS["all_legacy"])
        run_prefreeze_resolution(
            payload, strict_db=False, db_resolver=_CannedDb() if reachable else None
        )
        blobs[reachable] = json.dumps(
            payload["entities"]["compounds"], sort_keys=True, default=str
        )
        _ir, ir_report = build_pwml_ir(payload, strict_db=False)
        reported[reachable] = ir_report["db_resolution"]["available"]

    assert hashlib.sha256(blobs[True].encode()).hexdigest() == hashlib.sha256(
        blobs[False].encode()
    ).hexdigest(), "the population is no longer indistinguishable; this test is vacuous"
    assert reported[True] is True
    assert reported[False] is False


def test_a_reachable_db_with_no_matching_row_does_not_claim_it_was_unavailable() -> None:
    """A4. D-032 clause 6 rules ``preflight`` product-visible export content.

    At base this export states "Resolution DB unavailable (db_not_configured)"
    about a database that was consulted and answered -- a false statement,
    persisted to ``pwml_ir_report.json``.
    """

    payload = _shell_payload([{"name": "norbelladine", "hmdb_id": "HMDB9999999"}])
    prefreeze = run_prefreeze_resolution(
        payload, strict_db=False, db_resolver=_EmptyReachableDb()
    )
    assert (
        prefreeze["compounds"]["resolution_report"]["db_resolution"]["available"] is True
    ), "the fixture must model a REACHABLE db or this asserts nothing"

    _ir, ir_report = build_pwml_ir(payload, strict_db=False)

    assert ir_report["db_resolution"]["available"] is True
    assert "preflight" not in ir_report or ir_report["preflight"] is None
    assert not [
        issue for issue in ir_report["warnings"]
        if "Resolution DB unavailable" in str(issue.get("message") or "")
    ], ir_report["warnings"]


@pytest.mark.parametrize(
    ("resolver", "reason"), [(None, "db_not_configured"), (_DownDb(), "harvest_db_down")],
    ids=["not_configured", "configured_but_down"])
def test_the_same_row_still_warns_and_names_the_real_reason(resolver: Any, reason: str) -> None:
    """The other direction, so the fix is not "never warn again".

    ``configured_but_down`` is the second half of A4: carrying ``available``
    alone leaves the preflight defaulting the reason to ``db_not_configured``,
    a *different* false statement in the same product-visible sentence -- the
    DB was configured, and it was down. Both legs must name what happened.
    """

    payload = _shell_payload([{"name": "norbelladine", "hmdb_id": "HMDB9999999"}])
    run_prefreeze_resolution(payload, strict_db=False, db_resolver=resolver)
    assert payload[PREFREEZE_DB_RESOLUTION_FIELD] == {
        "available": False, "reason": reason}

    _ir, ir_report = build_pwml_ir(payload, strict_db=False)

    assert ir_report["db_resolution"] == {"compounds": [], "available": False,
                                          "reason": reason}
    assert ir_report["preflight"]["compounds"] == ["norbelladine"]
    assert ir_report["preflight"]["db_reason"] == reason
    assert [
        issue for issue in ir_report["warnings"]
        if issue.get("code") == "noncanonical_names_collision_risk"
        and reason in str(issue.get("message") or "")
    ]


def test_a_payload_no_prefreeze_stage_touched_keeps_the_report_shape() -> None:
    """The carrier is read, never manufactured.

    ``build_pwml_ir`` is called directly on committed leg fixtures by C-040's
    golden, which hashes this report. A payload carrying no marker must
    therefore keep exactly the ``db_resolution`` shape ``_new_report`` gives it
    -- no invented ``available``, in either direction.
    """

    payload = _shell_payload([{"name": "Glycine", "pathbank_compound_id": 78,
                               COMPOUND_RESOLUTION_VERDICT_FIELD: "legacy_id_unverified"}])
    assert PREFREEZE_DB_RESOLUTION_FIELD not in payload

    _ir, ir_report = build_pwml_ir(payload, strict_db=False)

    assert ir_report["db_resolution"] == {"compounds": []}
