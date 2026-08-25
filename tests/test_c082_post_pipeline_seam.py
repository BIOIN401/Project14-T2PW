"""C-082 / F-115 at the seam that actually died -- the real app, the real widget.

``run_post_pipeline_sbml_artifacts`` calls ``run_prefreeze_resolution`` with **no
``try``**, and the app's outermost handler turns anything that comes out of it
into ``st.error("Post-pipeline conversion failed: ...")``. ``t2pw.batch.driver``
reads that as ``status: error``, ``failure_kind: crash``, ``stage:
post_pipeline`` -- no ``release_status``, no preserved payload, no PWML. That is
what T-106 measured on ``PMC12444477/research`` and what permanent merge rule 7
forbids.

Everything else about the declination is unit-tested in
``test_c082_species_rename_declination.py``. *Where it lands* cannot be, which is
the same reason ``test_streamlit_quarantine_boundary.py`` exists -- and this file
borrows that module's harness rather than rebuilding it: the same autouse real
``streamlit`` fixture, the same offline mapping stub, the same seeded pipeline and
the same research-mode widget drive.

**Why its own file rather than three appended tests over there.** That module is
the whole ``qb`` component of ``chunk_d_gate.py``, whose node count is
``ENFORCED`` at 23. Adding to it moves an enforced sprint-gate baseline
(``qb`` 23 -> 24, ``TOTAL`` 187 -> 188), which is a merge-rule-4 move needing
orchestrator authorization and has nothing to do with species canonicalization.
Kept here, Chunk D's partition is byte-identical, in the same convention as every
other ``test_c0NN_*.py`` card file in this suite.

**G9.** This is a correction of pre-existing observable behaviour. On base
``e648287`` the test below fails with the T-106 crash string verbatim.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

# The harness, imported rather than duplicated. ``real_streamlit`` is declared
# ``autouse=True`` at its definition, so importing it into this namespace arms it
# for this module's tests too -- without it ``AppTest`` can be a ``MagicMock``
# left behind by an earlier test module, and the whole file would assert nothing.
from test_streamlit_quarantine_boundary import (  # noqa: F401
    _app_exceptions,
    _core_payload,
    _run_post_pipeline_research,
    offline_mapping,
    real_streamlit,
)

#: Two organism rows where one canonicalizes onto the other: the exact shape that
#: crashed ``PMC12444477/research`` on T-106 (``runs_verify/2026-08-24_1428``).
#: ``Escherichia coli K-12`` is taxonomy-identified and strain-qualified, so rung
#: 4 of the species ladder collapses it onto ``Escherichia coli`` -- offline, no
#: database, no name index -- and the pre-freeze stage then finds the binomial
#: already occupied by a second, genuinely different organism.
STRAIN = "Escherichia coli K-12"
BINOMIAL = "Escherichia coli"

#: The reason string ``_canonicalizer_verdict`` publishes for a declined rename.
#: Written out rather than imported: at base the symbol does not exist, and a test
#: that failed on an ``ImportError`` would be proving symbol absence instead of
#: behaviour (merge gate G9).
DECLINED_REASON = "species_rename_declined:AMBIGUOUS_RENAME_TARGET"


def _payload_with_a_strain_and_its_species() -> Dict[str, Any]:
    """The boundary suite's core payload plus the two colliding organism rows.

    Added to it rather than replacing its organism, so the run is the one that
    module already drives end to end and the ONLY new variable is the species
    pair. Nothing references them: species rows are not participants, which is
    exactly why the pre-freeze guard has to state the refusal itself
    (``_LOCATION_MEMBER_FIELDS`` has no species bucket, so no connectivity check
    can see the merge).
    """

    payload = _core_payload()
    payload["entities"]["species"].extend(
        [
            {"name": STRAIN, "taxonomy_id": "83333", "classification": "Bacteria"},
            {"name": BINOMIAL, "taxonomy_id": "562", "classification": "Bacteria"},
        ]
    )
    return payload


@pytest.mark.usefixtures("offline_mapping")
def test_an_ambiguous_species_rename_does_not_end_the_post_pipeline_leg() -> None:
    """The leg finishes, keeps its payload, records the ambiguity, merges nothing.

    Four assertions because four different things were lost when it crashed:
    the run, the artifacts, the record, and the two organisms. The last one is
    the refusal -- unchanged -- read off the canonical payload that ships rather
    than off a summary.
    """

    from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY

    at = _run_post_pipeline_research(_payload_with_a_strain_and_its_species())

    crashes = [str(e.value) for e in at.error if "Post-pipeline conversion failed" in str(e.value)]
    assert crashes == [], crashes
    assert _app_exceptions(at) == []

    artifacts = at.session_state["post_pipeline_artifacts"]
    assert artifacts, "the leg produced no post-pipeline artifacts"

    # Recorded, on the key this seam already publishes and the CLI already returns.
    review = artifacts["prefreeze_review_required"]
    assert review.get("species") == DECLINED_REASON, review

    # Preserved AND unmerged, in the payload that ships.
    species = artifacts[CANONICAL_PAYLOAD_KEY]["entities"]["species"]
    names = [row["name"] for row in species]
    assert STRAIN in names and BINOMIAL in names, names
    assert len(names) == len(set(names)), "two organisms were merged onto one name"
    taxonomies = {row["name"]: str(row.get("taxonomy_id") or "") for row in species}
    assert taxonomies[STRAIN] == "83333"
    assert taxonomies[BINOMIAL] == "562"
