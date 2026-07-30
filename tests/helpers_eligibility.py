"""Shared fixtures for the paper-eligibility tests.

Not a test module (pytest only collects ``test_*.py``). Holds the real paper
titles, the requested scopes and the small fakes that both
``test_paper_eligibility.py`` and ``test_paper_eligibility_corrections.py`` use,
so the two cannot drift on what "the Fournier paper" or "the requested scope"
means.

Every title here is verbatim from ``runs/2026-07-28_2122/plan.json``, HTML
entities included, because handling that noise is part of the job.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.rag.acquire import CandidatePaper  # noqa: E402
from t2pw.rag.eligibility import (  # noqa: E402
    EligibilityDecision,
    EligibilityThresholds,
    RequestedScope,
    screen_paper,
)

#: The shipped defaults, stated explicitly so no verdict in the tests can be
#: moved by an environment variable or a developer's ``.env``.
THRESHOLDS = EligibilityThresholds()

#: The stored run the dry-run tests evaluate against.
DRY_RUN_PLAN = ROOT / "runs" / "2026-07-28_2122"

# --- real titles from runs/2026-07-28_2122 ---------------------------------
T_FOURNIER = (
    "Ochrobactrum anthropi Causing Fournier's Gangrene: A Report of a Rare Case."
)
T_POULTRY = (
    "Genotypic Characterization of Virulence Factors in Extended-Spectrum "
    "Beta-Lactamase (ESBL)-Producing &lt;i&gt;Escherichia coli&lt;/i&gt; Strains "
    "from Chickens in Hungary."
)
T_TURKEY = (
    "Virulence Gene Profiles of Extended-Spectrum Beta-Lactamase (ESBL)-Producing "
    "&lt;i&gt;Escherichia coli&lt;/i&gt; Isolated from Turkeys in Hungary: A "
    "Whole-Genome Sequencing Study."
)
T_COVID = (
    "Association of gender and main comorbidities with expression of lncRNAs and "
    "mRNAs in COVID-19 patients"
)
T_ENTB = (
    "The enterobactin biosynthetic intermediate 2,3-dihydroxybenzoic acid is a "
    "competitive inhibitor of the Escherichia coli isochorismatase EntB."
)
T_MEND_LISTERIA = (
    "Structures of Listeria monocytogenes MenD in ThDP-bound and in-crystallo "
    "captured intermediate I-bound forms."
)
T_HEME_FEEDBACK = (
    "A reversible feedback mechanism regulating mitochondrial heme synthesis."
)
T_LIPID_A_REVIEW = "The regulation of lipid A biosynthesis."

# --- requested scopes, matching topics.txt --------------------------------
LIPID_A_ECOLI = RequestedScope(
    requested_pathway="lipid A biosynthesis", requested_organism="Escherichia coli"
)
ENTEROBACTIN_ECOLI = RequestedScope(
    requested_pathway="enterobactin biosynthesis", requested_organism="Escherichia coli"
)
MENAQUINONE_SUBTILIS = RequestedScope(
    requested_pathway="menaquinone biosynthesis", requested_organism="Bacillus subtilis"
)
HEME_HUMAN = RequestedScope(
    requested_pathway="heme biosynthesis", requested_organism="Homo sapiens"
)
CHOLESTEROL_HUMAN = RequestedScope(
    requested_pathway="cholesterol biosynthesis", requested_organism="Homo sapiens"
)

# --- the three reference sets from the 2026-07-28_2122 plan ---------------
#: Papers a pathway extractor could never have succeeded on, each of which cost a
#: full-text download plus two app runs in that night's batch.
JUNK_2122 = (
    "PMC12971581",  # Fournier's gangrene case report
    "PMC12649316",  # chicken ESBL virulence survey
    "PMC12737783",  # turkey ESBL whole-genome survey
    "PMC12797059",  # COVID-19 lncRNA comorbidity study
    "PMC13139079",  # river resistome surveillance
    "PMC12898691",  # gene-set evolution tool
)

#: Genuine Lpx / Men / PPOX / Ent reaction papers -- the acceptances that must
#: survive every tightening of the contextual rules.
GENUINE_2122 = (
    "PMC12444477",  # regulation of lipid A biosynthesis
    "PMC12096016",  # enterobactin EntB / isochorismatase
    "PMC13264790",  # PPOX, heme pathway
    "PMC12856317",  # mitochondrial heme synthesis feedback
    "PMC11946230",  # menaquinone-7 in B. subtilis
)

#: The five real cholesterol false-positive shapes: pathway genes named in an
#: omics hit list, or the molecule named with no local reaction evidence.
CHOLESTEROL_FALSE_POSITIVES_2122 = (
    "PMC12113831",  # Quercus robur keratinocyte proteomics
    "PMC12428349",  # cholesterol regulates airway differentiation
    "PMC12782028",  # CaSR antagonism in osteosarcoma, transcriptome
    "PMC12993329",  # CD79A/CD40 CAR-T metabolic pathway
    "PMC12705669",  # PRDM9 persister cells in glioblastoma
)


def screen(
    title: str, scope: RequestedScope, abstract: str = "", paper_id: str = "PMC1"
) -> EligibilityDecision:
    """Screen one title/abstract at the pinned thresholds."""
    return screen_paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        scope=scope,
        thresholds=THRESHOLDS,
    )


def searcher(mapping: Dict[str, List[CandidatePaper]]):
    """A fake ``search_candidates`` keyed by the topic (pathway_name)."""

    def _search(context: Dict[str, Any], **kwargs: Any) -> List[CandidatePaper]:
        status = kwargs.get("status")
        if isinstance(status, dict):
            status["query"] = f'"{context.get("pathway_name")}"'
        return list(mapping.get(str(context.get("pathway_name") or ""), []))

    return _search


def dry_run_module():
    """Import ``scripts/eligibility_dry_run.py`` without leaving it on the path."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import eligibility_dry_run  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    return eligibility_dry_run
