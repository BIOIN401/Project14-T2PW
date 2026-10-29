"""ORCH-793 (ORCH-789 C4, D-119 s2): UniProt's own gene name of an entry, in the pathway species, admits it.

Candidates are the UniProt answers the ORCH-784/785 legs recorded in their run caches
(``api-v9::pif1::tax:3702>3702``, ``api-fallback-v2::fur::tax:562>562``, ``api-v9::tak1::tax:9606>9606``;
gene fields confirmed against ``docs/pwml_recovery_sprint/evidence/orch793/uniprot_entries.json``).
No fuzzy matching is added and a real id is still required (D-106).

**G9 -- CORRECTION of observable behaviour** (``test_g9_*``): each fails on base ``f3a80e09``.
**PRESERVATION** (``test_preservation_*``): passes at base and tip.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List

from t2pw.mapping.map_ids import _resolve_ambiguous_protein_candidates, verify_real_protein_identity


def _c(acc: str, name: str, genes: List[str], primary: List[str], organism: str, reviewed: bool, score: float,
       taxon: str) -> Dict[str, Any]:
    return {"accession": acc, "protein_name": name, "gene_names": genes, "primary_gene_names": primary,
            "organism": organism, "taxonomy_id": taxon, "reviewed": reviewed, "score": score}


AT = "Arabidopsis thaliana"
PIF1 = [  # api-v9::pif1::tax:3702>3702
    _c("Q8GZM7", "Transcription factor PIF1", ["BHLH15", "EN101", "PIF1", "PIL5"], ["PIF1"], AT, True, 1.0, "3702"),
    _c("P93831", "Histone-lysine N-methyltransferase CLF", ["CLF", "ICU1", "PIF1", "PIF2", "SDG1", "SET1"], ["CLF"],
       AT, True, 1.0, "3702"),
    {"accession": "F4J4H6", "protein_name": "PIF1 helicase", "gene_names": [], "organism": AT, "taxonomy_id": "3702",
     "reviewed": False, "score": 0.425},
]
FUR = [
    _c("P0A9A9", "Ferric uptake regulation protein", ["fur"], ["fur"], "Escherichia coli (strain K12)", True, 0.85, "83333"),
    _c("P0A9B0", "Ferric uptake regulation protein", ["fur"], ["fur"],
       "Escherichia coli O6:H1 (strain CFT073 / ATCC 700928 / UPEC)", True, 0.85, "199310"),
    _c("P0A9B1", "Ferric uptake regulation protein", ["fur"], ["fur"], "Escherichia coli O157:H7", True, 0.85, "83334"),
    _c("B7MFV0", "Ferric uptake regulation protein", ["fur"], ["fur"], "Escherichia coli O45:K1 (strain S88 / ExPEC)",
       False, 0.8, "585035"),
]
HS = "Homo sapiens"
TAK1 = [
    _c("P49116", "Nuclear receptor subfamily 2 group C member 2", ["NR2C2", "TAK1", "TR4"], ["NR2C2"], HS, True, 1.0, "9606"),
    _c("O43318", "Mitogen-activated protein kinase kinase kinase 7", ["MAP3K7", "TAK1"], ["MAP3K7"], HS, True, 1.0, "9606"),
]


def _verify(name: str, candidates: List[Dict[str, Any]], accession: str, organism: str) -> Dict[str, Any]:
    return verify_real_protein_identity(name, candidates=candidates, mapped_ids={"uniprot": accession},
                                        organism=organism, source="api")


NARF = [
    _c("Q9UHQ1", "Nuclear prelamin A recognition factor", ["NARF"], ["NARF"], HS, True, 1.0, "9606"),
    _c("Q8WVD3", "E3 ubiquitin-protein ligase RNF138", ["NARF", "RNF138"], ["RNF138"], HS, True, 1.0, "9606"),
]


def test_preservation_a_primary_gene_name_does_not_outrank_another_entrys_synonym() -> None:
    """Why PIF1 (chl x3) is NOT recovered: CLF lists PIF1 as a synonym, Q8GZM7 as its gene
    name, and UniProt's gene fields cannot say which the paper means. "Primary beats synonym"
    would pick Q8GZM7 here -- and on wnt_ca pick Q9UHQ1 (prelamin A recognition factor) for
    NARF, which the paper means as RNF138 (NLK-associated RING finger). Both stay refused."""

    assert _verify("PIF1", PIF1, "Q8GZM7", AT)["verified"] is False
    assert _verify("NARF", NARF, "Q9UHQ1", HS)["verified"] is False


def test_g9_fur_strain_records_of_one_gene_in_the_species_are_one_answer() -> None:
    """E. coli Fur: P0A9A9 / P0A9B0 / P0A9B1 (+ unreviewed B7MFV0) are one gene, one name (entero x2)."""

    verdict = _verify("Fur", FUR, "P0A9A9", "Escherichia coli")
    assert verdict["verified"] is True, verdict["reason"]


def test_g9_ambiguous_same_gene_survivors_resolve_in_a_stable_order() -> None:
    """Whatever order UniProt lists them in, the same record is picked: never list order."""

    picks = set()
    for order in permutations(FUR[:3]):
        chosen, ambiguity = _resolve_ambiguous_protein_candidates(
            "Fur", organism="Escherichia coli", result={"source": "api", "candidates": list(order)})
        assert chosen is not None, ambiguity
        picks.add(chosen["accession"])
    assert picks == {"P0A9A9"}


def test_preservation_tak1_a_synonym_of_two_entries_stays_refused() -> None:
    """TAK1 is a gene SYNONYM of both NR2C2 and MAP3K7 and the primary name of neither:
    UniProt's gene fields cannot choose, so neither is admitted (D-119 s2 not met)."""

    for accession in ("O43318", "P49116"):
        assert _verify("TAK1", TAK1, accession, HS)["verified"] is False


def test_preservation_a_synonym_owner_is_still_refused_against_the_primary_owner() -> None:
    assert _verify("PIF1", PIF1, "P93831", AT)["verified"] is False


def test_preservation_same_gene_in_another_species_never_collapses() -> None:
    other = _c("Q8ZRL5", "Ferric uptake regulation protein", ["fur"], ["fur"], "Salmonella typhimurium", True, 0.85, "90371")
    assert _verify("Fur", [FUR[0], other], "Q8ZRL5", "Escherichia coli")["verified"] is False
