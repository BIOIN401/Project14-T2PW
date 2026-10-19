"""ORCH-747: a process participant resolves to the kind its registry declares.

This is a CORRECTION of pre-existing observable behaviour, not a new capability, so it
carries a behavioural proof that fails on the base commit:
``docs/pwml_recovery_sprint/evidence/orch747_graph_kind_base_proof.py`` imports
``draft_graph.py`` as committed at HEAD, builds the same payload, and shows the orphaned
and duplicated nodes this test forbids.

The defect: ``_resolve_actor_kind`` was given name sets for proteins, protein_complexes
and compounds only. A participant naming a nucleic acid or an element collection matched
none of them and took the ``compound`` default. Because a node id carries its kind, that
minted a SECOND node for the same entity, and every edge attached to the mis-typed one
while the correctly typed node from the registry loop sat with no edges at all.

Measured on a real run (araC autoregulation, PMC319880): all 8 DNA entities existed
twice, the 8 real nucleic-acid nodes were orphans, and 8 of the 15 compound nodes were
DNA. Downstream that means identifier mapping tries to resolve a promoter as a compound.
"""

from __future__ import annotations

import collections
from typing import Dict, List

from t2pw.pipeline.draft_graph import ENTITY_KIND_MAP, build_draft_graph


def _payload() -> Dict[str, object]:
    """One interaction and one reaction over every participant kind the schema allows."""

    return {
        "pathway_context": {"name": "ara operon", "organism": "Escherichia coli"},
        "entities": {
            "species": [{"name": "Escherichia coli"}],
            "compounds": [{"name": "L-arabinose", "class": "compound"}],
            "proteins": [{"name": "araC protein", "class": "protein"}],
            "protein_complexes": [{"name": "RNA polymerase holoenzyme",
                                   "class": "protein_complex"}],
            "nucleic_acids": [{"name": "A site", "class": "nucleic_acid"},
                              {"name": "Pc promoter", "class": "nucleic_acid"}],
            "element_collections": [{"name": "ara regulatory elements",
                                     "class": "element_collection"}],
        },
        "processes": {
            "interactions": [
                {"name": "araC protein binds A site", "entity_1": "araC protein",
                 "relationship": "binding", "entity_2": "A site", "evidence": "q"},
                {"name": "araC protein inhibits Pc promoter", "entity_1": "araC protein",
                 "relationship": "inhibition", "entity_2": "Pc promoter", "evidence": "q"},
                {"name": "polymerase binds the elements",
                 "entity_1": "RNA polymerase holoenzyme", "relationship": "binding",
                 "entity_2": "ara regulatory elements", "evidence": "q"},
            ],
            "reactions": [
                {"name": "transcription", "inputs": ["Pc promoter"],
                 "outputs": ["L-arabinose"],
                 "modifiers": [{"entity": "RNA polymerase holoenzyme", "role": "catalyst"}],
                 "evidence": "q"},
            ],
        },
    }


def _degrees(graph) -> Dict[str, int]:
    deg: Dict[str, int] = collections.Counter()
    for edge in graph.edges:
        deg[edge.source] += 1
        deg[edge.target] += 1
    return deg


def test_every_entity_registry_can_resolve_a_participant():
    """The map that CREATES entity nodes and the lookup that RESOLVES participants are
    now one object, so they cannot disagree about which kinds exist."""
    assert set(ENTITY_KIND_MAP.values()) == {
        "protein_complex", "protein", "compound", "nucleic_acid", "element_collection"}


def test_a_dna_participant_is_one_node_of_its_declared_kind():
    graph = build_draft_graph(_payload())
    by_label: Dict[str, List[str]] = collections.defaultdict(list)
    for node in graph.nodes:
        if node.kind != "interaction":
            by_label[node.label].append(node.kind)

    assert by_label["A site"] == ["nucleic_acid"]
    assert by_label["Pc promoter"] == ["nucleic_acid"]
    assert by_label["ara regulatory elements"] == ["element_collection"]
    assert by_label["araC protein"] == ["protein"]
    assert by_label["RNA polymerase holoenzyme"] == ["protein_complex"]
    assert by_label["L-arabinose"] == ["compound"]
    # The defect's signature: the same name present under two kinds.
    for label, kinds in by_label.items():
        assert len(kinds) == 1, f"{label} was minted under {kinds}"


def test_no_declared_entity_is_left_without_edges():
    """Every entity this payload declares takes part in a process, so an orphan here can
    only mean its edges went to a differently-typed twin."""
    graph = build_draft_graph(_payload())
    deg = _degrees(graph)
    orphans = [n.label for n in graph.nodes
               if n.kind != "interaction" and deg.get(n.id, 0) == 0]
    assert orphans == [], orphans


def test_an_unregistered_name_still_defaults_to_compound():
    """Unchanged behaviour: a participant no registry declares is still a compound, and a
    punctuated name is still guessed as a complex. The fix adds registries; it does not
    re-decide the fallbacks."""
    payload = _payload()
    payload["processes"]["interactions"].append(          # type: ignore[index]
        {"name": "mystery", "entity_1": "unlisted thing", "relationship": "binding",
         "entity_2": "MPC1/MPC2", "evidence": "q"})
    graph = build_draft_graph(payload)
    kinds = {n.label: n.kind for n in graph.nodes if n.kind != "interaction"}
    assert kinds["unlisted thing"] == "compound"
    assert kinds["MPC1/MPC2"] == "protein_complex"


def test_registry_precedence_is_unchanged_for_a_doubly_declared_name():
    """A name in two registries resolved as complex, then protein, then compound before
    the fix, and must still: the new registries are appended to that order, never ahead
    of it."""
    payload = _payload()
    payload["entities"]["compounds"].append({"name": "araC protein"})   # type: ignore[index]
    payload["entities"]["protein_complexes"].append({"name": "L-arabinose"})  # type: ignore[index]
    graph = build_draft_graph(payload)
    deg = _degrees(graph)
    # The participant edges must land on the higher-precedence kind.
    attached = {(n.label, n.kind) for n in graph.nodes
                if n.kind != "interaction" and deg.get(n.id, 0) > 0}
    assert ("araC protein", "protein") in attached
    assert ("L-arabinose", "protein_complex") in attached
