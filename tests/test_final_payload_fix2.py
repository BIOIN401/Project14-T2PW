"""Final payload P5 fix round 2: review 2 findings R2-1 (membrane suffix claims nothing), R2-2 (protein-coding -> DNA needs
positive paper evidence), R2-3 (the title has its own budget). Fakes only, no network."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "tests"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from clean_model_fill_fakes import FakeFetcher  # noqa: E402
from test_final_payload_meta import PUBMED, TITLE, _p  # noqa: E402
from test_final_payload_nucleic_lookup import Fake, G, P, TODAY, row, run  # noqa: E402
from test_final_payload_ontology_p5 import _build, _loc_names  # noqa: E402
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.service import final_payload as fp  # noqa: E402

SUFFIX = " (unresolved, no ontology id)"
ECOLI = {"name": "Escherichia coli", "pathwhiz_id": 1, "classification": "Prokaryote"}
HUMAN = {"name": "Homo sapiens", "pathwhiz_id": 2, "classification": "Eukaryote"}
UNLISTED = {"name": "Zzz zzz", "pathwhiz_id": 3}  # no classification, unlisted genus: the organism kind cannot be told


# ---- R2-1: the kept membrane's suffix says "unresolved", never "non-bacterial" ----------------------------------------------
def _membrane_payload(species_rows):
    """Each species has one state in the membrane and one in the cytosol, same tissue: dropping the membrane (no id) would make the
    first state a subset of the second, so the wildcard guard keeps the membrane."""
    locs = [{"name": "outer membrane"}, {"name": "cytosol", "pathwhiz_id": 3}]
    states = [{"name": f"{loc['name']} {sp['name']}", "species": sp["name"], "tissue": "t0", "subcellular_location": loc["name"]}
              for sp in species_rows for loc in locs]
    return {"entities": {"species": species_rows, "subcellular_locations": locs, "tissues": [{"name": "t0", "pathwhiz_id": 10}],
                         "compounds": [{"name": "A", "pathbank_compound_id": 103}]},
            "biological_states": states, "processes": {"reactions": [], "transports": [], "interactions": []}}


def _check_kept_membrane(monkeypatch, species_rows) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build(_membrane_payload(species_rows))
    names = _loc_names(root)
    assert "outer membrane" + SUFFIX in names and "Outer Membrane" not in names and "outer membrane" not in names
    assert not any("non-bacterial" in n for n in names)  # no claim about the organism
    assert b.layout_debug_counts["membranes_kept_unresolved"] == 1 and "nonbacterial_membranes_kept_by_name" not in b.layout_debug_counts
    kept = [o for o in b.vocabulary_outcomes if o["outcome"] == "membrane_kept_unresolved"]
    assert kept == [{"kind": "subcellular-location", "name": "outer membrane", "outcome": "membrane_kept_unresolved",
                     "written": "outer membrane" + SUFFIX}]  # the paper's own name stays on record


def test_r21_undetermined_organism_unlisted_genus(monkeypatch) -> None:
    _check_kept_membrane(monkeypatch, [UNLISTED])


def test_r21_mixed_bacterium_and_unknown(monkeypatch) -> None:
    _check_kept_membrane(monkeypatch, [ECOLI, UNLISTED])


def test_r21_mixed_bacterium_and_eukaryote(monkeypatch) -> None:
    _check_kept_membrane(monkeypatch, [ECOLI, HUMAN])


def test_r21_all_eukaryote(monkeypatch) -> None:
    _check_kept_membrane(monkeypatch, [HUMAN])


def test_r21_all_bacterial_membrane_is_still_the_pathwhiz_row(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    b, root = _build(_membrane_payload([ECOLI]))
    assert "Outer Membrane" in _loc_names(root) and "membranes_kept_unresolved" not in b.layout_debug_counts


def test_r21_flag_off_is_untouched(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    b, root = _build(_membrane_payload([ECOLI, HUMAN]))
    assert "outer membrane" in _loc_names(root) and not b.layout_debug_counts.get("membranes_kept_unresolved") and not b.vocabulary_outcomes


# ---- R2-2: protein-coding -> DNA needs a paper sentence that names the row and has no protein word -----------------------------
_CRD1 = [G(7, "CRD1", "protein-coding", "AT3G56940, CHL27", 3702, "Arabidopsis thaliana")]


def test_r22_no_paper_text_keeps_a_protein_coding_row_unknown() -> None:
    for text in ("", None):
        p = P({"name": "CHL27"}, taxid="3702")
        rep = run(p, Fake(_CRD1), text=text)
        r = row(p)
        assert r["nucleotide_type_status"] == "unknown" and r["nucleotide_type_unknown_lookup"] == "no_protein_context_evidence"
        assert "nucleotide_type" not in r and "nucleotide_type_source" not in r
        assert rep["lookup_codes"] == {"no_protein_context_evidence": 1} and rep["lookup_resolved"] == 0


def test_r22_text_that_never_names_the_row_keeps_it_unknown() -> None:
    p = P({"name": "CHL27"}, taxid="3702")
    rep = run(p, Fake(_CRD1), text="The cyclase subunit binds iron. Other genes were measured.")
    assert row(p)["nucleotide_type_unknown_lookup"] == "no_protein_context_evidence" and rep["lookup_resolved"] == 0


def test_r22_a_sentence_naming_the_row_without_a_protein_word_resolves_dna() -> None:
    p = P({"name": "CHL27"}, taxid="3702")
    rep = run(p, Fake(_CRD1), text="CHL27 was measured in leaves.")
    assert row(p)["nucleotide_type"] == "DNA" and rep["lookup_resolved"] == 1
    p = P({"name": "Foo1", "synonyms": ["CHL27"]}, taxid="3702")  # a synonym counts as naming the row
    run(p, Fake([G(8, "Foo1", "protein-coding", "", 3702)]), text="CHL27 was measured in leaves.")
    assert row(p)["nucleotide_type"] == "DNA"


def test_r22_one_protein_sentence_among_the_naming_ones_blocks_it() -> None:
    p = P({"name": "CHL27"}, taxid="3702")
    run(p, Fake(_CRD1), text="CHL27 was measured in leaves. The CHL27 subunit binds iron.")
    assert row(p)["nucleotide_type_unknown_lookup"] == "protein_context" and "nucleotide_type" not in row(p)


def test_r22_rna_gene_types_do_not_need_paper_text() -> None:
    p = P({"name": "RyhB"})
    rep = run(p, Fake([G(1, "ryhB", "ncRNA")]), text="")
    assert row(p)["nucleotide_type"] == "RNA" and rep["lookup_resolved"] == 1


# ---- R2-3: the title has its own budget; every step reports budget_exhausted -------------------------------------------------
class _SlowFake(FakeFetcher):
    """A fake whose every request except the pubmed title lookup takes 400 s of (fake) wall clock, and which counts the pubmed calls."""

    def __init__(self, routes, clock) -> None:
        super().__init__(routes)
        self.clock = clock

    def get_json(self, url, params=None):
        if not (params or {}).get("db") == "pubmed":
            self.clock.now += 400.0
        return self._find(url, params)

    get_text = get_json


def test_r23_a_spent_id_fill_budget_does_not_stop_the_title(monkeypatch) -> None:
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(fp, "time", SimpleNamespace(monotonic=lambda: clock.now))  # only this module sees the fake clock
    inner = _SlowFake([("esummary", {"db": "pubmed"}, PUBMED)], clock)
    monkeypatch.setattr("t2pw.clean_model.fill.http.UrllibFetcher", lambda throttle=None: inner)
    p = _p()
    rep = fp.maybe_apply_final_id_fill(p, {"likely_organism": "Pseudomonas putida"}, {"final_payload_paper": {"pmid": "555"}},
                                       cache=Cache(None), today=TODAY)
    assert rep["error"] is None
    assert rep["budget_exhausted"] is True  # the id fill spent its budget (its first request took 400 s of the 300 s)
    assert isinstance(rep["nucleic_types"]["budget_exhausted"], bool)
    meta = rep["pathway_meta"]
    assert meta["written"] and meta["title_source"] == "NCBI" and meta["has_reference"] and meta["budget_exhausted"] is False
    assert p[fp.FINAL_PAYLOAD_PATHWAY_FIELD]["title"]["value"] == TITLE


def test_r23_a_spent_budget_is_reported_and_the_title_still_falls_back(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(fp, "time", SimpleNamespace(monotonic=lambda: clock.now))
    assert fp.TITLE_BUDGET_SECONDS < fp.BUDGET_SECONDS
    f = fp._Budget(FakeFetcher([("esummary", {"db": "pubmed"}, PUBMED)]), fp.TITLE_BUDGET_SECONDS)
    assert f.exhausted is False
    clock.now = fp.TITLE_BUDGET_SECONDS + 1
    p = _p()
    rep = fp.apply_final_pathway_meta(p, {"pmid": "555", "title": "fallback"}, fetcher=f, cache=Cache(None), today=TODAY)
    assert f.exhausted is True and rep["budget_exhausted"] is True
    assert rep["title_source"] == "paper metadata"  # a spent budget falls back to the given title, nothing is invented
