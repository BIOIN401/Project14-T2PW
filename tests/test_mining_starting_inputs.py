"""Starting-input parsing, composability and upload determinism."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from t2pw.mining.acquisition import (  # noqa: E402
    PaperReferenceError,
    classify_evidence_type,
    make_paper_starting_input,
    parse_paper_reference,
    record_from_upload,
)
from t2pw.mining.config import MiningConfig, StartingInput  # noqa: E402
from t2pw.mining.models import EVIDENCE_PREPRINT, EVIDENCE_REVIEW  # noqa: E402


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12345678", ("pmid", "12345678")),
        ("PMID:12345678", ("pmid", "12345678")),
        ("pmid: 12345678", ("pmid", "12345678")),
        ("PMC1234567", ("pmcid", "PMC1234567")),
        ("pmc1234567", ("pmcid", "PMC1234567")),
        ("PMCID:PMC1234567", ("pmcid", "PMC1234567")),
        ("10.1038/s41586-024-00001-2", ("doi", "10.1038/s41586-024-00001-2")),
        ("doi:10.1038/abc", ("doi", "10.1038/abc")),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/", ("pmcid", "PMC7654321")),
        ("https://pubmed.ncbi.nlm.nih.gov/33445566/", ("pmid", "33445566")),
        ("https://doi.org/10.1016/j.cell.2024.01.001", ("doi", "10.1016/j.cell.2024.01.001")),
        ("https://example.org/some/paper.html", ("url", "https://example.org/some/paper.html")),
    ],
)
def test_every_accepted_reference_form(raw, expected):
    assert parse_paper_reference(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not a reference", "PMID:abc", "12"])
def test_an_unparseable_reference_is_refused_not_guessed(raw):
    """Guessing at a reference is how a run ends up citing the wrong paper."""
    with pytest.raises(PaperReferenceError):
        parse_paper_reference(raw)


def test_make_paper_starting_input_records_both_raw_and_parsed():
    item = make_paper_starting_input("PMID:12345678")
    assert item.kind == "paper"
    assert item.raw == "PMID:12345678"
    assert item.value == "12345678"
    assert item.id_type == "pmid"


def test_all_input_kinds_compose_in_one_run():
    config = MiningConfig.build(
        target_species="Arabidopsis thaliana",
        topic="jasmonate biosynthesis",
        starting_inputs=[
            StartingInput(kind="topic", raw="jasmonate biosynthesis", value="jasmonate biosynthesis"),
            StartingInput(kind="seed_metabolite", raw="12-OPDA", value="12-OPDA"),
            StartingInput(kind="seed_enzyme", raw="AOS", value="AOS"),
            StartingInput(kind="seed_reaction", raw="13-HPOT -> 12-OPDA", value="13-HPOT -> 12-OPDA"),
            make_paper_starting_input("PMID:12345678"),
            make_paper_starting_input("PMC7654321"),
            StartingInput(kind="pathway_graph", raw="graph.json", value="graph",
                          source_path="/tmp/graph.json"),
        ],
        overrides={"offline": True},
    )
    assert len(config.inputs_of("paper")) == 2
    assert len(config.seed_entities) == 3
    assert len(config.inputs_of("pathway_graph")) == 1
    assert config.topic == "jasmonate biosynthesis"


def test_target_species_is_mandatory():
    with pytest.raises(ValueError, match="target_species is required"):
        MiningConfig(target_species="   ")


def test_supplied_species_text_is_preserved_alongside_the_normalized_form():
    config = MiningConfig(target_species="  A. thaliana  ")
    assert config.target_species == "A. thaliana"
    assert config.target_species_normalized == "arabidopsis thaliana"


def test_uploads_are_content_addressed_and_deterministic():
    data = b"Results. A is converted to B in Arabidopsis thaliana."
    first = record_from_upload(data=data, file_name="paper.txt")
    second = record_from_upload(data=data, file_name="a-different-name.txt")
    assert first.record.paper_id == second.record.paper_id
    assert first.record.file_sha256 == second.record.file_sha256
    assert first.record.paper_id.startswith("FILE:")
    assert first.record.has_source_identifier is True
    assert first.text == data.decode()

    other = record_from_upload(data=data + b" Extra.", file_name="paper.txt")
    assert other.record.paper_id != first.record.paper_id


def test_a_paper_with_no_resolvable_identifier_fails_the_provenance_requirement():
    from t2pw.mining.models import PaperRecord

    assert PaperRecord(paper_id="TITLE:abc", title="Only a title").has_source_identifier is False
    assert PaperRecord(paper_id="PMID:1", pmid="1").has_source_identifier is True


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"article_type": "Review"}, EVIDENCE_REVIEW),
        ({"title": "A review of jasmonate signalling"}, EVIDENCE_REVIEW),
        ({"source": "biorxiv"}, EVIDENCE_PREPRINT),
        ({"url": "https://www.biorxiv.org/content/1"}, EVIDENCE_PREPRINT),
        ({"article_type": "journal article"}, "primary"),
        ({"title": "LOX2 converts alpha-linolenic acid"}, "unknown"),
    ],
)
def test_evidence_type_is_labelled_where_determinable_and_honest_otherwise(kwargs, expected):
    assert classify_evidence_type(**kwargs) == expected


def test_budget_and_model_overrides_flow_through_config(monkeypatch):
    monkeypatch.setenv("PATHWAY_MINING_MAX_ROUNDS", "3")
    monkeypatch.setenv("PATHWAY_MINING_MODEL", "vendor/some-model")
    config = MiningConfig.build(target_species="Homo sapiens")
    assert config.budgets.max_rounds == 3
    assert config.model.model == "vendor/some-model"

    explicit = MiningConfig.build(
        target_species="Homo sapiens",
        overrides={"budgets": {"max_rounds": 1}, "model": {"model": "vendor/other"}},
    )
    assert explicit.budgets.max_rounds == 1
    assert explicit.model.model == "vendor/other"


def test_default_model_is_the_configured_flash_model(monkeypatch):
    monkeypatch.delenv("PATHWAY_MINING_MODEL", raising=False)
    from t2pw.mining.config import DEFAULT_MINING_MODEL, MiningModelConfig

    assert DEFAULT_MINING_MODEL == "deepseek/deepseek-v4-flash"
    assert MiningModelConfig.from_env().model == DEFAULT_MINING_MODEL


def test_config_serializes_everything_the_manifest_needs():
    config = MiningConfig.build(
        target_species="Arabidopsis thaliana", topic="x", overrides={"offline": True}
    )
    payload = config.to_dict()
    for key in (
        "artifact_schema_version", "target_species", "target_species_normalized", "topic",
        "starting_inputs", "budgets", "model", "acquisition_sources", "vector_backend",
        "offline", "runs_dir", "enabled_via",
    ):
        assert key in payload
    assert "api_key" not in repr(payload).lower()
