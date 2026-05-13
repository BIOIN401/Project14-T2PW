from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

llm_client = types.ModuleType("t2pw.llm.client")
llm_client.chat = lambda *args, **kwargs: "{}"
llm_client.chat_with_tools = lambda *args, **kwargs: "{}"
sys.modules.setdefault("t2pw.llm.client", llm_client)

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _StubSession:
        def __init__(self) -> None:
            self.headers: Dict[str, str] = {}

        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("network disabled in test")

    requests_stub.Session = _StubSession
    requests_stub.HTTPError = RuntimeError
    sys.modules["requests"] = requests_stub

from t2pw.curation import gap_resolver  # noqa: E402


class FakeDb:
    last_error = ""

    def available(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def find_species(self, organism: str, *, taxonomy_id: str | None = None) -> Dict[str, Any]:
        assert organism == "Homo sapiens"
        assert taxonomy_id == "9606"
        return {
            "status": "mapped",
            "chosen_rule": "taxonomy_id",
            "confidence": 0.98,
            "candidates": [
                {
                    "pathbank_species_id": 1,
                    "name": "Homo sapiens",
                    "taxonomy_id": "9606",
                    "confidence": 0.98,
                }
            ],
        }

    def map_protein_row(self, row: Dict[str, Any], species: str) -> Dict[str, Any]:
        assert row["name"] == "MPC1"
        assert species == "Homo sapiens"
        return {
            "status": "mapped",
            "provider": "PathBankDB",
            "source": "db",
            "chosen_rule": "exact_protein_name_species",
            "confidence": 0.96,
            "pathbank_protein_id": 11,
            "mapped_ids": {"uniprot": "Q9Y5U8", "pathbank_protein_id": "11"},
            "candidates": [
                {
                    "pathbank_protein_id": 11,
                    "name": "Mitochondrial pyruvate carrier 1",
                    "gene_name": "MPC1",
                    "uniprot": "Q9Y5U8",
                    "species_id": 1,
                    "mapped_ids": {"uniprot": "Q9Y5U8", "pathbank_protein_id": "11"},
                    "score": 0.96,
                }
            ],
        }


def test_agent_tools_include_db_operator_surface() -> None:
    tool_names = {tool["function"]["name"] for tool in gap_resolver._ENRICHMENT_TOOLS}

    assert {
        "lookup_species",
        "lookup_subcellular_location",
        "lookup_biological_state",
        "lookup_compound_db",
        "lookup_protein_db",
        "lookup_protein_complex_db",
        "lookup_complex_by_component",
        "create_novel_compound",
        "create_novel_protein",
        "create_novel_complex",
        "propose_patch",
    }.issubset(tool_names)


def test_stage3_complex_issue_can_be_fixed_with_agent_tools(monkeypatch: Any) -> None:
    payload = {
        "entities": {
            "species": [{"name": "Homo sapiens", "taxonomy_id": "9606"}],
            "protein_complexes": [{"name": "MPC complex", "components": [{"name": "MPC1"}]}],
            "proteins": [],
            "compounds": [],
        },
        "biological_states": [],
        "element_locations": {"compound_locations": [], "protein_locations": []},
    }
    fake_db = FakeDb()
    seen_tools: List[str] = []

    def fake_chat_with_tools(*, tools: List[Dict[str, Any]], tool_executor: Any, **kwargs: Any) -> str:
        seen_tools.extend(tool["function"]["name"] for tool in tools)

        species_result = tool_executor(
            "lookup_species",
            {"name": "Homo sapiens", "taxonomy_id": "9606"},
        )
        assert "mapped_ids" not in species_result
        species = species_result["candidates"][0]

        protein_result = tool_executor(
            "lookup_protein_db",
            {"name": "MPC1", "species": species["name"]},
        )
        assert "mapped_ids" not in protein_result
        protein = protein_result["candidates"][0]

        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/species/0/pathbank_species_id",
                "value": species["pathbank_species_id"],
                "evidence": "lookup_species returned taxonomy_id 9606 as PathBank species 1.",
                "confidence": 0.98,
            },
        )
        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/protein_complexes/0/species",
                "value": species["name"],
                "evidence": "lookup_species matched Homo sapiens for the complex species.",
                "confidence": 0.98,
            },
        )
        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/protein_complexes/0/species_id",
                "value": species["pathbank_species_id"],
                "evidence": "lookup_species returned PathBank species ID 1.",
                "confidence": 0.98,
            },
        )
        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/protein_complexes/0/components/0/pathbank_protein_id",
                "value": protein["pathbank_protein_id"],
                "evidence": "lookup_protein_db returned MPC1 as PathBank protein 11.",
                "confidence": 0.96,
            },
        )
        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/protein_complexes/0/components/0/mapped_ids",
                "value": protein["mapped_ids"],
                "evidence": "lookup_protein_db returned UniProt Q9Y5U8 for MPC1.",
                "confidence": 0.96,
            },
        )
        tool_executor(
            "propose_patch",
            {
                "op": "add",
                "path": "/entities/protein_complexes/0/components/0/stoichiometry",
                "value": 1,
                "evidence": "Default stoichiometry for a named resolved component.",
                "confidence": 0.9,
            },
        )
        return '{"processed": 2, "patches_proposed": 6}'

    monkeypatch.setattr(gap_resolver.PathBankDbResolver, "from_env", classmethod(lambda cls, cfg=None: fake_db))
    monkeypatch.setattr(gap_resolver, "chat", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(gap_resolver, "chat_with_tools", fake_chat_with_tools)

    resolved, report = gap_resolver.resolve_gaps(
        payload,
        id_source="db",
        db_config={"host": "localhost", "user": "tester"},
        use_llm=True,
        enable_id_resolution=False,
    )

    assert "lookup_species" in seen_tools
    assert "lookup_protein_db" in seen_tools
    assert resolved["entities"]["species"][0]["pathbank_species_id"] == 1
    complex_row = resolved["entities"]["protein_complexes"][0]
    assert complex_row["species"] == "Homo sapiens"
    assert complex_row["species_id"] == 1
    assert complex_row["components"][0]["pathbank_protein_id"] == 11
    assert complex_row["components"][0]["mapped_ids"]["uniprot"] == "Q9Y5U8"
    assert complex_row["components"][0]["stoichiometry"] == 1
    assert report["enrichment"]["patch_application"]["summary"]["accepted_count"] == 6
