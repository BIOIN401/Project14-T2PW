from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

llm_client = types.ModuleType("t2pw.llm.client")
llm_client.chat = lambda *args, **kwargs: "{}"
llm_client.chat_with_tools = lambda *args, **kwargs: "{}"
sys.modules.setdefault("t2pw.llm.client", llm_client)

from t2pw.curation.gap_resolver import _collect_stage3_issues  # noqa: E402


def test_biological_state_without_species_fails_with_species_issue() -> None:
    payload = {
        "entities": {"compounds": [], "proteins": [], "protein_complexes": [], "species": []},
        "biological_states": [{"name": "cytosol", "subcellular_location": "cytosol"}],
        "element_locations": {},
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }

    issues = _collect_stage3_issues(payload, max_items=20)
    biological_state_issues = [issue for issue in issues if issue.get("entity_type") == "biological_state"]

    assert len(biological_state_issues) == 1
    assert biological_state_issues[0]["needs_species"] is True
    assert "biological_state_missing_species" in biological_state_issues[0]["reasons"]
