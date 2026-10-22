"""ORCH-762 offline replay harness (not a test module).

Drives a committed leg payload through the app's own mapping sequence --
Stage-2 ``map_payload`` (``streamlit_app.py:3487``) -> ``normalize_process_payload``
(``:3619``) -> Stage-6 ``map_payload`` (``:4248``, ``merge_identity_duplicates=True``) --
with every door to the outside shut or replayed:

* UniProt / UniParc / UniProt-taxonomy REST: answered from
  ``tests/fixtures/orch762/uniprot_responses.json``, recorded ONCE live by
  ``docs/pwml_recovery_sprint/evidence/orch762/record_fixtures.py``. A query that is not in
  the file is answered 404 and listed in ``unrecorded`` -- the tests assert that list is
  empty, so no assertion can pass on a silently missing answer.
* any other host (ChEBI, KEGG, HMDB, EuropePMC, NCBI): MOCKED, answered 404 and listed in
  ``mocked``. Compound lookups therefore end ``novel``; NCBI taxonomy backfill is off
  (``T2PW_SPECIES_NCBI=0``) and the fixtures carry the taxonomy id the leg's backfill wrote.
* the EuropePMC alias tier (``lookup_literature_protein_aliases``): MOCKED from a table the
  test passes (the leg's own recorded alias, e.g. ``holo-EntB -> EntB``).
* the model alias tier (``_ai_protein_synonym_lookup``) and the species LLM: never reached
  (counted, returns nothing).
* PathBank: no database. Optionally, rows of the payload's ``proteins`` bucket are answered
  with the resolution the leg recorded for them (``resolutions``), the ORCH-759a technique.

The audit and curator LLM rounds between Stage 3 and Stage 6 are not run: the Stage-6 input
is the normalised payload itself.
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if "t2pw" not in sys.modules and str(SRC) not in sys.path:
    # The fixture recorder imports ``t2pw`` from the BASE tree first, then this module.
    sys.path.insert(0, str(SRC))

FIXTURES = ROOT / "tests" / "fixtures" / "orch762"
RESPONSES = FIXTURES / "uniprot_responses.json"
RECORDED_HOSTS = ("https://rest.uniprot.org/",)


def request_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    return f"{url}?{urlencode(sorted((str(k), str(v)) for k, v in (params or {}).items()))}"


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self._body = body
        self.text = json.dumps(body) if body is not None else ""

    def json(self) -> Any:
        return deepcopy(self._body)


class ReplayHttp:
    """``HttpClient.get`` stand-in serving recorded UniProt answers; every request is logged."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None) -> None:
        self.responses = responses if responses is not None else load_responses()
        self.calls: List[str] = []
        self.unrecorded: List[str] = []
        self.mocked: List[str] = []

    def get(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Any = None) -> FakeResponse:
        key = request_key(url, params)
        self.calls.append(key)
        if not url.startswith(RECORDED_HOSTS):
            self.mocked.append(key)
            return FakeResponse(404, {})
        hit = self.responses.get(key)
        if hit is None:
            self.unrecorded.append(key)
            return FakeResponse(404, {})
        return FakeResponse(int(hit["status"]), hit["json"])

    def uniprot_queries(self) -> List[str]:
        return [key for key in self.calls if key.startswith("https://rest.uniprot.org/uniprotkb/search")]


def write_responses(responses: Dict[str, Any]) -> None:
    """One recorded answer per line, keys sorted: compact, and a re-record diffs by query."""

    lines = [
        f"{json.dumps(key)}: {json.dumps(responses[key], sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"
        for key in sorted(responses)
    ]
    RESPONSES.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8", newline="\n")


def load_responses() -> Dict[str, Any]:
    return json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Doors:
    """The patches every replay runs under, plus the counters a test reads."""

    def __init__(
        self,
        http: Any,
        *,
        literature_aliases: Optional[Dict[str, List[Dict[str, str]]]] = None,
        resolutions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.http = http
        self.literature_aliases = literature_aliases or {}
        self.resolutions = resolutions or {}
        self.ai_calls: List[Any] = []
        self.literature_calls: List[Any] = []
        self.strategy_calls: List[str] = []

    def __enter__(self) -> "Doors":
        from t2pw.mapping import map_ids

        real_strategy = map_ids._map_protein_with_strategy
        doors = self

        def literature(_client: Any, name: str, organism: str = "", *a: Any, **k: Any) -> Dict[str, Any]:
            doors.literature_calls.append((name, organism))
            return {"aliases": deepcopy(doors.literature_aliases.get(name, []))}

        def ai(name: str, organism: str = "", *a: Any, **k: Any) -> List[Any]:
            doors.ai_calls.append((name, organism))
            return []

        def strategy(**kwargs: Any) -> Dict[str, Any]:
            name = str(kwargs.get("name") or "")
            row = kwargs.get("protein_row")
            doors.strategy_calls.append(name)
            recorded = doors.resolutions.get(name)
            if recorded is not None and isinstance(row, dict) and set(row) != {"name"}:
                return deepcopy(recorded)
            return real_strategy(**kwargs)

        self._patches = [
            patch.dict(os.environ, {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}),
            patch.object(map_ids.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)),
            patch.object(map_ids, "_ai_protein_synonym_lookup", side_effect=ai),
            patch.object(map_ids, "lookup_literature_protein_aliases", side_effect=literature),
            patch.object(map_ids.HttpClient, "get", lambda _self, url, params=None, headers=None: doors.http.get(url, params, headers)),
        ]
        if self.resolutions:
            self._patches.append(patch.object(map_ids, "_map_protein_with_strategy", side_effect=strategy))
        for item in self._patches:
            item.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        for item in reversed(self._patches):
            item.__exit__(*exc)


def stage2(payload: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    """``map_payload`` exactly as the app's Stage-2 call makes it."""

    from t2pw.mapping import map_ids

    workdir.mkdir(parents=True, exist_ok=True)
    return map_ids.map_payload(
        deepcopy(payload),
        cache_path=workdir / "id_mapping_cache.json",
        id_source="hybrid",
        use_cache=True,
        allow_complex_wrapper_creation=False,
        allow_structural_cleanup=False,
    )


def stage6(payload: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    """``map_payload`` exactly as the app's Stage-6 call makes it (strict mode)."""

    from t2pw.mapping import map_ids

    workdir.mkdir(parents=True, exist_ok=True)
    return map_ids.map_payload(
        deepcopy(payload),
        cache_path=workdir / "id_mapping_cache_stage6.json",
        id_source="hybrid",
        use_cache=False,
        allow_complex_wrapper_creation=True,
        allow_structural_cleanup=True,
        merge_identity_duplicates=True,
    )


def replay(payload: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    """Stage 2 -> Stage 3 normalisation -> Stage 6, offline. Call inside :class:`Doors`."""

    from t2pw.pipeline.process_normalizer import normalize_process_payload

    first = stage2(payload, workdir)
    normalized, normalization_report = normalize_process_payload(first["payload"])
    final = stage6(normalized, workdir)
    return {"stage2": first, "normalized": normalized, "normalization_report": normalization_report, "stage6": final}


def shipped_uniprot(payload: Dict[str, Any], bucket: str = "proteins") -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in (payload.get("entities") or {}).get(bucket) or []:
        if isinstance(row, dict):
            out[str(row.get("name") or "")] = str((row.get("mapped_ids") or {}).get("uniprot") or "")
    return out


def recorded_resolutions(final_mapped: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """name -> the resolution a leg recorded on each protein row (identity admission undone).

    A withheld accession (``rejected_mapped_ids`` with an admission record) is put back: the
    replay must re-admit exactly what the resolver found, so that admission decides again.
    """

    out: Dict[str, Dict[str, Any]] = {}
    for row in (final_mapped.get("entities") or {}).get("proteins") or []:
        if not isinstance(row, dict):
            continue
        meta = row.get("mapping_meta") if isinstance(row.get("mapping_meta"), dict) else {}
        ids = dict(row.get("mapped_ids") or {})
        if meta.get("identity_admission"):
            ids.update(meta.get("rejected_mapped_ids") or {})
        if not ids.get("uniprot") and not ids.get("drugbank"):
            continue
        out[str(row.get("name") or "")] = {
            "status": "mapped",
            "mapped_ids": ids,
            "pathbank_protein_id": row.get("pathbank_protein_id"),
            "provider": meta.get("provider") or "PathBankDB",
            "source": meta.get("source") or "db",
            "confidence": meta.get("confidence", 1.0),
            "chosen_rule": meta.get("chosen_rule") or "pathbank_protein_id",
            "candidates": deepcopy(meta.get("candidates") or []),
            "resolution": deepcopy(meta.get("resolution") or {"status": "matched"}),
            "identity_verdict": deepcopy(meta.get("identity_verdict")),
            "resolved_name": meta.get("resolved_name"),
        }
    return out


# ── the replay scenarios (shared by the tests and the one-time fixture recorder) ──

#: The leg's own recorded EuropePMC alias for the compound row (ORCH-761 extract § C:
#: ``matched_alias=EntB alias_source=literature_alias``).
HOLO_ENTB_ALIAS = {"holo-EntB": [{"alias": "EntB", "source": "literature_alias"}]}


def scenario_kf147(http: Any, workdir: Path) -> Any:
    """ORCH-756 menaquinone leg, organism "Lactococcus lactis subsp. lactis KF147" (684738)."""

    with Doors(http) as doors:
        return replay(load_fixture("menaquinone_kf147_merged.json"), workdir), doors


def scenario_lactis_species(http: Any, workdir: Path) -> Any:
    """The same leg in the ORCH-755-night shape: organism "Lactococcus lactis" (1358)."""

    with Doors(http) as doors:
        return replay(load_fixture("menaquinone_lactis_species.json"), workdir), doors


def scenario_enterobactin_holo(http: Any, workdir: Path) -> Any:
    """ORCH-755 enterobactin leg (the holo-EntB night); PathBank answers replayed from the leg."""

    with Doors(
        http,
        literature_aliases=HOLO_ENTB_ALIAS,
        resolutions=load_fixture("enterobactin_orch755_resolutions.json"),
    ) as doors:
        return replay(load_fixture("enterobactin_orch755_merged.json"), workdir), doors


def scenario_ecoli_uniprot(http: Any, workdir: Path) -> Any:
    """The enterobactin payload's proteins through the UniProt tier itself (no PathBank)."""

    with Doors(http) as doors:
        return {"stage2": stage2(load_fixture("enterobactin_orch755_merged.json"), workdir)}, doors


SCENARIOS = {
    "kf147": scenario_kf147,
    "lactis_species": scenario_lactis_species,
    "enterobactin_holo": scenario_enterobactin_holo,
    "ecoli_uniprot": scenario_ecoli_uniprot,
}
