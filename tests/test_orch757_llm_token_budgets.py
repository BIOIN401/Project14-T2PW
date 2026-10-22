"""ORCH-757 (D-105 § 2): every LLM call requests the configured, fallback-aware budget.

WHY. ``ORCH-755-RESULT.md`` § 4: a reasoning model (``deepseek/deepseek-v4-flash``) spent
the small hardcoded budgets (300 to 12,000 tokens) on reasoning and returned empty
``finish_reason=length`` completions, so no strict PWML was written. D-105 lifts every
budget to one knob, ``LLM_MAX_TOKENS`` (code default 128,000), keeps the three per-stage
variables, and clamps each request to the smallest output cap among the models the call
can fail over to (the Stage-1 ladder's ``OPENROUTER_EXTRACTION_FALLBACK_MODEL``).

G9, BEHAVIOURAL. Every stage is driven through its real entry point with the provider
transport faked (``t2pw.llm.client._client``), so each number asserted is the
``max_tokens`` that would go on the wire. On base ``a43a2b12`` every ``test_g9_*`` case
fails: the stages sent their literals, the truncation retry was capped at 24,000, and the
extraction request was not clamped to the fallback's 65,536. The protected
``streamlit_app.py`` call sites are driven through the real orchestrator source by
``tests/test_streamlit_stage2_orchestration.py``'s harness; there the number asserted is
the argument the call site hands the stage (the stage's own wire test is above it).

PRESERVATION, NOT G9. ``test_requests_are_byte_identical_except_the_budget`` pins every
request minus ``max_tokens``/``timeout`` (messages, model, temperature, response format,
tools) to sha256 values measured at base. D-104 amendment 2: LLM inputs stay
byte-identical; only the budget moves. It passes at base by construction.
``test_per_stage_variables_still_win`` is preservation too (those variables worked at base).
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import t2pw.llm.client as client  # noqa: E402

DEEPSEEK = "deepseek/deepseek-v4-flash"  # max completion 943,717 (OpenRouter, 2026-10-21)
GEMINI = "google/gemini-3.8-flash"  # max completion 65,536 (OpenRouter, 2026-10-21)
CONFIGURED = 50_000
CODE_DEFAULT = 128_000

_MODEL_VARS = (
    "OPENROUTER_MODEL",
    "OPENROUTER_PREPROCESSOR_MODEL",
    "OPENROUTER_EXTRACTION_MODEL",
    "OPENROUTER_EXTRACTION_FALLBACK_MODEL",
    "OPENROUTER_INFERENCE_MODEL",
    "OPENROUTER_AUDIT_MODEL",
    "OPENROUTER_CURATOR_MODEL",
    "OPENROUTER_GAP_MODEL",
    "OPENROUTER_OVERWATCH_MODEL",
    "OPENROUTER_FINAL_COMPLETENESS_MODEL",
    "OPENROUTER_RAG_EXTRACT_MODEL",
)
_BUDGET_VARS = (
    "LLM_MAX_TOKENS",
    "OPENROUTER_PREPROCESSOR_MAX_TOKENS",
    "OPENROUTER_EXTRACTION_MAX_TOKENS",
    "OPENROUTER_INFERENCE_MAX_TOKENS",
)

TEXT = (
    "Glutamate-cysteine ligase joins L-glutamate and L-cysteine into gamma-glutamylcysteine "
    "using ATP. Glutathione synthetase adds glycine to form glutathione."
)


def _payload() -> Dict[str, Any]:
    return {
        "metadata": {"pathway_name": "Glutathione synthesis", "organism": "Homo sapiens"},
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "compounds": [{"name": "L-glutamate"}, {"name": "gamma-glutamylcysteine"}],
            "proteins": [{"name": "glutamate-cysteine ligase"}],
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "glutamate cysteine ligation",
                    "inputs": ["L-glutamate"],
                    "outputs": ["gamma-glutamylcysteine"],
                    "enzymes": [{"entity": "glutamate-cysteine ligase", "entity_type": "protein"}],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


# --- the faked transport -------------------------------------------------------------


class _Wire:
    """``_client`` stand-in: records every request, answers with ``reply``."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.reply: Any = "{}"
        self.finish_reason = "stop"
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = self.reply(kwargs) if callable(self.reply) else self.reply
        message = types.SimpleNamespace(content=content, tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return types.SimpleNamespace(choices=[choice], usage=None)

    def budgets(self) -> List[Any]:
        return [call.get("max_tokens") for call in self.calls]


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    return install_wire(monkeypatch)


def install_wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """OpenRouter, every stage on deepseek, LLM_MAX_TOKENS pinned, no ``.env`` leakage.

    A plain function so ``evidence/orch757_golden_capture.py`` measures GOLDEN with it."""

    monkeypatch.setattr(client, "PROVIDER", "openrouter")
    for name in _MODEL_VARS + _BUDGET_VARS + ("T2PW_OFFLINE_CURATOR",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", DEEPSEEK)
    monkeypatch.setenv("LLM_MAX_TOKENS", str(CONFIGURED))
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    fake = _Wire()
    monkeypatch.setattr(client, "_client", fake)
    import t2pw.stoich.agent as stoich_agent  # binds _client/_model at import

    monkeypatch.setattr(stoich_agent, "_client", fake)
    monkeypatch.setattr(stoich_agent, "_model", DEEPSEEK)
    return fake


def _quiet(call: Callable[[], Any]) -> None:
    """Run ``call``; a stage that raises after its request was recorded is fine here."""

    try:
        call()
    except Exception:  # noqa: BLE001 - the request is what is under test
        pass


# --- one driver per LLM call site ----------------------------------------------------

STAGES: Dict[str, Callable[[Path, _Wire], None]] = {}


def stage(name: str) -> Callable[[Callable[[Path, _Wire], None]], Callable[[Path, _Wire], None]]:
    def register(fn: Callable[[Path, _Wire], None]) -> Callable[[Path, _Wire], None]:
        STAGES[name] = fn
        return fn

    return register


@stage("client.chat")
def _client_chat(_tmp: Path, _wire: _Wire) -> None:
    client.chat([{"role": "user", "content": "hello"}])


@stage("client.chat_with_tools")
def _client_tools(_tmp: Path, wire: _Wire) -> None:
    wire.reply = "done"
    client.chat_with_tools([{"role": "user", "content": "hello"}], tools=[], tool_executor=lambda *_a: None)


@stage("preprocessor")
def _preprocessor(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.preprocessor import preprocess

    preprocess(TEXT)


@stage("extraction")
def _extraction(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.pipeline import run_extraction_pipeline

    _quiet(lambda: run_extraction_pipeline(TEXT))


@stage("inference")
def _inference(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.pipeline import run_inference_pipeline

    run_inference_pipeline(TEXT, _payload())


@stage("inference.chunking")
def _inference_chunking(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.pipeline import run_stage_two_with_chunking

    run_stage_two_with_chunking(TEXT, _payload(), enable_chunking=False)


@stage("audit")
def _audit(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.curation.audit_json_llm import audit_payload

    audit_payload(_payload(), use_llm=True)


@stage("completeness")
def _completeness(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.curation.completeness_audit import run_final_completeness_audit

    run_final_completeness_audit(
        source_text=TEXT,
        locked_manifest=[{"reaction_name": "glutamate cysteine ligation"}],
        final_pathway_json=_payload(),
    )


@stage("gap.species")
def _gap_species(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.curation.gap_resolver import infer_entity_species

    infer_entity_species(_payload(), entity_type="protein", entity_name="glutamate-cysteine ligase")


@stage("curator")
def _curator(tmp: Path, wire: _Wire) -> None:
    from t2pw.curation.pathway_curator import run_pathway_curator

    wire.reply = "done"
    src = tmp / "audited.json"
    src.write_text(json.dumps(_payload()), encoding="utf-8")
    run_pathway_curator(src, tmp / "curated.json", tmp / "curator_report.json",
                        reaction_summary="R1: L-glutamate -> gamma-glutamylcysteine",
                        pathway_name="Glutathione synthesis", organism="Homo sapiens")


@stage("interactive_curator")
def _interactive(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.curation.interactive_curator import run_interactive_curator_round

    run_interactive_curator_round(_payload(), b"", "rename the enzyme")


@stage("overwatch")
def _overwatch(tmp: Path, _wire: _Wire) -> None:
    from t2pw.sbml.overwatch import run_sbml_overwatch

    mapped, sbml, report = tmp / "mapped.json", tmp / "pathway.sbml", tmp / "validation.json"
    mapped.write_text(json.dumps(_payload()), encoding="utf-8")
    sbml.write_text("<sbml/>", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    run_sbml_overwatch(mapped, sbml, report, tmp / "overwatch.json")


@stage("rag.prose_extraction")
def _rag(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.rag.extract import extract_reactions_from_text

    extract_reactions_from_text(TEXT)


@stage("map_ids.alias")
def _alias(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.mapping.map_ids import _ai_protein_synonym_lookup

    _ai_protein_synonym_lookup("glutamate-cysteine ligase", "Homo sapiens")


@stage("map_ids.species")
def _map_species(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.mapping.map_ids import _infer_species_with_gap_resolver

    _infer_species_with_gap_resolver(_payload(), entity_type="protein",
                                     name="glutamate-cysteine ligase", use_llm=True)


@stage("focused_repair")
def _focused(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.focused_repair import _call_repair_llm

    _call_repair_llm("You repair pathways.", "Repair this.")


@stage("localized_repair.json")
def _repair_json(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.localized_repair import repair_json_text

    repair_json_text('{"entities": {"compounds": [', "Expecting value: line 1 column 29")


@stage("localized_repair.rows")
def _repair_rows(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.pipeline.localized_repair import repair_invalid_rows

    repair_invalid_rows(
        _payload(),
        [{"pointer": "/processes/reactions/0", "code": "missing_inputs", "message": "no inputs"}],
    )


@stage("stoich.classifier")
def _classifier(_tmp: Path, wire: _Wire) -> None:
    from t2pw.stoich.classifier import classify_reaction

    wire.reply = '{"class": "unknown_or_mechanistic", "confidence": "low", "reasoning": "r"}'
    classify_reaction({"name": "zzqx transformation", "inputs": [], "outputs": []})


@stage("stoich.agent")
def _stoich_agent(_tmp: Path, wire: _Wire) -> None:
    from t2pw.stoich.agent import run_stoich_agent

    wire.reply = json.dumps({
        "class": "unknown_or_mechanistic", "confidence": "low", "verdict": "add",
        "add_inputs": ["water"], "add_outputs": [], "reasoning": "r", "sources_used": ["kegg"],
    })
    _quiet(lambda: run_stoich_agent({"processes": {"reactions": [
        {"name": "zzqx transformation", "inputs": ["A"], "outputs": ["B"]}]}}))


@stage("pathwhiz_converter.llm_plan")
def _llm_plan(_tmp: Path, wire: _Wire) -> None:
    from t2pw.tools.pathwhiz_converter.llm_plan import request_patch_plan

    wire.reply = '{"objective": "tidy", "ops": []}'
    request_patch_plan(llm_client=client, input_model_summary={"species": []},
                       example_summaries=[], allow_add_reaction=False)


@stage("extraction.demo")
def _demo(_tmp: Path, _wire: _Wire) -> None:
    from t2pw.extraction.extract import run_demo

    run_demo()


@stage("legacy_sbml_pipeline")
def _legacy(tmp: Path, _wire: _Wire) -> None:
    """``pipeline.run_legacy_sbml_pipeline``: real audit and overwatch, offline middle."""

    import t2pw.curation.apply_audit_patch as apply_mod
    import t2pw.mapping.map_ids as map_mod
    import t2pw.sbml.json_to_sbml as sbml_mod
    from t2pw.pipeline.pipeline import run_legacy_sbml_pipeline

    def _apply(input_path: Path, _patch: Path, output_path: Path, **_kw: Any) -> Dict[str, Any]:
        Path(output_path).write_text(Path(input_path).read_text(encoding="utf-8"), encoding="utf-8")
        return {}

    def _map(input_path: Path, output_path: Path, report_path: Path, **_kw: Any) -> Dict[str, Any]:
        Path(output_path).write_text(Path(input_path).read_text(encoding="utf-8"), encoding="utf-8")
        Path(report_path).write_text("{}", encoding="utf-8")
        return {}

    def _sbml(_mapped: Path, sbml_file: Path, report_json: Path, report_txt: Path, **_kw: Any) -> Dict[str, Any]:
        Path(sbml_file).write_text("<sbml/>", encoding="utf-8")
        Path(report_json).write_text("{}", encoding="utf-8")
        Path(report_txt).write_text("", encoding="utf-8")
        return {}

    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(apply_mod, "run_apply", _apply)
        patch.setattr(map_mod, "run_mapping", _map)
        patch.setattr(sbml_mod, "build_sbml", _sbml)
        src = tmp / "final.json"
        src.write_text(json.dumps(_payload()), encoding="utf-8")
        run_legacy_sbml_pipeline(src, out_dir=tmp / "legacy", mapping_cache=tmp / "cache.json")
    finally:
        patch.undo()


def _drive(name: str, tmp_path: Path, wire: _Wire) -> _Wire:
    wire.calls.clear()
    wire.reply = "{}"
    work = tmp_path / name.replace(".", "_")
    work.mkdir(parents=True, exist_ok=True)
    STAGES[name](work, wire)
    assert wire.calls, f"{name}: the stage issued no request, so it proves nothing"
    return wire


# --- G9: the configured budget reaches the wire --------------------------------------


@pytest.mark.parametrize("name", sorted(STAGES))
def test_g9_every_stage_requests_llm_max_tokens(name: str, tmp_path: Path, wire: _Wire) -> None:
    """``LLM_MAX_TOKENS=50000``: every request of every stage asks for exactly 50,000."""

    assert set(_drive(name, tmp_path, wire).budgets()) == {CONFIGURED}


@pytest.mark.parametrize("name", sorted(STAGES))
def test_g9_with_nothing_configured_every_stage_requests_128000(
    name: str, tmp_path: Path, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No budget variable at all: the code default is 128,000 (D-105 § 1)."""

    monkeypatch.delenv("LLM_MAX_TOKENS")
    assert set(_drive(name, tmp_path, wire).budgets()) == {CODE_DEFAULT}


def test_g9_an_unusable_budget_falls_back_to_the_code_default(
    tmp_path: Path, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    for bad in ("lots", "0", "-5"):
        monkeypatch.setenv("LLM_MAX_TOKENS", bad)
        assert set(_drive("preprocessor", tmp_path, wire).budgets()) == {CODE_DEFAULT}, bad
        assert set(_drive("audit", tmp_path, wire).budgets()) == {CODE_DEFAULT}, bad


def test_g9_the_stage_variables_feed_their_stage_defaults(
    tmp_path: Path, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction and inference read their own variable when the caller passes nothing."""

    monkeypatch.setenv("OPENROUTER_EXTRACTION_MAX_TOKENS", "40000")
    monkeypatch.setenv("OPENROUTER_INFERENCE_MAX_TOKENS", "60000")
    assert set(_drive("extraction", tmp_path, wire).budgets()) == {40_000}
    assert set(_drive("inference", tmp_path, wire).budgets()) == {60_000}
    assert set(_drive("inference.chunking", tmp_path, wire).budgets()) == {60_000}


def test_g9_extraction_is_clamped_to_its_fallback_models_cap(
    tmp_path: Path, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ladder can fail over to gemini (65,536), so the deepseek request is clamped too."""

    from t2pw.pipeline.pipeline import run_extraction_pipeline

    monkeypatch.setenv("OPENROUTER_EXTRACTION_MODEL", DEEPSEEK)
    monkeypatch.setenv("OPENROUTER_EXTRACTION_FALLBACK_MODEL", GEMINI)
    monkeypatch.setenv("OPENROUTER_EXTRACTION_MAX_TOKENS", "128000")
    assert set(_drive("extraction", tmp_path, wire).budgets()) == {65_536}

    # The UI passes its number explicitly (``streamlit_app.py`` Stage 1 max tokens).
    wire.calls.clear()
    _quiet(lambda: run_extraction_pipeline(TEXT, max_tokens=128_000))
    assert set(wire.budgets()) == {65_536}
    assert {call["model"] for call in wire.calls} == {DEEPSEEK}

    # The row repair and the JSON repair on the extraction model are clamped the same way.
    assert set(_drive("localized_repair.rows", tmp_path, wire).budgets()) == {CONFIGURED}
    monkeypatch.setenv("LLM_MAX_TOKENS", "128000")
    assert set(_drive("localized_repair.rows", tmp_path, wire).budgets()) == {65_536}
    assert set(_drive("localized_repair.json", tmp_path, wire).budgets()) == {65_536}

    # Rung 3 itself, on the fallback model.
    wire.calls.clear()
    client.chat_detailed([{"role": "user", "content": "x"}], max_tokens=128_000,
                         model_env_var="OPENROUTER_EXTRACTION_FALLBACK_MODEL")
    assert wire.calls[-1]["model"] == GEMINI and wire.calls[-1]["max_tokens"] == 65_536


def test_g9_an_unlisted_model_gets_the_smallest_listed_cap(
    tmp_path: Path, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The open-question rule (ORCH-757-RESULT): unknown model -> 65,536, the tightest known cap."""

    monkeypatch.setenv("OPENROUTER_MODEL", "example/unlisted-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "128000")
    assert set(_drive("audit", tmp_path, wire).budgets()) == {65_536}
    monkeypatch.setenv("LLM_MAX_TOKENS", "20000")
    assert set(_drive("audit", tmp_path, wire).budgets()) == {20_000}


def test_g9_the_truncation_retry_is_not_capped_below_the_budget(
    tmp_path: Path, wire: _Wire
) -> None:
    """Stage 2's length retry escalated to ``min(24000, 1.5x)``: with the UI's 64,000 it
    SHRANK to 24,000. Now it grows 1.5x, bounded only by the model's cap.

    The reply is ORCH-755's measured shape: an EMPTY completion with
    ``finish_reason=length``. An empty reply gets no localized repair, so the parse error
    ("Expecting value") is the last error and the truncation branch is taken."""

    from t2pw.pipeline.pipeline import PipelineFailure, run_stage_two_with_chunking

    wire.reply, wire.finish_reason = "", "length"
    with pytest.raises(PipelineFailure):
        run_stage_two_with_chunking(TEXT, _payload(), enable_chunking=False, max_tokens=64_000)
    budgets = wire.budgets()
    assert budgets[0] == 64_000
    assert budgets[-1] == 96_000, budgets
    assert 24_000 not in budgets


# --- the protected streamlit_app.py call sites, through the real orchestrator ----------


def _app_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, clock: Any = None):
    import test_streamlit_stage2_orchestration as orchestration

    for name in _BUDGET_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_MAX_TOKENS", str(CONFIGURED))
    function, _observed = orchestration._orchestration_harness(tmp_path)
    namespace = function.__globals__
    seen: Dict[str, List[Dict[str, Any]]] = {}

    def spy(key: str) -> None:
        real = namespace[key]

        def recorder(*args: Any, **kwargs: Any) -> Any:
            seen.setdefault(key, []).append(kwargs)
            return real(*args, **kwargs)

        namespace[key] = recorder

    for key in ("run_audit", "run_gap_resolution", "run_pathway_curator", "run_sbml_overwatch"):
        spy(key)
    if clock is not None:
        namespace["time"] = clock
    kwargs = orchestration._invoke_kwargs(tmp_path)
    kwargs.update({
        "use_llm_audit": True,
        "audit_candidate_count": 2,
        "audit_timeout_seconds": 600,  # under 120 s left, the app drops to 1 candidate
        "use_gap_resolver": True,
        "use_llm_gap_resolver": True,
        "build_legacy_sbml": True,
        "use_sbml_overwatch": True,
    })
    return function(orchestration._paper_payload(), **kwargs), seen


def test_g9_app_audit_gap_curator_and_overwatch_sites_read_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``streamlit_app.py`` audit (base ``min(8000, 3600+700r)`` / ``min(10000, +offset)``,
    deterministic scoring 1200), gap resolver (``min(1400, 700+120r)``), curator (2000),
    overwatch (3000): each now hands its stage the configured budget, and the round record
    says so."""

    result, seen = _app_harness(tmp_path, monkeypatch)
    audits = seen["run_audit"]
    llm_audits = [k["llm_max_tokens"] for k in audits if k["use_llm"]]
    scoring = [k["llm_max_tokens"] for k in audits if not k["use_llm"]]
    assert llm_audits == [CONFIGURED, CONFIGURED]
    assert scoring == [CONFIGURED, CONFIGURED]
    assert [k["llm_max_tokens"] for k in seen["run_gap_resolution"]] == [CONFIGURED]
    assert [k["llm_max_tokens"] for k in seen["run_pathway_curator"]] == [CONFIGURED]
    assert [k["llm_max_tokens"] for k in seen["run_sbml_overwatch"]] == [CONFIGURED]
    assert result["audit_iterations"][0]["max_tokens"] == CONFIGURED


def test_g9_app_fallback_candidate_sites_read_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timed-out-round fallback (two deterministic audits at 1200, recorded as 1200)."""

    ticks = iter([0.0, 0.0])  # loop start and the round check; every later read is late
    clock = types.SimpleNamespace(time=lambda: next(ticks, 10_000.0))
    result, seen = _app_harness(tmp_path, monkeypatch, clock=clock)
    audits = seen["run_audit"]
    assert [k["use_llm"] for k in audits] == [False, False], "the fallback path was not reached"
    assert [k["llm_max_tokens"] for k in audits] == [CONFIGURED, CONFIGURED]
    assert result["audit_iterations"][0]["max_tokens"] == CONFIGURED


# --- preservation (not G9) -------------------------------------------------------------


def test_per_stage_variables_still_win(
    tmp_path: Path, wire: _Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worked at base and must keep working: the preprocessor's own variable, and an
    explicit ``max_tokens`` from a caller."""

    monkeypatch.setenv("OPENROUTER_PREPROCESSOR_MAX_TOKENS", "70000")
    assert set(_drive("preprocessor", tmp_path, wire).budgets()) == {70_000}
    wire.calls.clear()
    client.chat([{"role": "user", "content": "x"}], max_tokens=321)
    assert wire.budgets() == [321]


def _request_digest(call: Dict[str, Any]) -> str:
    kept = {k: v for k, v in call.items() if k not in ("max_tokens", "timeout")}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, default=repr).encode("utf-8")).hexdigest()


#: sha256 of every request each stage sends, minus ``max_tokens`` and ``timeout``,
#: measured on base ``a43a2b12`` with this file's fixture (ORCH-757 evidence).
GOLDEN: Dict[str, List[str]] = {
    "audit": ["5c0b3479048985d9836e5b153ba8aba178065e17ed62a3b3ee4f285b0f8011ca"],
    "client.chat": ["0ac5ec9b65792596e928666ecb34227ac3a92a0d32cc5b5a4cfd28bc99915380"],
    "client.chat_with_tools": ["0ac5ec9b65792596e928666ecb34227ac3a92a0d32cc5b5a4cfd28bc99915380"],
    "completeness": ["3792147b463e772c899b298ff0b160b8a125e89e128c3ed4bb6da615fdb4ec28"],
    "curator": ["21a36f90117d565e833d6f752b5ec37d2f60b52b920c6e6ced0448f0a0115c89"],
    "extraction": ["b4ec1b8f44082ba2527439cac90c0766767e3a6778d0a1cebb7a89fd46bc76e7", "6ae567d4d71134a1b9b37f8f41dc08fed463bb0f212f95dcb6a9aa61f104291b"],
    "extraction.demo": ["b103910a190e9e16b93bd0bdcab55cbda15b5f267b6ac9f73cf09050a7ff422b"],
    "focused_repair": ["35ec27090442c91dccf3b3f1ef04a62ef8bd323181bb8900333daffd6b8f1dfe"],
    "gap.species": ["2b12c03909774d312212f3721a97eb0600f463900b29c0945976a01139d0c9cd"],
    "inference": ["27a8ad55093ffbf9288e1ffed590162484f207d23929880259c1f4ec4cae2537"],
    "inference.chunking": ["f76555d7661a38c09a25e2cd4534d8b76d239877113f236dd75bf98e8b510f01"],
    "interactive_curator": ["7eb7db6e90cf549a9926ea7462e005da6ea2daa3395931620d804941cb21f67e", "49b3a17edc6d0b616377a335d84be9eddf9e6922c47bf01c1277f056bc52fb1d"],
    "legacy_sbml_pipeline": ["5c0b3479048985d9836e5b153ba8aba178065e17ed62a3b3ee4f285b0f8011ca", "d634c2b60df9a71b4603c96f221f51c4252bc4ef99cdabaf07729080de80f119"],
    "localized_repair.json": ["f02c8f6d3991ada58d598f224ae99dbcf3de50dc61bfc362fd251e472ad8989f"],
    "localized_repair.rows": ["a21a049f8f712e23387c0f0c472f84740457ce7478ae09246e9d9c6d7fa365ea"],
    "map_ids.alias": ["dc7e6674e257237a13871ad820d95a7480ac5d5b18f145dbd880bbd0047de71d"],
    "map_ids.species": ["2b12c03909774d312212f3721a97eb0600f463900b29c0945976a01139d0c9cd"],
    "overwatch": ["d634c2b60df9a71b4603c96f221f51c4252bc4ef99cdabaf07729080de80f119"],
    "pathwhiz_converter.llm_plan": ["606ccd87c1f29970bd279e6e5c64d3571f5499f2d68fbec13d6620321efc39df"],
    "preprocessor": ["920f56a9507cf9dda0fc3c5de310f2ffdecb9353a9cb1ba2805398a9d59e19b0"],
    "rag.prose_extraction": ["735ffa29cd7671311ea2c372b4d1163861aa8e370e7bc9c0e0f3e2bebde2939d"],
    "stoich.agent": ["7f385f1d4df3fa7470d5fdb2c918db186d1b082c75b09b0b3ba05a31745019db", "bfcb6934455119ef77af3f31bd8b481b8b67b6843f637d3bca2f44561d2ced51", "18ce5423bd23a9716dc59e71c517f63d7d74dc92b170dcafa9702952b4d9fe82"],
    "stoich.classifier": ["6fe6b6aaf71bf6b65ea08a29dc9d972d259b81ce64bbaa78cec7cee0035767ab"],
}


def request_digests(tmp_path: Path, wire: _Wire) -> Dict[str, List[str]]:
    return {name: [_request_digest(c) for c in _drive(name, tmp_path, wire).calls]
            for name in sorted(STAGES)}


def test_requests_are_byte_identical_except_the_budget(tmp_path: Path, wire: _Wire) -> None:
    assert request_digests(tmp_path, wire) == GOLDEN
