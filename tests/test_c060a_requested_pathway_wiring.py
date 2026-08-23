"""C-060a -- the requested-pathway wiring for the C-060 entity-admission gate.

C-060 gave ``merge_additions`` a ``pathway_context`` keyword and proved the
advisory phase against it. **No production caller ever passed one.** All three
are positional two-argument calls, so C-060's F4 protection -- the currency a
removed assay composite strands (R-003 4.F3/F4.6) -- was unreachable outside the
test suite.

``test_g9_*`` -- **G9 behavioural base proofs**, driving the REAL production call
    shape: ``run_stage_two_with_feedback_loop`` for the pipeline site, and for
    the two app sites the actual source bytes of the statement, lifted out of
    ``streamlit_app.py`` by AST and executed. Nothing here supplies an argument
    the production source does not. At base ``f2f7599`` all three fail with an
    ``AssertionError`` naming currency rows that survived -- never on an import,
    never on a missing symbol.

``test_new_acceptance_*`` -- **explicitly labelled new acceptance tests**: the
    Stage-0 -> ``PathwayContext`` factory and the ``_unlocatable`` typographic
    hardening. **The hardening is prophylactic** -- ``_unlocatable`` keys off
    ``seed_text``, deliberately NOT wired, so that code is inert in production
    and can carry no base-SHA failure. None is claimed or fabricated below.

Pure, offline, deterministic: no LLM, no network, no database, no Streamlit.
"""

from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline import entity_admission as ea  # noqa: E402
from t2pw.pipeline.cofactor_policy import PathwayContext, classify_entity  # noqa: E402
from t2pw.pipeline.pipeline import merge_additions  # noqa: E402

APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"
PIPELINE_PATH = SRC / "t2pw" / "pipeline" / "pipeline.py"
#: C-060a's line-range ownership in the app module (WIRING-PACK header): a
#: concurrent card owns <= 4530, so every statement lifted here is >= this line.
OWNED_FLOOR = 4640
GATE_MODULE = "t2pw.pipeline.entity_admission"


# --------------------------------- lifting the REAL production statements ---
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _merge_calls(tree: ast.Module) -> list:
    """Every ``<name> = merge_additions(...)`` statement, ordered by line."""
    return sorted(
        (
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "merge_additions"
        ),
        key=lambda node: node.lineno,
    )


def _app_sites() -> list:
    sites = [n for n in _merge_calls(_tree(APP_PATH)) if n.lineno >= OWNED_FLOOR]
    assert len(sites) == 2, f"expected two owned app merge sites: {[s.lineno for s in sites]}"
    return sites


def _production_sites() -> list:
    """All three: the two app sites plus ``run_stage_two_with_feedback_loop``."""
    sites = _app_sites() + [n for n in _merge_calls(_tree(PIPELINE_PATH)) if n.lineno > 700]
    assert len(sites) == 3, [n.lineno for n in sites]
    return sites


def _exec_nodes(namespace: dict, *nodes: ast.stmt) -> dict:
    module = ast.Module(body=list(nodes), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(APP_PATH), "exec"), namespace)  # noqa: S102
    return namespace


def _run_app_site(index: int, namespace: dict) -> dict:
    """Execute app merge site *index* from the source bytes, in *namespace*.

    The body is the gate-module imports the app performs between the previous
    merge site (or the ownership floor) and this one, then the merge statement
    itself. At base there are no such imports and the statement is the bare
    two-positional call -- this helper supplies no argument the production source
    does not, which is what makes the base failure a real one.
    """
    sites = _app_sites()
    site, lower = sites[index], (sites[index - 1].lineno if index else OWNED_FLOOR - 1)
    imports = [
        node for node in ast.walk(_tree(APP_PATH))
        if isinstance(node, ast.ImportFrom) and node.module == GATE_MODULE
        and lower < node.lineno < site.lineno
    ]
    return _exec_nodes(namespace, *imports, site)


# ------------------- R-003 4.F3/F4.6: the composite and the currency it strands
ASSAY_SPAN = (
    "LDH-catalyzed conversion of pyruvate to lactate is then monitored by loss of "
    "OD 340 following oxidation of NADH to NAD+"
)
ENTEROBACTIN_CTX = {"pathway_name": "enterobactin biosynthesis", "likely_organism": "Escherichia coli"}
#: The same two species, in the one pathway that is ABOUT them.
NAD_CTX = {"pathway_name": "NAD biosynthesis", "likely_organism": "Escherichia coli"}
GENUINE = "EntB isochorismatase reaction"
COMPOSITE = "EntB isochorismatase reaction (NADH-dependent coupled assay)"
SURVIVORS = ["isochorismate", "2,3-diDHB", "pyruvate"]


def _f4_base() -> dict:
    return {
        "entities": {"compounds": [{"name": n} for n in SURVIVORS + ["NADH"]],
                     "proteins": [{"name": "EntB"}]},
        "processes": {"reactions": [
            {"name": GENUINE, "inputs": ["isochorismate"], "outputs": ["2,3-diDHB", "pyruvate"]}]},
    }


def _f4_additions() -> dict:
    return {"additions": {
        "entities": {"compounds": [{"name": "NAD", "provenance": "inferred"}]},
        "processes": {"reactions": [{
            "name": COMPOSITE, "inputs": ["isochorismate", "NADH"],
            "outputs": ["2,3-diDHB", "pyruvate", "NAD"], "evidence": ASSAY_SPAN}]},
    }}


def _f1_payload() -> dict:
    """R-003 4.F1.6 -- the reagent form of a species already present."""
    return {
        "entities": {"compounds": [{"name": "heme"}, {"name": "hemin"}],
                     "proteins": [{"name": "ALAS2"}]},
        "processes": {"interactions": [
            {"entity_1": "heme", "entity_2": "ALAS2", "relationship": "binds_to"},
            {"entity_1": "hemin", "entity_2": "ALAS2", "relationship": "binds_to"}]},
    }


#: The app's local name for the seed paper at both merge sites
#: (``streamlit_app.py:5679`` stores it as ``pipeline_source_text``). C-075 armed
#: ``merge_additions(source_text=...)`` at the Stage-2 site, so the lifted
#: statement now references this name and the exec namespace has to carry it or
#: every arm below dies with ``NameError`` before reaching its assertion.
#:
#: EMPTY ON PURPOSE, and this is the only thing that changed in this file. This
#: instrument owns C-060a's ``pathway_context`` wiring; supplying a paper here
#: would arm ANOTHER card's rule inside it and change the gate's evidence base
#: out from under the properties below. With ``""`` the gate writes no index and
#: behaves exactly as C-060a pinned it. No assertion in this file was touched.
APP_SOURCE_TEXT = ""


def _stage_two_site(base, additions, ctx=ENTEROBACTIN_CTX):
    return _run_app_site(0, {
        "merge_additions": merge_additions, "stage_one_in_scope": base,
        "stage_two": additions, "pathway_context": dict(ctx),
        "text": APP_SOURCE_TEXT})["final_payload"]


def _rag_site(base, envelope, ctx=ENTEROBACTIN_CTX):
    return _run_app_site(1, {
        "merge_additions": merge_additions, "final_payload": base,
        "rag_envelope": envelope, "pathway_context": dict(ctx),
        "text": APP_SOURCE_TEXT})["merged"]


def _names(payload, bucket="compounds"):
    return [row.get("name") for row in payload.get("entities", {}).get(bucket, [])]


def _reactions(payload):
    return [row.get("name") for row in payload.get("processes", {}).get("reactions", [])]


# ========================================================== G9 behavioural base
def test_g9_f4_currency_orphans_close_through_run_stage_two_with_feedback_loop(monkeypatch):
    """F4 through ``pipeline.py:723`` -- a real production function, positional.

    The merged payload is not returned; it becomes ``working_stage_one``, the
    payload the NEXT QA round is handed. At base that still carries NADH and NAD,
    because the loop's own ``pathway_context`` dict never reached the gate."""
    from t2pw.pipeline import pipeline as pl

    handed = []

    def _stage_two(_text, stage_one, **_kwargs):
        handed.append(copy.deepcopy(stage_one))
        return copy.deepcopy(_f4_additions()), []

    monkeypatch.setattr(pl, "run_stage_two_with_chunking", _stage_two)
    pl.run_stage_two_with_feedback_loop(
        "seed text", _f4_base(), pathway_context=dict(ENTEROBACTIN_CTX),
        qa_rounds=2, enable_chunking=False,
    )

    assert len(handed) == 2, "the second QA round must run for the merge to be observable"
    assert _reactions(handed[1]) == [GENUINE]
    assert _names(handed[1]) == SURVIVORS, (
        "currency stranded by the removed assay composite survived into QA round 2"
    )


def test_g9_f4_currency_orphans_close_through_the_app_stage_two_merge():
    """F4 through the first app site -- the real statement, executed as written."""
    merged = _stage_two_site(_f4_base(), _f4_additions())
    assert _reactions(merged) == [GENUINE]
    assert _names(merged) == SURVIVORS, (
        "the app's Stage-2 merge admitted currency stranded by the assay composite"
    )


def test_g9_f4_currency_orphans_close_through_the_app_rag_remerge():
    """F4 through the second app site -- the RAG conform/merge re-entry."""
    merged = _rag_site(_f4_base(), _f4_additions())
    assert _reactions(merged) == [GENUINE]
    assert _names(merged) == SURVIVORS, (
        "the app's RAG re-merge admitted currency stranded by the assay composite"
    )


# ================================================ new acceptance (no base claim)
def test_new_acceptance_f1_and_f3_still_close_through_both_app_sites():
    """Acceptance 1. F1/F3 need no context and closed at base; pin that the
    wiring disturbed neither."""
    first = _stage_two_site(_f1_payload(), {})
    assert _names(first) == ["heme"]
    assert len(first["processes"]["interactions"]) == 1
    assert COMPOSITE not in _reactions(_rag_site(_f4_base(), _f4_additions()))


def test_new_acceptance_no_over_removal_when_the_currency_is_the_subject():
    """Acceptance 2. Identical rows and wiring, a pathway that is ABOUT NAD:
    ``classify_entity`` answers ``participant`` and phase 2 removes nothing. The
    gate subtracts because of the CONTEXT, never because of the name."""
    ctx = ea.pathway_context_from_stage_zero(NAD_CTX)
    assert classify_entity("NADH", ctx).verdict == "participant"

    merged = _stage_two_site(_f4_base(), _f4_additions(), NAD_CTX)
    assert _names(merged) == SURVIVORS + ["NADH", "NAD"]
    assert all(e["phase"] != ea.PHASE_ADVISORY for e in merged[ea.LEDGER_KEY]["removed"])


def test_new_acceptance_a_blank_stage_zero_context_changes_nothing():
    """No new refusal path. Stage 0 can come back with nothing usable; the factory
    then yields ``PathwayContext("")``, which ``_screen_advisory`` already coerces
    ``None`` to -- so the wired call and the unwired call agree row for row."""
    blank = ea.pathway_context_from_stage_zero({"pathway_name": "", "likely_organism": ""})
    assert isinstance(blank, PathwayContext) and blank.requested_pathway == ""

    wired = merge_additions(_f4_base(), _f4_additions(), pathway_context=blank)
    unwired = merge_additions(_f4_base(), _f4_additions())
    assert _names(wired) == _names(unwired) == SURVIVORS + ["NADH", "NAD"]
    assert wired[ea.LEDGER_KEY] == unwired[ea.LEDGER_KEY]


def test_new_acceptance_a0_c5_ordering_survives_the_wiring(monkeypatch):
    """Acceptance 4. Replace the classifier with one returning ``participant`` /
    ``high`` for EVERYTHING -- the most protective verdict there is -- and drive
    the wired production statement. hemin still goes, in phase 1, and the
    classifier never saw it: the hallucination gate runs first and independently."""
    seen = []

    def _maximally_protective(name, _context):
        seen.append(str(name))
        return classify_entity("succinyl-CoA", PathwayContext("heme biosynthesis"))

    monkeypatch.setattr(ea, "classify_entity", _maximally_protective)
    merged = _stage_two_site(_f1_payload(), {})

    assert _names(merged) == ["heme"]
    assert "hemin" not in seen, "phase 1 consulted cofactor_policy on the removed row"
    hemin = next(e for e in merged[ea.LEDGER_KEY]["removed"] if e["entity"] == "hemin")
    assert hemin["phase"] == ea.PHASE_HALLUCINATION


def test_new_acceptance_the_two_pass_ledger_stays_idempotent_once_wired():
    """Acceptance 3. A RAG leg runs both app sites in sequence and the second
    call's BASE is the first call's output. The first pass's removals must still
    be on the record, must not double, and a new removal must still append."""
    first = _stage_two_site(_f4_base(), _f4_additions())
    recorded = [(e["entity"], e["rule"], e["phase"]) for e in first[ea.LEDGER_KEY]["removed"]]
    assert ("NADH", ea.RULE_CURRENCY_NOT_SUBJECT, ea.PHASE_ADVISORY) in recorded

    second = _rag_site(first, {})
    assert [
        (e["entity"], e["rule"], e["phase"]) for e in second[ea.LEDGER_KEY]["removed"]
    ] == recorded
    assert second[ea.LEDGER_KEY]["counts"]["removed"] == len(recorded)

    third = _rag_site(
        second, {"additions": {"entities": {"compounds": [{"name": "heme"}, {"name": "hemin"}]}}})
    assert [e["entity"] for e in third[ea.LEDGER_KEY]["removed"]] == [
        e[0] for e in recorded] + ["hemin"]


def test_new_acceptance_every_production_call_site_builds_a_real_pathway_context():
    """The trap this card exists for, plus the scope line it must not cross.

    ``_screen_advisory`` gates on ``isinstance(context, PathwayContext)``, so the
    Stage-0 **dict** in scope at every site would be silently inert: each site
    must call the shared factory. ``seed_text`` stays absent on purpose -- it is
    ``_unlocatable``'s only input, and supplying it at ``pipeline.py:723`` would
    switch evidence-span removal on for every Stage-2 QA round: a pinned-baseline
    move and a separate chartered decision."""
    for node in _production_sites():
        keywords = {kw.arg: kw.value for kw in node.value.keywords}
        assert "seed_text" not in keywords, f"line {node.lineno} wired seed_text"
        value = keywords.get("pathway_context")
        assert value is not None, f"line {node.lineno} still passes no context"
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name), (
            f"line {node.lineno} does not construct its context")
        assert value.func.id == "pathway_context_from_stage_zero", (
            f"line {node.lineno} re-derives the request instead of calling the factory")


def test_new_acceptance_the_factory_reproduces_maybe_run_rags_own_derivation():
    """DERIVATION DIVERGENCE. ``maybe_run_rag`` derives the request once, under a
    comment stating no consumer may re-derive it differently -- but it is a local
    (``streamlit_app.py:468-473``) the C-060a call sites cannot see. So this
    evaluates the app's OWN source for those two assignments, alongside the app's
    own ``_safe_dict``, and requires the factory to agree field for field. If
    either derivation moves without the other, this fails."""
    named = {
        node.name: node for node in _tree(APP_PATH).body
        if isinstance(node, ast.FunctionDef) and node.name in ("maybe_run_rag", "_safe_dict")
    }
    wanted = {"_requested_pathway", "_requested_organism"}
    assigns = [
        node for node in ast.walk(named["maybe_run_rag"])
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
    ]
    assert {t.id for a in assigns for t in a.targets} == wanted, assigns

    for sample in (
        {"pathway_name": "enterobactin biosynthesis", "likely_organism": "Escherichia coli"},
        {"pathway_name": "heme biosynthesis", "organism": "Homo sapiens"},
        {"pathway_name": "x", "likely_organism": "", "organism": "fallback"},
        {"pathway_name": "", "likely_organism": ""}, {"likely_organism": "Escherichia coli"},
        {"pathway_name": None, "likely_organism": None},
        {"pathway_name": 12, "likely_organism": ["a"]}, {}, None, "not a dict",
    ):
        ns = _exec_nodes({"pathway_context": sample}, named["_safe_dict"], *assigns)
        built = ea.pathway_context_from_stage_zero(sample)
        assert built.requested_pathway == ns["_requested_pathway"], sample
        assert built.organism == ns["_requested_organism"], sample

    already = PathwayContext("heme biosynthesis", "Homo sapiens")
    assert ea.pathway_context_from_stage_zero(already) is already, "double-derived"


# ------------------------------------------------------------------ prophylactic
#: ``_unlocatable`` reads ``seed_text``, which no production caller supplies, so
#: the test below is reachable only from a test today. Hardening against the day
#: ``seed_text`` is wired -- NOT a correction of observed behaviour, and no
#: base-SHA failure is claimed for it.
_CURLY_SEED = (
    "The enzyme’s activity was measured for the 2,3–dihydroxybenzoate "
    "“coupled” step across   several   replicates."
)


def _screen_one(evidence, name="x"):
    return ea.screen_additions(
        {"entities": {"compounds": [{"name": name, "evidence": evidence}]},
         "processes": {"reactions": []}},
        seed_text=_CURLY_SEED,
    )


def test_new_acceptance_the_hardening_folds_typography_and_nothing_else():
    """The seed carries a curly apostrophe, curly quotes, an en-dash and doubled
    spaces; quoting the same sentence back in ASCII is a transcription
    difference, not a fabrication, so the row stays. Everything else the seed
    genuinely does not contain is still condemned: the fold is a fixed character
    map, so no synonym, paraphrase or same-punctuation compound rides in on it."""
    kept, ledger = _screen_one(
        "The enzyme's activity was measured for the 2,3-dihydroxybenzoate "
        '"coupled" step across several replicates.',
        name="2,3-diDHB",
    )
    assert _names(kept) == ["2,3-diDHB"] and ledger["removed"] == []

    for evidence in (
        "ALAS synthesizes dALA from succinyl-CoA and the amino acid glycine",
        "The enzyme's activity was measured for ATP synthase in the same assay.",
    ):
        kept, ledger = _screen_one(evidence)
        assert _names(kept) == []
        assert [e["rule"] for e in ledger["removed"]] == [ea.RULE_UNLOCATABLE_EVIDENCE]
