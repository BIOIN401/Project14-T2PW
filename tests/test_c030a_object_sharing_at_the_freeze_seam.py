"""C-030a: A0-C7 -- object sharing across the freeze seam, proved not assumed.

**G9 label, up front, because it is the thing most easily got wrong.** Every test
here is a **GUARD ON PRE-EXISTING OBSERVABLE BEHAVIOUR that PASSES AT THE BASE
SHA.** This card is TEST-ONLY: it changes no production line, so there is no
behaviour change to prove and **no base failure is claimed or offered** --
manufacturing one would be a fabrication. Same split
``tests/test_c052_prefreeze_report_at_the_streamlit_seams.py`` part 5 draws for
A0-C8's ``test_a0c8_guard_*`` names. What is *new* is the **discharge of a
requirement**, not a capability: A0-C7 has never been discharged by any test.

**Where this is, and where it deliberately is not.** ``MASTER_PLAN`` section 9
scopes C-030 to ``freeze_canonical_payload``, and **F-008 proved A0-C7
undischargeable inside it**: the canonical payload there is a ``deepcopy`` that
nothing in the seam's own returned dict aliases, so the documented share->copy
mutant is indistinguishable from the tip on all nine seam observables across all
39 legs. A test written against it would be vacuous by construction, and this
file touches it not at all. The **real** sharing sites are the four places
``run_post_pipeline_sbml_artifacts`` publishes the frozen object into the
artifact set, plus the one line binding the seam's return to the local they read.

**Why identity and not equality.** A0-C7 records ``final_mapped is
result["payload"]`` as *tautological* -- True on the 35 legs that freeze a payload
and False on the 4 that refuse, tracking ``quarantine_ok`` and nothing else, and a
share->copy mutant survives it. Equality is weaker still: a copy is equal to its
original by definition. So the contract is stated as identity against the two
reference objects a test can actually capture -- the object the seam returned, and
the **pre-seam caller-owned payload** the orchestrator held before calling it --
and re-proved behaviourally by mutating through one alias and observing the change
through the others.

**Non-vacuity is a permanent gate here, not a claim in a report.** The
``test_a0c7_mutant_*`` tests rebuild the orchestrator from its own AST with
``deepcopy`` at exactly one sharing site and assert the **exact** clause set that
goes red. F-008's lesson is precisely that a test which cannot distinguish sharing
from copying discharges nothing.

No ``AppTest`` is needed: the orchestrator under test makes zero ``st.`` calls.
"""

from __future__ import annotations

import ast
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT / "src", ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

# ``t2pw.app.streamlit_app`` runs a Streamlit script at import and opens a
# ``with st.form(...)`` at module scope. The stub is this repository's
# established convention, copied in shape from
# ``tests/test_c052_prefreeze_report_at_the_streamlit_seams.py``; it is sound for
# the same reason it is sound there -- the orchestrator makes no Streamlit call
# -- and the guard matters: if the real streamlit is already loaded, stubbing it
# would break whatever loaded it.
if "streamlit" not in sys.modules:  # pragma: no cover - import-time wiring
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st

from t2pw.pipeline.gate_reports import CANONICAL_PAYLOAD_KEY, payload_sha256  # noqa: E402
from t2pw.pwml import prefreeze_resolution as prefreeze_module  # noqa: E402
import test_streamlit_stage2_orchestration as orchestration  # noqa: E402

APP_PATH = ROOT / "src" / "t2pw" / "app" / "streamlit_app.py"
EP1 = "run_post_pipeline_sbml_artifacts"
SEAM = "freeze_canonical_payload"

#: The local the seam's return is bound to, and the name the four publication
#: sites read. Every success-arm mutant below copies exactly this object.
CANONICAL_LOCAL = "canonical_export_payload"
#: The pre-seam caller-owned payload: built by ``json.loads`` above the seam,
#: canonicalized in place, handed to the seam, and republished unchanged.
PRE_SEAM_LOCAL = "final_export_payload"
#: The name the mutants call. Injected into the harness namespace by
#: :func:`_harness`; absent from production, so a mutant cannot run by accident.
COPY_HOOK = "_c030a_copy"

#: The artifact-set keys that publish the frozen object. Named as data because
#: the mutants and the contract must not be able to disagree about which sites
#: they are. ``CANONICAL_PAYLOAD_KEY`` is written as a NAME node in the source
#: and the other three as string constants; :class:`_CopyAtKeys` renders both.
CANONICAL_KEY_NODE = "CANONICAL_PAYLOAD_KEY"
ALIAS_KEYS = ("final_mapped", "final_mapped_quarantined", "final_export_input")
#: Where the pre-seam object is republished, and where the refusal arm falls back.
ENRICHED_KEY = "final_mapped_enriched"


# -- offline doubles --------------------------------------------------------
#
# Passed explicitly. A worktree now carries a ``.env`` (F-051), so "offline by
# accident" is not available and has to be arranged: an identity test must not
# acquire a dependency on whether a database happened to be reachable.


class _OfflineResolver:
    """Reachable enough to be consulted, and reports itself unavailable."""

    last_error = "db_not_configured_in_test"

    def available(self) -> bool:
        return False


class _EmptyNameIndex:
    """An index that resolves nothing, so no canonical name is ever fetched."""

    def compound_canonical(self, **_ids: Any) -> Optional[Dict[str, Any]]:
        return None


def _offline_prefreeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the REAL pre-freeze seam through offline doubles.

    Forwarding rather than stubbing: the call site is untouched and the real
    canonicalizers run, so the in-place mutation contract D-015 documents above
    that call is exercised rather than assumed away. Only the resolver and index
    the call site leaves to the ambient environment are filled in.
    """

    real = prefreeze_module.run_prefreeze_resolution

    def spy(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("db_resolver", _OfflineResolver())
        kwargs.setdefault("name_index", _EmptyNameIndex())
        return real(payload, **kwargs)

    monkeypatch.setattr(prefreeze_module, "run_prefreeze_resolution", spy)


def _refusing_quarantine_boundary(payload: Dict[str, Any], **_kwargs: Any) -> Dict[str, Any]:
    """The refusal arm: the boundary declines, so no canonical payload exists.

    The harness pass-through's shape with ``ok`` false and a reason -- what
    ``freeze_canonical_payload`` reads to leave ``canonical_export_payload``
    empty and ``canonical_json_path`` unset.
    """

    return {
        "ok": False,
        "payload": payload,
        "quarantine_report": {},
        "quarantine_artifacts": {},
        "coverage": {},
        "refusal_reasons": ["c030a_refusal_arm"],
        "review_flags": [],
    }


# -- the mutants: the orchestrator, rebuilt from its own AST with one copy --


def _orchestrator_node() -> ast.FunctionDef:
    source = APP_PATH.read_text(encoding="utf-8")
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == EP1)


class _WrapName(ast.NodeTransformer):
    """Wrap every load of ``target`` in a call to the copy hook. Counts hits."""

    def __init__(self, target: str) -> None:
        self.target, self.hits = target, 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.target:
            self.hits += 1
            return ast.Call(func=ast.Name(id=COPY_HOOK, ctx=ast.Load()),
                            args=[node], keywords=[])
        return node


class _CopyAtKeys(ast.NodeTransformer):
    """Copy ``target`` where the artifact-set dict publishes it under ``keys``."""

    def __init__(self, keys: Sequence[str], target: str) -> None:
        self.keys, self.target, self.hits = set(keys), target, 0

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        self.generic_visit(node)
        for index, key in enumerate(node.keys):
            rendered = (key.value if isinstance(key, ast.Constant)
                        else key.id if isinstance(key, ast.Name) else None)
            if rendered in self.keys:
                inner = _WrapName(self.target)
                node.values[index] = inner.visit(node.values[index])
                self.hits += inner.hits
        return node


class _CopyAtSeamBinding(ast.NodeTransformer):
    """Copy the seam's return where the orchestrator binds it to its local.

    ``canonical_export_payload = _freeze["payload"]`` -- the ONE line that makes
    the artifact set's four aliases the seam's own object rather than a
    lookalike. A copy here is invisible to every "the four keys agree" check,
    because all four would agree, on the copy; only the clause that compares the
    published object against the object the seam actually returned can see it.
    """

    def __init__(self) -> None:
        self.hits = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if targets == [CANONICAL_LOCAL] and isinstance(node.value, ast.Subscript):
            self.hits += 1
            node.value = ast.Call(func=ast.Name(id=COPY_HOOK, ctx=ast.Load()),
                                  args=[node.value], keywords=[])
        return node


def _install_mutant(namespace: Dict[str, Any], transformer: ast.NodeTransformer) -> Any:
    """Exec the mutated orchestrator into the harness namespace and return it."""

    node = transformer.visit(_orchestrator_node())
    assert getattr(transformer, "hits", 0) >= 1, (
        f"{type(transformer).__name__} matched no site -- the mutant is a no-op "
        "and would 'discriminate' by not existing")
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(APP_PATH), "exec"), namespace)  # noqa: S102
    return namespace[EP1]


# -- the harness ------------------------------------------------------------


def _harness(
    tmp_path: Path,
    *,
    mutant: Optional[ast.NodeTransformer] = None,
    refuse: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """``run_post_pipeline_sbml_artifacts``, callable, plus the seam's own view.

    Reuses ``tests/test_streamlit_stage2_orchestration.py``'s harness -- the
    established vehicle for calling this orchestrator without a browser, a DB or
    an LLM. ``seen`` records the two reference objects no artifact-set key can
    supply: the payload handed to the seam, and the dict it handed back.
    """

    function, _observed = orchestration._orchestration_harness(tmp_path)
    namespace = function.__globals__
    namespace[COPY_HOOK] = deepcopy
    if refuse:
        namespace["run_quarantine_boundary"] = _refusing_quarantine_boundary

    seen: Dict[str, Any] = {}
    real_seam = namespace[SEAM]

    def seam_spy(pre_seam_payload: Dict[str, Any], **kwargs: Any) -> Any:
        seen["pre_seam"] = pre_seam_payload
        result = real_seam(pre_seam_payload, **kwargs)
        seen["frozen"] = result
        seen["calls"] = seen.get("calls", 0) + 1
        return result

    namespace[SEAM] = seam_spy
    if mutant is not None:
        function = _install_mutant(namespace, mutant)
    return function, seen


def _run(tmp_path: Path, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    function, seen = _harness(tmp_path, **kwargs)
    result = function(orchestration._paper_payload(),
                      **orchestration._invoke_kwargs(tmp_path))
    assert seen.get("calls") == 1, "the seam did not run exactly once"
    assert isinstance(result, dict) and result.get("gate_failed") is False, (
        f"the orchestrator did not reach its success return: {result.get('error')!r}")
    return result, seen


# -- the contract, as named clauses -----------------------------------------
#
# Returned as a list of violated clause names rather than asserted inline, so a
# mutant test can state the EXACT set it expects to go red. ``pytest.raises`` was
# rejected for this: it would pass just as happily on an unrelated AssertionError
# from the harness, which is precisely the "the test cannot tell" failure this
# card exists to rule out.


def _sharing_violations(result: Dict[str, Any], seen: Dict[str, Any]) -> List[str]:
    """Every A0-C7 clause the SUCCESS arm breaks. Empty is the contract held."""

    pre_seam, frozen = seen["pre_seam"], seen["frozen"]
    canonical = result.get(CANONICAL_PAYLOAD_KEY)
    broken: List[str] = []

    # 1. The artifact set publishes the seam's OWN object. Everything else is
    #    downstream of this: four keys can agree perfectly on a lookalike.
    if canonical is not frozen["payload"]:
        broken.append("artifact_set_publishes_the_object_the_seam_returned")
    # 2-4. Every consumer-facing alias is that same one object.
    for key in ALIAS_KEYS:
        if result.get(key) is not canonical:
            broken.append(f"{key}_aliases_the_canonical_object")
    # 5. It is NOT the pre-seam caller-owned payload -- the seam deepcopies the
    #    quarantined payload, so a tip that published the pre-seam object would
    #    be publishing something the final gate never validated.
    if canonical is pre_seam:
        broken.append("canonical_object_is_not_the_pre_seam_object")
    # 6. The pre-seam object is still reachable, under its own name. This is the
    #    "capture the pre-seam caller-owned payload reference" half of A0-C7,
    #    made observable from outside the orchestrator.
    if result.get(ENRICHED_KEY) is not pre_seam:
        broken.append("enriched_key_republishes_the_pre_seam_object")
    # 7. Sharing, observed as behaviour rather than as ``is``: a mutation made
    #    through one handle is visible through every other. This is the property
    #    the seam's own deepcopy comment says its consumers depend on.
    sentinel = "_c030a_sharing_probe"
    canonical[sentinel] = True
    try:
        if not all(isinstance(result.get(key), dict) and sentinel in result[key]
                   for key in ALIAS_KEYS):
            broken.append("a_mutation_through_one_alias_is_visible_through_all")
    finally:
        canonical.pop(sentinel, None)
    return broken


def _refusal_violations(result: Dict[str, Any], seen: Dict[str, Any]) -> List[str]:
    """Every A0-C7 clause the REFUSAL arm breaks. Empty is the contract held."""

    pre_seam = seen["pre_seam"]
    broken: List[str] = []
    if result.get(CANONICAL_PAYLOAD_KEY) != {}:
        broken.append("a_refusal_publishes_no_canonical_payload")
    if result.get("quarantine_refused") is not True:
        broken.append("a_refusal_is_reported_as_one")
    # The documented fallback: with no canonical payload the two consumer keys
    # publish the last complete payload ITSELF, not a copy of it.
    for key in ("final_mapped", "final_export_input", ENRICHED_KEY):
        if result.get(key) is not pre_seam:
            broken.append(f"refusal_{key}_is_the_pre_seam_object")
    return broken


# == A0-C7 GUARDS. These pass at the base SHA; see the module docstring. =====


def test_a0c7_guard_the_artifact_set_shares_one_canonical_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A0-C7, success arm: all seven clauses of the sharing contract, at once."""

    _offline_prefreeze(monkeypatch)
    result, seen = _run(tmp_path)

    assert result["quarantine_ok"] is True, "the freeze refused; this would be vacuous"
    assert _sharing_violations(result, seen) == []


def test_a0c7_guard_the_shared_object_is_the_one_that_was_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published object is the object whose digest the seam published.

    Identity alone would not say this, and the digest alone would not say it
    either: the two together are the claim the freeze seam exists to make.
    """

    _offline_prefreeze(monkeypatch)
    result, seen = _run(tmp_path)

    canonical = result[CANONICAL_PAYLOAD_KEY]
    assert canonical is seen["frozen"]["payload"]
    assert result["canonical_payload_sha256"] == seen["frozen"]["payload_hash"]
    assert payload_sha256(canonical) == result["canonical_payload_sha256"]
    assert result["sbml_input_source"] == CANONICAL_PAYLOAD_KEY


def test_a0c7_guard_a_refusal_publishes_the_pre_seam_object_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other arm -- and the reason the tautological assertion is tautological.

    ``final_mapped is result["payload"]`` is True on exactly the legs that freeze
    a payload and False on exactly the legs that refuse, i.e. a restatement of
    ``quarantine_ok``. Here the refusal arm is driven deliberately and its own
    sharing relationship asserted: the fallback publishes the last complete
    payload ITSELF, so the two arms share *different* objects and neither arm's
    assertion can be satisfied by the other arm's object.
    """

    _offline_prefreeze(monkeypatch)
    result, seen = _run(tmp_path, refuse=True)

    assert result["quarantine_ok"] is False, "the boundary did not refuse"
    assert _refusal_violations(result, seen) == []
    # And the success arm's contract is genuinely NOT satisfied here. Stated as
    # the EXACT measured difference so the two arms cannot quietly become the
    # same test: the two consumer keys publish the pre-seam object while the
    # canonical key publishes the empty dict the seam returned, which no
    # success-arm clause would tolerate.
    assert sorted(_sharing_violations(result, seen)) == sorted([
        "final_mapped_aliases_the_canonical_object",
        "final_export_input_aliases_the_canonical_object",
        "a_mutation_through_one_alias_is_visible_through_all"])
    # ``final_mapped_quarantined`` is absent from that list on purpose: it and
    # the canonical key are both the seam's one empty dict, so the refusal arm
    # still shares an object -- it shares a different one.
    assert result["final_mapped_quarantined"] is result[CANONICAL_PAYLOAD_KEY]
    assert result[CANONICAL_PAYLOAD_KEY] is seen["frozen"]["payload"]


# == MUTANT DISCRIMINATION. The card: a test that cannot tell sharing from ===
#    copying discharges nothing. Each case inserts ``deepcopy`` at exactly ONE
#    site and pins the EXACT clause set that goes red.


@pytest.mark.parametrize(
    "site,expected",
    [
        pytest.param(
            "final_mapped",
            ["final_mapped_aliases_the_canonical_object",
             "a_mutation_through_one_alias_is_visible_through_all"],
            id="copy-at-final_mapped"),
        pytest.param(
            "final_mapped_quarantined",
            ["final_mapped_quarantined_aliases_the_canonical_object",
             "a_mutation_through_one_alias_is_visible_through_all"],
            id="copy-at-final_mapped_quarantined"),
        pytest.param(
            "final_export_input",
            ["final_export_input_aliases_the_canonical_object",
             "a_mutation_through_one_alias_is_visible_through_all"],
            id="copy-at-final_export_input"),
        pytest.param(
            CANONICAL_KEY_NODE,
            ["artifact_set_publishes_the_object_the_seam_returned",
             "final_mapped_aliases_the_canonical_object",
             "final_mapped_quarantined_aliases_the_canonical_object",
             "final_export_input_aliases_the_canonical_object",
             "a_mutation_through_one_alias_is_visible_through_all"],
            id="copy-at-canonical-payload-key"),
    ],
)
def test_a0c7_mutant_a_copy_at_one_publication_site_goes_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str, expected: List[str]
) -> None:
    """Share->copy at one artifact-set key, and the clauses that must notice.

    Asserted as an EXACT set, not as "something failed": a mutant that broke more
    clauses than its site can account for would mean the harness moved and not
    the sharing, and that is the failure mode F-008 warns about.
    """

    _offline_prefreeze(monkeypatch)
    result, seen = _run(tmp_path, mutant=_CopyAtKeys([site], CANONICAL_LOCAL))

    assert sorted(_sharing_violations(result, seen)) == sorted(expected)


def test_a0c7_mutant_a_copy_at_the_seam_binding_goes_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutant the four-keys-agree check CANNOT see, and clause 1 can.

    **This is the mutant the card turns on, and the reason A0-C7's own form is
    insufficient.** ``canonical_export_payload = deepcopy(_freeze["payload"])``
    leaves all four artifact-set keys in perfect agreement -- with each other,
    on a lookalike the seam never hashed, gated or serialized. So every check
    of the shape *"the artifact set is internally consistent"* still passes, and
    so does A0-C7's ``final_mapped is <the canonical payload>``: both operands
    are the copy, so the tautological form reads ``True`` while the object that
    shipped is not the object the freeze froze. That is exactly F-041's
    *"a share->copy mutant survives it"*, and it is what makes an assertion
    against the artifact set alone unable to discharge this requirement.

    Only a clause that compares the published object against the object the seam
    **actually returned** -- captured through the spy, which no artifact-set key
    can supply -- can see it. Exactly one clause goes red, which is what proves
    that clause is load-bearing rather than decoration.
    """

    _offline_prefreeze(monkeypatch)
    result, seen = _run(tmp_path, mutant=_CopyAtSeamBinding())

    assert _sharing_violations(result, seen) == [
        "artifact_set_publishes_the_object_the_seam_returned"]
    # The four keys still agree with each other, which is the whole point.
    canonical = result[CANONICAL_PAYLOAD_KEY]
    assert all(result[key] is canonical for key in ALIAS_KEYS)


def test_a0c7_mutant_a_copy_in_the_refusal_fallback_goes_red(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal arm has its own sharing site, and its own mutant.

    ``"final_mapped": canonical_export_payload or final_export_payload`` -- with
    no canonical payload the right operand is what ships, so a copy there is
    invisible to every success-arm clause.
    """

    _offline_prefreeze(monkeypatch)
    result, seen = _run(
        tmp_path, refuse=True,
        mutant=_CopyAtKeys(["final_mapped"], PRE_SEAM_LOCAL))

    assert _refusal_violations(result, seen) == ["refusal_final_mapped_is_the_pre_seam_object"]


def test_a0c7_mutant_the_tautological_assertion_survives_every_one_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A0-C7's own claim, measured on this tip rather than quoted.

    ``final_mapped is <the canonical payload>`` -- the assertion A0-C7 records as
    tautological -- is evaluated under the seam-binding mutant, where it is still
    True while the payload that shipped is one the seam never froze. So the
    record's *"a share->copy mutant survives it"* is re-established here as a
    measurement, and the clause set above is shown to be strictly stronger rather
    than merely differently worded.
    """

    _offline_prefreeze(monkeypatch)
    result, seen = _run(tmp_path, mutant=_CopyAtSeamBinding())

    tautological = result["final_mapped"] is result[CANONICAL_PAYLOAD_KEY]
    assert tautological is True, "the tautological form did not survive; re-measure"
    assert _sharing_violations(result, seen) != []
