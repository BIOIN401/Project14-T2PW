"""C-096 / F-129 -- a caller must be able to say "no database", explicitly.

**G9 labelling, stated once and precisely.**

*Every test in this file is a NEW ACCEPTANCE test for NEW capability.* The value
they pass -- :data:`t2pw.pwml.compound_resolution.NO_DB_RESOLVER` -- does not
exist at the base SHA ``7862fcc``, so none of them can carry a *behavioural* base
failure and none is offered as one. Under G9 that is the honest half: a genuinely
new capability carries an explicitly labelled new acceptance test and needs no
fabricated base failure.

The **correction** half of this card is elsewhere and is labelled as a correction:
the four repaired fixtures in ``tests/test_prefreeze_third_export_seam.py``. Those
assert pre-existing observable behaviour, they fail behaviourally on the base SHA
with a live PathBank up, and that failure is committed --
``evidence/g11/ORCH-705/02-f129-baseline.json`` (4 failed / 8 passed, exit 1),
reproduced in this worktree as ``evidence/g11/C-096/03-base-seam-live.json``.

What the sentinel is for
------------------------
``db_resolver`` had two states and needed three. ``None`` means *unspecified --
open the ambient PathBank connection*, which is contract-bound: ``PRODUCT_CONTRACT``
§8 forbids the exporter opening one, so the pre-freeze call must
(``prefreeze_resolution.resolve_compounds_prefreeze`` docstring, D-015, D-032
clause 6). ``None`` was **also** the only value available to a caller meaning
*resolve nothing against a database*, so that meaning was unreachable: such a
caller silently got the ambient live database and nothing in the report said so.
:data:`NO_DB_RESOLVER` adds the third state. ``None`` is not redefined, and
:func:`test_none_still_opens_the_ambient_resolver` is what holds that line.

On the two ``from_env`` doubles below
-------------------------------------
The ambient resolver is *installed by the test*, not inherited from the machine.
That is the point: F-129's whole complaint is that these outcomes were green or red
depending on whether a developer's PathBank happened to be running. A monkeypatched
``from_env`` makes the ambient database **guaranteed present and guaranteed
answering**, which is strictly stronger than hoping one is up, and it is not a way
of hiding a live one -- :func:`test_the_sentinel_ignores_whatever_the_real_ambient_database_is`
runs against the real ambient, unpatched, and pins the sentinel's answer against it.
"""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import t2pw.mapping.map_ids as map_ids  # noqa: E402
import t2pw.pwml.compound_resolution as cr  # noqa: E402
from t2pw.pwml.compound_resolution import (  # noqa: E402
    DB_RESOLUTION_DISABLED_REASON,
    NO_DB_RESOLVER,
)
from t2pw.pwml.ir import PREFREEZE_DB_RESOLUTION_FIELD, build_pwml_ir  # noqa: E402
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution  # noqa: E402


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

#: The row the canned PathBank answers with. It carries the SAME ``hmdb_id`` the
#: payload row carries, so ``PathWhizCompoundResolver`` matches it on
#: ``hmdb_id_exact`` at 0.95 -- above ``DB_MATCH_CONFIDENCE_FLOOR`` and admitted
#: by D-028 rule 3 -- and the rename ``glycolate`` -> ``Glycolic acid`` is
#: APPLIED. A consulted database is therefore visible in the row itself, not only
#: in the report, which is what lets every assertion below be behavioural.
_GLYCOLIC_ACID_ROW: Dict[str, Any] = {
    "id": 4108, "name": "Glycolic acid", "short_name": "Glycolate",
    "hmdb_id": "HMDB0000115", "kegg_id": "C00160", "chebi_id": "17497",
    "pubchem_cid": "757", "cas": "79-14-1", "biocyc_id": "GLYCOLLATE",
    "chemspider_id": "737", "drugbank_id": "", "pwc_id": "PW_C000115",
    "description": "canned", "synonyms": "Glycolic acid; Glycolate",
}


class _CannedReachableDb:
    """A REACHABLE PathBank that answers every query with the glycolate row."""

    def __init__(self) -> None:
        self.queries: List[str] = []

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        self.queries.append(str(sql))
        return [dict(_GLYCOLIC_ACID_ROW)]


class _DownDb:
    """CONFIGURED and unreachable. ``last_error`` names the real reason."""

    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


class _AmbientTripwire:
    """Stands in for ``PathBankDbResolver.from_env`` and records every call.

    Returns a resolver that is reachable AND would rename the payload row, so
    "the ambient database was consulted" is observable two independent ways: this
    counter, and the row's own name.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.db = _CannedReachableDb()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "_AmbientTripwire":
        def _from_env(_cls: Any, overrides: Any = None) -> Any:  # noqa: ARG001
            self.calls += 1
            return self.db

        monkeypatch.setattr(
            map_ids.PathBankDbResolver, "from_env", classmethod(_from_env)
        )
        return self


def _install_unconfigured_ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``from_env`` answer "nothing is configured", deterministically.

    Needed only by :func:`test_the_three_reasons_are_three_different_reasons`,
    which has to assert the ``db_not_configured`` arm, and that arm is by
    definition unreachable while an ambient PathBank is configured. This is a
    stub inside one assertion about the reason VOCABULARY -- it is not how any
    test in this file or in ``test_prefreeze_third_export_seam.py`` obtains its
    "no database" outcome. Those use :data:`NO_DB_RESOLVER`, which needs no stub
    and works with the database running.
    """

    monkeypatch.setattr(
        map_ids.PathBankDbResolver,
        "from_env",
        classmethod(lambda _cls, overrides=None: None),  # noqa: ARG005
    )


def _glycolate_rows() -> List[Dict[str, Any]]:
    return [{"name": "glycolate", "hmdb_id": "HMDB0000115"}]


def _payload(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": copy.deepcopy(compounds),
        },
        "biological_states": [
            {"name": "cyto_state", "species": "Homo sapiens",
             "subcellular_location": "cytosol"}
        ],
        "element_locations": {"compound_locations": []},
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _resolve(db_resolver: Any) -> Dict[str, Any]:
    """One ``_resolve_compound_rows`` call over one glycolate row.

    ``name_index=None`` on purpose: the real offline index would also rename, and
    then a rename would no longer prove the DATABASE was consulted.
    """

    report: Dict[str, Any] = {}
    rows = cr._resolve_compound_rows(
        _glycolate_rows(),
        db_resolver=db_resolver,
        strict_db=False,
        report=report,
        pointer_prefix="/entities/compounds",
        name_index=None,
    )
    return {"rows": rows, "report": report,
            "db_resolution": report["db_resolution"]}


# ---------------------------------------------------------------------------
# 1. The sentinel resolves nothing -- with a live ambient database available
# ---------------------------------------------------------------------------


def test_the_sentinel_resolves_nothing_though_the_ambient_database_is_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. The central test of the card.

    Constructed so it FAILS if the ambient database is consulted: the ambient
    resolver installed here is reachable and would rename ``glycolate`` ->
    ``Glycolic acid`` and stamp four identifiers on the row. Its control is
    :func:`test_none_still_opens_the_ambient_resolver`, which passes ``None``
    through this same tripwire and gets exactly that rename -- so a change that
    made the sentinel silently stop working would turn this red rather than
    leaving it vacuously green.
    """

    tripwire = _AmbientTripwire().install(monkeypatch)

    outcome = _resolve(NO_DB_RESOLVER)

    assert tripwire.calls == 0, "the ambient resolver was constructed anyway"
    assert tripwire.db.queries == [], "the ambient database was queried anyway"
    assert outcome["db_resolution"]["available"] is False
    assert outcome["db_resolution"]["reason"] == DB_RESOLUTION_DISABLED_REASON
    # Behavioural, not just reported: the row keeps the name the paper gave it
    # and gains no DB identity.
    row = outcome["rows"][0]
    assert row["name"] == "glycolate"
    assert row.get("db_row") is None
    assert row.get("pathwhiz_id") is None


def test_the_sentinel_ignores_whatever_the_real_ambient_database_is() -> None:
    """NEW ACCEPTANCE. Same claim, against the REAL ambient resolver, unpatched.

    Deterministic in both environments and vacuous in neither, because it pins
    two things against each other: the sentinel's answer is a CONSTANT, while the
    ``None`` answer tracks whatever this machine's ambient PathBank says. That is
    precisely the property F-129 found missing -- an explicitly offline caller is
    now independent of ambient infrastructure while ``None`` still reports it.
    """

    ambient = map_ids.PathBankDbResolver.from_env()
    ambient_available = ambient is not None and bool(ambient.available())

    disabled = _resolve(NO_DB_RESOLVER)
    assert disabled["db_resolution"]["available"] is False
    assert disabled["db_resolution"]["reason"] == DB_RESOLUTION_DISABLED_REASON
    assert disabled["rows"][0]["name"] == "glycolate"

    unspecified = _resolve(None)
    assert unspecified["db_resolution"]["available"] is ambient_available


# ---------------------------------------------------------------------------
# 2. `None` is unchanged -- the preservation obligation
# ---------------------------------------------------------------------------


def test_none_still_opens_the_ambient_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. The preservation test, and the card's main risk.

    ``PRODUCT_CONTRACT`` §8 -- "exporters perform no network or database lookups"
    -- is satisfied by the pre-freeze call opening the connection instead. If
    ``None`` stopped opening it, the lookup would move back into the exporter or
    vanish. Asserted three ways: ``from_env`` is CALLED, the database is QUERIED,
    and the resolution is APPLIED to the row.
    """

    tripwire = _AmbientTripwire().install(monkeypatch)

    outcome = _resolve(None)

    assert tripwire.calls == 1
    assert tripwire.db.queries, "the ambient resolver was never queried"
    assert outcome["db_resolution"]["available"] is True
    assert "reason" not in outcome["db_resolution"]
    row = outcome["rows"][0]
    assert row["name"] == "Glycolic acid"
    assert row["raw_name"] == "glycolate"
    assert row["db_row"]["id"] == 4108


def test_an_omitted_db_resolver_is_the_same_as_none_through_the_public_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. Every production caller that omits the argument is safe.

    ``run_prefreeze_resolution``'s default is ``db_resolver=None`` and that
    default is what production passes. Omitting it must still open the ambient
    connection -- i.e. the default may not have drifted to the sentinel.
    """

    tripwire = _AmbientTripwire().install(monkeypatch)
    payload = _payload(_glycolate_rows())

    report = run_prefreeze_resolution(payload, strict_db=False, name_index=None)

    assert tripwire.calls >= 1
    assert report["compounds"]["resolution_report"]["db_resolution"]["available"] is True
    assert payload[PREFREEZE_DB_RESOLUTION_FIELD] == {"available": True}
    assert payload["entities"]["compounds"][0]["name"] == "Glycolic acid"


# ---------------------------------------------------------------------------
# 3. An explicitly passed resolver is untouched
# ---------------------------------------------------------------------------


def test_an_explicitly_passed_resolver_is_still_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. The third state changes nothing about the first."""

    tripwire = _AmbientTripwire().install(monkeypatch)
    given = _CannedReachableDb()

    outcome = _resolve(given)

    assert tripwire.calls == 0, "an explicit resolver must not be second-guessed"
    assert given.queries, "the caller's own resolver was not used"
    assert outcome["db_resolution"]["available"] is True
    assert outcome["rows"][0]["name"] == "Glycolic acid"


# ---------------------------------------------------------------------------
# 4. Three causes, three reasons
# ---------------------------------------------------------------------------


def test_the_three_reasons_are_three_different_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. "I disabled it" is not "it is down" is not "it is absent".

    All three are ``available: False``, and before this card the first was not
    expressible at all. Each arm is asserted separately AND the three are
    asserted pairwise distinct, because a reason vocabulary that collapses is
    exactly how a caller's decision gets reported as an infrastructure failure --
    which ``PRODUCT_CONTRACT`` §8's neighbouring rule, "a lookup failure is not
    evidence that an accession is false", exists to keep apart.
    """

    disabled = _resolve(NO_DB_RESOLVER)["db_resolution"]
    assert disabled["available"] is False
    assert disabled["reason"] == DB_RESOLUTION_DISABLED_REASON

    down = _resolve(_DownDb())["db_resolution"]
    assert down["available"] is False
    assert down["reason"] == "harvest_db_down"

    _install_unconfigured_ambient(monkeypatch)
    absent = _resolve(None)["db_resolution"]
    assert absent["available"] is False
    assert absent["reason"] == "db_not_configured"

    reasons = [disabled["reason"], down["reason"], absent["reason"]]
    assert len(set(reasons)) == 3, reasons


def test_the_disabled_reason_reaches_the_exporter_and_the_product_visible_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NEW ACCEPTANCE. The distinction survives to where a reader sees it.

    D-032 clause 6 (LOCKED) rules ``preflight`` and its warning product-visible
    export content. A run that resolved nothing because the caller said so must
    not tell its reader the resolution DB was unavailable for some other cause,
    and the exporter may not re-derive the fact (``PRODUCT_CONTRACT`` §8) -- so it
    has to be carried, and this asserts it arrives.
    """

    tripwire = _AmbientTripwire().install(monkeypatch)
    payload = _payload([{"name": "norbelladine", "hmdb_id": "HMDB9999999"}])

    run_prefreeze_resolution(
        payload, strict_db=False, db_resolver=NO_DB_RESOLVER, name_index=None
    )

    assert tripwire.calls == 0
    assert payload[PREFREEZE_DB_RESOLUTION_FIELD] == {
        "available": False, "reason": DB_RESOLUTION_DISABLED_REASON}

    _ir, ir_report = build_pwml_ir(payload, strict_db=False)

    assert ir_report["db_resolution"]["available"] is False
    assert ir_report["db_resolution"]["reason"] == DB_RESOLUTION_DISABLED_REASON
    assert ir_report["preflight"]["db_reason"] == DB_RESOLUTION_DISABLED_REASON
    assert [
        issue for issue in ir_report["warnings"]
        if issue.get("code") == "noncanonical_names_collision_risk"
        and DB_RESOLUTION_DISABLED_REASON in str(issue.get("message") or "")
    ], ir_report["warnings"]


# ---------------------------------------------------------------------------
# 5. Injection -- the value is carried, not inferred
# ---------------------------------------------------------------------------


def test_the_same_payload_reports_two_verdicts_from_the_selection_alone() -> None:
    """NEW ACCEPTANCE. Injection test, on the indistinguishable population.

    Every row carries a ``pathbank_compound_id``, so compound resolution takes the
    legacy-id branch and ``continue``s BEFORE the resolver is consulted: the rows
    that come out are byte-identical on both legs. The exporter must still report
    the two different answers, which only a carried value can do -- the same
    argument D-032 makes for ``available``, now extended to the reason. The
    byte-identity is ASSERTED, so the test cannot go vacuous if the fixture drifts.
    """

    rows = [{"name": "Glycine", "pathbank_compound_id": 78},
            {"name": "Pyruvic acid", "pathbank_compound_id": 91}]
    verdicts: Dict[str, Any] = {}
    blobs: Dict[str, str] = {}

    for leg, resolver in (("disabled", NO_DB_RESOLVER), ("canned", _CannedReachableDb())):
        payload = _payload(rows)
        run_prefreeze_resolution(
            payload, strict_db=False, db_resolver=resolver, name_index=None
        )
        blobs[leg] = repr(payload["entities"]["compounds"])
        _ir, ir_report = build_pwml_ir(payload, strict_db=False)
        verdicts[leg] = ir_report["db_resolution"]

    assert blobs["disabled"] == blobs["canned"], (
        "the population is no longer indistinguishable; this test is vacuous")
    assert verdicts["canned"]["available"] is True
    assert "reason" not in verdicts["canned"]
    assert verdicts["disabled"]["available"] is False
    assert verdicts["disabled"]["reason"] == DB_RESOLUTION_DISABLED_REASON


# ---------------------------------------------------------------------------
# 6. The sentinel's own shape
# ---------------------------------------------------------------------------


def test_the_sentinel_is_a_singleton_that_survives_copying() -> None:
    """NEW ACCEPTANCE. Identity is the whole mechanism, so identity must hold.

    ``resolve_compounds_prefreeze`` deep-copies the rows it resolves and a caller
    could as easily deep-copy a kwargs dict. A sentinel that lost its identity
    under ``deepcopy`` would fail OPEN -- back onto the ambient database, silently
    -- which is the failure this card exists to remove.
    """

    assert copy.copy(NO_DB_RESOLVER) is NO_DB_RESOLVER
    assert copy.deepcopy(NO_DB_RESOLVER) is NO_DB_RESOLVER
    assert pickle.loads(pickle.dumps(NO_DB_RESOLVER)) is NO_DB_RESOLVER

    # Not a bare False and not a magic string: either would collide with a real
    # value some caller could already be passing, and neither can be told apart
    # from data by identity.
    assert NO_DB_RESOLVER is not None
    assert not isinstance(NO_DB_RESOLVER, (bool, str, int))
    assert repr(NO_DB_RESOLVER) == "NO_DB_RESOLVER"

    # Truthy on purpose. ``if not db_resolver:`` idioms must read it as "a
    # selection was made", never as "nothing was passed"; a falsy sentinel would
    # fall through such a guard straight back to the ambient connection.
    assert bool(NO_DB_RESOLVER) is True


def test_the_sentinel_carries_no_resolver_shape_to_be_mistaken_for() -> None:
    """NEW ACCEPTANCE. It is recognised by ``is``, so it must not be duck-typable.

    ``_resolve_compound_rows`` reads ``available`` and ``last_error`` off whatever
    it is given. If the sentinel answered either, there would be two ways to mean
    the same thing and a second seam able to drift -- which is the shape of the
    defect being fixed, not of its fix.
    """

    for attribute in ("available", "resolve", "last_error", "_query"):
        assert not hasattr(NO_DB_RESOLVER, attribute), attribute
