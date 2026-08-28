"""The semantic coverage validator: seven checks, one verdict per paper.

What this is for
================

The strict export gate answers "is this payload shaped the way PathWhiz's
importer wants?". It does not answer "is this the requested pathway, and is every
claim in it traceable to the paper?". Those are the questions here, and they are
scored *separately* from the export gate on purpose: a payload can be perfect
biology that PathWhiz refuses (missing a UniProt accession) or perfect shape that
is scientifically worthless (a connected graph of invented metabolites). Reporting
one number for both is how a run scores 0/16 with nobody able to say whether the
biology was right.

The eight checks
----------------

``requested_pathway_anchors_present``  the payload is about the pathway that was
    asked for, evidenced by the gold case's anchor terms appearing in it
``reaction_source_carrier_present``  every retained reaction carries a citation
    FIELD -- hygiene only. It was previously named
    ``evidence_source_for_every_retained_reaction``, which claimed far more than
    it measured: a bare ``evidence: "the paper says"`` satisfies it, as does a
    ``source_ref`` pointing at a passage about something else, so it could report
    a hallucinated reaction as fully sourced.
``retained_reactions_match_supported_signatures``  the check that actually
    measures support. Retained chemistry is matched against reaction signatures
    read out of the paper by hand, each backed by a quote that is verified against
    the stored ``01_source_text.txt`` at scoring time. Reports reaction-level
    precision and recall. Inapplicable -- never a silent pass -- when the paper
    text is missing, because an unverifiable quote is an assertion.
``organism_compatible``  no retained reaction is attributed to an organism the
    gold case forbids
``no_real_id_or_name_conflict``  no entity carries an external accession it has
    not earned, and no accession is claimed by two different entities
``no_rejected_rag_reaction_reintroduced``  a claim the admission gate refused did
    not come back in through synthesis or gap-fill
``minimum_connected_core``  enough reactions form ONE chemically connected chain
``placeholder_identities_distinguished``  Unknown-backed rows admit what they are
    and never pose as real mappings

Each check produces findings, not just a boolean, because "organism check failed"
is not actionable and "reaction 12 is attributed to Listeria monocytogenes in a
run requested for Bacillus subtilis" is.

Connectivity model
------------------

Two reactions are adjacent when they share a **non-cofactor** metabolite. Both
halves of that rule matter. Enzymes are excluded because two unrelated steps
sharing a promiscuous enzyme are not a pathway -- this is the same rule
``rag.admission`` enforces via ``participant_names()``, and deliberately *not*
what ``pipeline.qa_graph.build_graph`` computes (it adds enzyme edges, so its
component counts run optimistic). Cofactors are excluded because ATP, water and
NAD+ appear in most reactions, and a graph linked through ATP reports any bag of
unrelated reactions as one connected pathway. The cofactor list is imported from
``t2pw.rag.synthesize.COFACTOR_NAMES`` rather than restated, so this module and
the admission gate can never disagree about what counts as a link.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from t2pw.bench.goldset import (
    GoldCase,
    GoldTerm,
    MATCH_NONE,
    SupportedReaction,
    canonical_text,
    fold_for_quote,
    normalize_name,
)


# ---------------------------------------------------------------------------
# Check identifiers. Stable strings -- they are keys in the stored report and a
# renamed check silently drops out of a historical comparison.
# ---------------------------------------------------------------------------
CHECK_ANCHORS = "requested_pathway_anchors_present"

#: Renamed from ``evidence_source_for_every_retained_reaction``, which claimed
#: more than it measured. It checks that a reaction *carries a source carrier* --
#: an ``evidence`` string, a ``rag_provenance.source_id``, a ``source_ref``. A
#: bare ``"the paper says"`` satisfies it, and so does a ``source_ref`` pointing
#: at a passage about something else, so it can report a hallucinated reaction as
#: fully sourced. Carrying a citation is a necessary hygiene property, not
#: scientific support; :data:`CHECK_SUPPORTED_REACTIONS` is the one that measures
#: support.
CHECK_SOURCE_CARRIER = "reaction_source_carrier_present"

#: Retained chemistry versus the reactions the paper actually states, matched as
#: signatures and backed by a quote verified against the stored paper text.
CHECK_SUPPORTED_REACTIONS = "retained_reactions_match_supported_signatures"

CHECK_ORGANISM = "organism_compatible"
CHECK_ID_CONFLICT = "no_real_id_or_name_conflict"
CHECK_RAG_REINTRODUCTION = "no_rejected_rag_reaction_reintroduced"
CHECK_CONNECTED_CORE = "minimum_connected_core"
CHECK_PLACEHOLDER_IDENTITY = "placeholder_identities_distinguished"

#: An ACTOR -- an ``enzymes[]`` / ``modifiers[]`` / ``transporters[]`` sub-entry on a
#: retained reaction -- must be named by the span that same sub-entry cites as its own
#: evidence. Distinct from :data:`CHECK_SOURCE_CARRIER` (does a citation field exist at
#: all) and from :data:`CHECK_SUPPORTED_REACTIONS` (the CHEMISTRY, gold-only). This one
#: asks what the payload answers about itself: the row says *"protein P does this, and
#: here is the sentence I took it from"* and the sentence names another protein. F-058
#: and F-079 are both that shape -- ``EntE`` over *"secreted ... by a TolC-dependent
#: process"*. Produced by ``bench.semantic_production`` only; every ``CHECK_ORDER``
#: consumer already guards on membership, so its absence from a gold report costs nothing.
CHECK_ACTOR_EVIDENCE = "actor_named_in_its_own_cited_span"

CHECK_ORDER: Tuple[str, ...] = (
    CHECK_ANCHORS,
    CHECK_SOURCE_CARRIER,
    CHECK_SUPPORTED_REACTIONS,
    CHECK_ORGANISM,
    CHECK_ID_CONFLICT,
    CHECK_RAG_REINTRODUCTION,
    CHECK_CONNECTED_CORE,
    CHECK_PLACEHOLDER_IDENTITY,
    CHECK_ACTOR_EVIDENCE,
)

#: Scientific error categories, reported as separate counts. These are NOT
#: severities -- they are different kinds of wrongness, and collapsing them into
#: one "errors" number is what hid the identity problems behind the gate noise.
ERR_FALSE_REAL_IDENTIFIERS = "false_real_identifiers"
ERR_PLACEHOLDER_BACKED_PROTEINS = "placeholder_backed_proteins"

#: Renamed from ``unsupported_reactions``. This counts reactions carrying no
#: citation *field at all* -- a hygiene defect. It never established that the
#: remaining reactions were supported.
ERR_MISSING_SOURCE_CARRIER = "reactions_missing_source_carrier"

#: The real thing: a retained reaction whose chemistry matches no signature the
#: paper supports. Only counted when support auditing was applicable.
ERR_UNSUPPORTED_REACTIONS = "unsupported_reactions"

ERR_CROSS_ORGANISM_REACTIONS = "cross_organism_reactions"
ERR_ORPHANED_REFERENCES = "orphaned_references"
ERR_MISSING_PATHWAY_ANCHORS = "missing_pathway_anchors"
ERR_MISSING_SUPPORTED_REACTIONS = "missing_supported_reactions"
ERR_QUARANTINED_PROCESSES = "quarantined_processes"

ERROR_ORDER: Tuple[str, ...] = (
    ERR_FALSE_REAL_IDENTIFIERS,
    ERR_PLACEHOLDER_BACKED_PROTEINS,
    ERR_UNSUPPORTED_REACTIONS,
    ERR_MISSING_SOURCE_CARRIER,
    ERR_CROSS_ORGANISM_REACTIONS,
    ERR_ORPHANED_REFERENCES,
    ERR_MISSING_PATHWAY_ANCHORS,
    ERR_MISSING_SUPPORTED_REACTIONS,
    ERR_QUARANTINED_PROCESSES,
)

#: Buckets whose rows are real biological entities (species, subcellular
#: locations and biological states are scaffolding and are not graded).
_ENTITY_BUCKETS: Tuple[str, ...] = ("compounds", "proteins", "protein_complexes")


def _schema() -> Any:
    """The ONE participant-key and participant-slot source (C-089, Ruling 7).

    Imported LAZILY, and that is load-bearing rather than stylistic.
    ``tests/test_semantic_production_no_gold.py:159`` pins this module's import
    graph to exactly five ``t2pw`` modules, so a module-scope
    ``from t2pw.pipeline...`` would put ``t2pw.pipeline`` in the production
    scorer's import surface and break that pin. Every other ``t2pw.pipeline``
    import in this file is function-local for the same reason (``:1016``,
    ``:1357``). ``participant_schema`` is a stdlib-only leaf, so the cost is one
    ``sys.modules`` lookup after the first call.
    """

    from t2pw.pipeline import participant_schema

    return participant_schema


def _reaction_buckets() -> Tuple[str, ...]:
    """Buckets on ``payload["processes"]`` that carry chemistry.

    C-089: a derived view of ``participant_schema.PARTICIPANT_SLOTS``, so the
    bucket set and the slot map cannot drift apart. Unchanged in content --
    ``interactions`` is deliberately still absent, and whether it belongs is
    D-069 / C-091's ruling, not this card's.
    """

    return tuple(_schema().PARTICIPANT_SLOTS)


def _cofactor_names() -> FrozenSet[str]:
    """The ubiquitous-cofactor set, reused from RAG synthesis (single source).

    Imported lazily and defensively for the same reason ``rag.provenance`` does
    it: this module must stay importable in a checkout where the RAG extras are
    not installed, and an empty set degrades the connectivity check toward
    *pessimism* (fewer links, smaller core) rather than toward a false pass.
    """

    try:
        from t2pw.rag.synthesize import COFACTOR_NAMES

        return frozenset(COFACTOR_NAMES)
    except Exception:  # pragma: no cover - defensive, mirrors rag.provenance
        return frozenset()


def _dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _names(value: Any) -> List[str]:
    """Participant names from a reaction slot, which may hold strings or rows.

    C-089 replaced the five keys spelled out here with the shared list, which was
    the right move for ``element``: 416 dict-shaped participant entries in the
    committed corpus carry ``element`` and nothing else, all of them in
    ``transports.elements_with_states``.

    C-097 / F-131: the key list is
    ``participant_schema.PARTICIPANT_SCHEMA_NAME_KEYS`` -- the eight keys
    ``payload_models.py`` actually declares as carrying an ENTITY NAME -- and not
    the full ``PARTICIPANT_NAME_KEYS`` union. The union appends
    ``PARTICIPANT_LEGACY_NAME_KEYS`` (``ref``, ``id``), which appear in no
    participant model and exist for ``identity_admission``, where a reader that
    stops seeing them can only start stripping identities. This reader has the
    opposite need: it wants entity *names*, and an ``id`` is not one. Reading the
    union here was a widening C-089's charter did not authorise; corpus impact
    measured 0 (REV-089).

    ``identity_admission`` keeps consuming the full union, unchanged. First key
    present wins, exactly as before; the schema tuple's order begins ``name``,
    ``entity``, ``compound``, ``protein``, ``protein_complex`` so no shape this
    reader already understood changes hands.
    """

    out: List[str] = []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return out
    keys = _schema().PARTICIPANT_SCHEMA_NAME_KEYS
    for item in value:
        if isinstance(item, str):
            text = canonical_text(item)
        elif isinstance(item, dict):
            picked: Any = None
            for key in keys:
                candidate = item.get(key)
                if candidate:
                    picked = candidate
                    break
            text = canonical_text(picked)
        else:
            text = ""
        if text:
            out.append(text)
    return out


def _enzyme_names(reaction: Mapping[str, Any]) -> List[str]:
    """Names in the slots whose occupant acts ON the process.

    C-089: a derived view of ``participant_schema.ENZYME_ROLE_SLOTS``. Same
    three slots as
    before -- ``catalysts`` is in no model and is carried in the shared module's
    legacy tail rather than dropped here.
    """

    out: List[str] = []
    for key in _schema().ENZYME_ROLE_SLOTS:
        out.extend(_names(reaction.get(key)))
    return out


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """One check's verdict plus the findings that justify it."""

    name: str
    ok: bool
    summary: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    #: Set when the check could not be evaluated -- e.g. the admission report was
    #: never persisted, so reintroduction is unknowable. An inapplicable check is
    #: NOT a pass; it is reported separately so a missing artifact cannot be
    #: mistaken for a clean result.
    inapplicable_reason: str = ""

    @property
    def applicable(self) -> bool:
        return not self.inapplicable_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "applicable": self.applicable,
            "inapplicable_reason": self.inapplicable_reason,
            "summary": self.summary,
            "findings": self.findings,
        }


@dataclass
class CoverageDetail:
    """How much of an expected term list the payload actually covered."""

    label: str
    expected: int
    found: int
    matched: List[Dict[str, str]] = field(default_factory=list)
    missing: List[Dict[str, str]] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return (self.found / self.expected) if self.expected else 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "expected": self.expected,
            "found": self.found,
            "fraction": round(self.fraction, 4),
            "matched": self.matched,
            "missing": self.missing,
        }


@dataclass
class SemanticReport:
    """The full semantic verdict for one paper in one mode."""

    paper_id: str
    mode: str
    checks: Dict[str, CheckResult] = field(default_factory=dict)
    coverage: Dict[str, CoverageDetail] = field(default_factory=dict)
    scientific_errors: Dict[str, int] = field(default_factory=dict)
    identity_census: Dict[str, int] = field(default_factory=dict)
    graph: Dict[str, Any] = field(default_factory=dict)
    #: Reaction-level precision/recall against paper-verified signatures.
    support: Dict[str, Any] = field(default_factory=dict)
    evaluated: bool = True
    not_evaluated_reason: str = ""

    @property
    def ok(self) -> bool:
        """Every *applicable* check passed.

        An inapplicable check cannot make a paper pass. If any check could not be
        evaluated the paper is not semantically confirmed -- see
        :attr:`confirmed`, which is what the acceptance report counts.
        """

        return self.evaluated and all(c.ok for c in self.checks.values() if c.applicable)

    @property
    def confirmed(self) -> bool:
        """Passed AND every check was evaluable. The honest success criterion."""

        return self.ok and all(c.applicable for c in self.checks.values())

    @property
    def failed_checks(self) -> List[str]:
        return [n for n in CHECK_ORDER if n in self.checks and not self.checks[n].ok]

    @property
    def inapplicable_checks(self) -> List[str]:
        return [n for n in CHECK_ORDER if n in self.checks and not self.checks[n].applicable]

    def error_total(self) -> int:
        return sum(self.scientific_errors.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "mode": self.mode,
            "evaluated": self.evaluated,
            "not_evaluated_reason": self.not_evaluated_reason,
            "ok": self.ok,
            "confirmed": self.confirmed,
            "failed_checks": self.failed_checks,
            "inapplicable_checks": self.inapplicable_checks,
            "checks": {name: self.checks[name].to_dict() for name in CHECK_ORDER if name in self.checks},
            "coverage": {key: value.to_dict() for key, value in self.coverage.items()},
            "scientific_errors": {key: self.scientific_errors.get(key, 0) for key in ERROR_ORDER},
            "identity_census": dict(self.identity_census),
            "graph": dict(self.graph),
            "support": dict(self.support),
        }


# ---------------------------------------------------------------------------
# Payload accessors.
# ---------------------------------------------------------------------------
def _entities(payload: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    raw = payload.get("entities")
    if not isinstance(raw, dict):
        return {bucket: [] for bucket in _ENTITY_BUCKETS}
    return {bucket: _dicts(raw.get(bucket)) for bucket in _ENTITY_BUCKETS}


def _processes(payload: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = _reaction_buckets()
    raw = payload.get("processes")
    if not isinstance(raw, dict):
        return {bucket: [] for bucket in buckets}
    return {bucket: _dicts(raw.get(bucket)) for bucket in buckets}


def _declared_names(payload: Mapping[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for rows in _entities(payload).values():
        for row in rows:
            name = normalize_name(row.get("name"))
            if name:
                out.add(name)
            for alias in row.get("same_as") or ():
                alias_norm = normalize_name(alias)
                if alias_norm:
                    out.add(alias_norm)
    return out


def _payload_vocabulary(payload: Mapping[str, Any]) -> List[str]:
    """Every name the payload mentions -- entities, processes, participants.

    Anchors are searched against this rather than against a JSON dump: a dump
    also contains evidence quotes, and an anchor found only inside a quoted span
    from the paper proves the retriever saw the term, not that the pathway was
    built around it.
    """

    out: List[str] = []
    for rows in _entities(payload).values():
        for row in rows:
            name = canonical_text(row.get("name"))
            if name:
                out.append(name)

    # Compartments and species live in their own buckets, and a compartment IS a
    # legitimate pathway anchor -- "mitochondrial matrix" is the whole point of a
    # paper about where heme synthesis is initiated and terminated. Omitting these
    # buckets made every compartment anchor unmatchable by construction, which
    # reads as a coverage failure and is really a hole in the search.
    raw_entities = payload.get("entities")
    if isinstance(raw_entities, dict):
        for bucket in ("subcellular_locations", "species", "cell_types", "tissues"):
            for row in _dicts(raw_entities.get(bucket)):
                name = canonical_text(row.get("name"))
                if name:
                    out.append(name)
    for row in _dicts(payload.get("biological_states")):
        for key in ("name", "subcellular_location", "species"):
            text = canonical_text(row.get(key))
            if text:
                out.append(text)
    for bucket, rows in _processes(payload).items():
        for row in rows:
            name = canonical_text(row.get("name"))
            if name:
                out.append(name)
            out.extend(_names(row.get("inputs")))
            out.extend(_names(row.get("outputs")))
            out.extend(_enzyme_names(row))
    pathway = payload.get("pathway")
    if isinstance(pathway, dict):
        for key in ("name", "subject", "description"):
            text = canonical_text(pathway.get(key))
            if text:
                out.append(text)
    return out


# ---------------------------------------------------------------------------
# Coverage of a term list.
# ---------------------------------------------------------------------------
def _cover(label: str, terms: Sequence[GoldTerm], vocabulary: Sequence[str]) -> CoverageDetail:
    matched: List[Dict[str, str]] = []
    missing: List[Dict[str, str]] = []
    for term in terms:
        rule, value = term.match_any(vocabulary)
        if rule != MATCH_NONE:
            matched.append({"expected": term.name, "matched": value, "rule": rule})
        else:
            missing.append({"expected": term.name, "quote": term.quote[:240]})
    return CoverageDetail(label=label, expected=len(terms), found=len(matched), matched=matched, missing=missing)


# ---------------------------------------------------------------------------
# Check 1 -- anchors.
# ---------------------------------------------------------------------------
def _check_anchors(case: GoldCase, vocabulary: Sequence[str], coverage: Dict[str, CoverageDetail]) -> CheckResult:
    detail = _cover("pathway_anchors", case.expected_pathway_anchors, vocabulary)
    coverage["pathway_anchors"] = detail
    coverage["expected_enzymes"] = _cover("expected_enzymes", case.expected_enzymes, vocabulary)
    coverage["expected_metabolites"] = _cover("expected_metabolites", case.expected_metabolites, vocabulary)

    ok = not detail.missing
    summary = (
        f"{detail.found}/{detail.expected} requested-pathway anchors present"
        if ok
        else f"{len(detail.missing)} of {detail.expected} requested-pathway anchors absent: "
        + ", ".join(m["expected"] for m in detail.missing[:6])
    )
    return CheckResult(
        name=CHECK_ANCHORS,
        ok=ok,
        summary=summary,
        findings=[{"missing_anchor": m["expected"], "paper_says": m["quote"]} for m in detail.missing],
    )


# ---------------------------------------------------------------------------
# Check 2 -- evidence for every retained reaction.
# ---------------------------------------------------------------------------
def _has_source(row: Mapping[str, Any]) -> bool:
    """Whether a row resolves to a citable source.

    Mirrors ``rag.provenance._has_resolvable_source`` but is restated here rather
    than imported, for one reason: that function is the *additive RAG* contract
    and accepts RAG keys only. A reaction extracted straight from the seed paper
    carries a plain ``evidence`` string and no ``rag_provenance``, and calling it
    unsupported would report every non-RAG run as 100% fabricated.
    """

    rag = row.get("rag_provenance")
    if isinstance(rag, dict) and (rag.get("source_id") or rag.get("source_uri")):
        return True
    for paper in row.get("source_papers") or ():
        if isinstance(paper, dict) and (paper.get("source_id") or paper.get("uri")):
            return True
    evidence = row.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        return True
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and (item.get("source_id") or item.get("source_uri") or item.get("text")):
                return True
            if isinstance(item, str) and item.strip():
                return True
    for ref in row.get("source_refs") or ():
        if isinstance(ref, str) and ref.strip():
            return True
    return False


def _check_source_carrier(processes: Mapping[str, List[Dict[str, Any]]]) -> Tuple[CheckResult, int]:
    """Hygiene only: does each reaction carry a citation field at all?

    Deliberately does NOT claim the reaction is supported. A bare
    ``evidence: "the paper says"`` passes this, which is why the check and its
    error count were renamed to say what they measure.
    """

    findings: List[Dict[str, Any]] = []
    checked = 0
    for bucket, rows in processes.items():
        for index, row in enumerate(rows):
            checked += 1
            if _has_source(row):
                continue
            findings.append(
                {
                    "pointer": f"/processes/{bucket}/{index}",
                    "name": canonical_text(row.get("name")) or "(unnamed)",
                    "reason": "retained reaction carries no source field of any kind",
                    "inputs": _names(row.get("inputs"))[:6],
                    "outputs": _names(row.get("outputs"))[:6],
                }
            )
    ok = not findings
    summary = (
        f"all {checked} retained reaction(s) carry a source field "
        "(a carrier only -- see the supported-signature check for actual support)"
        if ok
        else f"{len(findings)} of {checked} retained reaction(s) carry no source field at all"
    )
    return CheckResult(name=CHECK_SOURCE_CARRIER, ok=ok, summary=summary, findings=findings), len(findings)


# ---------------------------------------------------------------------------
# Check 2b -- retained chemistry versus what the paper actually states.
# ---------------------------------------------------------------------------
def _side_matches(expected: Sequence[GoldTerm], observed: Sequence[str]) -> bool:
    """Every expected participant is present on this side of the reaction.

    One-directional on purpose: the retained reaction may carry *extra*
    participants (cofactors the paper did not spell out) without losing the
    match, but it may not be missing one the paper states.
    """

    return all(term.match_any(observed)[0] != MATCH_NONE for term in expected)


def _signature_matches(signature: SupportedReaction, row: Mapping[str, Any]) -> bool:
    inputs = _names(row.get("inputs"))
    outputs = _names(row.get("outputs"))
    forward = _side_matches(signature.inputs, inputs) and _side_matches(signature.outputs, outputs)
    backward = (
        signature.reversible
        and _side_matches(signature.inputs, outputs)
        and _side_matches(signature.outputs, inputs)
    )
    if not (forward or backward):
        return False
    if signature.enzyme is not None:
        # The enzyme is optional evidence: a paper-stated catalyst must not be
        # contradicted, but a reaction that omits it is still the same chemistry.
        enzymes = _enzyme_names(row)
        if enzymes and signature.enzyme.match_any(enzymes)[0] == MATCH_NONE:
            return False
    return True


def _check_supported_reactions(
    case: GoldCase,
    processes: Mapping[str, List[Dict[str, Any]]],
    paper_text: Optional[str],
) -> Tuple[CheckResult, Dict[str, Any], int, int]:
    """Reaction-level precision and recall against paper-verified signatures.

    Returns ``(check, metrics, unsupported_count, missing_count)``.

    Inapplicable -- never a silent pass -- on three routes: the case states no
    signatures; the stored paper text is missing, because a quote that cannot be
    verified is an assertion; or the signature set is a SUBSET and at least one
    retained row matched nothing, so the unsupported-reaction verdict was never
    reached. The negative-control ceiling is applied by the caller and can make
    that third route measurable again.
    """

    if not case.supported_reactions:
        return (
            CheckResult(
                name=CHECK_SUPPORTED_REACTIONS,
                ok=True,
                summary="not evaluated: this gold case states no supported reaction signatures",
                inapplicable_reason="the gold case carries no supported_reactions to audit against",
            ),
            {},
            0,
            0,
        )

    folded_text = fold_for_quote(paper_text) if paper_text else ""
    if not folded_text:
        return (
            CheckResult(
                name=CHECK_SUPPORTED_REACTIONS,
                ok=True,
                summary="not evaluated: the stored paper text was not available for quote verification",
                inapplicable_reason=(
                    "01_source_text.txt was not found for this paper, so no gold quote could be "
                    "verified; an unverified quote is an assertion, not evidence"
                ),
            ),
            {},
            0,
            0,
        )

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for bucket, bucket_rows in processes.items():
        for index, row in enumerate(bucket_rows):
            rows.append((f"/processes/{bucket}/{index}", row))

    findings: List[Dict[str, Any]] = []
    unverifiable: List[Dict[str, str]] = []
    matched_signatures: List[Dict[str, Any]] = []
    missing_signatures: List[Dict[str, Any]] = []
    matched_pointers: Set[str] = set()

    for signature in case.supported_reactions:
        quote_ok = fold_for_quote(signature.quote) in folded_text
        if not quote_ok:
            # A gold-set defect, not a pipeline defect. Reported loudly and the
            # signature is excluded from scoring rather than held against the run.
            unverifiable.append(
                {
                    "signature": signature.describe(),
                    "quote": signature.quote[:160],
                    "reason": "this gold quote does not occur in the stored paper text",
                }
            )
            continue
        hits = [pointer for pointer, row in rows if _signature_matches(signature, row)]
        if hits:
            matched_pointers.update(hits)
            matched_signatures.append(
                {
                    "signature": signature.describe(),
                    "quote_hash": signature.quote_hash(),
                    "matched_by": hits[:6],
                }
            )
        else:
            missing_signatures.append(
                {
                    "signature": signature.describe(),
                    "quote": signature.quote[:200],
                    "quote_hash": signature.quote_hash(),
                    "reason": "the paper states this reaction and the payload does not contain it",
                }
            )

    scored = [s for s in case.supported_reactions if fold_for_quote(s.quote) in folded_text]
    unsupported_rows = [pointer for pointer, _row in rows if pointer not in matched_pointers]

    true_positives = len(matched_pointers)
    false_positives = len(unsupported_rows)
    false_negatives = len(missing_signatures)
    precision = true_positives / (true_positives + false_positives) if rows else None
    recall = len(matched_signatures) / len(scored) if scored else None

    for pointer in unsupported_rows[:40]:
        row = dict(rows[[p for p, _ in rows].index(pointer)][1])
        findings.append(
            {
                "pointer": pointer,
                "name": canonical_text(row.get("name")) or "(unnamed)",
                "kind": "unsupported_retained_reaction",
                "inputs": _names(row.get("inputs"))[:6],
                "outputs": _names(row.get("outputs"))[:6],
                "reason": "this retained reaction matches no signature the paper supports",
            }
        )
    for entry in missing_signatures:
        findings.append({**entry, "kind": "missing_supported_reaction"})
    for entry in unverifiable:
        findings.append({**entry, "kind": "gold_quote_not_found_in_paper"})

    complete = case.supported_reactions_complete

    # Was the *unsupported-reaction verdict* actually reached? It is reached in
    # exactly two situations:
    #
    #   * the signature set is exhaustive, so an unmatched row is by definition
    #     unsupported and ``false_positives`` counts it; or
    #   * every retained row matched a quote-verified signature, so "zero
    #     unsupported" is a measurement rather than an assumption -- a row that
    #     matches hand-read chemistry is supported whether or not the set is
    #     exhaustive.
    #
    # With a SUBSET set and at least one unmatched row it is not reached at all:
    # those findings are deleted below and the returned count is a hard zero. That
    # zero used to be reported as a clean result, which collapses "not evaluated"
    # into "passed" -- forbidden by PRODUCT_CONTRACT 11 -- and is what left
    # acceptance priority 2 structurally unable to return a non-zero count on
    # eight of the ten pinned papers. The suppression below is correct and stays;
    # what changes is that the check now SAYS it did not evaluate.
    unsupported_verdict_evaluated = bool(complete or not unsupported_rows)

    metrics = {
        "signature_set_complete": complete,
        "unsupported_verdict_evaluated": unsupported_verdict_evaluated,
        "signatures_declared": len(case.supported_reactions),
        "signatures_quote_verified": len(scored),
        "signatures_unverifiable": len(unverifiable),
        "signatures_matched": len(matched_signatures),
        "retained_reactions": len(rows),
        "true_positives": true_positives,
        "unattributed_reactions": false_positives,
        "false_positives": false_positives if complete else None,
        "false_negatives": false_negatives,
        "attribution_rate": round(precision, 4) if precision is not None else None,
        "precision": round(precision, 4) if (complete and precision is not None) else None,
        "recall": round(recall, 4) if recall is not None else None,
        "matched": matched_signatures,
        "missing": missing_signatures,
        "unverifiable_quotes": unverifiable,
    }

    # Unmatched rows only become *errors* when the signature set is exhaustive.
    # Otherwise they are unattributed: the subset was hand-read, and a
    # cross-paper RAG addition cannot match a seed-paper signature by
    # construction. Calling those hallucinations would have reported 227
    # fabricated reactions in a run that produced far fewer.
    if not complete:
        findings = [f for f in findings if f.get("kind") != "unsupported_retained_reaction"]

    ok = not missing_signatures and not unverifiable and (not complete or not unsupported_rows)
    label = "precision" if complete else "attribution rate (signature set is a SUBSET)"
    summary = (
        f"{label} {true_positives}/{true_positives + false_positives}"
        + (f" ({precision * 100:.0f}%)" if precision is not None else "")
        + f", recall {len(matched_signatures)}/{len(scored)}"
        + (f" ({recall * 100:.0f}%)" if recall is not None else "")
    )
    if unverifiable:
        summary += f"; {len(unverifiable)} GOLD QUOTE(S) NOT FOUND IN THE PAPER"

    # The attribution rate and the recall above ARE measured and stay in the
    # summary; only the unsupported-reaction verdict is withheld.
    inapplicable_reason = ""
    if not unsupported_verdict_evaluated:
        summary += "; UNSUPPORTED-REACTION VERDICT NOT EVALUATED"
        inapplicable_reason = (
            f"the unsupported-reaction verdict was NOT EVALUATED: this gold case's signature "
            f"set is a SUBSET (supported_reactions_complete is false), so the "
            f"{len(unsupported_rows)} retained reaction(s) that matched no signature cannot be "
            "judged either way -- an unmatched row may be invented, may be a step the paper "
            "states in a form no signature was written for, or may be a legitimate cross-paper "
            "RAG addition, and those are indistinguishable from the signature list alone. The "
            "attribution rate and recall in the summary ARE measured. Only an exhaustive "
            "supported_reactions list, or a max_retained_reactions ceiling, can make this "
            "question answerable for this paper."
        )

    return (
        CheckResult(
            name=CHECK_SUPPORTED_REACTIONS,
            ok=ok,
            summary=summary,
            findings=findings,
            inapplicable_reason=inapplicable_reason,
        ),
        metrics,
        false_positives if complete else 0,
        false_negatives,
    )


# ---------------------------------------------------------------------------
# Check 3 -- organism compatibility.
# ---------------------------------------------------------------------------
def _organism_of(row: Mapping[str, Any]) -> str:
    for key in ("organism", "species", "target_organism"):
        text = canonical_text(row.get(key))
        if text:
            return text
    rag = row.get("rag_provenance")
    if isinstance(rag, dict):
        text = canonical_text(rag.get("organism"))
        if text:
            return text
    meta = row.get("mapping_meta")
    if isinstance(meta, dict):
        for key in ("species", "target_organism"):
            text = canonical_text(meta.get(key))
            if text:
                return text
    return ""


def _organism_conflicts(observed: str, case: GoldCase) -> str:
    """Why ``observed`` is incompatible, or ``""``.

    Only a *forbidden* organism is a violation. An absent organism is not: the
    payload routinely leaves it blank and ``pipeline.propagate_context_organism``
    fills it later, so counting blanks here would report a cross-organism error
    for every early-stage payload. Blanks are surfaced by the identity census
    instead, where they belong.
    """

    if not observed:
        return ""
    norm = normalize_name(observed)
    for forbidden in case.forbidden_organisms:
        forbidden_norm = normalize_name(forbidden)
        if not forbidden_norm:
            continue
        if norm == forbidden_norm or norm.startswith(forbidden_norm + " ") or forbidden_norm.startswith(norm + " "):
            return f"attributed to '{observed}', which this case forbids"
    return ""


def _check_organism(
    case: GoldCase,
    processes: Mapping[str, List[Dict[str, Any]]],
    entities: Mapping[str, List[Dict[str, Any]]],
) -> Tuple[CheckResult, int]:
    findings: List[Dict[str, Any]] = []
    for bucket, rows in processes.items():
        for index, row in enumerate(rows):
            reason = _organism_conflicts(_organism_of(row), case)
            if reason:
                findings.append(
                    {
                        "pointer": f"/processes/{bucket}/{index}",
                        "name": canonical_text(row.get("name")) or "(unnamed)",
                        "reason": reason,
                    }
                )
    reaction_violations = len(findings)

    # Proteins are reported too, but do not count as cross-organism *reactions*.
    # A placeholder legitimately carries the PathBank sentinel species, and
    # counting it as a cross-organism reaction would double-charge the same
    # defect that `placeholder_backed_proteins` already counts.
    for index, row in enumerate(entities.get("proteins", [])):
        reason = _organism_conflicts(_organism_of(row), case)
        if reason:
            findings.append(
                {
                    "pointer": f"/entities/proteins/{index}",
                    "name": canonical_text(row.get("name")) or "(unnamed)",
                    "reason": reason,
                    "counts_as_reaction_violation": False,
                }
            )

    ok = not findings
    summary = (
        f"no retained reaction is attributed to a forbidden organism (requested: {case.requested_organism})"
        if ok
        else f"{reaction_violations} reaction(s) and {len(findings) - reaction_violations} protein(s) "
        f"attributed to a forbidden organism"
    )
    return CheckResult(name=CHECK_ORGANISM, ok=ok, summary=summary, findings=findings), reaction_violations


# ---------------------------------------------------------------------------
# Check 4 -- real-ID / name conflicts.
# ---------------------------------------------------------------------------
def _external_ids(row: Mapping[str, Any]) -> Dict[str, str]:
    """Every real external accession the row carries, by namespace."""

    out: Dict[str, str] = {}
    containers = [row]
    for key in ("mapped_ids", "ids", "mapping_meta"):
        value = row.get(key)
        if isinstance(value, dict):
            containers.append(value)
    for container in containers:
        for key in ("uniprot", "uniprot_id", "drugbank", "drugbank_id", "hmdb", "kegg", "chebi", "pubchem"):
            value = canonical_text(container.get(key))
            if not value or value.casefold() in ("unknown", "unmapped", "none", "n/a"):
                continue
            out.setdefault(key.replace("_id", ""), value)
    return out


def accession_claimed_across_kinds(claimants: Iterable[Tuple[str, str]]) -> bool:
    """Do two claimants of one accession differ in **kind** AND in **name**?

    THE SINGLE DEFINITION of the concept for both semantic seams. C-080 MOVED it
    here out of :func:`_check_id_conflicts`, unchanged, so that
    ``semantic_production._audit_entities`` -- the copy that gates real runs --
    reads the same predicate the acceptance scorer does instead of carrying a
    third. Its body is byte-for-byte the pipeline's own in
    ``mapping.identity_admission.find_kind_conflicting_accessions``, whose rule
    this is the ``bench`` side of; that function is not called directly because it
    folds accessions with ``accession_key`` while these two seams key on
    ``_external_ids`` plus ``casefold``, and grouping the rows differently would
    change which claims are compared.

    ``claimants`` is ``(kind class, normalized name)`` per claiming row -- caller's
    choice of kind vocabulary, since the only requirement is that both seams read
    it from ``identity_admission.entity_kind_class``.

    Both halves are load-bearing, and neither is new here:

    * **Different kinds** is the incompatibility. One accession cannot denote a
      protein and a metabolite at once. Same-kind agreement is **D-035 clause 3c**
      evidence that the rows are the same entity -- ``EntB``/``holo-EntB``, or
      ``EntE``/``enterobactin synthase`` -- and is left completely alone.
    * **Different names** keeps a routing artefact from reading as a type error.
      One entity resolved into both ``compounds`` and ``proteins`` is written
      twice, not a protein masquerading as a metabolite.

    Duplicate claimants collapse and the order of comparison is fixed by
    ``sorted``, so the verdict does not depend on payload order.
    """

    kinds = sorted(set(claimants))
    return any(
        left[0] != right[0] and left[1] != right[1]
        for position, left in enumerate(kinds)
        for right in kinds[position + 1:]
    )


def _check_id_conflicts(
    case: GoldCase,
    entities: Mapping[str, List[Dict[str, Any]]],
) -> Tuple[CheckResult, int]:
    """Three distinct forgeries, all of which end as a wrong pathway that passes.

    1. a **forbidden identifier carrying a real accession** -- the gold set says
       this string is not a distinct identity, and the payload has given it one;
    2. a **placeholder posing as a real mapping** -- delegated to
       ``entity_identity.placeholder_claims_real_identity``, the one gate that
       can see a fabricated accession on an Unknown-backed row;
    3. one accession claimed by two entities **of incompatible kind** -- a
       name/ID collision, which silently fuses a protein and a metabolite in the
       exported graph.

    C-076 / F-102 narrowed (3), and exempted one case from (1). Both rules used
    to fire on a *name* difference alone. That is the predicate C-073's review
    rejected in the pipeline, for contradicting **D-035 clause 3c**: a matching
    stable external identifier is *proof* that two differently-named rows are
    the same entity. The pipeline was corrected to a cross-kind rule; the scorer
    was not, so it went on demoting real legs with the rejected rule.

    * (3) now needs an **incompatible kind** as well as a different name.
      Protein-ish versus compound-ish, read from
      ``identity_admission.entity_kind_class`` -- the pipeline's own definition,
      imported rather than restated so the two cannot drift apart. ``EntB`` and
      ``holo-EntB``, or ``EntE`` and ``enterobactin synthase``, sharing one
      UniProt accession is agreement. ``drugbank:DB00114`` on both ``ALAS2``
      (protein) and ``Pyridoxal 5'-phosphate`` (compound) is still a collision,
      and that is the one true defect C-073 fixed upstream.
    * (1) exempts a gold entry of kind ``modification_state`` -- a holo/apo or
      otherwise post-translationally modified form of ONE polypeptide -- but
      **only when every accession it carries is also claimed by a real,
      differently-named parent row of the same kind**. Sharing the parent's
      accession is the legitimate case the ruling protects. Inventing an
      accession of its own is still a fabricated identity, and a
      ``strain_or_construct`` -- the ``R196A`` point mutant is a *different
      polypeptide sequence* -- is never exempt.

    Deliberately unchanged: an exempt row is silent rather than reported, because
    a finding of any kind would keep ``ok`` false and so keep gating the leg,
    which is exactly what the ruling forbids; a forbidden identifier carrying NO
    accession still reports ``forbidden_identity_present_unmapped``; and
    ``false_real`` still counts (1) and (2) only -- the collision branch has
    never incremented it and still does not.
    """

    from t2pw.mapping.identity_admission import entity_kind_class
    from t2pw.pipeline.entity_identity import placeholder_claims_real_identity

    #: Gold ``kind`` for a modification state of one polypeptide (holo/apo,
    #: phosphopantetheinylated, metal-loaded). Function-local: this is the only
    #: reader, and C-076's boundary owns nothing else in this module.
    modification_state = "modification_state"

    findings: List[Dict[str, Any]] = []
    false_real = 0
    by_accession: Dict[Tuple[str, str], List[str]] = {}
    #: Same keys as ``by_accession``, carrying what the verdicts below need that a
    #: display name cannot answer: ``(kind class, normalized name, is forbidden)``.
    claimants: Dict[Tuple[str, str], List[Tuple[str, str, bool]]] = {}

    # -- pass 1: who claims which accession ---------------------------------
    # Two of the three rules are about a row's CO-CLAIMANTS, so no verdict can be
    # reached before every row has been seen. Hence two passes over the same
    # rows: ``_external_ids`` and ``forbidden_match`` are both pure, and emitting
    # every finding in the second pass keeps them in payload order, exactly as a
    # single pass did.
    for bucket in _ENTITY_BUCKETS:
        kind_class = entity_kind_class(bucket)
        for index, row in enumerate(entities.get(bucket, [])):
            name = canonical_text(row.get("name"))
            holder = name or f"/entities/{bucket}/{index}"
            is_forbidden = case.forbidden_match(name) is not None
            for namespace, accession in _external_ids(row).items():
                key = (namespace, accession.casefold())
                by_accession.setdefault(key, []).append(holder)
                claimants.setdefault(key, []).append((kind_class, normalize_name(holder), is_forbidden))

    def _shares_a_parent_accession(kind_class: str, holder: str, ids: Mapping[str, str]) -> bool:
        """Is EVERY accession on this row also claimed by a real parent row?

        "Real parent" is same kind, a different normalized name, and **not itself
        forbidden** -- so two modification states cannot vouch for each other's
        invented accession, and one that carries an accession nothing else claims
        is not sharing anything.
        """

        if not ids:
            return False
        norm = normalize_name(holder)
        for namespace, accession in ids.items():
            if not any(
                other_kind == kind_class and other_name != norm and not other_forbidden
                for other_kind, other_name, other_forbidden in claimants.get((namespace, accession.casefold()), ())
            ):
                return False
        return True

    # -- pass 2: one verdict per row, in payload order ----------------------
    for bucket in _ENTITY_BUCKETS:
        kind_class = entity_kind_class(bucket)
        for index, row in enumerate(entities.get(bucket, [])):
            name = canonical_text(row.get("name"))
            pointer = f"/entities/{bucket}/{index}"
            ids = _external_ids(row)

            forbidden = case.forbidden_match(name)
            if forbidden is not None:
                if ids:
                    exempt = forbidden.kind == modification_state and _shares_a_parent_accession(
                        kind_class, name or pointer, ids
                    )
                    if not exempt:
                        false_real += 1
                        findings.append(
                            {
                                "pointer": pointer,
                                "name": name,
                                "kind": "false_real_identifier",
                                "forbidden_kind": forbidden.kind,
                                "identifiers": ids,
                                "reason": forbidden.reason
                                or f"'{name}' is not a real distinct identity ({forbidden.kind}) but carries {ids}",
                            }
                        )
                else:
                    findings.append(
                        {
                            "pointer": pointer,
                            "name": name,
                            "kind": "forbidden_identity_present_unmapped",
                            "forbidden_kind": forbidden.kind,
                            "reason": forbidden.reason
                            or f"'{name}' is not a real distinct identity ({forbidden.kind})",
                        }
                    )

            claim = placeholder_claims_real_identity(row)
            if claim:
                false_real += 1
                findings.append(
                    {
                        "pointer": pointer,
                        "name": name,
                        "kind": "placeholder_claims_real_identity",
                        "rule": claim,
                        "identifiers": ids,
                        "reason": f"placeholder row is posing as a real mapping ({claim})",
                    }
                )

    for (namespace, accession), holders in sorted(by_accession.items()):
        distinct = sorted({normalize_name(h) for h in holders if h})
        if len(distinct) < 2:
            continue
        # D-035 clause 3c. A shared stable accession is PROOF that two
        # differently-named rows are the same entity, so a name difference alone
        # is agreement, not collision. Only an incompatible KIND turns it back
        # into one: an accession cannot denote a protein and a metabolite at
        # once. Both halves are load-bearing, exactly as they are in
        # ``identity_admission.find_kind_conflicting_accessions``, whose rule
        # this is now the scorer's side of. C-080 MOVED the comparison itself up
        # to :func:`accession_claimed_across_kinds` so the production release gate
        # can call the same definition; the rule is unchanged.
        cross_kind = accession_claimed_across_kinds(
            (kind, folded) for kind, folded, _ in claimants.get((namespace, accession), ())
        )
        if not cross_kind:
            continue
        findings.append(
            {
                "kind": "accession_claimed_by_multiple_entities",
                "namespace": namespace,
                "identifier": accession,
                "entities": sorted(set(holders)),
                "reason": (
                    f"{namespace}:{accession} is claimed by {len(distinct)} differently-named "
                    "entities of incompatible kind (protein-ish vs compound-ish)"
                ),
            }
        )

    ok = not findings
    summary = (
        "no real-identifier or name conflict"
        if ok
        else f"{false_real} false real identifier(s); {len(findings)} identity conflict finding(s) total"
    )
    return CheckResult(name=CHECK_ID_CONFLICT, ok=ok, summary=summary, findings=findings), false_real


# ---------------------------------------------------------------------------
# Check 5 -- rejected RAG reactions must not reappear.
# ---------------------------------------------------------------------------
def _claim_key(inputs: Iterable[Any], outputs: Iterable[Any], enzymes: Iterable[Any], reversible: Any) -> str:
    """An order-independent claim key, mirroring ``admission.claim_identity``.

    Restated with this module's normalizer rather than importing the private
    ``admission._fold``: both sides of the comparison are folded here, so
    internal consistency is what matters, and depending on a private helper would
    make the benchmark break on an unrelated refactor.
    """

    ins = "+".join(sorted(filter(None, (normalize_name(v) for v in _names(list(inputs))))))
    outs = "+".join(sorted(filter(None, (normalize_name(v) for v in _names(list(outputs))))))
    enz = "+".join(sorted(filter(None, (normalize_name(v) for v in _names(list(enzymes))))))
    return f"{ins}=>{outs}|{enz}|{int(bool(reversible))}"


def _check_rag_reintroduction(
    processes: Mapping[str, List[Dict[str, Any]]],
    admission: Optional[Mapping[str, Any]],
) -> CheckResult:
    if not isinstance(admission, dict) or "rejected" not in admission:
        return CheckResult(
            name=CHECK_RAG_REINTRODUCTION,
            ok=True,
            summary="not evaluated: no RAG admission report was persisted for this run",
            inapplicable_reason=(
                "AdmissionReport.rejected is not written to disk by the batch driver; "
                "without it a reintroduced claim is undetectable"
            ),
        )

    rejected = _dicts(admission.get("rejected"))
    index: Dict[str, Dict[str, Any]] = {}
    for row in rejected:
        key = _claim_key(row.get("inputs") or (), row.get("outputs") or (), row.get("enzymes") or (), row.get("reversible"))
        if key and key != "=>||0":
            index.setdefault(key, row)

    findings: List[Dict[str, Any]] = []
    for bucket, rows in processes.items():
        for position, row in enumerate(rows):
            key = _claim_key(
                row.get("inputs") or (),
                row.get("outputs") or (),
                _enzyme_names(row),
                row.get("reversible"),
            )
            hit = index.get(key)
            if hit is None:
                continue
            findings.append(
                {
                    "pointer": f"/processes/{bucket}/{position}",
                    "name": canonical_text(row.get("name")) or "(unnamed)",
                    "claim_key": key,
                    "rejected_for": list(hit.get("reasons") or ()),
                    "gap_id": canonical_text(hit.get("gap_id")),
                    "reason": "a claim the admission gate refused is present in the exported payload",
                }
            )

    ok = not findings
    summary = (
        f"none of the {len(index)} rejected RAG claim(s) reappear in the payload"
        if ok
        else f"{len(findings)} rejected RAG claim(s) were reintroduced"
    )
    return CheckResult(name=CHECK_RAG_REINTRODUCTION, ok=ok, summary=summary, findings=findings)


# ---------------------------------------------------------------------------
# Check 6 -- minimum connected core.
# ---------------------------------------------------------------------------
def _connected_core(
    processes: Mapping[str, List[Dict[str, Any]]],
    cofactors: FrozenSet[str],
) -> Dict[str, Any]:
    """Largest set of reactions connected through shared non-cofactor metabolites."""

    nodes: List[Tuple[str, Set[str]]] = []
    for bucket, rows in processes.items():
        for index, row in enumerate(rows):
            metabolites = {
                normalize_name(name)
                for name in (*_names(row.get("inputs")), *_names(row.get("outputs")))
            }
            metabolites = {
                m for m in metabolites if m and m not in cofactors and canonical_text(m).casefold() not in cofactors
            }
            nodes.append((f"/processes/{bucket}/{index}", metabolites))

    adjacency: Dict[int, Set[int]] = {i: set() for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i][1] & nodes[j][1]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    seen: Set[int] = set()
    components: List[List[int]] = []
    for start in range(len(nodes)):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(group))

    components.sort(key=len, reverse=True)
    largest = components[0] if components else []
    return {
        "n_reactions": len(nodes),
        "n_components": len(components),
        "largest_core_size": len(largest),
        "largest_core_pointers": [nodes[i][0] for i in largest][:40],
        "isolated_reactions": sum(1 for c in components if len(c) == 1),
        "cofactors_excluded": len(cofactors),
    }


def _check_connected_core(case: GoldCase, graph: Mapping[str, Any]) -> CheckResult:
    """Floor for a real case, ceiling for a negative control.

    Both directions are the same question asked from opposite ends: does the
    payload contain the amount of chemistry the paper actually supports? A floor
    catches under-extraction, and it is the only half that a hallucinating
    pipeline can satisfy for free. The ceiling -- set only on negative controls,
    where the correct answer is zero reactions -- is the half that cannot be
    satisfied by inventing anything.
    """

    largest = int(graph.get("largest_core_size", 0))
    retained = int(graph.get("n_reactions", 0))
    needed = case.min_connected_reactions
    ceiling = case.max_retained_reactions
    findings: List[Dict[str, Any]] = []

    over_ceiling = ceiling is not None and retained > ceiling
    under_floor = largest < needed

    if over_ceiling:
        findings.append(
            {
                "ceiling": ceiling,
                "retained": retained,
                "reason": (
                    "this paper contains no reaction of the requested pathway, so every retained "
                    "reaction is unsupported by the source"
                ),
                "examples": list(graph.get("largest_core_pointers") or ())[:10],
            }
        )
    if under_floor:
        findings.append(
            {
                "required": needed,
                "observed": largest,
                "n_reactions": retained,
                "n_components": graph.get("n_components", 0),
                "isolated_reactions": graph.get("isolated_reactions", 0),
                "reason": "reactions do not join into one chain through shared non-cofactor metabolites",
            }
        )

    ok = not findings
    if over_ceiling:
        summary = (
            f"negative control: {retained} reaction(s) retained where the paper supports {ceiling}"
        )
    elif under_floor:
        summary = (
            f"largest chemically connected core is {largest} reaction(s), below the {needed} this paper "
            f"supports ({retained} reaction(s) in {graph.get('n_components', 0)} component(s))"
        )
    elif ceiling == 0:
        summary = "negative control: no reaction was retained, which is the correct outcome"
    else:
        summary = f"largest chemically connected core is {largest} reaction(s); this case requires {needed}"

    return CheckResult(name=CHECK_CONNECTED_CORE, ok=ok, summary=summary, findings=findings)


# ---------------------------------------------------------------------------
# Check 7 -- placeholder identities distinguished from real ones.
# ---------------------------------------------------------------------------
def _check_placeholder_identity(
    case: GoldCase,
    entities: Mapping[str, List[Dict[str, Any]]],
) -> Tuple[CheckResult, Dict[str, int], int]:
    from t2pw.pipeline.entity_identity import (
        IDENTITY_PLACEHOLDER,
        IDENTITY_UNRESOLVED,
        IDENTITY_VERIFIED,
        identity_status,
        is_pathbank_unknown_protein,
        placeholder_claims_real_identity,
    )

    census = {
        IDENTITY_VERIFIED: 0,
        IDENTITY_PLACEHOLDER: 0,
        IDENTITY_UNRESOLVED: 0,
        "pathbank_unknown_sentinel": 0,
        "proteins_total": 0,
        "proteins_with_organism": 0,
        "compounds_total": len(entities.get("compounds", [])),
    }
    findings: List[Dict[str, Any]] = []
    placeholder_backed = 0

    for bucket in ("proteins", "protein_complexes"):
        for index, row in enumerate(entities.get(bucket, [])):
            census["proteins_total"] += 1
            if _organism_of(row):
                census["proteins_with_organism"] += 1
            status = identity_status(row) or IDENTITY_UNRESOLVED
            census[status] = census.get(status, 0) + 1
            if is_pathbank_unknown_protein(row):
                census["pathbank_unknown_sentinel"] += 1
            if status != IDENTITY_PLACEHOLDER:
                continue
            placeholder_backed += 1
            name = canonical_text(row.get("name"))
            pointer = f"/entities/{bucket}/{index}"
            claim = placeholder_claims_real_identity(row)
            if claim:
                findings.append(
                    {
                        "pointer": pointer,
                        "name": name,
                        "kind": "placeholder_not_distinguished",
                        "rule": claim,
                        "reason": f"placeholder is indistinguishable from a real mapping ({claim})",
                    }
                )
            elif not case.unknown_backed_proteins_acceptable:
                findings.append(
                    {
                        "pointer": pointer,
                        "name": name,
                        "kind": "unknown_backed_protein_not_acceptable",
                        "reason": (
                            "this case expects every protein to resolve to a real identity; "
                            f"'{name}' is Unknown-backed"
                        ),
                    }
                )

    # The summary is CONSTRUCTED FROM THE THREE COUNTS, never asserted.
    #
    # The previous wording said "every protein carries a real external identity"
    # whenever `placeholder_backed == 0`. That is a different claim: zero
    # placeholders with 222 *unresolved* rows means nothing was resolved at all,
    # not that everything resolved. On the 2026-07-28 artifacts -- pre-mapping
    # payloads with no accessions of any kind -- it read as a clean bill of health
    # for a file that cannot answer the question.
    verified = int(census.get(IDENTITY_VERIFIED, 0) or 0)
    unresolved = int(census.get(IDENTITY_UNRESOLVED, 0) or 0)
    total = int(census.get("proteins_total", 0) or 0)
    breakdown = (
        f"{verified} verified / {placeholder_backed} Unknown-backed placeholder / "
        f"{unresolved} unresolved, of {total} protein row(s)"
    )

    ok = not findings
    if not ok:
        summary = (
            f"{len(findings)} placeholder identity problem(s); {breakdown}"
        )
    elif total == 0:
        summary = "no protein rows to audit"
    elif placeholder_backed and case.unknown_backed_proteins_acceptable:
        summary = f"{breakdown}; placeholders are correctly marked and this case accepts them"
    elif unresolved:
        summary = (
            f"{breakdown}; no placeholder forged an identity, but {unresolved} row(s) carry no "
            "identity at all -- this is NOT 'every protein resolved'"
        )
    else:
        summary = f"{breakdown}; every protein row carries a real external identity"
    return (
        CheckResult(name=CHECK_PLACEHOLDER_IDENTITY, ok=ok, summary=summary, findings=findings),
        census,
        placeholder_backed,
    )


# ---------------------------------------------------------------------------
# Referential integrity (feeds the orphaned-references error count).
# ---------------------------------------------------------------------------
def _orphaned_references(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Names a process references that no entity bucket declares.

    This is acceptance priority 3. A process pointing at an undeclared name means
    the exported graph has an edge to nothing -- the importer either drops the
    reaction or renders a dangling node, and either way the pathway shown is not
    the pathway that was validated.

    C-089 / F-125: the slot list is ``participant_schema.participant_slots``,
    per bucket, not
    one uniform ``inputs``/``outputs``/enzyme list applied to all three. Before
    this, the absolute gate for referential integrity read **none** of ``cargo``,
    ``transporters`` or ``elements_with_states`` -- that is, every participant
    slot a ``TransportModel`` or ``ReactionCoupledTransportModel`` actually has.
    Measured over the committed corpus it counted **3** orphans where a
    schema-complete reader counts **6**; the three it could not see include a
    leaked JSON pointer ``/entities/proteins/0`` sitting in a transporter name
    slot. The reader only widened: every slot it read before it still reads, via
    ``participant_schema``'s legacy tail.
    """

    schema = _schema()
    acts_on_slots = frozenset(schema.ENZYME_ROLE_SLOTS)
    declared = _declared_names(payload)
    findings: List[Dict[str, Any]] = []
    for bucket, rows in _processes(payload).items():
        for index, row in enumerate(rows):
            for slot in schema.participant_slots(bucket):
                acts_on = slot in acts_on_slots
                for name in _names(row.get(slot)):
                    if normalize_name(name) in declared:
                        continue
                    findings.append(
                        {
                            # An enzyme-role orphan has always been reported at
                            # the bucket's ``enzymes`` pointer whichever of the
                            # three slots held it. Kept: correcting the pointer
                            # would move an observable this card was not sent to
                            # move.
                            "pointer": "/processes/{0}/{1}/{2}".format(
                                bucket, index, "enzymes" if acts_on else slot),
                            "name": name,
                            "reason": (
                                "enzyme is not declared in any entity bucket"
                                if acts_on
                                else "participant is not declared in any entity bucket"
                            ),
                        }
                    )
    return findings


def _quarantined_count(payload: Mapping[str, Any], quarantine_report: Optional[Mapping[str, Any]]) -> int:
    total = 0
    if isinstance(quarantine_report, dict):
        quarantined = quarantine_report.get("quarantined")
        if isinstance(quarantined, list):
            total = max(total, len(quarantined))
        counts = quarantine_report.get("counts")
        if isinstance(counts, dict):
            try:
                total = max(total, int(counts.get("quarantined", 0) or 0))
            except (TypeError, ValueError):
                pass
    locked = payload.get("quarantined_locked_reactions")
    if isinstance(locked, list):
        total = max(total, len(locked))
    return total


# ---------------------------------------------------------------------------
# The entry point.
# ---------------------------------------------------------------------------
def validate_semantic_coverage(
    case: GoldCase,
    payload: Optional[Mapping[str, Any]],
    *,
    mode: str = "",
    admission: Optional[Mapping[str, Any]] = None,
    quarantine_report: Optional[Mapping[str, Any]] = None,
    paper_text: Optional[str] = None,
) -> SemanticReport:
    """Score one payload against one gold case. Deterministic, offline, no LLM.

    ``payload`` may be ``None`` -- a run that died before producing one is
    reported as *not evaluated* rather than as a semantic failure, because those
    are different facts and only the first one is the extractor's problem.
    """

    report = SemanticReport(paper_id=case.paper_id, mode=mode)
    if not isinstance(payload, dict) or not payload:
        report.evaluated = False
        report.not_evaluated_reason = "the run produced no payload to score"
        report.scientific_errors = {key: 0 for key in ERROR_ORDER}
        return report

    entities = _entities(payload)
    processes = _processes(payload)
    vocabulary = _payload_vocabulary(payload)

    anchors = _check_anchors(case, vocabulary, report.coverage)
    carrier, missing_carrier = _check_source_carrier(processes)
    supported, support_metrics, unsupported, missing_supported = _check_supported_reactions(
        case, processes, paper_text
    )
    report.support = support_metrics
    organism, cross_organism = _check_organism(case, processes, entities)
    conflicts, false_real = _check_id_conflicts(case, entities)
    reintroduction = _check_rag_reintroduction(processes, admission)

    report.graph = _connected_core(processes, _cofactor_names())
    connected = _check_connected_core(case, report.graph)

    placeholder, census, placeholder_backed = _check_placeholder_identity(case, entities)
    report.identity_census = census

    orphans = _orphaned_references(payload)

    # Reactions beyond the gold ceiling are unsupported *by the requested
    # pathway*, however well-quoted they are. The carrier check asks "does this
    # reaction cite something?" and a reaction lifted from the paper's unrelated
    # subject matter answers yes; only the gold case knows how much of the
    # requested pathway the paper actually contains. On a negative control the
    # ceiling is 0, so this reduces to "every retained reaction is unsupported".
    # The ceiling is kept alongside signature matching, not replaced by it: it is
    # the only rule that works for a paper with no supportable chemistry at all.
    if case.max_retained_reactions is not None:
        over = int(report.graph.get("n_reactions", 0)) - case.max_retained_reactions
        unsupported = max(unsupported, max(0, over))
        # The ceiling answers the same question by a route that needs no signature
        # set at all, so a case carrying one HAS an evaluated unsupported-reaction
        # verdict even where the signature check could not reach one. This is what
        # keeps the negative controls counting.
        report.support["unsupported_verdict_evaluated"] = True
    # The two early returns above yield empty metrics; an absent key there means
    # the question was never asked, not that the answer was zero.
    report.support.setdefault("unsupported_verdict_evaluated", False)

    for result in (
        anchors,
        carrier,
        supported,
        organism,
        conflicts,
        reintroduction,
        connected,
        placeholder,
    ):
        report.checks[result.name] = result

    report.scientific_errors = {
        ERR_FALSE_REAL_IDENTIFIERS: false_real,
        ERR_PLACEHOLDER_BACKED_PROTEINS: placeholder_backed,
        ERR_UNSUPPORTED_REACTIONS: unsupported,
        ERR_MISSING_SOURCE_CARRIER: missing_carrier,
        ERR_CROSS_ORGANISM_REACTIONS: cross_organism,
        ERR_ORPHANED_REFERENCES: len(orphans),
        ERR_MISSING_PATHWAY_ANCHORS: len(report.coverage["pathway_anchors"].missing),
        ERR_MISSING_SUPPORTED_REACTIONS: missing_supported,
        ERR_QUARANTINED_PROCESSES: _quarantined_count(payload, quarantine_report),
    }
    report.graph["orphaned_reference_examples"] = orphans[:20]
    return report


__all__ = [
    "CHECK_ANCHORS",
    "CHECK_CONNECTED_CORE",
    "CHECK_SOURCE_CARRIER",
    "CHECK_SUPPORTED_REACTIONS",
    "CHECK_ID_CONFLICT",
    "CHECK_ORDER",
    "CHECK_ORGANISM",
    "CHECK_PLACEHOLDER_IDENTITY",
    "CHECK_RAG_REINTRODUCTION",
    "ERROR_ORDER",
    "ERR_CROSS_ORGANISM_REACTIONS",
    "ERR_FALSE_REAL_IDENTIFIERS",
    "ERR_MISSING_PATHWAY_ANCHORS",
    "ERR_ORPHANED_REFERENCES",
    "ERR_PLACEHOLDER_BACKED_PROTEINS",
    "ERR_QUARANTINED_PROCESSES",
    "ERR_MISSING_SOURCE_CARRIER",
    "ERR_MISSING_SUPPORTED_REACTIONS",
    "ERR_UNSUPPORTED_REACTIONS",
    "CheckResult",
    "CoverageDetail",
    "SemanticReport",
    "validate_semantic_coverage",
]
