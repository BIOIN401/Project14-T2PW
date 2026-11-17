"""D-W24 (5): merge near-synonym biological states, then write exact duplicate reactions once (final payload step).

Runs last in ``final_payload.maybe_apply_final_id_fill`` (flag ``T2PW_FINAL_PAYLOAD``; off = the payload is untouched), so in a
run it comes before the pre-freeze resolver, the freeze and the strict quarantine, and the IR, the PWML, the SBML and the text
record all read the same merged payload. No network, no LLM. Fails open: on any error the payload is exactly what it was. Runs
once: a payload that already carries ``final_payload_merge`` is left alone. When nothing merges or is flagged, the payload is
not touched at all (deferrals are only in the returned report).

States. Two states merge only when they mean the same place: the same organism (a state with no species never merges), the
same location after :func:`location_key` (the table below; parenthetical text is kept), the same cell type and tissue (or
both absent), the same :data:`POSITION_WORDS`, AND names that agree: the name's words minus every spelling of its location,
the organism, the cell type, the tissue and :data:`NAME_STOPWORDS` must carry the same place words (:data:`PLACE_WORDS`:
"thylakoid membrane" is not "chloroplast", "blood" is not "urine") and the other words of one must be a subset of the other's
("hepatocyte cytosol" is not "neuron cytosol"). Agreement is checked against every member of a group, and a state that agrees
with two states that disagree with each other ("cytosol" beside both) never merges, so the result never depends on row order. A
placeholder (``__auto_state__``, ``AutoState_*``) has no name words. A state with no location field is keyed by its name; a
state with neither never merges. A eukaryote "cell" / "intracellular" state is the cytosol only when nothing else
distinguishes them, and never merges without a cell type or tissue. The kept state is the first of its group (row 0, the
default state, never moves), or its first stated member when the first is a placeholder; a placeholder at row 0 with a
stated member is not merged (the stated state would have to move row 0). Merged-away names go on the kept
row's ``aliases``; the rows, whole, into the ``final_payload_merge`` record. Every reference by name (processes,
element_locations, entity ``element_states``) is re-pointed; one re-pointed off a placeholder is marked
``compartment_source: "estimated"`` with a ``compartment_estimate`` reason (lane L7's field). Guards, each deferring and saying
why: names that disagree; a transport whose two ends would become one state; a kept name a later surviving state also carries.

Reactions. After the state merge, reactions with the same state, the same non-empty left and right members with stoichiometry
(read as the IR reads them), the same direction, reversibility, spontaneous flag and out-of-scope verdict are written once.
Enzymes and modifiers of every copy are unioned on the kept row (isozymes merge too, as PathWhiz does on import;
``isozyme_groups`` counts them). A copy with no enzyme and a catalysed copy are conflicting claims: neither is folded into the
other, both carry a ``review_reasons`` entry. The kept row is the copy with the most enzymes and modifiers, then a locked one,
then the first. Each merged-away row is recorded whole, as it was before this step, in the ``final_payload_merge`` record AND
in ``quarantined_locked_reactions`` with the reason ``duplicate_locked_reaction`` (the one ``dedupe_processes`` uses; KI-014,
"merged duplicate, not counted"), so its lock stays accounted for, its evidence stays in the text record and the coverage does
not fall; ``merged_into`` names the kept row. A merged-away name no surviving reaction carries leaves ``reaction_order``. Zero
loss guards: a copy (the kept one included) takes part only when every enzyme and modifier of it passes the strict
quarantine's own actor check under strict_db AND is a declared protein complex or a bare protein the quarantine will wrap
(``strict_quarantine.wrappable_bare_enzymes``, lane L5, D-W24 3; idx 113's AtCYP73A5). A copy that fails a guard stays as it
was, under ``deferred``.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

MERGE_FIELD = "final_payload_merge"
MERGED_REASON = "duplicate_locked_reaction"  # the reason dedupe_processes uses; KI-014 "merged duplicate, not counted"

# ---- normalisation table ---------------------------------------------------------------------------------------------
#: Spelling variants of one location -> one spelling (whole string, after lower-casing, "_" -> " ", a parenthetical, a
#: leading "the" and a trailing " state" removed, last word singular). Different organelles never share a value here.
LOCATION_SYNONYMS: Dict[str, str] = {
    "extracellular": "extracellular space", "extracellular space": "extracellular space",
    "extracellular region": "extracellular space", "extracellular environment": "extracellular space",
    "extracellular medium": "extracellular space", "extracellular milieu": "extracellular space",
    "cytoplasm": "cytoplasm", "cytoplasmic": "cytoplasm",
    "cytosol": "cytosol", "cytosolic": "cytosol",
    "intracellular": "intracellular", "intracellular space": "intracellular",
    "cell": "cell", "cellular": "cell", "whole cell": "cell", "bacterial cell": "cell",
    "mitochondrion": "mitochondrion", "mitochondria": "mitochondrion", "mitochondrial": "mitochondrion",
    "nucleus": "nucleus", "nuclear": "nucleus", "nuclei": "nucleus",
    "er": "endoplasmic reticulum", "endoplasmic reticulum": "endoplasmic reticulum",
    "golgi": "golgi apparatus", "golgi apparatus": "golgi apparatus", "golgi complex": "golgi apparatus",
    "plasma membrane": "plasma membrane", "cell membrane": "plasma membrane", "cytoplasmic membrane": "plasma membrane",
    "periplasm": "periplasm", "periplasmic": "periplasm", "periplasmic space": "periplasm",
    "peroxisome": "peroxisome", "peroxisomal": "peroxisome",
    "vacuole": "vacuole", "vacuolar": "vacuole",
    "lysosome": "lysosome", "lysosomal": "lysosome",
    "plastid": "plastid", "plastidic": "plastid",
    "chloroplast": "chloroplast", "chloroplastic": "chloroplast",
}
#: In a prokaryote these all name the inside of the cell: the cytoplasm.
PROKARYOTE_CYTOPLASM = frozenset({"cytoplasm", "cytosol", "intracellular", "cell"})
#: In a eukaryote these name "inside the cell, compartment not stated"; they are NOT the cytosol (see ``_state_groups``).
EUKARYOTE_UNSTATED = frozenset({"intracellular", "cell"})
#: A eukaryote "cell" state is the cytosol only when its organism's states name no place outside this set.
EUKARYOTE_CELL_AS_CYTOSOL_WHEN_ONLY = frozenset({"cell", "cytosol", "extracellular space"})
#: Words that place a state inside a coarser location ("apical extracellular" vs "synaptic extracellular": both
#: "extracellular" in the location field, different places). Read from the name and the location field; two states merge
#: only when they carry the same set of these.
POSITION_WORDS = frozenset({
    "apical", "basolateral", "basal", "lateral", "luminal", "lumen", "synaptic", "presynaptic", "postsynaptic",
    "inner", "outer", "intermembrane", "matrix", "stroma", "cis", "trans", "medial", "proximal", "distal", "perinuclear",
    "pericellular", "early", "late", "rough", "smooth", "dendritic", "axonal", "somatic",
})
_STRAIN_WORDS = frozenset({"strain", "str", "str."})
#: Name words that carry no place: dropped before two names are compared.
NAME_STOPWORDS = frozenset({"of", "in", "the", "a", "an", "and", "state", "from", "at", "on", "within", "inside", "to", "for",
                            "with", "by", "strain", "str", "subsp", "sp", "spp", "var", "serovar", "pv"})
#: Words that name a place (besides the table's own words). A name word from this set that is not the state's own location
#: must be carried by both names: "thylakoid membrane" (location chloroplast) is not "chloroplast", "blood" is not "urine".
EXTRA_PLACE_WORDS = frozenset({
    "thylakoid", "envelope", "nucleolus", "nucleoplasm", "membrane", "lumen", "stroma", "matrix", "vesicle", "granule",
    "endosome", "cytoskeleton", "chromatin", "cleft", "pore", "space", "compartment", "wall", "apoplast", "etioplast",
    "amyloplast", "glyoxysome", "microsome", "mitochondrial", "organelle", "inclusion", "body", "droplet", "lipid",
    "blood", "urine", "plasma", "serum", "saliva", "bile", "milk", "cerebrospinal", "fluid", "sweat", "feces", "lymph",
    "synapse", "axon", "dendrite", "cilium", "flagellum", "spine",
})


def position_words(state: Dict[str, Any]) -> frozenset:
    text = _text(f"{state.get('name') or ''} {_state_loc(state) or ''}")
    return frozenset(re.findall(r"[a-z0-9]+", text)) & POSITION_WORDS


def _word(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "is", "us")) else t


def _tokens(value: Any) -> set:
    """Lower-case words, last-letter plural dropped; parenthetical text is kept ("chloroplast (envelope)")."""
    return {_word(t) for t in re.findall(r"[a-z0-9]+", _text(value))}


_PLACEHOLDER = re.compile(r"^_*auto_?state", re.I)


def is_placeholder(state: Dict[str, Any]) -> bool:
    """A generated state with no stated place (``__auto_state__``, ``AutoState_<species><location>``)."""
    return bool(_PLACEHOLDER.match(str(state.get("name") or "").strip()))


def _norm(value: Any) -> str:
    from t2pw.pwml.ir import _canonical, _norm as ir_norm  # the IR's state key: a reference means what the IR resolves

    return ir_norm(_canonical(value))


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ")).strip().casefold()


def _singular(text: str) -> str:
    head, _, last = text.rpartition(" ")
    if len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "is", "us")):
        last = last[:-1]
    return (head + " " + last).strip()


PLACE_WORDS = frozenset(set().union(*(_tokens(k) | _tokens(v) for k, v in LOCATION_SYNONYMS.items()))
                        | {_word(w) for w in EXTRA_PLACE_WORDS} | POSITION_WORDS)


def location_key(value: Any, prokaryote: bool) -> str:
    """One spelling per place. ``"X state"`` = ``"X"``; plural = singular; the table above; in a prokaryote "cell",
    "intracellular", "cytoplasm" and "cytosol" are the cytoplasm; in a eukaryote "cell" and "intracellular" are one
    unstated place, kept apart from the cytosol."""
    s = _text(value)  # a parenthetical is kept: "chloroplast (thylakoid membrane)" is not "chloroplast"
    s = re.sub(r"^the ", "", s)
    s = re.sub(r" state$", "", s)
    s = LOCATION_SYNONYMS.get(s, LOCATION_SYNONYMS.get(_singular(s), _singular(s)))
    if prokaryote and s in PROKARYOTE_CYTOPLASM:
        return "cytoplasm"
    if not prokaryote and s in EUKARYOTE_UNSTATED:
        return "cell"
    return s


def context_key(value: Any) -> str:
    """Cell type / tissue: case, spacing and a plural last word do not matter ("Type III cells" = "Type III cell")."""
    s = _text(value)
    return _singular(s) if s else ""


# ---- organisms -------------------------------------------------------------------------------------------------------
def _species_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in ((payload.get("entities") or {}).get("species") or []) if isinstance(r, dict)]


def _species_row(name: Any, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    n = _norm(name)
    if not n:
        return None
    for r in rows:
        if n in {_norm(x) for x in [r.get("name"), *(r.get("synonyms") or []), *(r.get("aliases") or [])] if isinstance(x, str)}:
            return r
    return None


def _binomial(name: Any) -> Optional[Tuple[str, str, Tuple[str, ...]]]:
    toks = [t.strip("().,;") for t in _text(name).replace("(", " ").replace(")", " ").split()]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return None
    extras = tuple(t for t in toks[2:] if t not in _STRAIN_WORDS)
    return toks[0], toks[1], extras


def _genus_match(a: str, b: str) -> bool:
    if a == b:
        return True
    for x, y in ((a, b), (b, a)):
        if len(x) == 2 and x.endswith(".") and y[:1] == x[:1] and len(y) > 2:
            return True
    return False


class _Organisms:
    """Same organism: same name; or same taxonomy id; or the same binomial with the same strain words; or a species-level
    name and a strain-level name of the same binomial when the payload's states name exactly one strain of it."""

    def __init__(self, payload: Dict[str, Any], names: List[str]) -> None:
        self.rows = _species_rows(payload)
        self.strains: Dict[Tuple[str, str], set] = {}
        for n in names:
            b = _binomial(n)
            if b and b[2]:
                self.strains.setdefault((b[0].rstrip(".")[:1], b[1]), set()).add(b[2])

    def taxon(self, name: Any) -> str:
        r = _species_row(name, self.rows)
        return str((r or {}).get("taxonomy_id") or "").strip()

    def prokaryote(self, name: Any) -> bool:
        r = _species_row(name, self.rows)
        if r is None:  # an unlisted spelling: borrow the class of a listed species of the same binomial
            b = _binomial(name)
            for row in self.rows:
                rb = _binomial(row.get("name"))
                if b and rb and rb[1] == b[1] and _genus_match(rb[0], b[0]):
                    r = row
                    break
        return str((r or {}).get("classification") or "").strip().casefold() == "prokaryote"

    def same(self, a: Any, b: Any) -> bool:
        if _norm(a) == _norm(b):
            return True
        ta, tb = self.taxon(a), self.taxon(b)
        if ta and ta == tb:
            return True
        ba, bb = _binomial(a), _binomial(b)
        if not ba or not bb or ba[1] != bb[1] or not _genus_match(ba[0], bb[0]):
            return False
        if ba[2] == bb[2]:
            return True
        if ba[2] and bb[2]:
            return False
        return len(self.strains.get((ba[0].rstrip(".")[:1], ba[1]), ())) == 1


# ---- states ----------------------------------------------------------------------------------------------------------
def _state_loc(row: Dict[str, Any]) -> Any:
    return row.get("subcellular_location") or row.get("subcellular-location") or row.get("compartment") or row.get("compartment_canonical")


def _iter_state_refs(payload: Dict[str, Any]):
    """(container, key) of every reference to a state by name: processes, element_locations and entities (``element_states``,
    an entity's own ``biological_state``) -- any string value under a key naming a biological state."""
    def walk(v: Any):
        if isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str) and "biological_state" in k and (isinstance(x, str) or x is None):
                    yield v, k
                else:
                    yield from walk(x)
        elif isinstance(v, list):
            for x in v:
                yield from walk(x)
    for top in ("processes", "element_locations", "entities"):
        yield from walk(payload.get(top) or {})


def _resolver(states: List[Dict[str, Any]]):
    """The IR's reading of a state reference: empty or unknown -> row 0 (the default state); a name -> the LAST row with it."""
    last: Dict[str, int] = {}
    for i, s in enumerate(states):
        n = _norm(s.get("name"))
        if n:
            last[n] = i

    def row(name: Any) -> int:
        return last.get(_norm(name), 0) if _norm(name) else 0
    return row, last


def _name_residue(state: Dict[str, Any], loc_words: set, payload_species: List[Dict[str, Any]]) -> set:
    """The words of a state's name that say something its location, organism, cell type and tissue do not."""
    if is_placeholder(state):
        return set()
    strip = set(loc_words) | NAME_STOPWORDS
    sp = state.get("species")
    row = _species_row(sp, payload_species)
    for n in [sp, *([row.get("name"), *(row.get("synonyms") or [])] if row else [])]:
        if isinstance(n, str) and n.strip():
            toks = re.findall(r"[a-z0-9]+", _text(n))
            strip |= {_word(t) for t in toks}
            if toks:
                strip.add(toks[0][:1])  # "L. lactis", "E. coli"
    for f in ("cell_type", "cell-type", "tissue"):
        strip |= _tokens(state.get(f))
    return _tokens(state.get("name")) - strip


def _names_agree(a: set, b: set) -> bool:
    """Place words beyond the location must be the same in both; any other word may only be added, never swapped."""
    pa, pb = a & PLACE_WORDS, b & PLACE_WORDS
    oa, ob = a - PLACE_WORDS, b - PLACE_WORDS
    return pa == pb and (oa <= ob or ob <= oa)


def _state_groups(payload: Dict[str, Any], states: List[Dict[str, Any]], report: Dict[str, Any]) -> List[int]:
    """group[i] = index of the kept state for row i."""
    org = _Organisms(payload, [str(s.get("species") or "") for s in states])
    pro = [org.prokaryote(s.get("species")) for s in states]
    loc = []
    for i, s in enumerate(states):
        k = location_key(_state_loc(s) or s.get("name"), pro[i])
        if not k or not str(s.get("species") or "").strip():
            k = "\0unkeyed:%d" % i  # no location and no name, or no species: never merges
        loc.append(k)
    ctx = [(context_key(s.get("cell_type") or s.get("cell-type")), context_key(s.get("tissue")), position_words(s)) for s in states]
    # Eukaryote: "cell" is the cytosol only when nothing else distinguishes them -- that organism's states name no place
    # other than the cell, the cytosol and the outside, and a cytosol state with the same cell type and tissue exists.
    cell_as_cytosol = []
    for i, s in enumerate(states):
        if pro[i] or loc[i] != "cell":
            continue
        kin = [j for j in range(len(states)) if not pro[j] and org.same(s.get("species"), states[j].get("species"))]
        if {loc[j] for j in kin} <= EUKARYOTE_CELL_AS_CYTOSOL_WHEN_ONLY and any(loc[j] == "cytosol" and ctx[j] == ctx[i] for j in kin):
            cell_as_cytosol.append(i)
    for i in cell_as_cytosol:
        loc[i] = "cytosol"
        report["eukaryote_cell_as_cytosol"].append(str(states[i].get("name")))
    # Eukaryote "cell" with no cell type and no tissue says nothing about which cell: only its name does (idx 150:
    # "afferent nerve fiber state" and an __auto_state__ placeholder, both "cell"). It never merges.
    for i, s in enumerate(states):
        if not pro[i] and loc[i] == "cell" and ctx[i][:2] == ("", ""):
            if any(j != i and not pro[j] and loc[j] == "cell" and ctx[j][:2] == ("", "") and org.same(s.get("species"), states[j].get("species"))
                   for j in range(len(states))):
                report["deferred"].append({"kind": "state", "state": str(s.get("name")), "row": i, "reason": "eukaryote_cell_without_cell_type_or_tissue"})
            loc[i] = "\0unkeyed:%d" % i
    # The name must agree too (F1): the words of the location (every spelling of it in the table), the organism, the cell
    # type, the tissue and stopwords are dropped; what is left must agree (``_names_agree``).
    spelled: Dict[str, set] = {}
    for phrase in set(LOCATION_SYNONYMS) | set(LOCATION_SYNONYMS.values()):
        for p in (True, False):
            spelled.setdefault(("p" if p else "e") + location_key(phrase, p), set()).update(_tokens(phrase))
    species_rows = _species_rows(payload)
    residue = []
    for i, s in enumerate(states):
        words = _tokens(_state_loc(s)) | spelled.get(("p" if pro[i] else "e") + loc[i], set())
        if i in cell_as_cytosol:
            words |= spelled.get("ecell", set())
        residue.append(_name_residue(s, words, species_rows))

    def same_place(i: int, j: int) -> bool:
        return (pro[i] == pro[j] and loc[i] == loc[j] and ctx[i] == ctx[j]
                and org.same(states[i].get("species"), states[j].get("species")) and _names_agree(residue[i], residue[j]))

    # R1 (fail closed): a state that agrees with two states which disagree with each other ("cytosol" with "hepatocyte
    # cytosol" and "neuron cytosol") has no one place it is the same as; it never merges, whatever the row order. With these
    # removed, what is left is order-independent: every state that agrees with two others has them agree with each other.
    blocked: set = set()
    for i in range(len(states)):
        partners = [j for j in range(len(states)) if j != i and same_place(i, j)]
        clashing = sorted({str(states[b].get("name")) for a in partners for b in partners if a < b and not same_place(a, b)})
        if clashing:
            blocked.add(i)
            report["deferred"].append({"kind": "state", "state": str(states[i].get("name")), "row": i,
                                       "reason": "agrees_with_states_that_disagree", "with": clashing})
    row, _ = _resolver(states)
    while True:
        group = list(range(len(states)))
        members: Dict[int, List[int]] = {}
        for i in range(len(states)):
            if i in blocked or group[i] != i:
                continue
            members[i] = [i]
            for j in range(i + 1, len(states)):
                if j in blocked or group[j] != j:
                    continue
                if pro[i] == pro[j] and loc[i] == loc[j] and ctx[i] == ctx[j] and org.same(states[i].get("species"), states[j].get("species")):
                    if all(_names_agree(residue[m], residue[j]) for m in members[i]):
                        group[j] = i
                        members[i].append(j)
                    else:
                        report["deferred"].append({"kind": "state", "state": str(states[j].get("name")), "row": j,
                                                   "with": str(states[i].get("name")), "reason": "names_disagree",
                                                   "words": sorted(residue[j] ^ residue[i])})
        # Guard: a transport whose two ends are different states must not become a same-state transport.
        clash = set()
        for t in ((payload.get("processes") or {}).get("transports") or []):
            if not isinstance(t, dict):
                continue
            pairs = [(t.get("from_biological_state"), t.get("to_biological_state"))]
            for c in t.get("transport_elements") or []:
                if isinstance(c, dict):
                    pairs.append((c.get("left_biological_state") or c.get("from_biological_state") or t.get("from_biological_state"),
                                  c.get("right_biological_state") or c.get("to_biological_state") or t.get("to_biological_state")))
            for a, b in pairs:
                ra, rb = row(a), row(b)
                if ra != rb and group[ra] == group[rb]:
                    clash |= {ra, rb}
        if not clash:
            break
        report["deferred"] = [d for d in report["deferred"] if d.get("reason") != "names_disagree"]  # recomputed next pass
        for i in sorted(clash):
            report["deferred"].append({"kind": "state", "state": str(states[i].get("name")), "row": i, "reason": "transport_ends_would_merge"})
        blocked |= clash
    # The kept state is a stated one rather than a placeholder. R2: row 0 (the default state) never moves, so a placeholder at
    # row 0 with a stated member is not merged at all (the box would otherwise keep the placeholder's name).
    for k, ms in members.items():
        if len(ms) > 1 and is_placeholder(states[k]):
            stated = [m for m in ms if not is_placeholder(states[m])]
            if stated and k == 0:
                for m in ms:
                    group[m] = m
                report["deferred"].append({"kind": "state", "state": str(states[0].get("name")), "row": 0,
                                           "reason": "default_state_is_a_placeholder",
                                           "with": [str(states[m].get("name")) for m in stated]})
            elif stated:
                for m in ms:
                    group[m] = stated[0]
    # Guard: the kept state's name must still resolve to it (no later surviving row may carry it).
    kept = {i for i in range(len(states)) if group[i] == i}
    for k in sorted(kept):
        ms = [i for i in range(len(states)) if group[i] == k]
        if len(ms) < 2:
            continue
        n = _norm(states[k].get("name"))
        if n and any(j > k and j in kept and _norm(states[j].get("name")) == n for j in range(len(states))):
            for i in ms:
                group[i] = i
            report["deferred"].append({"kind": "state", "state": str(states[k].get("name")), "row": k, "reason": "kept_name_not_unique"})
    return group


def _merge_states(payload: Dict[str, Any], report: Dict[str, Any]):
    """Merge the states in place; returns the post-merge resolver (reference -> kept row object) for the reaction step."""
    states = list(payload.get("biological_states") or [])
    if not states or not all(isinstance(s, dict) for s in states):
        return None
    group = _state_groups(payload, states, report)
    _, last = _resolver(states)
    target = {n: states[group[i]] for n, i in last.items()}  # name -> kept row object
    default = states[0]

    def resolve(name: Any) -> Dict[str, Any]:
        return target.get(_norm(name), default) if _norm(name) else default

    if all(group[i] == i for i in range(len(states))):
        return resolve
    rename = {n: str(states[group[i]].get("name")) for n, i in last.items() if group[i] != i}
    placeholder = {n for n, i in last.items() if group[i] != i and is_placeholder(states[i])}
    for holder, key in list(_iter_state_refs(payload)):
        n = _norm(holder.get(key))
        if n in rename and _norm(rename[n]) != n:
            old = holder[key]
            holder[key] = rename[n]
            report["state_references_repointed"] += 1
            if n in placeholder:  # F7: a placeholder had no stated compartment; where it now sits is an estimate (L7's field)
                kept = states[group[last[n]]]
                holder["compartment_source"] = "estimated"
                holder.setdefault("compartment_estimate", []).append({
                    "field": key, "from_state": old, "to_state": rename[n],
                    "reason": f"placeholder state '{old}' had no stated compartment; merged into '{rename[n]}' "
                              f"({_state_loc(kept)}, {kept.get('species')}) as the same place (D-W24 5, L6)"})
                report["placeholder_references_estimated"] += 1
    originals = copy.deepcopy(states)
    for i, s in enumerate(states):
        k = group[i]
        if k == i:
            continue
        kept = states[k]
        aliases = kept.setdefault("aliases", [])
        for a in [s.get("name"), *(s.get("aliases") or [])]:
            if isinstance(a, str) and a and _norm(a) != _norm(kept.get("name")) and a not in aliases:
                aliases.append(a)
        report["states"].append({"state": s.get("name"), "row": i, "into": kept.get("name"), "into_row": k,
                                 "placeholder": is_placeholder(s),
                                 "location": _state_loc(s), "into_location": _state_loc(kept),
                                 "species": s.get("species"), "into_species": kept.get("species"), "original": originals[i]})
    payload["biological_states"] = [s for i, s in enumerate(states) if group[i] == i]
    return resolve


# ---- reactions -------------------------------------------------------------------------------------------------------
def _side(value: Any) -> Tuple[Tuple[str, int], ...]:
    from t2pw.pwml.ir import _coerce_participants

    return tuple(sorted((_norm(p["name"]), int(p.get("stoichiometry") or 1)) for p in _coerce_participants(value)))


def _actor_key(actor: Any) -> Tuple[str, str]:
    from t2pw.pwml.ir import _actor_name_and_hint

    name, _hint, role = _actor_name_and_hint(actor)
    return _norm(name), role


def _actor_name(actor: Any) -> str:
    from t2pw.pwml.ir import _actor_name_and_hint

    return _actor_name_and_hint(actor)[0]


def _lock(row: Dict[str, Any]) -> str:
    v = row.get("locked_reaction_id")
    return str(v).strip() if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""


def _merge_reactions(payload: Dict[str, Any], resolve: Any, originals: List[Any], report: Dict[str, Any]) -> None:
    from t2pw.mapping.map_ids import _normalize_name
    from t2pw.pipeline.strict_quarantine import (
        _actor_verdict, _build_registry, _scope_is_out_of_scope, wrappable_bare_enzymes)
    from t2pw.pwml.ir import _direction

    procs = payload.get("processes") or {}
    reactions = procs.get("reactions") if isinstance(procs, dict) else None
    if not isinstance(reactions, list):
        return
    registry = _build_registry(payload)
    # L5 (D-W24 3): the quarantine wraps a bare protein with a database identity in its single-protein complex, so
    # such an actor WILL be written as a complex; it no longer defers its copy. Same predicate as the quarantine's.
    will_wrap = set(wrappable_bare_enzymes(payload))
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for i, r in enumerate(reactions):
        if not isinstance(r, dict):
            continue
        state = resolve(r.get("biological_state")) if resolve else None
        sig = (id(state), _direction(r.get("direction")), str(r.get("reversible")), bool(r.get("spontaneous")),
               _side(r.get("inputs")), _side(r.get("outputs")), _scope_is_out_of_scope(r))
        groups.setdefault(sig, []).append(i)
    drop: set = set()
    for sig, members in groups.items():
        if len(members) < 2:
            continue
        if not sig[4] or not sig[5]:  # a side is empty: not a reaction anything can be proven equal to
            for i in members:
                report["deferred"].append({"kind": "reaction", "reaction": reactions[i].get("name"), "row": i, "reason": "empty_side"})
            continue
        ok = []
        for i in members:
            verdict = _actor_verdict("reactions", reactions[i], registry, strict_db=True)
            bare = [_actor_name(a) for f in ("enzymes", "modifiers") for a in reactions[i].get(f) or []
                    if (registry.resolve(_actor_name(a)) or ("",))[0] != "protein_complexes"
                    and _normalize_name(_actor_name(a)) not in will_wrap]
            if verdict is None and not bare:
                ok.append(i)
            else:
                report["deferred"].append({"kind": "reaction", "reaction": reactions[i].get("name"), "row": i,
                                           "reason": ("actor_not_admissible_under_strict:" + str(verdict[1])) if verdict
                                           else "actor_not_a_protein_complex:" + "|".join(bare)})
        if len(ok) < 2:
            continue
        # F3: a copy with no enzyme and a catalysed copy are conflicting claims (idx 20: "Spontaneous elimination" vs the
        # MenH step). Neither is folded into the other; both are flagged. Copies on the same side still merge.
        n_actors = {i: len(reactions[i].get("enzymes") or []) + len(reactions[i].get("modifiers") or []) for i in ok}
        catalysed, uncatalysed = [i for i in ok if n_actors[i]], [i for i in ok if not n_actors[i]]
        if catalysed and uncatalysed:
            names = [str(reactions[i].get("name")) for i in ok]
            for i in ok:
                reactions[i].setdefault("review_reasons", []).append(
                    "same_step_with_and_without_enzyme: written as " + " / ".join(repr(n) for n in names) + "; kept apart (D-W24 5, L6)")
                report["flagged"].append({"reaction": reactions[i].get("name"), "row": i, "reason": "same_step_with_and_without_enzyme"})
        for sub in ((catalysed, uncatalysed) if catalysed and uncatalysed else (ok,)):
            if len(sub) >= 2:
                _merge_copies(payload, reactions, sub, n_actors, originals, drop, report)
    if drop:
        procs["reactions"] = [r for i, r in enumerate(reactions) if i not in drop]
        # F6: a merged-away name no surviving reaction carries leaves reaction_order
        alive = {_norm(r.get("name")) for r in procs["reactions"] if isinstance(r, dict)}
        gone = {_norm(originals[i].get("name")) for i in drop} - alive
        order = payload.get("reaction_order")
        if gone and isinstance(order, list):
            payload["reaction_order"] = [x for x in order if not (isinstance(x, str) and _norm(x) in gone)]
            report["reaction_order_removed"] = len(order) - len(payload["reaction_order"])


def _merge_copies(payload: Dict[str, Any], reactions: List[Any], ok: List[int], n_actors: Dict[int, int], originals: List[Any],
                  drop: set, report: Dict[str, Any]) -> None:
    """Write the copies ``ok`` once. Kept: the copy with the most enzymes and modifiers (named after its catalyst), then a
    locked one, then the first."""
    k = min(ok, key=lambda i: (-n_actors[i], not _lock(reactions[i]), i))
    kept = reactions[k]
    isozymes = False
    for i in ok:
        if i == k:
            continue
        dup = reactions[i]
        to_add = []
        differs = False
        for fld in ("enzymes", "modifiers"):
            have = {_actor_key(a) for a in (kept.get(fld) or [])}
            differs = differs or {_actor_key(a) for a in (dup.get(fld) or [])} != have
            for a in dup.get(fld) or []:
                if _actor_key(a) not in have:
                    to_add.append((fld, a))
                    have.add(_actor_key(a))
        for fld, a in to_add:
            kept.setdefault(fld, []).append(copy.deepcopy(a))
        original = originals[i]
        # F2: the existing reason for a duplicate folded into a kept row (KI-014: listed, not counted in the coverage)
        record = {"reaction_name": str(original.get("name") or "<unnamed>"), "reason": MERGED_REASON,
                  "action": "merged_into_reaction", "json_pointer": f"/processes/reactions/{i}",
                  "merged_into": {"reaction_name": kept.get("name"), "locked_reaction_id": _lock(kept) or None,
                                  "note": f"written once as '{kept.get('name')}' (D-W24 5)"},
                  "original_reaction": original}
        if _lock(original):
            record["locked_reaction_id"] = _lock(original)
        if not isinstance(payload.get("quarantined_locked_reactions"), list):
            payload["quarantined_locked_reactions"] = []
        payload["quarantined_locked_reactions"].append(record)
        drop.add(i)
        isozymes = isozymes or differs
        report["reactions"].append({"reaction": original.get("name"), "row": i, "lock": _lock(original) or None,
                                    "into": kept.get("name"), "into_row": k, "into_lock": _lock(kept) or None,
                                    "biological_state": original.get("biological_state"),
                                    "actors_added": [[f, _actor_name(a)] for f, a in to_add]})
    if isozymes:
        report["isozyme_groups"] += 1


def merge_states_and_duplicates(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the merge to ``payload`` in place; the report is also written under ``payload[MERGE_FIELD]`` when anything
    merged (deferrals alone stay in the returned report only). Never raises: any error restores the payload and is reported."""
    report: Dict[str, Any] = {"schema": 1, "states": [], "reactions": [], "deferred": [], "isozyme_groups": 0,
                              "eukaryote_cell_as_cytosol": [], "state_references_repointed": 0,
                              "placeholder_references_estimated": 0, "flagged": [], "error": None}
    if not isinstance(payload, dict) or MERGE_FIELD in payload:
        report["skipped"] = "already_merged" if isinstance(payload, dict) else "not_a_payload"
        return report
    snapshot = copy.deepcopy(payload)
    try:
        reactions = (payload.get("processes") or {}).get("reactions")
        originals = copy.deepcopy(reactions) if isinstance(reactions, list) else []
        states_before = len(payload.get("biological_states") or [])
        reactions_before = len(originals)
        resolve = _merge_states(payload, report)
        _merge_reactions(payload, resolve, originals, report)
        report["counts"] = {"states_before": states_before, "states_after": len(payload.get("biological_states") or []),
                            "reactions_before": reactions_before,
                            "reactions_after": len(((payload.get("processes") or {}).get("reactions") or []))}
        if report["states"] or report["reactions"] or report["flagged"]:  # else the payload stays byte-identical
            payload[MERGE_FIELD] = copy.deepcopy(report)
    except Exception as exc:  # fail open
        payload.clear()
        payload.update(snapshot)
        report.update(states=[], reactions=[], flagged=[], error=f"{type(exc).__name__}: {exc}")
    return report
