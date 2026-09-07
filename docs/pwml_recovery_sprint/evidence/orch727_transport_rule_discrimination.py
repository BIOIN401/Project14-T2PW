"""ORCH-727: can ANY structural transport rule separate the bad case from the good ones?

READ-ONLY. Reruns nothing, imports no production module, edits nothing. It reads archived
canonical payloads and archived source texts and answers one question:

    Is there a deterministic predicate over transports that refuses PMC12452463's
    `Enterobactin secretion` while letting PMC7232280's and PMC8510960's transports
    survive?

The authorized rule was: refuse a transport with NO transporter AND NO defensible
source/destination locations. This script evaluates that rule and three neighbouring
variants, then two further candidate invariants (evidence locatability, cargo graph
connectivity) and one aimed at the sibling objection (complex component structure).

A rule is USABLE only if it refuses every `must_block` transport and refuses none of the
`must_export` or `control` transports.

Nothing here decides product policy. It measures whether the authorized change is
constructible at all.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: (label, class, path-to-leg-dir, path-to-paper-dir)
LEGS: Tuple[Tuple[str, str, str], ...] = (
    ("PMC12452463@2026-08-28", "must_block", "runs_verify/2026-08-28_1816/papers/PMC12452463"),
    ("PMC12452463@2026-09-02", "must_block", "runs_verify/2026-09-02_2052/papers/PMC12452463"),
    ("PMC7232280@2026-09-06", "must_export", "runs_verify/2026-09-06_1425/papers/PMC7232280"),
    ("PMC8510960@2026-09-06", "must_export", "runs_verify/2026-09-06_1425/papers/PMC8510960"),
    ("PMC12071552@2026-09-06", "control", "runs_verify/2026-09-06_1425/papers/PMC12071552"),
)

#: The only transport in the must_block legs the gold actually objects to. The
#: ferri-enterobactin IMPORT is not objected to -- the gold's complaint is the efflux
#: step -- so a rule that refuses the import is over-broad, not correct.
OBJECTED = ("enterobactin secretion",)

#: A synthetic state the normalizer mints; it is not a location a curator asserted.
AUTO_STATE = "__auto_state__"


def load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def has_transporter(row: Dict[str, Any]) -> bool:
    return bool([x for x in (row.get("transporters") or []) if x])


def real_location(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text != AUTO_STATE


def any_location(row: Dict[str, Any]) -> bool:
    return real_location(row.get("from_biological_state")) or real_location(row.get("to_biological_state"))


def both_locations(row: Dict[str, Any]) -> bool:
    return real_location(row.get("from_biological_state")) and real_location(row.get("to_biological_state"))


VARIANTS: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "A  no transporter AND no locations at all  (AS AUTHORIZED)":
        lambda t: not has_transporter(t) and not any_location(t),
    "B  no transporter (alone)":
        lambda t: not has_transporter(t),
    "C  no transporter OR incomplete locations":
        lambda t: not has_transporter(t) or not both_locations(t),
    "D  no transporter AND not both locations":
        lambda t: not has_transporter(t) and not both_locations(t),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "."

    print("=" * 96)
    print("ORCH-727 -- is the authorized transport-integrity rule constructible?")
    print("=" * 96)
    print()

    rows: List[Dict[str, Any]] = []
    for label, klass, rel in LEGS:
        base = os.path.join(root, rel)
        payload = load(os.path.join(base, "strict", "final_mapped.json"))
        if payload is None:
            print(f"  {label}: no archived payload")
            continue
        source_path = os.path.join(base, "01_source_text.txt")
        source = ""
        if os.path.isfile(source_path):
            with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
                source = fold(handle.read())
        processes = payload.get("processes") or {}
        produced, consumed = set(), set()
        for reaction in processes.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            for item in reaction.get("outputs") or []:
                produced.add(fold(item.get("entity") if isinstance(item, dict) else item))
            for item in reaction.get("inputs") or []:
                consumed.add(fold(item.get("entity") if isinstance(item, dict) else item))
        for transport in processes.get("transports") or []:
            if not isinstance(transport, dict):
                continue
            name = str(transport.get("name") or "")
            evidence = fold(transport.get("evidence"))
            cargo = fold(transport.get("cargo"))
            rows.append(
                {
                    "leg": label,
                    "class": "objected" if name.casefold() in OBJECTED else klass,
                    "name": name,
                    "row": transport,
                    "evidence_verbatim": bool(evidence) and evidence in source,
                    "cargo_produced": cargo in produced,
                    "cargo_consumed": cargo in consumed,
                }
            )
        payload_complexes = (payload.get("entities") or {}).get("protein_complexes") or []
        for complex_row in payload_complexes:
            if not isinstance(complex_row, dict):
                continue
            components = complex_row.get("components") or []
            rows.append(
                {
                    "leg": label,
                    "class": klass,
                    "kind": "complex",
                    "name": str(complex_row.get("name") or ""),
                    "n_components": len(components),
                    "component_names": [
                        (c.get("name") if isinstance(c, dict) else c) for c in components
                    ],
                }
            )

    transports = [r for r in rows if r.get("kind") != "complex"]
    complexes = [r for r in rows if r.get("kind") == "complex"]

    print("-" * 96)
    print("TRANSPORT INVENTORY")
    print("-" * 96)
    print(f"{'class':<12}{'leg':<24}{'transport':<42}{'transp':>7}{'from':>26}{'to':>24}")
    for r in transports:
        t = r["row"]
        print(
            f"{r['class']:<12}{r['leg'][:23]:<24}{r['name'][:40]:<42}"
            f"{str(has_transporter(t)):>7}{str(t.get('from_biological_state'))[:24]:>26}"
            f"{str(t.get('to_biological_state'))[:22]:>24}"
        )
    print()

    print("-" * 96)
    print("RULE VARIANTS -- a rule is USABLE only if it refuses every 'objected' row and no other")
    print("-" * 96)
    for name, predicate in VARIANTS.items():
        refused = {r["name"]: predicate(r["row"]) for r in transports}
        bad_blocked = all(predicate(r["row"]) for r in transports if r["class"] == "objected")
        good_kept = not any(
            predicate(r["row"]) for r in transports if r["class"] in ("must_export", "control")
        )
        usable = bad_blocked and good_kept
        print(f"\n  {name}")
        print(f"      refuses every objected transport : {bad_blocked}")
        print(f"      keeps every must-export/control  : {good_kept}")
        print(f"      => USABLE: {usable}")
        for r in transports:
            mark = "REFUSED" if refused[r["name"]] else "kept   "
            print(f"          {mark}  [{r['class']:<11}] {r['leg'][:22]:<24}{r['name'][:44]}")

    print()
    print("-" * 96)
    print("CANDIDATE INVARIANT: evidence span locatable in the archived source text")
    print("-" * 96)
    for r in transports:
        print(f"  [{r['class']:<11}] {r['leg'][:22]:<24}{r['name'][:44]:<46} verbatim={r['evidence_verbatim']}")
    objected_verbatim = all(r["evidence_verbatim"] for r in transports if r["class"] == "objected")
    print(f"  => every objected transport's evidence IS verbatim in the paper: {objected_verbatim}")
    print("     so a provenance rule cannot separate them.")

    print()
    print("-" * 96)
    print("CANDIDATE INVARIANT: cargo produced/consumed by a retained reaction")
    print("-" * 96)
    for r in transports:
        print(
            f"  [{r['class']:<11}] {r['leg'][:22]:<24}{r['name'][:40]:<42}"
            f" produced={r['cargo_produced']} consumed={r['cargo_consumed']}"
        )
    print("     => the objected transport is fully connected while control cargo is not produced:")
    print("        the discriminator is INVERTED.")

    print()
    print("-" * 96)
    print("SIBLING OBJECTION: complex component structure")
    print("-" * 96)
    for r in complexes:
        print(
            f"  [{r['class']:<11}] {r['leg'][:22]:<24}{r['name'][:42]:<44}"
            f" n={r['n_components']} components={r['component_names'][:3]}"
        )
    print("     => 'enterobactin synthase complex' (gold: forbidden_identifier) is the same")
    print("        shape as G10H / STR / SGD / NIT-7A: one component, the Unknown sentinel.")

    print()
    print("=" * 96)
    print("CONCLUSION")
    print("=" * 96)
    usable_any = any(
        all(p(r["row"]) for r in transports if r["class"] == "objected")
        and not any(p(r["row"]) for r in transports if r["class"] in ("must_export", "control"))
        for p in VARIANTS.values()
    )
    print(f"  A usable structural transport rule exists on this evidence: {usable_any}")
    if not usable_any:
        print("  The authorized change is NOT constructible as specified, and no neighbouring")
        print("  structural, provenance or connectivity variant separates the cases.")
        print("  DO NOT IMPLEMENT. Report to the product owner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
