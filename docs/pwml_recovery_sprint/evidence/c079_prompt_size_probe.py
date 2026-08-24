"""C-079 / F-105 -- what the interactive curator prompt actually carries.

Run the SAME bytes of this script against a base checkout and against the tip and
compare the two reports. It answers three questions with numbers rather than with
symbol lookups, which is what gate G9 asks of a correction to pre-existing
observable behaviour:

1. THE BEHAVIOUR. Does ``build_interactive_curator_messages`` -- the production
   serializer ``run_interactive_curator_round`` calls -- put the source index into
   the prompt text? On base it does; at tip it does not. Reported as the byte size
   of the serialized user message, its sha256, and a literal substring test.

2. NON-VACUITY. A payload carrying NO omitted key must serialize BYTE-IDENTICALLY
   on base and at tip. The ``prompt_without_index`` sha256 in the two reports must
   match exactly. A filter that changed that number would be changing prompts it
   has no business touching.

3. THE LAYERING RULE that decided where the shared constant lives. Reported as
   sys.modules membership: importing ``interactive_curator`` must not drag in the
   OpenAI SDK, and importing ``audit_json_llm`` does. The last section shows the
   failure mode that makes the difference matter -- ``t2pw.llm.client`` raises
   while being imported when no usable provider is configured.

No LLM call is made and no network is touched: ``build_interactive_curator_messages``
is a pure serializer, and the provider probe only re-executes a module body.

Usage:
    python c079_prompt_size_probe.py --tree C:/t/c079
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

#: Exit 98 is reserved sprint-wide for "the tree under measurement is not the tree
#: that was asked for" (TEST_MATRIX.md § 0 rule 10). A probe that measured the
#: wrong checkout would prove the opposite of what it claims, so it refuses.
EXIT_MEASUREMENT_REFUSED = 98

#: A real committed paper, so the index blob is a real one and not a literal
#: invented to make the delta look large. PMC12180156 is the gold set's designated
#: hallucination test and its source text is committed with the run.
SOURCE_RELPATH = "runs/2026-08-02_2130/papers/PMC12180156/01_source_text.txt"

#: The SDK weight that decided C-079's structure. If ``interactive_curator`` ever
#: starts pulling these in at import, a pure string filter has become unimportable
#: without LLM credentials. The last two are the whole cost C-079 does add, and are
#: tracked here so the delta is reported rather than glossed: two pure-``re``/
#: ``typing`` modules, neither of which can raise on import.
SDK_MODULES = (
    "openai",
    "httpx",
    "dotenv",
    "t2pw.llm.client",
    "t2pw.curation.prompt_payload_keys",
    "t2pw.mapping.identity_admission",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_text(curator: Any, payload: Dict[str, Any]) -> str:
    """The user message ``run_interactive_curator_round`` would send, verbatim.

    An empty PNG keeps the image part a constant so every byte counted here is a
    byte of serialized payload.
    """
    messages = curator.build_interactive_curator_messages(
        working_json=payload,
        graph_png_bytes=b"",
        user_request="add the missing transport step",
        qa_report={"errors": [], "warnings": []},
        mapping_misses=[],
        history=[],
    )
    return messages[1]["content"][0]["text"]


def _pathway_payload() -> Dict[str, Any]:
    """A small but structurally real payload, so the prompt is not only index."""
    return {
        "pathway_name": "heme biosynthesis",
        "entities": {
            "compounds": [
                {"name": "succinyl-CoA", "mapped_id": "HMDB0001022"},
                {"name": "glycine", "mapped_id": "HMDB0000123"},
            ],
            "proteins": [{"name": "ALAS2", "mapped_ids": ["P22557"]}],
        },
        "processes": [{"name": "ALAS2 condensation"}],
        "reactions": [
            {
                "id": "r1",
                "inputs": ["succinyl-CoA", "glycine"],
                "outputs": ["5-aminolevulinate"],
                "modifiers": ["ALAS2"],
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-079 interactive prompt size probe")
    parser.add_argument("--tree", required=True, help="checkout to measure")
    args = parser.parse_args(argv)

    tree = Path(args.tree).resolve()
    src = tree / "src"
    sys.path.insert(0, str(src))

    import t2pw  # noqa: PLC0415 -- resolution is the thing being measured
    from t2pw.curation import interactive_curator as curator  # noqa: PLC0415

    package = (src / "t2pw").resolve()
    resolved = {
        "t2pw": str(Path(t2pw.__file__).resolve()),
        "interactive_curator": str(Path(curator.__file__).resolve()),
    }
    for name, path in resolved.items():
        if package not in Path(path).parents:
            print("T2PW_MEASUREMENT_TREE_REFUSED")
            print(json.dumps({"expected_package": str(package), "resolved": resolved}, indent=2))
            return EXIT_MEASUREMENT_REFUSED

    # 3a. layering, BEFORE anything imports the audit module
    sdk_after_curator = {name: (name in sys.modules) for name in SDK_MODULES}

    from t2pw.mapping import identity_admission  # noqa: PLC0415

    source_path = tree / SOURCE_RELPATH
    if not source_path.exists():
        print(f"MISSING SOURCE ARTIFACT: {source_path}")
        return 2
    raw = source_path.read_text(encoding="utf-8", errors="replace")
    index = identity_admission.build_source_index(raw)

    without = _pathway_payload()
    with_index = _pathway_payload()
    with_index[identity_admission.SOURCE_INDEX_KEY] = index

    text_without = _prompt_text(curator, without)
    text_with = _prompt_text(curator, with_index)

    # 3b. what importing the audit module would have cost the interactive one
    importlib.import_module("t2pw.curation.audit_json_llm")
    sdk_after_audit = {name: (name in sys.modules) for name in SDK_MODULES}

    report = {
        "tree": str(tree),
        "resolved": resolved,
        "source_artifact": {
            "path": str(source_path),
            "chars": len(raw),
        },
        "index_blob_bytes": len(
            json.dumps(index, indent=2, ensure_ascii=False).encode("utf-8")
        ),
        "prompt_with_index_bytes": len(text_with.encode("utf-8")),
        "prompt_without_index_bytes": len(text_without.encode("utf-8")),
        "prompt_delta_bytes": len(text_with.encode("utf-8")) - len(text_without.encode("utf-8")),
        "prompt_with_index_sha256": _sha256(text_with),
        "prompt_without_index_sha256": _sha256(text_without),
        "source_index_key_in_prompt": identity_admission.SOURCE_INDEX_KEY in text_with,
        "paper_prefix_in_prompt": identity_admission.normalize_text(raw)[:60] in text_with,
        "prompts_are_byte_identical": text_with == text_without,
        "sdk_in_sys_modules_after_interactive_curator": sdk_after_curator,
        "sdk_in_sys_modules_after_audit_json_llm": sdk_after_audit,
    }

    # 3c. the failure mode that makes the layering rule real. Re-executing the
    # module body with no usable provider is exactly what a fresh interpreter
    # would do on `import t2pw.curation.audit_json_llm`.
    os.environ["LLM_PROVIDER"] = "__c079_probe_invalid__"
    try:
        importlib.reload(sys.modules["t2pw.llm.client"])
        report["llm_client_import_with_bad_provider"] = "imported without raising"
    except Exception as exc:  # noqa: BLE001 -- the exception IS the measurement
        report["llm_client_import_with_bad_provider"] = f"{type(exc).__name__}: {exc}"

    print("C079_PROBE_REPORT")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
