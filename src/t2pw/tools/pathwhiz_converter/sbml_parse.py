from __future__ import annotations

from typing import Any, Dict, List, Tuple


def parse_sbml_bytes(sbml_bytes: bytes) -> Tuple[Any, Any]:
    try:
        import libsbml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("python-libsbml is required for PathWhiz converter.") from exc

    try:
        text = sbml_bytes.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"SBML payload is not valid UTF-8: {exc}") from exc

    try:
        doc = libsbml.readSBMLFromString(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"libSBML parse failure: {exc}") from exc

    if doc is None:
        raise ValueError("libSBML returned no document.")
    model = doc.getModel()
    if model is None:
        raise ValueError("SBML document has no model.")
    return doc, model


def validate_document(doc: Any) -> Dict[str, Any]:
    check_count = int(doc.checkConsistency())
    messages: List[Dict[str, Any]] = []
    severe_count = 0
    for idx in range(int(doc.getNumErrors())):
        err = doc.getError(idx)
        if err is None:
            continue
        entry = {
            "severity": int(err.getSeverity()),
            "category": int(err.getCategory()),
            "line": int(err.getLine()),
            "message": str(err.getMessage() or ""),
        }
        messages.append(entry)
        if entry["severity"] >= 2:
            severe_count += 1
    return {
        "check_count": check_count,
        "error_count": len(messages),
        "severe_count": severe_count,
        "has_errors": severe_count > 0,
        "messages": messages,
    }


def serialize_document(doc: Any) -> bytes:
    try:
        import libsbml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("python-libsbml is required for PathWhiz converter.") from exc
    xml_text = libsbml.writeSBMLToString(doc)
    if not isinstance(xml_text, str) or not xml_text.strip():
        raise ValueError("Failed to serialize SBML document.")
    return xml_text.encode("utf-8")

