"""Format validators (development only): SBML (libsbml), SBGN-ML (XSD + libSBGN PD schematron), PWML (structure).

    validate_sbml(bytes)  -> {ok, errors, warnings}
    validate_sbgn(bytes)  -> {ok, xsd_errors, schematron_failures}
    validate_pwml(bytes)  -> {ok, errors}

``libsbml`` and ``lxml`` are imported inside the functions: this module imports without them.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

SBGN_DATA = Path(__file__).resolve().parents[4] / "tests" / "data" / "vision" / "sbgn"
_NS = {"s": "http://sbgn.org/libsbgn/0.3"}
_SVRL = {"svrl": "http://purl.oclc.org/dsdl/svrl"}
_ISO = "http://purl.oclc.org/dsdl/schematron"


# ---------------------------------------------------------------- SBML

def validate_sbml(data: bytes, max_messages: int = 200) -> Dict[str, Any]:
    """libsbml read + consistency checks (all categories but units: no units are claimed, the checks only add noise,
    as in the clean_model validation script). ``errors`` = error / fatal messages (incl. read errors), ``warnings`` = warnings."""
    import libsbml
    doc = libsbml.readSBMLFromString(data.decode("utf-8", "replace") if isinstance(data, bytes) else data)
    if doc is None:
        return {"ok": False, "errors": [{"id": 0, "line": 0, "msg": "libsbml could not read the document"}], "warnings": []}
    n_read = doc.getNumErrors()
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    doc.checkConsistency()
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        rec = {"id": e.getErrorId(), "line": e.getLine(), "msg": e.getMessage().strip()[:300]}
        if e.getSeverity() in (libsbml.LIBSBML_SEV_ERROR, libsbml.LIBSBML_SEV_FATAL):
            errors.append(rec)
        elif e.getSeverity() == libsbml.LIBSBML_SEV_WARNING:
            warnings.append(rec)
    if doc.getModel() is None:
        errors.append({"id": 0, "line": 0, "msg": "no model"})
    return {"ok": not errors, "errors": errors[:max_messages], "warnings": warnings[:max_messages],
            "n_errors": len(errors), "n_warnings": len(warnings), "n_read_messages": n_read}


# ---------------------------------------------------------------- SBGN

@lru_cache(maxsize=2)
def _sbgn_schemas(data_dir: str):
    """(XSD, schematron). The schematron is libSBGN sbgn_pd.sch, 'basic' phase, loaded by lxml after four in-memory
    adjustments (the file on disk is untouched): title moved before ns and digit-led pattern ids prefixed (lxml checks ISO
    RelaxNG); assert attributes lxml's schema does not know dropped (the assert id stays as the message prefix); pattern
    pd10133 removed (XPath 2.0 distinct-values, lxml is XSLT 1.0), re-checked in validate_sbgn() as duplicate arcs."""
    from lxml import etree, isoschematron
    d = Path(data_dir)
    root = etree.parse(str(d / "sbgn_pd.sch")).getroot()
    title = root.find("{%s}title" % _ISO)
    root.remove(title)
    root.find("{%s}ns" % _ISO).addprevious(title)
    for p in root.iter("{%s}pattern" % _ISO):
        p.set("id", "p" + p.get("id"))
    for a in list(root.iter("{%s}active" % _ISO)):
        if a.get("pattern") == "pd10133":
            a.getparent().remove(a)
        else:
            a.set("pattern", "p" + a.get("pattern"))
    for a in root.iter("{%s}assert" % _ISO):
        a.text = (a.get("id") or "") + ": " + (a.text or "").strip()
        for k in ("id", "name", "see", "diagnostics", "role"):
            a.attrib.pop(k, None)
    return etree.XMLSchema(etree.parse(str(d / "SBGN.xsd"))), isoschematron.Schematron(root.getroottree(), store_report=True)


def validate_sbgn(data: bytes, data_dir=None) -> Dict[str, Any]:
    """SBGN-ML 0.3 Process Description: XSD errors, libSBGN schematron failed asserts (message starts with the assert id,
    e.g. ``pd10112``), and duplicate arcs (pattern pd10133)."""
    from lxml import etree
    xsd, sch = _sbgn_schemas(str(data_dir or SBGN_DATA))
    try:
        doc = etree.fromstring(data if isinstance(data, bytes) else data.encode("utf-8"))
    except etree.XMLSyntaxError as e:
        return {"ok": False, "xsd_errors": ["not well-formed: %s" % e], "schematron_failures": []}
    xsd_errors = [str(e) for e in xsd.error_log] if not xsd.validate(doc) else []
    sch.validate(doc)
    fails = [f.text.strip()[:120] for f in sch.validation_report.xpath("//svrl:failed-assert/svrl:text", namespaces=_SVRL)]
    ends = Counter((a.get("class"), a.get("source"), a.get("target")) for a in doc.iterfind(".//s:arc", _NS))
    fails += ["pd10133: duplicate arc %s" % (k,) for k, n in ends.items() if n > 1]
    return {"ok": not xsd_errors and not fails, "xsd_errors": xsd_errors, "schematron_failures": fails}


# ---------------------------------------------------------------- PWML

@lru_cache(maxsize=1)
def _pwml_signature():
    from t2pw.pwml.validate import discover_structure_signature
    from t2pw.vision.generate import _reference
    return discover_structure_signature(_reference())


# Fields the writer leaves out on purpose unless the PathBank DB supplied them (writer.py: pwc-id, short-name, pwbs-id are
# global database identities, never invented). Offline, they are absent from every export; counted, not errors.
DB_IDENTITY_FIELDS = ("pwc-id", "short-name", "pwbs-id")


def validate_pwml(data: bytes, ir: Optional[dict] = None) -> Dict[str, Any]:
    """Well-formed XML, then the repo's structural validator (pwml/validate.py: the section / field layout of the reference
    PWML). A missing DB_IDENTITY_FIELDS field is counted under ``db_identity_absent``, not as an error. With ``ir`` (the PWML
    IR the export returned) also pwml/ir.py ``validate_pwml_ir`` (its errors are reported as ``ir:<code>``)."""
    from lxml import etree
    from t2pw.pwml.validate import validate_generated_tree
    try:
        root = etree.fromstring(data, etree.XMLParser(remove_blank_text=True, huge_tree=True))
    except etree.XMLSyntaxError as e:
        return {"ok": False, "errors": [{"kind": "not-well-formed", "message": str(e)}], "n_errors": 1, "db_identity_absent": {}}
    errors, absent = [], Counter()
    for i in validate_generated_tree(etree.ElementTree(root), _pwml_signature())["issues"]:
        field = i["message"].partition("<")[2].rstrip(">") if i["kind"] == "missing-field" else ""
        if field in DB_IDENTITY_FIELDS:
            absent[field] += 1
        else:
            errors.append(i)
    if ir is not None:
        from t2pw.pwml.ir import validate_pwml_ir
        errors += [dict(kind="ir:" + str(e.get("code")), message=e.get("message")) for e in validate_pwml_ir(ir)["errors"]]
    return {"ok": not errors, "errors": errors[:200], "n_errors": len(errors), "db_identity_absent": dict(absent)}
