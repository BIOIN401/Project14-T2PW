"""On-demand generation (docs/vision/PLAN.md section 2.3): every format from ONE scoped view.

``generate(big, scope, fmt)`` builds ``view(big, scope)`` once. PWML is exported with the production path the
replays use (``service.postpipeline.run_pwml_export``, layout v2, offline); the PWML IR that export returns is
what the SVG/HTML diagram and the SBGN coordinates use, so the formats cannot disagree. Pure over its inputs:
no DB, no network, no LLM.

    python -m t2pw.vision.generate <run_dir | big_payload.json> --scope paper|default|full|hops:N --fmt pwml --out f
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from t2pw.vision.view import DEFAULT, FULL, PAPER_ONLY, ScopeSpec, view

FORMATS = ("pwml", "sbml", "sbgn", "kgml", "svg", "html", "json")
_REF_NAMES = ("PW000001.pwml", "PW012926.pwml")  # structure reference the PWML writer learns its skeleton from
_DB_ENV = tuple("PATHBANK_DB_" + k for k in ("HOST", "PORT", "USER", "PASSWORD", "SCHEMA",
                                              "CONNECT_TIMEOUT", "READ_TIMEOUT", "WRITE_TIMEOUT"))
_MEMO: Dict[str, Tuple[Optional[bytes], dict, Optional[str]]] = {}  # export key -> (pwml bytes, ir, error); newest 4


class GenerateError(RuntimeError):
    """The PWML export refused or failed for this view (the message is the export's own)."""


def parse_scope(text: str) -> ScopeSpec:
    t = (text or "default").strip().lower()
    if t in ("paper", "paper_only"):
        return PAPER_ONLY
    if t == "default":
        return DEFAULT
    if t == "full":
        return FULL
    if t.startswith("hops:") and t[5:].isdigit():
        return ScopeSpec(False, int(t[5:]))
    raise ValueError("scope must be paper, default, full or hops:N, got %r" % text)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_options_from_run(run_dir) -> Dict[str, Any]:
    """The export controls a stored run was taken under (strict_db, export mode, Stage-0 context), read the way the
    replay harness reads them. Without a quarantine report: the exporter's defaults."""
    p = Path(run_dir) / "quarantine_report.json"
    if not p.exists():
        return {}
    qr = json.loads(p.read_text(encoding="utf-8"))
    di = qr.get("decision_inputs") or {}
    terms = list(dict.fromkeys(list(di.get("requested_core") or []) + list(di.get("pathway_context_core") or [])))
    opts: Dict[str, Any] = {"strict_db": bool(qr.get("strict_db", True))}
    if di.get("export_mode"):
        opts["export_mode"] = di["export_mode"]
    if terms:
        opts["pathway_context"] = {"key_compounds": terms}
    return opts


@contextlib.contextmanager
def _offline():
    """No PathBank DB and no live curator for the duration of an export. Blank (not delete) the DB variables so a
    later ``.env`` load cannot refill them (load_dotenv override=False keeps what is already set)."""
    keep = {k: os.environ.get(k) for k in _DB_ENV + ("T2PW_OFFLINE_CURATOR",)}
    try:
        for k in _DB_ENV:
            os.environ[k] = ""
        os.environ["T2PW_OFFLINE_CURATOR"] = "1"
        yield
    finally:  # ponytail: process-wide env swap, not thread-safe; a threaded server would serialise generate()
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _reference() -> Path:
    import t2pw
    root = Path(t2pw.__file__).resolve().parents[2]
    for n in _REF_NAMES:
        if (root / "reference" / n).exists():
            return root / "reference" / n
    raise GenerateError("no reference PWML under %s" % (root / "reference"))


def _export_pwml(v: Dict[str, Any], opts: Dict[str, Any]) -> Tuple[Optional[bytes], dict, Optional[str]]:
    """(pwml bytes, pwml IR, error). The same call the replays make: run_pwml_export on a temp project root."""
    from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE, coerce_mode
    from t2pw.service import postpipeline
    ref = _reference()
    with tempfile.TemporaryDirectory(prefix="t2pw-gen-") as td, _offline():
        res = postpipeline.run_pwml_export(
            v, opts.get("pathway_name", "Generated Pathway"), "", "Metabolic", Path(td), ref, 3200, 1400, "#FFFFFF",
            None, bool(opts.get("strict_db", True)), None, None, opts.get("pathway_context"),
            coerce_mode(opts.get("export_mode", DEFAULT_EXPORT_MODE)))
    if res.get("ok") and res.get("xml_bytes"):
        return res["xml_bytes"], res.get("pwml_ir") or {}, None
    return None, res.get("pwml_ir") or {}, str(res.get("error") or "PWML export failed")


def _sbml(v: Dict[str, Any], meta: Dict[str, Any]) -> str:
    from t2pw.clean_model.builder import build_from_payload
    from t2pw.clean_model.export.sbml import sbml_text
    return sbml_text(build_from_payload(v, meta))


def _kgml(v: Dict[str, Any], meta: Dict[str, Any]) -> str:
    from t2pw.clean_model.builder import build_from_payload
    from t2pw.clean_model.export.kgml import kgml_text
    return kgml_text(build_from_payload(v, meta))[0]


def generate(big: Dict[str, Any], scope: ScopeSpec = DEFAULT, fmt: str = "pwml", *,
             cache_dir=None, export: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None) -> bytes:
    """One format of one scoped view, as bytes.

    ``export``: PWML export controls (pathway_name, strict_db, export_mode, pathway_context); the default is the
    exporter's own. ``meta``: paper metadata for SBML/KGML (clean model ``read_run_meta``). Both are part of the
    cache key next to the view hash, so a different control never returns another control's bytes.
    Raises GenerateError when the PWML export refuses the view (pwml, svg and html need it; sbgn falls back to its grid).
    """
    if fmt not in FORMATS:
        raise ValueError("fmt must be one of %s, got %r" % (", ".join(FORMATS), fmt))
    v = view(big, scope)
    opts, meta = dict(export or {}), dict(meta or {})
    vh, oh = _sha(_canon(v)), _sha(_canon([opts, meta]))[:8]
    cache = Path(cache_dir) if cache_dir else None
    hit = cache / ("%s.%s.%s" % (vh, oh, fmt)) if cache else None
    if hit is not None and hit.exists():
        return hit.read_bytes()

    def pwml() -> Tuple[Optional[bytes], dict, Optional[str]]:
        k = vh + oh
        if k not in _MEMO:
            if len(_MEMO) >= 4:
                _MEMO.pop(next(iter(_MEMO)))
            _MEMO[k] = _export_pwml(v, opts)
        return _MEMO[k]

    if fmt == "json":
        out = json.dumps(v, ensure_ascii=False, indent=1).encode("utf-8")
    elif fmt == "sbml":
        out = _sbml(v, meta).encode("utf-8")
    elif fmt == "kgml":
        out = _kgml(v, meta).encode("utf-8")
    else:
        xml, ir, err = pwml()
        if fmt == "pwml":
            if xml is None:
                raise GenerateError(err)
            out = xml
        elif fmt == "sbgn":
            from t2pw.vision.export_sbgn import sbgn_text
            out = sbgn_text(v, ir=ir if xml is not None else None).encode("utf-8")
        else:
            if xml is None:
                raise GenerateError(err)
            from t2pw.vision import diagram
            out = (diagram.render_svg(ir) if fmt == "svg" else diagram.render_html(ir)).encode("utf-8")
    if hit is not None:
        cache.mkdir(parents=True, exist_ok=True)
        hit.write_bytes(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m t2pw.vision.generate", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="a run directory (final_mapped.json) or a big payload .json")
    ap.add_argument("--scope", default="default", help="paper | default | full | hops:N")
    ap.add_argument("--fmt", required=True, choices=FORMATS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="cache directory (optional)")
    a = ap.parse_args(argv)
    from t2pw.vision.payload import big_from_run, load_big_payload
    src, export, meta = Path(a.source), None, None
    if src.is_dir():
        from t2pw.clean_model.builder import read_run_meta
        big, export, meta = big_from_run(src), export_options_from_run(src), read_run_meta(str(src))
    else:
        big = load_big_payload(src)
    try:
        data = generate(big, parse_scope(a.scope), a.fmt, cache_dir=a.cache, export=export, meta=meta)
    except GenerateError as e:
        print("generate: %s" % e, file=sys.stderr)
        return 2
    Path(a.out).write_bytes(data)
    print("%s: %d bytes" % (a.out, len(data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
