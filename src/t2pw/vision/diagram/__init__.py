"""Pathway diagram from a PWML IR dict. Moved as-is from the v2 prototype (render.py); screenshots and the batch/page
drivers are not part of the package."""
import re
from html import escape as _esc
from importlib import resources

from . import metrics as _M
from . import render as _R
from .selfcheck import check_svg

__all__ = ["render_svg", "render_html", "diagram_metrics", "selfcheck"]


def _css():
    return resources.files(__package__).joinpath("diagram.css").read_text(encoding="utf-8")


def _draw(ir, style=None, ident="b"):
    style = style or _R.pick_style(_R.metrics(ir))[0]
    svg, m, lay = _R.render(ir, style, ident)
    return style, svg, m, lay


def render_svg(ir: dict, *, style: str | None = None, ident: str = "b") -> str:
    """Standalone SVG with the diagram CSS inlined. style: bands|cell|spine (default: the prototype's auto-pick)."""
    svg = _draw(ir, style, ident)[1]
    return re.sub(r"(<svg [^>]*>)", lambda m: m.group(1) + "<style>" + _css() + "</style>", svg, count=1)


def render_html(ir: dict, *, title: str | None = None, style: str | None = None) -> str:
    """One self-contained page: the SVG (with its regulation toggle and legend) inside a minimal shell."""
    t = _esc(title or (ir.get("pathway") or {}).get("name") or "T2PW pathway diagram")
    svg = _draw(ir, style)[1]
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{t}</title><style>{_css()}\nbody{{background:var(--bg);color:var(--ink);margin:0;padding:16px;font-family:system-ui,sans-serif}}"
            f"h1{{font-size:1.2rem}}.scroll{{overflow:auto;background:var(--canvas)}}.scroll svg{{max-width:none}}</style></head>"
            f'<body><h1>{t}</h1><div class="scroll">{svg}</div></body></html>')


def diagram_metrics(ir: dict, *, style: str | None = None) -> dict:
    """Crossings, bends, enzyme links of the default view (plus full_view / hidden_links when regulation is hidden)."""
    style, _, m, lay = _draw(ir, style)
    out = dict(style=style, W=round(lay["W"]), H=round(lay["H"]), **_M.model_metrics(m))
    if getattr(m, "full", None) is not None:
        fm = _M.model_metrics(m.full)
        out["full_view"] = {k: fm[k] for k in ("crossings", "bends", "enzyme_links_gt200")}
        out["hidden_links"] = len(m.reg["keys"])
    return out


def selfcheck(ir: dict, svg: str) -> dict:
    """Faithfulness of the drawing against the IR. {'failures': [...]}; empty list = pass."""
    f = check_svg(ir, svg)
    return {"ok": not f, "failures": f}
