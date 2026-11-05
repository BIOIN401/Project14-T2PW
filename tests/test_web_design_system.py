"""NEW ACCEPTANCE TEST (W5, new capability).

The public UI design system: Jinja2 templates under ``src/t2pw/web/templates``
and hand-written CSS under ``src/t2pw/web/static/css``. Nothing here existed
before W5, so there is no base-commit failure to show (G9); these tests are the
acceptance contract W6 builds on:

* ``base.html`` and ``styleguide.html`` render with jinja2 (StrictUndefined);
* every public macro in ``macros/components.html`` is exercised;
* every CSS custom property used is defined, and the light, dark (system) and
  dark (override) theme blocks define the same token set;
* text colour pairs meet WCAG AA (4.5:1), UI borders and focus rings 3:1;
* no ``<script>``, no external URL, no ``@import`` or ``url()``.
"""

from __future__ import annotations

import re
from pathlib import Path

import jinja2
import pytest

WEB = Path(__file__).resolve().parents[1] / "src" / "t2pw" / "web"
TEMPLATES = WEB / "templates"
CSS_DIR = WEB / "static" / "css"
TOKENS = CSS_DIR / "tokens.css"
COMPONENTS = TEMPLATES / "macros" / "components.html"

TONES = ("neutral", "info", "success", "warning", "error")


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES)),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )


def _render(name: str, **ctx) -> str:
    return _env().get_template(name).render(**ctx)


def _macro(src: str) -> str:
    """Render one snippet with the component macros imported as ``c``."""
    return _env().from_string('{% import "macros/components.html" as c %}' + src).render()


def _css_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(CSS_DIR.glob("*.css")))


def _decls(block: str) -> dict:
    return {k: v.strip() for k, v in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block)}


def _theme_blocks() -> dict:
    css = TOKENS.read_text(encoding="utf-8")
    patterns = {
        "light": r':root,\s*\[data-theme="light"\]\s*\{([^}]*)\}',
        "dark_system": r'@media \(prefers-color-scheme: dark\)\s*\{\s*'
        r':root:not\(\[data-theme="light"\]\)\s*\{([^}]*)\}',
        "dark_override": r'\n\[data-theme="dark"\]\s*\{([^}]*)\}',
    }
    blocks = {}
    for name, pat in patterns.items():
        m = re.search(pat, css)
        assert m, f"tokens.css: theme block {name!r} not found"
        blocks[name] = _decls(m.group(1))
    return blocks


# ── Rendering ────────────────────────────────────────────────────────────


def test_base_renders_with_defaults():
    html = _render("base.html")
    assert html.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in html  # no data-theme: follow the system
    assert '<main id="main"' in html and 'href="#main"' in html
    assert 'href="/static/css/tokens.css"' in html and 'href="/static/css/site.css"' in html
    nav = re.search(r'<nav class="site-nav" aria-label="Main">(.*?)</nav>', html, re.S).group(1)
    assert re.findall(r">([^<]+)</a>", nav) == ["Submit", "Examples", "Help", "About / Cite"]
    assert "aria-current" not in nav
    assert "Citation placeholder" in html and "Lab credit placeholder" in html


def test_base_theme_override_and_active_nav():
    html = _render("base.html", theme="dark", active_nav="help", static_url="rel/static")
    assert '<html lang="en" data-theme="dark">' in html
    assert '<a href="/help" aria-current="page">Help</a>' in html
    assert 'href="rel/static/css/site.css"' in html
    # an unknown theme value is ignored, never echoed into the attribute
    assert "data-theme" not in _render("base.html", theme="purple")


def test_styleguide_renders_both_themes():
    html = _render("styleguide.html")
    assert '<section class="sg-panel" data-theme="light"' in html
    assert '<section class="sg-panel" data-theme="dark"' in html
    # ids are unique, so labels still point at one control each
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)
    for i in re.findall(r'<label[^>]*\sfor="([^"]+)"', html):
        assert i in ids, f"label for={i!r} has no control"


def test_every_public_macro_is_exercised_by_the_styleguide():
    src = COMPONENTS.read_text(encoding="utf-8")
    macros = [m for m in re.findall(r"\{%-?\s*macro\s+(\w+)\s*\(", src) if not m.startswith("_")]
    assert set(macros) >= {
        "button", "text_field", "textarea_field", "select_field", "checkbox_field",
        "file_field", "radio_group", "card", "status_badge", "steps", "alert",
        "data_table", "download_link", "chip",
    }
    guide = (TEMPLATES / "styleguide.html").read_text(encoding="utf-8")
    missing = [m for m in macros if not re.search(rf"\bc\.{m}\(", guide)]
    assert not missing, f"macros not shown in styleguide.html: {missing}"
    html = _render("styleguide.html")
    for marker in (
        'class="btn btn--primary"', 'class="btn btn--danger"', 'class="btn btn--primary btn--sm"',
        'type="email"', "<textarea", "<select", 'type="checkbox"', 'type="file"', 'type="radio"',
        'class="card"', 'class="card card--raised"', 'class="steps"', 'class="table"',
        'class="download"', 'class="chip chip--code"', 'class="chip chip--accession"',
        'class="alert alert--info"', 'class="alert alert--warning"', 'class="alert alert--error"',
    ):
        assert marker in html, marker
    for status in ("queued", "running", "done", "failed", "review_required"):
        assert f"badge--{status}" in html, status


# ── Macro behaviour ──────────────────────────────────────────────────────


def test_field_label_hint_and_error_are_wired():
    html = _macro('{{ c.text_field("n", "Name", hint="H", error="E", required=True) }}')
    assert '<label class="field__label" for="n">' in html
    assert 'aria-describedby="n-hint n-error"' in html and 'aria-invalid="true"' in html
    assert 'id="n-hint"' in html and 'id="n-error"' in html and " required" in html
    plain = _macro('{{ c.text_field("n", "Name") }}')
    assert "aria-describedby" not in plain and "aria-invalid" not in plain


def test_select_options_and_selection():
    html = _macro('{{ c.select_field("s", "S", ["a", ("b", "Bee")], selected="b") }}')
    assert '<option value="a">a</option>' in html
    assert '<option value="b" selected>Bee</option>' in html


def test_checkbox_defaults_and_attrs_passthrough():
    html = _macro('{{ c.checkbox_field("ok", "OK", checked=True, attrs={"hx-get": "/x"}) }}')
    assert " checked" in html and 'hx-get="/x"' in html
    assert " checked" not in _macro('{{ c.checkbox_field("ok", "OK") }}')


def test_button_link_and_disabled():
    assert _macro('{{ c.button("Go", href="/x") }}') == '<a class="btn btn--primary" href="/x">Go</a>'
    assert "disabled" in _macro('{{ c.button("Go", disabled=True) }}')


def test_status_badge_labels_and_unknown_status():
    assert ">Review required<" in _macro('{{ c.status_badge("review_required") }}')
    unknown = _macro('{{ c.status_badge("<odd>") }}')
    assert 'class="badge badge--neutral badge--unknown"' in unknown
    assert ">&lt;odd&gt;<" in unknown and "<odd>" not in unknown  # escaped text, never a class


def _status_badge(status: str) -> str:
    return _env().from_string(
        '{% import "macros/components.html" as c %}{{ c.status_badge(s) }}'
    ).render(s=status)


def test_status_badge_accepts_every_pathway_result_status():
    """Every PathwayResult.status value is a known badge, so a new value added to
    t2pw.service.pathway (W0 vocabulary) without a badge mapping fails here."""
    from t2pw.service import pathway

    statuses = {v for k, v in vars(pathway).items() if k.startswith("STATUS_")}
    assert statuses == {"pass", "fail", "timeout", "error", "scope_conflict"}
    expected_tone = {
        pathway.STATUS_PASS: "success",
        pathway.STATUS_FAIL: "error",
        pathway.STATUS_TIMEOUT: "error",
        pathway.STATUS_ERROR: "error",
        pathway.STATUS_SCOPE_CONFLICT: "warning",
    }
    labels = set()
    for status in statuses:
        html = _status_badge(status)
        assert f"badge--{expected_tone[status]}" in html, (status, html)
        assert f"badge--{status}" in html and "badge--unknown" not in html, (status, html)
        labels.add(re.search(r">([^<]+)</span>", html).group(1))
    assert len(labels) == len(statuses)  # fail / timeout / error keep their own text


def test_steps_mark_current_and_state_text():
    html = _macro('{{ c.steps([{"label": "A", "state": "done"}, {"label": "B", "state": "current"}, {"label": "C"}]) }}')
    assert html.count('aria-current="step"') == 1
    assert "(completed)" in html and "(in progress)" in html and "(not started)" in html


def test_alert_error_is_announced_and_others_are_not():
    assert 'role="alert"' in _macro('{% call c.alert("error") %}x{% endcall %}')
    assert "role=" not in _macro('{% call c.alert("warning", title="T") %}x{% endcall %}')


def test_table_caption_headers_and_markup_cells():
    html = _macro('{{ c.data_table(["A"], [[c.chip("X:1")], ["<b>"]], caption="Cap") }}')
    assert "<caption>Cap</caption>" in html and '<th scope="col">A</th>' in html
    assert '<code class="chip chip--code">X:1</code>' in html  # macro output not re-escaped
    assert "&lt;b&gt;" in html  # plain strings are escaped


def test_download_link_and_chip_link():
    html = _macro('{{ c.download_link("/f", "Get", filename="p.pwml", meta="PWML") }}')
    assert 'href="/f" download="p.pwml"' in html and "download__meta" in html
    assert 'href="/x"' in _macro('{{ c.chip("ID", kind="accession", href="/x") }}')


# ── Tokens ───────────────────────────────────────────────────────────────


def test_every_used_custom_property_is_defined():
    css = _css_text()
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    rendered = _render("styleguide.html") + _render("base.html")
    used = set(re.findall(r"var\(\s*(--[\w-]+)", css + rendered))
    assert used, "no var() found"
    assert not used - defined, f"used but undefined: {sorted(used - defined)}"


def test_themes_define_the_same_token_set():
    b = _theme_blocks()
    assert set(b["light"]) == set(b["dark_system"]) == set(b["dark_override"])
    assert b["dark_system"] == b["dark_override"], "the two dark blocks must be identical"
    assert len(b["light"]) >= 30


def test_styleguide_shows_every_colour_token():
    html = _render("styleguide.html")
    colours = [k for k in _theme_blocks()["light"] if k.startswith("--color-")]
    missing = [k for k in colours if f"var({k})" not in html]
    assert not missing, missing


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    assert re.fullmatch(r"[0-9a-fA-F]{6}", h), hex_colour
    chans = []
    for i in (0, 2, 4):
        c = int(h[i : i + 2], 16) / 255
        chans.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = chans
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# (foreground, background) token pairs, as the components use them.
TEXT_PAIRS = [
    ("text", "bg"), ("text", "surface"), ("text", "surface-muted"), ("text", "primary-subtle"),
    ("text-muted", "bg"), ("text-muted", "surface"), ("text-muted", "surface-muted"),
    ("primary", "bg"), ("primary", "surface"), ("primary", "primary-subtle"),
    ("primary-hover", "bg"), ("primary-hover", "surface"),
    ("on-primary", "primary"), ("on-primary", "primary-hover"),
    ("on-danger", "danger"), ("on-danger", "danger-hover"),
    ("error-fg", "surface"), ("info-fg", "surface"),
    ("surface", "success-fg"), ("surface", "error-fg"),
] + [(f"{t}-fg", f"{t}-bg") for t in TONES] + [("text", f"{t}-bg") for t in TONES]

NON_TEXT_PAIRS = [
    ("border-strong", "surface"), ("border-strong", "bg"),
    ("focus", "bg"), ("focus", "surface"), ("focus", "surface-muted"),
]


@pytest.mark.parametrize("theme", ["light", "dark_override"])
def test_wcag_aa_contrast(theme):
    tokens = _theme_blocks()[theme]
    colour = lambda name: tokens[f"--color-{name}"]  # noqa: E731
    failures = []
    for pairs, minimum in ((TEXT_PAIRS, 4.5), (NON_TEXT_PAIRS, 3.0)):
        for fg, bg in pairs:
            ratio = _contrast(colour(fg), colour(bg))
            if ratio < minimum:
                failures.append(f"{fg} on {bg}: {ratio:.2f} < {minimum}")
    assert not failures, failures


# ── Hygiene and accessibility basics ─────────────────────────────────────


HTMX_TAG = '<script src="/static/js/htmx.min.js" defer></script>'


def test_no_script_no_external_urls():
    files = sorted(TEMPLATES.rglob("*.html")) + sorted(CSS_DIR.glob("*.css"))
    assert len(files) >= 5
    texts = {p.name: p.read_text(encoding="utf-8") for p in files}
    assert "layout.html" in texts  # W6 carve-out: the one page layout must exist and is checked below
    texts["rendered styleguide"] = _render("styleguide.html")
    texts["rendered base"] = _render("base.html", theme="dark")
    for name, text in texts.items():
        if name == "layout.html":
            # W6 carve-out: the page layout carries exactly one script, the vendored htmx tag.
            assert text.count("<script") == 1 and text.count(HTMX_TAG) == 1, name
        else:
            assert not re.search(r"<script", text, re.I), name
        assert not re.search(r"https?:", text, re.I), name
        assert not re.search(r"""(?:src|href)\s*=\s*["']//""", text, re.I), name
        assert not re.search(r"@import|url\(", text, re.I), name


def test_focus_and_reduced_motion_rules_exist():
    css = _css_text()
    assert ":focus-visible" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ".visually-hidden" in css and ".skip-link:focus" in css
