"""Render the W5 styleguide to a static file next to this script (W5 evidence).

Usage, from the repo root:  python docs/webserver/evidence/W5/render_styleguide.py [theme]
Writes styleguide.html (theme follows the system) or styleguide-<theme>.html.
The CSS is linked by a relative path into src/t2pw/web/static, so open the file
from inside a checkout.
"""

import sys
from pathlib import Path

import jinja2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
theme = sys.argv[1] if len(sys.argv) > 1 else ""
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(ROOT / "src/t2pw/web/templates")),
    autoescape=True,
    undefined=jinja2.StrictUndefined,
)
html = env.get_template("styleguide.html").render(
    static_url="../../../../src/t2pw/web/static", theme=theme
)
out = HERE / (f"styleguide-{theme}.html" if theme else "styleguide.html")
out.write_text(html, encoding="utf-8", newline="\n")
print(out)
