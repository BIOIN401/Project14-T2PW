"""Render the W6 pages to static HTML for screenshots (no server is started).

Uses fastapi TestClient on a temp DB/work root under C:/t/wsW6tmp, the W0 stub (nothing
runs), and store calls to put one job in "running" and one in "done". The stylesheet
links are rewritten to file:// paths, the htmx script tag is dropped, and the diagram
<img> points at a local placeholder PNG drawn here (NOT a real pathway rendering).

Usage: python render_pages.py <outdir> <workdir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from t2pw.service import organisms
from t2pw.service.pathway import PathwayResult
from t2pw.web.app import create_app
from t2pw.web.jobs import JobStore, result_to_dict

out, work = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
static = (Path(__file__).resolve().parents[4] / "src" / "t2pw" / "web" / "static").as_uri()

app = create_app(work / "jobs.sqlite", work / "work", "https://t2pw.example", rate_secret=b"x")
c = TestClient(app, follow_redirects=False)
org = organisms.get_by_label("Homo sapiens").label


def post(**kw):
    data = {"mode": "name", "organism_1": org, "consent": "1", "name_1": "glycolysis"}
    data.update(kw)
    return re.search(r"/job/([0-9a-f]{32})", c.post("/submit", data=data).headers["location"]).group(1)


def save(name: str, html: str, img: str = "") -> None:
    html = html.replace('src="/static/js/htmx.min.js" defer', "").replace("<script >", "")
    html = re.sub(r"<script[^>]*></script>", "", html)
    html = html.replace('"/static/', f'"{static}/')
    if img:
        html = re.sub(r'src="/job/[^"]+/file/image"', f'src="{img}"', html)
    (out / name).write_text(html, encoding="utf-8")


png = out / "placeholder-diagram.png"
im = Image.new("RGB", (760, 300), "white")
d = ImageDraw.Draw(im)
for i, label in enumerate(["metabolite A", "metabolite B", "metabolite C", "metabolite D"]):
    x = 20 + i * 190
    d.rounded_rectangle([x, 110, x + 150, 190], 12, outline="#0b5f7f", width=3)
    d.text((x + 28, 144), label, fill="#18212b")
    if i:
        d.line([x - 40, 150, x, 150], fill="#18212b", width=3)
im.save(png)

save("submit.html", c.get("/").text)

# job 1: finished, flagged for review, with files and a diagram
done = post(name_1="urea cycle")
st = JobStore(work / "jobs.sqlite")
it = st.claim_next(3600)
files = work / "work" / f"item-{it['id']}" / "attempt-1" / "out"
files.mkdir(parents=True)
(files / "pathway.pwml").write_text("<pwml/>")
(files / "pathway.png").write_bytes(png.read_bytes())
(files / "citations.json").write_text("{}")
r = result_to_dict(PathwayResult(status="pass"))
r.update(release_status={"status": "review_required"}, pwml_path=str(files / "pathway.pwml"),
         image_path=str(files / "pathway.png"), report_paths={"citations.json": str(files / "citations.json")},
         warnings=["Two reactions have no supporting citation and need a curator check."],
         message="Finished with items flagged for review.")
st.finish(it["id"], it["attempt"], r, 0)
save("results-done.html", c.get(f"/job/{done}/0").text, img=png.as_uri())

# job 2: first pathway running, second queued
running = post(name_1="glycolysis", name_2="tricarboxylic acid cycle", organism_2=org)
st.claim_next(3600)
save("job-running.html", c.get(f"/job/{running}").text)
st.close()
print("wrote", sorted(p.name for p in out.iterdir()))
