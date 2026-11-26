"""VIS-4: compound structure pictures (t2pw.render.structures) and the cropped, readable render. No network."""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import pytest

from t2pw.render import pathbank, structures

REAL_DOWNLOAD = structures.download  # the fixture below replaces the module attribute


def _png(w: int = 60, h: int = 40, ink=(0, 0, 0)) -> bytes:
    """A PubChem-like picture: grey 245 backdrop with a dark bar in the middle."""
    from PIL import Image

    im = Image.new("RGB", (w, h), (245, 245, 245))
    for x in range(w // 4, 3 * w // 4):
        for y in range(h // 3, 2 * h // 3):
            im.putpixel((x, y), ink)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(structures.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(structures, "download", lambda cid: pytest.fail(f"network fetch of {cid}"))
    monkeypatch.setattr(structures.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network in a test"))
    return tmp_path / "cache"


def test_fetches_caches_and_then_reads_the_cache(offline: Path):
    calls = []

    def fetch(cid):
        calls.append(cid)
        return _png()

    got = structures.structures(["CID:5950", "CID:5950", "CHEBI:17333", "CID:abc", "", "CID:0", "KEGG:1", "x"], 5, fetch)
    assert sorted(got) == ["CHEBI:17333", "CID:5950"] and sorted(calls) == ["CHEBI:17333", "CID:5950"]
    assert (offline / "CID_5950.png").read_bytes() == _png()
    again = structures.structures(["CID:5950", "CHEBI:17333"], 5, lambda cid: pytest.fail("cache not used"))
    assert again == got


def test_key_prefers_chebi_because_our_cids_can_be_wrong():
    # item-6: isochorismate carries CID 4141 (5-Methylfurtrethonium) but ChEBI 29780 (isochorismate)
    assert structures.key_for("29780", "4141") == "CHEBI:29780"
    assert structures.key_for("CHEBI:29780", "") == "CHEBI:29780"
    assert structures.key_for("", "5950") == "CID:5950"
    assert structures.key_for("", "") is None and structures.key_for("0", "x") is None
    assert structures.key_for("\u00b2", "5950") == "CID:5950"  # a broken ChEBI id falls back to the CID


def test_misses_are_cached_for_good_and_failures_for_a_day(offline: Path, monkeypatch):
    def fetch(cid):
        if cid == "CID:1":
            raise structures.Missing(cid)
        raise OSError("network down")

    assert structures.structures(["CID:1", "CID:2"], 5, fetch) == {}
    assert (offline / "CID_1.miss").exists() and (offline / "CID_2.fail").exists()
    assert structures.structures(["CID:1", "CID:2"], 5, lambda cid: pytest.fail("asked again")) == {}
    old = time.time() - structures.FAIL_RETRY_S - 60
    os.utime(offline / "CID_2.fail", (old, old))
    asked = []
    structures.structures(["CID:1", "CID:2"], 5, lambda cid: asked.append(cid) or _png())
    assert asked == ["CID:2"]  # the failure is asked again after FAIL_RETRY_S, the miss never


def test_one_retry_then_requests_are_paced(offline: Path):
    seen = []

    def flaky(cid):
        seen.append((cid, time.monotonic()))
        if len([c for c, _ in seen if c == cid]) == 1:
            raise OSError("503 busy")
        return _png()

    got = structures.structures(["CID:11", "CID:12", "CID:13"], 10, flaky)
    assert sorted(got) == ["CID:11", "CID:12", "CID:13"] and len(seen) == 6
    starts = sorted(t for _, t in seen)
    assert all(b - a >= structures.MIN_GAP_S * 0.8 for a, b in zip(starts, starts[1:]))


def test_budget_bounds_a_slow_network_and_zero_means_cache_only(offline: Path):
    release = threading.Event()

    def slow(cid):
        release.wait(10)
        return _png()

    t0 = time.perf_counter()
    try:
        assert structures.structures(["CID:7", "CID:8"], 0.2, slow) == {}
        assert time.perf_counter() - t0 < 2
    finally:
        release.set()
    assert structures.structures(["CID:9"], 0, lambda cid: pytest.fail("fetched with budget 0")) == {}


def test_download_maps_http_answers(monkeypatch):
    import urllib.error

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(structures.urllib.request, "urlopen", lambda req, timeout: Resp(_png()))
    assert REAL_DOWNLOAD("CID:5950").startswith(structures.PNG_MAGIC)
    urls = []
    monkeypatch.setattr(structures.urllib.request, "urlopen", lambda req, timeout: urls.append(req.full_url) or Resp(_png()))
    REAL_DOWNLOAD("CHEBI:29780")
    assert urls == ["https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/CHEBI:29780/PNG"]
    # review F3: a 200 that is not a PNG (a maintenance page) is retryable, never a permanent miss
    monkeypatch.setattr(structures.urllib.request, "urlopen", lambda req, timeout: Resp(b"<html>maintenance</html>"))
    with pytest.raises(structures.Unusable):
        REAL_DOWNLOAD("CID:5950")

    def http(code, body=None):
        def urlopen(req, timeout):
            assert req.full_url.endswith("/cid/5950/PNG") and timeout == structures.REQUEST_TIMEOUT_S
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(body) if body is not None else None)
        return urlopen

    for code, body in ((404, None), (400, b"Status: 404\nCode: PUGREST.NotFound\nMessage: No CID found")):
        monkeypatch.setattr(structures.urllib.request, "urlopen", http(code, body))
        with pytest.raises(structures.Missing):
            REAL_DOWNLOAD("CID:5950")
    for code, body in ((400, b"Code: PUGREST.BadRequest"), (403, None), (410, None), (429, None), (503, None)):
        monkeypatch.setattr(structures.urllib.request, "urlopen", http(code, body))
        with pytest.raises(urllib.error.HTTPError):
            REAL_DOWNLOAD("CID:5950")


def _ihdr(w: int, h: int) -> bytes:
    """Just a PNG signature and an IHDR claiming w x h: enough for the size check, nothing to decode."""
    return structures.PNG_MAGIC + b"\x00\x00\x00\x0dIHDR" + w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"


def test_oversized_or_truncated_pngs_are_never_decoded_or_cached(offline: Path, monkeypatch):
    # review F2: a 2 MB PNG can decode to a multi-GB array; the IHDR is checked first
    assert structures.png_ok(_png()) and structures.png_ok(_ihdr(2000, 300))
    assert not structures.png_ok(_ihdr(2001, 300)) and not structures.png_ok(_ihdr(300, 50000))
    assert not structures.png_ok(b"\x89PNG") and not structures.png_ok(b"GIF89a" + b"\x00" * 30)
    monkeypatch.setattr(pathbank.mpimg, "imread", lambda *a, **k: pytest.fail("decoded an oversized PNG"))
    assert pathbank._picture(_ihdr(40000, 40000)) is None

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    for body in (_ihdr(9000, 9000), _png() + b"\x00" * structures.MAX_BYTES):  # too large; body over the read cap
        monkeypatch.setattr(structures.urllib.request, "urlopen", lambda req, timeout, b=body: Resp(b))
        with pytest.raises(structures.Unusable):
            REAL_DOWNLOAD("CID:5950")
    # whatever a fetcher returns, only a PNG that passes the check is cached as a picture
    assert structures.structures(["CID:7"], 5, lambda key: _ihdr(9000, 9000)) == {}
    assert not (offline / "CID_7.png").exists() and (offline / "CID_7.fail").exists()


def test_unusable_answers_and_blocks_are_retryable_failures_not_misses(offline: Path):
    # review F3: one IP block or maintenance page must not mark every compound missing for good
    import urllib.error

    def blocked(key):
        if key == "CID:1":
            raise urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        raise structures.Unusable(key)

    assert structures.structures(["CID:1", "CID:2"], 5, blocked) == {}
    assert sorted(x.name for x in offline.iterdir()) == ["CID_1.fail", "CID_2.fail"]


@pytest.mark.parametrize("chebi, cid", [("\u00b2", ""), ("\u0663", ""), ("9" * 5000, ""), ("", "\u0663"),
                                        ("", "12345678901"), ("-5", "+7"), ("1e3", " "), ("CHEBI:\u00b2", "\u00b9")])
def test_key_for_takes_ascii_ids_only_and_never_raises(chebi, cid):
    # review F1: str.isdigit() accepts "\u00b2" (int() raises) and "\u0663" (int() gives 3: the wrong molecule)
    assert structures.key_for(chebi, cid) is None
    assert structures.structures([f"CID:{cid}", f"CHEBI:{chebi}"], 0) == {}


def test_picture_drops_the_backdrop_and_crops_to_the_drawing():
    pic = pathbank._picture(_png(60, 40))
    assert pic.shape[2] == 4
    assert pic.shape[0] < 40 and pic.shape[1] < 60  # cropped to the bar plus a small pad
    assert pic[0, 0, 3] == 0  # backdrop transparent
    assert pic[pic.shape[0] // 2, pic.shape[1] // 2, 3] == 1  # ink kept
    assert pathbank._picture(b"not a png") is None
    assert pathbank._picture(_png(20, 20, ink=(245, 245, 245))) is None  # nothing drawn


ONE = """<?xml version="1.0"?>
<super-pathway-visualization><pathway-visualization-contexts><pathway-visualization-context><pathway-visualization>
 <height>3000</height><width>4000</width>
 <pathway><compounds>
  <compound><id>7</id><name>L-Alanine</name><pubchem-cid>5950</pubchem-cid></compound>
  <compound><id>8</id><name>ATP</name><pubchem-cid>5957</pubchem-cid></compound>
 </compounds></pathway>
 <compound-locations>
  <compound-location><id>1</id><compound-id>7</compound-id><visualization-template-id>81</visualization-template-id>
    <hidden>false</hidden><x>1000</x><y>1000</y><zindex>10</zindex><width>200</width><height>190</height></compound-location>
  <compound-location><id>2</id><compound-id>8</compound-id><visualization-template-id>42</visualization-template-id>
    <hidden>false</hidden><x>1300</x><y>1000</y><zindex>10</zindex><width>50</width><height>30</height></compound-location>
 </compound-locations>
</pathway-visualization></pathway-visualization-context></pathway-visualization-contexts></super-pathway-visualization>
"""


@pytest.fixture
def one(tmp_path: Path) -> Path:
    p = tmp_path / "one.pwml"
    p.write_text(ONE, encoding="utf-8")
    return p


def _image(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def test_structure_box_shows_the_picture_and_pills_ask_for_none(one: Path, offline: Path, monkeypatch):
    asked = []
    monkeypatch.setattr(structures, "download", lambda cid: asked.append(cid) or _png(60, 40, ink=(255, 0, 0)))
    im = _image(pathbank.render_pwml(one, "png", structure_budget_s=5))
    assert asked == ["CID:5950"]  # the ATP pill draws no structure, so its CID never leaves the machine
    # cropped to the drawing: x 1000..1350, y 1000..1190, plus the 40-unit margin, at scale 1
    assert abs(im.size[0] - (350 + 80)) <= 1 and abs(im.size[1] - (190 + 80)) <= 1
    reds = [im.getpixel((x, y)) for x in range(40, 240) for y in range(80, 230)]
    assert any(r > 200 and g < 60 and b < 60 for r, g, b in reds)  # the picture is in the box body


def test_without_a_picture_the_box_stays_empty(one: Path, monkeypatch):
    monkeypatch.setattr(structures, "download", lambda cid: (_ for _ in ()).throw(OSError("offline")))
    im = _image(pathbank.render_pwml(one, "png", structure_budget_s=5))
    body = [im.getpixel((x, y)) for x in range(60, 220) for y in range(100, 210)]
    assert all(px == (255, 255, 255) for px in body)


def test_render_is_deterministic_with_a_warm_cache(one: Path, monkeypatch):
    monkeypatch.setattr(structures, "download", lambda cid: _png())
    a = pathbank.render_pwml(one, "png", structure_budget_s=5)
    monkeypatch.setattr(structures, "download", lambda cid: pytest.fail("cache not used"))
    assert pathbank.render_pwml(one, "png", structure_budget_s=5) == a


@pytest.mark.parametrize("w, h, want", [(1000, 800, 1.0), (4800, 1000, 0.85), (20000, 500, 0.4), (2400, 2400, 1.0)])
def test_output_scale_keeps_labels_readable_under_a_hard_cap(w, h, want):
    assert pathbank.output_scale(w, h) == pytest.approx(want)
    assert pathbank.output_scale(w, h, max_px=500) <= 500 / max(w, h) + 1e-9


ORPHAN = """<?xml version="1.0"?>
<super-pathway-visualization><pathway-visualization-contexts><pathway-visualization-context><pathway-visualization>
 <height>500</height><width>500</width>
 <pathway><proteins><protein><id>1</id><name>EntB</name></protein></proteins></pathway>
 <protein-locations><protein-location><id>1</id><protein-id>1</protein-id><visualization-template-id>2</visualization-template-id>
   <x>100</x><y>100</y><width>150</width><height>70</height><zindex>8</zindex></protein-location></protein-locations>
 <edges>
  <edge><id>1</id><path>M0 135 L100 135</path><visualization-template-id>5</visualization-template-id><zindex>18</zindex></edge>
  <edge><id>2</id><path>M100 135 L250 135</path><visualization-template-id>83</visualization-template-id><zindex>18</zindex></edge>
 </edges>
 <reaction-visualizations><reaction-visualization><id>1</id><edge-id>1</edge-id></reaction-visualization></reaction-visualizations>
</pathway-visualization></pathway-visualization-context></pathway-visualization-contexts></super-pathway-visualization>
"""


def test_an_edge_nothing_refers_to_is_not_drawn_over_the_enzyme(tmp_path: Path):
    # VIS-1 writes a connector from each enzyme to its own reaction as a shaft through the enzyme; no reaction,
    # transport or interaction refers to it and PathWhiz's viewer drops it. Drawn, it hid the enzyme's label.
    p = tmp_path / "orphan.pwml"
    p.write_text(ORPHAN, encoding="utf-8")
    root = pathbank.read_xml_with_repairs(str(p))
    items, _, _ = pathbank._items(root, pathbank.find_first_visualization(root))
    assert len(items) == 2  # the protein and edge 1; edge 2 (the orphan) skipped


def test_compound_header_wraps_like_pathbank_and_breaks_after_hyphens():
    # PathBank PW012911.svg: Verdana 20, "3,8-divinyl" / "chlorophyllide a"; one line for "Oxoglutaric acid"
    assert pathbank._header_lines("3,8-divinyl chlorophyllide a", 200) == (["3,8-divinyl", "chlorophyllide a"], 20.0)
    assert pathbank._header_lines("Oxoglutaric acid", 200) == (["Oxoglutaric acid"], 20.0)
    # item-6: textwrap never broke "dihydro-2,3", so the name shrank to 10 units
    lines, size = pathbank._header_lines("2,3-dihydro-2,3-dihydroxybenzoate", 200)
    assert lines == ["2,3-dihydro-2,3-", "dihydroxybenzoate"] and size >= 16
    # review F6: every line is measured to fit inside the box (0.55 em per character let it cross the border)
    for name in ("2,3-dihydro-2,3-dihydroxybenzoate", "2,3-dihydroxybenzoic acid", "Oxoglutaric acid"):
        lines, size = pathbank._header_lines(name, 200)
        assert all(pathbank._text_width(t, size) <= 190 for t in lines)
    lines, size = pathbank._header_lines("tetrahydrogeranylgeranyl diphosphate", 200)
    assert lines == ["tetrahydrogeranylgeranyl", "diphosphate"] and 12 <= size < 20  # one word too wide: shrink
