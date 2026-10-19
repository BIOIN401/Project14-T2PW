"""ORCH-747 helper: flatten a scanned two-column PDF into a text a gold drafter can
quote from. Writes nothing into the repository.

Why this is not ``t2pw.extraction.pdf_parser``: that reader groups words into lines by
vertical position, and on a two-column page the columns share a baseline. Its output
interleaves them, so sentences break mid-word and continue with text from the other
column. Quote verification folds whitespace away, so a shredded text still hashes and
still "verifies" quotes that are themselves shredded. The defect is silent.

Two rules, both learned by measurement on PMC319880:

* The gutter is found from the page's own whitespace, never from the page midpoint. A
  long word in the left column crosses the midpoint often enough that a midpoint rule
  misclassifies whole blocks as full-width, reproducing the interleaving. With a midpoint
  rule, page 4 still read "the bind either P1 P2, whereas the A sitecan or B site".
* A journal's column geometry is constant across pages, but figure labels and tables fill
  the gutter on some of them, hiding it. So the gutter is taken from the pages that do
  reveal it and applied to every page. Pages 1 to 3 of this paper are figure-heavy and
  reveal no gutter of their own.

Reading order per page: full-width lines in place (title block, spanning figure labels),
then the left column top to bottom, then the right column.

Usage: python extract_paper_text.py <pdf> <out_txt>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pdfplumber

LINE_TOLERANCE = 3.0          # pt; words within this vertical distance are one line
MIN_GUTTER_WIDTH = 8          # pt; narrower than this is word spacing, not a gutter
MIN_COLUMN_GAP = 10           # pt; the measured inter-column gap falls to 22 pt on
                              # page 3, so a 24 pt threshold rejects a real column break
SEARCH_BAND = (0.28, 0.72)    # fraction of page width to search for the gutter


def _lines(words):
    out = []
    for w in sorted(words, key=lambda w: (round(w["top"] / LINE_TOLERANCE), w["x0"])):
        if out and abs(out[-1][0]["top"] - w["top"]) <= LINE_TOLERANCE:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in out]


def _gutter(words, width, n_lines):
    """The ``(start, end)`` of this page's column gutter, or ``None``.

    Coverage is how many words span each x. A two-column body leaves a band of near-zero
    coverage; the few words that cross it belong to full-width lines, so a small non-zero
    minimum is tolerated rather than read as "no gutter".
    """
    lo, hi = int(width * SEARCH_BAND[0]), int(width * SEARCH_BAND[1])
    if hi <= lo:
        return None
    cov = [0] * (hi - lo + 1)
    for w in words:
        a, b = max(lo, int(w["x0"])), min(hi, int(w["x1"]))
        for x in range(a, b + 1):
            cov[x - lo] += 1
    floor = min(cov)
    if floor > max(1, int(0.10 * max(1, n_lines))):
        return None
    best_len, best_start, run = 0, None, 0
    for i, c in enumerate(cov):
        if c <= floor:
            run += 1
            if run > best_len:
                best_len, best_start = run, i - run + 1
        else:
            run = 0
    if best_start is None or best_len < MIN_GUTTER_WIDTH:
        return None
    return lo + best_start, lo + best_start + best_len


def page_text(words, width, shared=None):
    """``(text, diagnostics)`` for one page's words, in reading order."""
    if not words:
        return "", {"columns": 0}
    lines = _lines(words)
    g = shared if shared is not None else _gutter(words, width, len(lines))
    if g is None:
        return ("\n".join(" ".join(w["text"] for w in l) for l in lines),
                {"columns": 1, "gutter": None, "full_width_lines": len(lines)})

    g0, g1 = g
    boundary = (g0 + g1) / 2.0
    emitted, left, right = [], [], []
    for line in lines:
        # Split where the line's own whitespace crosses the boundary. Using the measured
        # band as a no-go zone instead fails when a page's right column starts inside the
        # shared band: page 3 of PMC319880 begins its right column at x=335 while the
        # median band runs to 340, which made the whole line read as full width and
        # re-interleaved the paper's central autoregulation sentence.
        cut, cut_gap = None, 0.0
        for i in range(len(line) - 1):
            gap = line[i + 1]["x0"] - line[i]["x1"]
            if line[i]["x1"] <= boundary <= line[i + 1]["x0"] and gap > cut_gap:
                cut, cut_gap = i, gap
        if cut is not None and cut_gap >= MIN_COLUMN_GAP:
            left.append(line[:cut + 1])
            right.append(line[cut + 1:])
            continue
        x0, x1 = min(w["x0"] for w in line), max(w["x1"] for w in line)
        if x0 < boundary and x1 > boundary:
            emitted.append(line)      # runs continuously across the boundary
            continue
        (left if (x0 + x1) / 2.0 < boundary else right).append(line)

    def render(ls):
        return "\n".join(" ".join(w["text"] for w in l) for l in ls)

    text = "\n".join(x for x in (render(emitted), render(left), render(right)) if x.strip())
    return text, {"columns": 2, "gutter": [g0, g1], "full_width_lines": len(emitted),
                  "left_lines": len(left), "right_lines": len(right)}


def shared_gutter(per_page):
    """The median gutter of the pages that reveal one, or ``None``."""
    found = [g for g in (_gutter(w, width, len(_lines(w))) if w else None
                         for width, w in per_page) if g]
    if not found:
        return None
    return (sorted(g[0] for g in found)[len(found) // 2],
            sorted(g[1] for g in found)[len(found) // 2])


def main() -> int:
    pdf, out_txt = sys.argv[1], sys.argv[2]
    with pdfplumber.open(pdf) as doc:
        per_page = [(p.width, p.extract_words(use_text_flow=False, keep_blank_chars=False))
                    for p in doc.pages]
    shared = shared_gutter(per_page)

    pages, diag = [], []
    for i, (width, words) in enumerate(per_page, 1):
        t, d = page_text(words, width, shared)
        pages.append(t)
        diag.append({"page": i, **d})
    flat = re.sub(r"\s+", " ", "\n".join(pages)).strip()
    Path(out_txt).write_text(flat, encoding="utf-8")

    toks = flat.split()
    alpha = [w for w in toks if any(c.isalpha() for c in w)]
    junk = [w for w in alpha if re.search(r"[^A-Za-z0-9\-'(),.;:/%+\[\]]", w)]

    def fold(s):
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    folded = fold(flat)
    # Phrases whose contiguity proves the columns were separated rather than interleaved.
    checks = ["direct correlation",
              "anti-inducer D-fucose",
              "catabolite gene-activator protein",
              "The procedures for their isolation, end labeling, and sequence analysis",
              "enzymes required for the metabolism of L-arabinose are coded by the araBAD operon"]
    print(json.dumps({
        "pdf_sha256": hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(Path(out_txt).read_bytes()).hexdigest(),
        "text_chars": len(flat), "words": len(toks), "pages": len(pages),
        "shared_gutter": shared,
        "junk_token_share": round(len(junk) / max(1, len(alpha)), 4),
        "per_page": diag,
        "contiguity_checks": {c: (fold(c) in folded) for c in checks},
        "body_sample": flat[2300:2900],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
