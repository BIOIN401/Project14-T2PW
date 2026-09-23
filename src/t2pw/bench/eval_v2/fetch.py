"""Read-only EuropePMC full-text fetch for gold exhaustiveness checks (ORCH-742c).

EVALUATION-ONLY. Stdlib HTTP, no key, no LLM. Full texts are written OUTSIDE the
repository (default ``C:/t/eval-fulltext``) because committing whole papers would
bloat ``.git``; what is committed is the manifest this prints: PMCID, source URL,
fetch date and the sha256 of every fetched byte stream, so a later reader can
re-fetch and prove it read the same text.

Usage::

    python -m t2pw.bench.eval_v2.fetch --out C:/t/eval-fulltext PMC12096016 ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

EUROPEPMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def xml_to_text(xml: str) -> str:
    """Flatten JATS XML to plain text (tags dropped, entities decoded, whitespace folded).

    References are KEPT: the gold author needs to see what a paper cites, and quote
    verification in :mod:`t2pw.bench.eval_v2.gold` is done on folded text anyway.
    """

    text = re.sub(r"<[^>]+>", " ", xml)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def fetch_one(pmcid: str, out_dir: Path, timeout: float = 60.0) -> Dict[str, object]:
    url = EUROPEPMC_XML.format(pmcid=pmcid)
    req = urllib.request.Request(url, headers={"User-Agent": "t2pw-eval-v2/1 (read-only)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        raw = resp.read()
    xml_path = out_dir / f"{pmcid}.xml"
    xml_path.write_bytes(raw)
    text = xml_to_text(raw.decode("utf-8", "replace"))
    txt_path = out_dir / f"{pmcid}.txt"
    txt_path.write_text(text, encoding="utf-8")
    return {
        "pmcid": pmcid,
        "source_url": url,
        "fetch_date": _dt.date.today().isoformat(),
        "xml_sha256": hashlib.sha256(raw).hexdigest(),
        "xml_bytes": len(raw),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_chars": len(text),
        "local_xml": str(xml_path),
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pmcids", nargs="+")
    ap.add_argument("--out", default="C:/t/eval-fulltext")
    ap.add_argument("--manifest", default=None, help="write the fetch manifest JSON here")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = 0
    for pmcid in args.pmcids:
        try:
            row = fetch_one(pmcid, out_dir)
        except Exception as exc:  # noqa: BLE001 - report and continue; exit code says it
            failed += 1
            row = {"pmcid": pmcid, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps(row))
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
