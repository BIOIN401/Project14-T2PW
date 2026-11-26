"""PathBank pathway lookup for the smart search (W10).

Runtime reads a local SQLite index only; it never touches the PathBank database or
pathbank.org. The index is built once by ``scripts/webserver/export_pathbank_index.py``
from the PathBank MySQL database (read-only SELECT) into
``data/webserver/pathbank_index.sqlite`` (not committed: 520k public pathways).

Public accessions: ``PWxxxxxx`` is ``super_pathway_visualizations.pw_id`` (never
``pathways.id``, which returns other pathways) and ``SMPxxxxxxx`` its ``smpdb_id``.

Diagrams and PWML come from an optional local mirror of PathBank's attachment store
(``T2PW_PATHBANK_FILES``), laid out as PathBank keeps it:
``<root>/simple_vector_images/<PW>.svg`` and ``<root>/pwml_markups/<PW>.pwml``.
pathbank.org itself sits behind a Cloudflare browser challenge, so the server cannot
fetch from it (W10 RESULT, evidence).
"""

from __future__ import annotations

import difflib
import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from t2pw.paths import PROJECT_ROOT

DEFAULT_INDEX = PROJECT_ROOT / "data" / "webserver" / "pathbank_index.sqlite"
PW_RE = re.compile(r"PW\d{6,7}")
VIEW_URL = "https://pathbank.org/view/{smp}"  # PathBank's own page for one pathway

STRONG = 0.99  # an exact name match (1.0), or an exact match through a synonym (0.99)
MIN_SCORE = 0.5  # below this a token match is noise

#: PathBank stores no pathway synonyms; these are the common names its titles miss.
#: ponytail: a hand list; grow it from search logs, or ask PathBank for a synonym table.
SYNONYMS: Tuple[Tuple[str, ...], ...] = (
    ("tca cycle", "citric acid cycle", "krebs cycle", "tricarboxylic acid cycle"),
    ("glycolysis", "glycolytic pathway", "embden meyerhof pathway", "embden meyerhof parnas pathway"),
    ("pentose phosphate pathway", "pentose phosphate shunt", "hexose monophosphate shunt"),
    ("urea cycle", "ornithine cycle"),
)
_ALIASES: Dict[str, Tuple[str, ...]] = {n: g for g in SYNONYMS for n in g}

SCHEMA = """
CREATE TABLE names (id INTEGER PRIMARY KEY, name TEXT NOT NULL, norm TEXT NOT NULL UNIQUE);
CREATE TABLE pathways (pw_id TEXT PRIMARY KEY, smp_id TEXT, name_id INTEGER NOT NULL,
                       species_id INTEGER, subject TEXT);
CREATE INDEX pathways_name ON pathways (name_id, species_id);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def normalize(text: str) -> str:
    """Case-folded words: ``"TCA cycle - E. coli"`` -> ``"tca cycle e coli"``."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text).casefold()).split())


def build_index(rows: Iterable[Mapping], path: Path | str, meta: Optional[Mapping[str, str]] = None) -> int:
    """Write a fresh index from rows with ``pw_id, smp_id, name, species_id, subject``.
    Returns the pathway count. Rows without a PW id or a name are skipped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(str(tmp))
    try:
        db.executescript(SCHEMA)
        name_ids: Dict[str, int] = {}
        n = 0
        for r in rows:
            pw, name = (r.get("pw_id") or "").strip(), (r.get("name") or "").strip()
            norm = normalize(name)
            if not PW_RE.fullmatch(pw) or not norm:
                continue
            if norm not in name_ids:  # first spelling wins (rows arrive in PW order)
                name_ids[norm] = db.execute("INSERT INTO names (name, norm) VALUES (?, ?)",
                                            (name, norm)).lastrowid
            db.execute("INSERT OR IGNORE INTO pathways VALUES (?, ?, ?, ?, ?)",
                       (pw, (r.get("smp_id") or "").strip() or None, name_ids[norm],
                        r.get("species_id"), (r.get("subject") or "").strip() or None))
            n += 1
        db.executemany("INSERT INTO meta VALUES (?, ?)", sorted((meta or {}).items()))
        db.commit()
    finally:
        db.close()
    os.replace(tmp, path)
    return n


@dataclass(frozen=True)
class Hit:
    pw_id: str
    smp_id: Optional[str]
    name: str
    species_id: Optional[int]
    subject: Optional[str]
    score: float
    organisms: int  # how many species PathBank has this pathway name for

    @property
    def url(self) -> Optional[str]:
        return VIEW_URL.format(smp=self.smp_id) if self.smp_id else None


def _pw_number(pw: str) -> int:
    return int(pw[2:])


class PathBankIndex:
    """Read-only lookups. One short-lived connection per call (sync handlers run in a pool)."""

    def __init__(self, path: Path | str = DEFAULT_INDEX) -> None:
        self.path = Path(path)

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def get(self, pw_id: str) -> Optional[Hit]:
        if not PW_RE.fullmatch(pw_id or "") or not self.available:
            return None
        db = self._db()
        try:
            r = db.execute("SELECT p.*, n.name, (SELECT COUNT(*) FROM pathways q WHERE q.name_id = p.name_id) c "
                           "FROM pathways p JOIN names n ON n.id = p.name_id WHERE p.pw_id = ?",
                           (pw_id,)).fetchone()
        finally:
            db.close()
        return Hit(r["pw_id"], r["smp_id"], r["name"], r["species_id"], r["subject"], 1.0, r["c"]) if r else None

    def search(self, query: str, species_ids: Optional[Sequence[int]] = None, limit: int = 3) -> List[Hit]:
        """Best pathways for ``query``, best first, at most one per distinct name.

        Exact name (1.0), then an exact synonym (0.99), then names containing every query
        word, then close spellings (``difflib``); scored by ``SequenceMatcher`` ratio. With
        ``species_ids`` only pathways drawn for those species count. Ties: lower PW id first
        (the older, curated originals)."""
        q = normalize(query)
        if not q or not self.available:
            return []
        db = self._db()
        try:
            found: Dict[int, Tuple[float, str]] = {}  # name id -> (score, name)

            def take(rows, score_fn):
                for r in rows:
                    s = score_fn(r["norm"])
                    if s >= MIN_SCORE and s > found.get(r["id"], (-1.0, ""))[0]:
                        found[r["id"]] = (s, r["name"])

            take(db.execute("SELECT id, name, norm FROM names WHERE norm = ?", (q,)), lambda _: 1.0)
            for alias in _ALIASES.get(q, ()):
                if alias != q:
                    take(db.execute("SELECT id, name, norm FROM names WHERE norm = ?", (alias,)), lambda _: STRONG)
            words = [w for w in q.split() if w not in {"pathway", "the", "of", "and"}] or q.split()
            sql = "SELECT id, name, norm FROM names WHERE " + " AND ".join(["norm LIKE ?"] * len(words)) + " LIMIT 2000"
            ratio = lambda norm: round(difflib.SequenceMatcher(None, q, norm).ratio() * 0.98, 4)  # noqa: E731
            take(db.execute(sql, [f"%{w}%" for w in words]), ratio)
            if not found:  # a typo: close spellings over every distinct name
                close = difflib.get_close_matches(q, _all_norms(str(self.path)), n=10, cutoff=0.8)
                for norm in close:
                    take(db.execute("SELECT id, name, norm FROM names WHERE norm = ?", (norm,)), ratio)
            hits: List[Hit] = []
            for name_id, (score, name) in found.items():
                rows = db.execute("SELECT * FROM pathways WHERE name_id = ?", (name_id,)).fetchall()
                pick = [r for r in rows if species_ids is None or r["species_id"] in set(species_ids)]
                if not pick:
                    continue
                r = min(pick, key=lambda r: _pw_number(r["pw_id"]))
                hits.append(Hit(r["pw_id"], r["smp_id"], name, r["species_id"], r["subject"], score, len(rows)))
        finally:
            db.close()
        hits.sort(key=lambda h: (-h.score, _pw_number(h.pw_id)))
        return hits[:limit]


@lru_cache(maxsize=2)
def _all_norms(path: str) -> Tuple[str, ...]:
    db = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        return tuple(r[0] for r in db.execute("SELECT norm FROM names"))
    finally:
        db.close()


def is_strong(hits: Sequence[Hit]) -> bool:
    """One clear answer: the best hit is an exact (or synonym) name match."""
    return bool(hits) and hits[0].score >= STRONG


# -- local PathBank files --------------------------------------------------------------


def files_root() -> Optional[Path]:
    root = os.environ.get("T2PW_PATHBANK_FILES", "").strip()
    return Path(root) if root else None


def local_file(root: Optional[Path], kind: str, pw_id: str) -> Optional[Path]:
    """PathBank's own SVG (``kind="svg"``: the full-colour drawing PathBank shows by default, else
    the simple one) or PWML (``"pwml"``) for ``pw_id``, if mirrored."""
    if root is None or not PW_RE.fullmatch(pw_id or ""):
        return None
    subs = {"svg": ("full_vector_images", "simple_vector_images"), "pwml": ("pwml_markups",)}[kind]
    ext = ".svg" if kind == "svg" else ".pwml"
    return next((p for p in (root / s / (pw_id + ext) for s in subs) if p.is_file()), None)
