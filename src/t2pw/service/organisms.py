"""Organism drop-down data (task W2, D-W04): every PathBank species.

Loads the committed ``data/webserver/organisms.json`` (written by
``scripts/webserver/export_organisms.py``). Never touches the database.
``scope`` is the string to pass as ``RequestedScope.requested_organism``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "webserver" / "organisms.json"


@dataclass(frozen=True)
class Organism:
    id: int  # PathBank species.id
    label: str  # unique drop-down text
    scope: str  # RequestedScope.requested_organism
    taxonomy_id: Optional[str] = None
    common_name: Optional[str] = None
    classification: Optional[str] = None
    has_pathways: bool = False


@lru_cache(maxsize=None)
def load_organisms(path: Path = DEFAULT_PATH) -> Tuple[Organism, ...]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(Organism(**e) for e in doc["organisms"])


def get_by_id(species_id: int, path: Path = DEFAULT_PATH) -> Optional[Organism]:
    return next((o for o in load_organisms(path) if o.id == int(species_id)), None)


def get_by_label(label: str, path: Path = DEFAULT_PATH) -> Optional[Organism]:
    """Case-insensitive exact label match (the label is unique in the file)."""
    want = str(label).strip().casefold()
    return next((o for o in load_organisms(path) if o.label.casefold() == want), None)


def scope_for(key: Union[int, str], path: Path = DEFAULT_PATH) -> str:
    """Scope string for a PathBank species id or a label. Raises ``KeyError`` if unknown."""
    org = get_by_id(key, path) if isinstance(key, int) else get_by_label(key, path)
    if org is None:
        raise KeyError(f"unknown organism: {key!r}")
    return org.scope
