"""Rebuild the frozen payload from a CleanModel (P-23: ``to_payload(build(p)) == p``)."""

from __future__ import annotations

import copy
from typing import Any, Dict

from .model import ENTITY_LISTS, PROCESS_LISTS, CleanModel


def to_payload(model: CleanModel) -> Dict[str, Any]:
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in model.extras["top"].items()}
    if model.states is not None:
        out["biological_states"] = [s.to_raw() for s in model.states]
    for container, table in (("entities", ENTITY_LISTS), ("processes", PROCESS_LISTS)):
        if container not in model.containers:
            continue
        d = {k: copy.deepcopy(v) for k, v in model.extras[container].items()}
        for key, (attr, _) in table.items():
            items = getattr(model, attr)
            if items is not None:
                d[key] = [i.to_raw() for i in items]
        out[container] = d
    if "element_locations" in model.containers:
        d = {k: copy.deepcopy(v) for k, v in model.extras["element_locations"].items()}
        for key, rows in model.placements.items():
            d[key] = [r.to_raw() for r in rows]
        out["element_locations"] = d
    return out
