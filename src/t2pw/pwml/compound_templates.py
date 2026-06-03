from __future__ import annotations

import re
from typing import Any


COMPOUND_ID_TO_TEMPLATE: dict[int, int] = {
    414: 42,  # Adenosine triphosphate
    1034: 43,  # Adenosine diphosphate
    32: 44,  # Adenosine monophosphate
    170: 45,  # Pyrophosphate
    1104: 46,  # Phosphate
    463: 47,  # Hydrogen carbonate / bicarbonate
    1420: 49,  # Water
    374: 50,  # Chlorine / Chloride
    353: 51,  # Calcium
    1316: 52,  # Carbon dioxide
    936: 53,  # Guanosine diphosphate (GDP)
    986: 54,  # Guanosine triphosphate (GTP)
    1051: 55,  # Hydrogen / Hydrogen ion (H+)
    40034: 55,  # Hydrogen ion (alternate PW id)
    1783: 56,  # Hydrogen peroxide
    457: 57,  # Potassium
    458: 58,  # Sodium
    721: 59,  # NAD+
    1144: 60,  # NADH
    143: 61,  # NADP+
    146: 62,  # NADPH
    35: 63,  # Ammonia
    1867: 64,  # Nitric oxide
    1065: 65,  # Oxygen
    1099: 85,  # Coenzyme A
    964: 137,  # FAD
    932: 138,  # FADH2
    846: 139,  # Coenzyme Q10 / Ubiquinone-10
    1006: 140,  # Ubiquinol / QH2
    423: 181,  # Magnesium
    22608: 293,  # Ammonium
    1611: 294,  # Boron
    1050: 295,  # Carbon monoxide
    472: 296,  # Cobalt
    514: 297,  # Copper
    544: 298,  # Iron 2+ (Fe2+)
    8133: 299,  # Iron 3+ (Fe3+)
    2607: 300,  # Lithium
    1027: 301,  # Manganese
    1637: 302,  # Nickel
    9795: 303,  # Zinc
}

COMPOUND_NAME_TO_TEMPLATE: dict[str, int] = {
    # ATP
    "adenosine triphosphate": 42,
    "atp": 42,
    # ADP
    "adenosine diphosphate": 43,
    "adp": 43,
    # AMP
    "adenosine monophosphate": 44,
    "amp": 44,
    # PPi
    "pyrophosphate": 45,
    "ppi": 45,
    "inorganic pyrophosphate": 45,
    "diphosphate": 45,
    # Pi
    "phosphate": 46,
    "pi": 46,
    "inorganic phosphate": 46,
    "orthophosphate": 46,
    # Bicarbonate
    "hydrogen carbonate": 47,
    "bicarbonate": 47,
    "hco3": 47,
    "hco3-": 47,
    # Water
    "water": 49,
    "h2o": 49,
    # Chloride
    "chlorine": 50,
    "chloride": 50,
    "cl": 50,
    # Calcium
    "calcium": 51,
    "ca2": 51,
    "calcium ion": 51,
    "calcium(2+)": 51,
    # CO2
    "carbon dioxide": 52,
    "co2": 52,
    # GDP
    "guanosine diphosphate": 53,
    "gdp": 53,
    # GTP
    "guanosine triphosphate": 54,
    "gtp": 54,
    # H+
    "hydrogen": 55,
    "hydrogen ion": 55,
    "h": 55,
    "proton": 55,
    "hydron": 55,
    # H2O2
    "hydrogen peroxide": 56,
    "h2o2": 56,
    # K+
    "potassium": 57,
    "potassium ion": 57,
    # Na+
    "sodium": 58,
    "sodium ion": 58,
    # NAD+
    "nad": 59,
    "nad+": 59,
    "nicotinamide adenine dinucleotide": 59,
    # NADH
    "nadh": 60,
    # NADP+
    "nadp": 61,
    "nadp+": 61,
    "nicotinamide adenine dinucleotide phosphate": 61,
    # NADPH
    "nadph": 62,
    # Ammonia
    "ammonia": 63,
    "nh3": 63,
    # NO
    "nitric oxide": 64,
    "no": 64,
    # O2
    "oxygen": 65,
    "o2": 65,
    "dioxygen": 65,
    # CoA
    "coenzyme a": 85,
    "coa": 85,
    "coa-sh": 85,
    "hscoa": 85,
    "coenzyme-a": 85,
    # FAD
    "fad": 137,
    "flavin adenine dinucleotide": 137,
    # FADH2
    "fadh2": 138,
    "fadh₂": 138,
    # Ubiquinone
    "coenzyme q10": 139,
    "ubiquinone": 139,
    "ubiquinone-10": 139,
    "coenzyme q": 139,
    # Ubiquinol
    "ubiquinol": 140,
    "qh2": 140,
    "coenzyme qh2": 140,
    # Mg
    "magnesium": 181,
    "mg2": 181,
    "magnesium ion": 181,
    # Ammonium
    "ammonium": 293,
    "nh4": 293,
    # Boron
    "boron": 294,
    # CO
    "carbon monoxide": 295,
    "co": 295,
    # Cobalt
    "cobalt": 296,
    "cobalt(2+)": 296,
    # Copper
    "copper": 297,
    "cu2": 297,
    "copper(2+)": 297,
    # Fe2+
    "iron": 298,
    "fe2": 298,
    "iron(2+)": 298,
    "ferrous": 298,
    # Fe3+
    "fe3": 299,
    "iron(3+)": 299,
    "ferric": 299,
    # Li
    "lithium": 300,
    # Mn
    "manganese": 301,
    "mn2": 301,
    # Ni
    "nickel": 302,
    "ni2": 302,
    # Zn
    "zinc": 303,
    "zn2": 303,
    "zinc(2+)": 303,
}

_CHARGE_SUFFIX_RE = re.compile(r"[\s(]*[+-]?\d*[+-]+\s*\)?$")


def _normalise_for_lookup(name: str) -> str:
    """Lowercase, strip ionic-charge suffix, collapse whitespace."""
    text = name.strip().casefold()
    text = _CHARGE_SUFFIX_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def select_compound_template_id(compound_record: dict[str, Any]) -> int:
    pw_id = compound_record.get("pathwhiz_id") or compound_record.get("db_id")
    if pw_id is not None:
        try:
            template_id = COMPOUND_ID_TO_TEMPLATE.get(int(pw_id))
            if template_id is not None:
                return template_id
        except (TypeError, ValueError):
            pass

    name = str(compound_record.get("name") or "")
    name_template_id = COMPOUND_NAME_TO_TEMPLATE.get(_normalise_for_lookup(name))
    if name_template_id is not None:
        return name_template_id

    mapped_ids = compound_record.get("mapped_ids") or {}
    if isinstance(mapped_ids, dict) and (mapped_ids.get("drugbank") or compound_record.get("drugbank_id")):
        return 157

    return 81
