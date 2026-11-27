"""VIS-9 review fixes F1-F5 in writer.py (exact-text replacements; fails loudly if the text moved)."""
from pathlib import Path
p = Path("C:/t/vis-pw9/src/t2pw/pwml/writer.py"); s = p.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
a = s.index("# PathBank compound ids that must not get a derived pwc-id")
b = s.index("def _mapped_compound_pwc_id(")
rep(s[a:b], '''_PWC_ID_SKIP_PATH = PROJECT_ROOT / "data" / "pwc_id_skip.json"
_GREEK_LETTERS = {"\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta", "\u03b5": "epsilon",
                  "\u03c9": "omega"}
_STEREO_PREFIX = re.compile(r"^(l|d|\(r\)|\(s\)|\(e\)|\(z\))-")


@lru_cache(maxsize=1)
def _pwc_id_skip() -> Optional[frozenset]:
    """PathBank compound ids that must not get a derived pwc-id (``data/pwc_id_skip.json``, built by
    ``python -m t2pw.pwml.refresh_pwc_id_skip``): rows whose pwc_id is not ``PW_C%06d(id)`` and rows with a wrong
    structure. ``None`` when the file is missing or unreadable: then no pwc-id is derived at all (fail closed)."""
    try:
        data = json.loads(_PWC_ID_SKIP_PATH.read_text(encoding="utf-8"))
        return frozenset(int(i) for i in data["derivation_mismatch"]) | frozenset(int(i) for i in data["structure_errors"])
    except Exception:  # noqa: BLE001 - a missing skip list must never let a wrong picture through
        return None


def _compound_name_parts(value: Any) -> Tuple[str, str]:
    """(leading stereo descriptor, comparison key) of a compound name. Greek letters are spelled out (alpha, beta, ...),
    so beta-alanine never reads as alanine; one leading L- / D- / (R)- / (S)- / (E)- / (Z)- is split off; a trailing
    "ic acid" reads as "ate"; punctuation and spaces drop, letters of any script stay."""
    text = str(value or "").casefold().strip()
    for letter, word in _GREEK_LETTERS.items():
        text = text.replace(letter, word)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    match = _STEREO_PREFIX.match(text)
    prefix = match.group(1) if match else ""
    text = re.sub(r"ic acid$", "ate", text[match.end():] if match else text)
    return prefix, re.sub(r"[\W_]+", "", text)


def _same_compound_name(ours: Any, theirs: Any) -> bool:
    """Same name up to case, punctuation, acid / -ate and a stereo descriptor only ONE side states (serine vs L-Serine);
    two different descriptors (D- vs L-, (S)- vs (R)-) never match."""
    (p1, k1), (p2, k2) = _compound_name_parts(ours), _compound_name_parts(theirs)
    return bool(k1) and k1 == k2 and (not p1 or not p2 or p1 == p2)


''')
rep('''    pid = _to_positive_int(record.get("pw_compound_id") or record.get("pathbank_compound_id"))
    if pid is None or pid != rid or pid in _PWC_ID_SKIP:
        return None''', '''    pid = _to_positive_int(record.get("pw_compound_id") or record.get("pathbank_compound_id"))
    skip = _pwc_id_skip()
    if pid is None or pid != rid or skip is None or pid in skip:
        return None''')
rep('''    ours = _compound_name_key(db_row.get("name") or record.get("name"))
    if not ours or ours not in {_compound_name_key(n) for n in names if n}:
        return None''', '''    ours = db_row.get("name") or record.get("name")
    if not any(_same_compound_name(ours, n) for n in names if n):
        return None''')
rep('''        written_pwc_ids: set = set()
''', '''        # VIS-9: every trusted pwc-id first, so a derived one never repeats one (whichever record comes first).
        written_pwc_ids: set = set()
        for record in entities.get("compounds", []) if isinstance(entities.get("compounds"), list) else []:
            if isinstance(record, dict):
                held_row = record.get("db_row") if isinstance(record.get("db_row"), dict) else {}
                held_ids = record.get("mapped_ids") if isinstance(record.get("mapped_ids"), dict) else {}
                held = _trusted_compound_pwc_id(record, held_row, held_ids)
                if held is not None:
                    written_pwc_ids.add(held)
''')
rep('''                if pwc_id in written_pwc_ids:  # a second record under the same id: a repeat pwc-id fails PathWhiz's uniqueness
                    pwc_id = None
            if pwc_id is not None:
                written_pwc_ids.add(pwc_id)''', '''                if pwc_id in written_pwc_ids:  # a repeat pwc-id fails PathWhiz's uniqueness: keep the first / trusted one
                    pwc_id = None
                if pwc_id is not None:
                    written_pwc_ids.add(pwc_id)''')
p.write_text(s, encoding="utf-8")
print("patched")
