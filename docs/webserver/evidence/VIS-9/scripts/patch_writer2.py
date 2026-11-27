from pathlib import Path
p = Path("C:/t/vis-pw9/src/t2pw/pwml/writer.py"); s = p.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep('import json\nimport math\n', 'import json\nimport logging\nimport math\n')
rep('_STEREO_PREFIX = re.compile(r"^(l|d|\(r\)|\(s\)|\(e\)|\(z\))-")',
    '_STEREO_PREFIX = re.compile(r"^(l|d|\(r\)|\(s\)|\(e\)|\(z\)|\(\+\)|\(-\))-")')
rep('''    structure. ``None`` when the file is missing or unreadable: then no pwc-id is derived at all (fail closed)."""
    try:
        data = json.loads(_PWC_ID_SKIP_PATH.read_text(encoding="utf-8"))
        return frozenset(int(i) for i in data["derivation_mismatch"]) | frozenset(int(i) for i in data["structure_errors"])
    except Exception:  # noqa: BLE001 - a missing skip list must never let a wrong picture through
        return None''', '''    structure. ``None`` when the file is missing, unreadable or has an empty mismatch list: then no pwc-id is derived
    at all (fail closed), with one warning."""
    try:
        data = json.loads(_PWC_ID_SKIP_PATH.read_text(encoding="utf-8"))
        mismatch = frozenset(int(i) for i in data["derivation_mismatch"])
        if mismatch:
            return mismatch | frozenset(int(i) for i in data["structure_errors"])
    except Exception:  # noqa: BLE001 - a missing skip list must never let a wrong picture through
        pass
    logging.getLogger(__name__).warning("VIS-9: %s missing, unreadable or empty: no pwc-id is derived", _PWC_ID_SKIP_PATH.name)
    return None''')
p.write_text(s, encoding="utf-8"); print("patched")
