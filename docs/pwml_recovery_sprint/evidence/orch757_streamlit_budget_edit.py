"""ORCH-757: the ONLY bytes changed in the protected streamlit_app.py (D-105 § 2).

Byte-level replacements on the CRLF working-tree file; each old line must occur
exactly the stated number of times, and the result must differ from the input in
exactly those lines and nowhere else.
"""
import sys
from pathlib import Path

p = Path("src/t2pw/app/streamlit_app.py")
before = p.read_bytes()
B = "llm_client_module.configured_max_tokens()"
SUBS = [
    (b"            base_max_tokens = min(8000, 3600 + 700 * (round_idx - 1))\r\n",
     f"            base_max_tokens = {B}\r\n".encode(), 1),
    (b"                cand_max_tokens = min(10000, base_max_tokens + token_offsets[cand_idx])\r\n",
     b"                cand_max_tokens = base_max_tokens\r\n", 1),
    (b"                    llm_max_tokens=1200,\r\n",
     b"                    llm_max_tokens=base_max_tokens,\r\n", 3),
    (b'                        "max_tokens": 1200,\r\n',
     b'                        "max_tokens": base_max_tokens,\r\n', 1),
    (b"                gap_tokens = min(1400, 700 + 120 * (round_idx - 1))\r\n",
     f"                gap_tokens = {B}\r\n".encode(), 1),
    (b"                llm_max_tokens=2000,\r\n",
     f"                llm_max_tokens={B},\r\n".encode(), 1),
    (b"                    llm_max_tokens=3000,\r\n",
     f"                    llm_max_tokens={B},\r\n".encode(), 1),
]
after = before
for old, new, count in SUBS:
    n = after.count(old)
    if n != count:
        sys.exit(f"expected {count} of {old!r}, found {n}")
    after = after.replace(old, new)

old_lines, new_lines = before.split(b"\r\n"), after.split(b"\r\n")
assert len(old_lines) == len(new_lines)
changed = [i + 1 for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
assert len(changed) == 9, changed
assert after.count(b"\r\n") == before.count(b"\r\n") and b"\n" not in after.replace(b"\r\n", b"")
p.write_bytes(after)
print("changed lines:", changed)
