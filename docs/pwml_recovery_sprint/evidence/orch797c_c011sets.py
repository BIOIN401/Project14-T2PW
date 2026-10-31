import json
import sys
from pathlib import Path

W = Path(sys.argv[1])
fixture = json.loads((W / "docs/pwml_recovery_sprint/evidence/c011_freeze_seam_before.json").read_text(encoding="utf-8"))["legs"]
pre = json.loads(Path("C:/t/orch797c/c011-pre.json").read_text(encoding="utf-8"))
tip = json.loads(Path("C:/t/orch797c/c011-tip0.json").read_text(encoding="utf-8"))
c057 = {n for n in fixture if fixture[n]["canonical_payload_sha256"] != (pre[n]["payload"] or "")}
c797 = {n for n in fixture if (pre[n]["payload"] or "") != (tip[n]["payload"] or "")}
conn = {n for n in fixture if pre[n]["connectivity"] != tip[n]["connectivity"]}
print("C-057 legs (fixture != pre-797a):", len(c057))
for n in sorted(c057):
    print("  ", n)
print("797a legs (pre-797a != tip):", len(c797))
for n in sorted(c797):
    print("  ", n, "(also C-057)" if n in c057 else "")
print("connectivity moved:", sorted(conn))
