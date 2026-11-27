"""VIS-9 (orchestrator decision on Q7): re-record the sour-taste PWML pins to the new intended output.
Touches only: tests/fixtures/orch768/legs.json and docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json
(leg ORCH-767/sour_taste_v2-strict/2026-10-24_0042, pathway.review_required.pwml: sha256 + bytes), and
tests/fixtures/equivalence/sour_taste_v2_strict.json.gz (recorded.pwml_sha256, pwml_bytes, pwml_files).
New values come from the PWMLs the tests themselves wrote at HEAD (rerec/*_new.pwml); the old ones must match.
usage: rerecord.py TREE"""
import gzip, hashlib, json, sys
from pathlib import Path
T = Path(sys.argv[1]); R = Path("C:/t/vis-pw9-work/rerec")
LEG, NAME = "ORCH-767/sour_taste_v2-strict/2026-10-24_0042", "pathway.review_required.pwml"
sha = lambda b: hashlib.sha256(b).hexdigest()
old768, new768 = (R / "orch768_sour_old.pwml").read_bytes(), (R / "orch768_sour_new.pwml").read_bytes()
oldeq, neweq = (R / "eqapp_old.pwml").read_bytes(), (R / "eqapp_new.pwml").read_bytes()
log = []
for rel in ("tests/fixtures/orch768/legs.json", "docs/pwml_recovery_sprint/evidence/orch768_corpus_manifest.json"):
    p = T / rel; raw = p.read_bytes(); crlf = b"\r\n" in raw
    d = json.loads(raw)
    leg = next(l for l in d["legs"] if l["id"] == LEG); m = leg["inputs"][NAME]
    assert m["sha256"] == sha(old768) and m["bytes"] == len(old768), (rel, m)
    m["sha256"], m["bytes"] = sha(new768), len(new768)
    text = json.dumps(d, indent=1, ensure_ascii=True) + "\n"
    p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode())
    log.append(f"{rel}: {LEG} {NAME} sha256 {sha(old768)} -> {sha(new768)}, bytes {len(old768)} -> {len(new768)}")
p = T / "tests/fixtures/equivalence/sour_taste_v2_strict.json.gz"
F = json.loads(gzip.decompress(p.read_bytes()))
r = F["recorded"]
assert r["pwml_sha256"] == sha(oldeq) and r["pwml_bytes"] == len(oldeq) and r["pwml_files"] == {"pathway.pwml": sha(oldeq)}, r["pwml_files"]
r["pwml_sha256"], r["pwml_bytes"], r["pwml_files"] = sha(neweq), len(neweq), {"pathway.pwml": sha(neweq)}
p.write_bytes(gzip.compress(json.dumps(F, indent=1, ensure_ascii=False).encode("utf-8"), mtime=0))
log.append(f"{p.relative_to(T).as_posix()}: recorded.pwml_sha256 / pwml_files[pathway.pwml] {sha(oldeq)} -> {sha(neweq)}, pwml_bytes {len(oldeq)} -> {len(neweq)}")
(R / "rerecord_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print("\n".join(log))
