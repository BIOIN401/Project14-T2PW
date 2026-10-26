"""ORCH-773: build tests/fixtures/orch773/ from captured freeze inputs.

Usage: python make_fixtures.py <capture dir> <fixture dir>

Each fixture is ``<slug>.freeze_input.json.gz``: the exact ``{"payload", "pathway_context"}`` the ORCH-768
replay handed ``freeze_canonical_payload`` on that corpus leg at base 3ef3fa9c (written by
capture_freeze_inputs.py), gzipped with mtime 0 so the bytes are reproducible. Prints the sha256 of each
uncompressed file (the test asserts it) and of each .gz. Stdlib only.
"""

import gzip
import hashlib
import sys
from pathlib import Path

LEGS = [
    "ORCH-755_enterobactin-strict_2026-10-20_2201",
    "ORCH-767_wnt_ca-strict_2026-10-24_0314",
    "ORCH-759_enterobactin-strict_2026-10-22_1037",
    "runs_verify_2026-08-21_2057_PMC12452463",
    "ORCH-755_sour_taste-strict_2026-10-21_0139",
    "runs_verify_2026-08-21_2057_PMC12096016",
    "runs_verify_2026-08-25_1216_PMC12856317",
    "runs_verify_2026-08-21_2239_PMC12180156",
]


def main():
    cap, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for slug in LEGS:
        raw = (cap / slug / "freeze_input.json").read_bytes()
        gz = gzip.compress(raw, compresslevel=9, mtime=0)
        (out / f"{slug}.freeze_input.json.gz").write_bytes(gz)
        print(f"{slug}  raw_sha256={hashlib.sha256(raw).hexdigest()}  raw_bytes={len(raw)}  "
              f"gz_sha256={hashlib.sha256(gz).hexdigest()}  gz_bytes={len(gz)}")


if __name__ == "__main__":
    main()
