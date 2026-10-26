"""ORCH-773 diagnosis: what the per-row pass omits on the test suites' own payloads.

Usage: python probe_toy_omissions.py --src <tree>/src --tests <tree>/tests

Imports payload builders from the test modules (read-only) and prints, per payload, every process the pass
omitted and every entity it removed, with gate and reason. Offline; prints only.
"""

import argparse
import socket
import sys
from pathlib import Path


def _block(*_a, **_k):
    raise OSError("ORCH773 probe: network blocked")


socket.socket.connect = _block
socket.create_connection = _block
socket.getaddrinfo = _block


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tests", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    sys.path.insert(0, str(Path(args.tests).resolve()))
    from t2pw.pipeline.strict_quarantine import quarantine_and_close

    import test_strict_quarantine_contract_alignment as CA
    import test_orch766_unreferenced_species as S766

    cases = [("contract_alignment._base", CA._base(), None, True)]
    for kind in S766.ALL_KINDS[:3]:
        cases.append((f"orch766._kind_payload({kind})", S766._kind_payload(kind), S766._context(kind), True))
    cases.append(("contract_alignment._base strict_db=False", CA._base(), None, False))
    for label, payload, ctx, strict_db in cases:
        res = quarantine_and_close(payload, strict_db=strict_db, pathway_context=ctx, mode="pathwhiz")
        print(f"\n## {label}: ok={res.ok} refusal={res.refusal_reasons}")
        for row in res.quarantine_report["quarantined"]:
            print(f"   quarantined {row['name']!r} {row['state']} {row['reason']} {(row.get('detail') or {}).get('omission')}")
        for row in res.removed_entity_report["removed_entities"]:
            print(f"   removed {row['bucket']} {row['name']!r} {row['reason']} {row.get('detail')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
