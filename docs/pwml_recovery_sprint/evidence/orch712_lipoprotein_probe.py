"""REV-100 finding 3: does PMC12444477 carry `Lpp` as well as the generic `lipoprotein`?

The gold tolerates a generic `lipoprotein` on the stated ground that the paper
"discusses proteins that will not cleanly resolve". But the body names **Lpp** --
E. coli murein lipoprotein, a specific resolvable protein. If the payloads carry
only the generic and never `Lpp`, the tolerance is excusing a LOST IDENTITY
rather than an inherent resolution difficulty.

Every count here is reported with a positive control, per the standard in
LEDGER.md: a probe that reports zero must first show it can report non-zero.

ASCII-only output.
"""
import glob
import io
import json
import os
import re
import sys

PAPER = "PMC12444477"
LPP = re.compile(r"\bLpp\b")
GENERIC = re.compile(r"\blipoprotein\b", re.I)
CONTROL = re.compile(r"\bLpxC\b")


def source_texts():
    seen = set()
    for pat in ("runs*/**/papers/%s/*.txt" % PAPER, "runs*/**/papers/%s/**/*.txt" % PAPER):
        for fp in glob.glob(pat, recursive=True):
            p = os.path.abspath(fp)
            if p not in seen:
                seen.add(p)
                yield fp


def payload_files():
    for pat in ("runs*/**/papers/%s/*/*.json" % PAPER,):
        for fp in sorted(glob.glob(pat, recursive=True)):
            yield fp


def entity_names(payload):
    out = []
    ents = payload.get("entities")
    if not isinstance(ents, dict):
        return out
    for bucket in ("proteins", "protein_complexes"):
        for row in ents.get(bucket) or []:
            if isinstance(row, dict):
                nm = str(row.get("name") or "").strip()
                if nm:
                    out.append((bucket, nm))
    return out


def main():
    print("=" * 74)
    print("L1. The paper body -- Lpp vs the generic, with a control")
    print("=" * 74)
    files = 0
    tot_lpp = tot_gen = tot_ctl = 0
    for fp in sorted(source_texts()):
        try:
            text = io.open(fp, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        files += 1
        n_lpp = len(LPP.findall(text))
        n_gen = len(GENERIC.findall(text))
        n_ctl = len(CONTROL.findall(text))
        tot_lpp += n_lpp
        tot_gen += n_gen
        tot_ctl += n_ctl
        if n_lpp or n_gen:
            print("   %-58s Lpp=%-3d lipoprotein=%-3d [LpxC=%d]"
                  % (os.path.basename(os.path.dirname(fp)) + "/" + os.path.basename(fp),
                     n_lpp, n_gen, n_ctl))
    print("")
    print("   files scanned      : %d" % files)
    print("   Lpp   total        : %d" % tot_lpp)
    print("   lipoprotein total  : %d" % tot_gen)
    print("   CONTROL LpxC total : %d   <- non-zero proves the grep and the corpus are live"
          % tot_ctl)
    if tot_ctl == 0:
        print("   *** CONTROL FAILED -- the zero above is meaningless. Refusing to report. ***")
        return 1

    print("")
    print("=" * 74)
    print("L2. Do the payloads carry Lpp, the generic, or only one of them?")
    print("=" * 74)
    any_lpp = any_generic = 0
    scanned = 0
    for fp in payload_files():
        try:
            payload = json.load(io.open(fp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        names = entity_names(payload)
        if not names:
            continue
        scanned += 1
        lpp_rows = [(b, n) for b, n in names if LPP.search(n)]
        gen_rows = [(b, n) for b, n in names if GENERIC.search(n)]
        if lpp_rows or gen_rows:
            leg = os.path.basename(os.path.dirname(fp))
            run = fp.replace("\\", "/").split("/papers/")[0]
            print("   %s :: %s :: %s" % (run, leg, os.path.basename(fp)))
            for b, n in lpp_rows:
                print("        Lpp        -> %-22s [%s]" % (n, b))
            for b, n in gen_rows:
                print("        generic    -> %-22s [%s]" % (n, b))
        any_lpp += 1 if lpp_rows else 0
        any_generic += 1 if gen_rows else 0

    print("")
    print("   payload files with entities : %d" % scanned)
    print("   files carrying an Lpp row   : %d" % any_lpp)
    print("   files carrying a generic row: %d" % any_generic)
    print("")
    if any_generic and not any_lpp:
        print("   VERDICT: the payloads carry ONLY the generic. The tolerance is excusing a")
        print("            LOST IDENTITY -- the paper names Lpp and the pipeline shipped")
        print("            'lipoprotein' instead.")
    elif any_lpp and any_generic:
        print("   VERDICT: both are present. The generic is an ADDITIONAL row, not a")
        print("            replacement, so no identity was lost by degradation.")
    elif not any_generic:
        print("   VERDICT: no generic row in any payload -- the gold entry is defensive")
        print("            rather than describing a measured occurrence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
