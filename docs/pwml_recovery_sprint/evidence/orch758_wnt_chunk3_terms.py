"""ORCH-758: is the WNT paper's chunk 3 a no-content chunk? Stdlib only, reads only.

Usage: python orch758_wnt_chunk3_terms.py <PMC10425814 01_source_text.txt>

Chunk boundaries are taken from orch758_chlorophyll_lexicon.txt (chunk_text at 10000/1600):
each chunk is located by its first words and its character length. Counts Wnt/Ca2+ and PCP
mechanism terms per chunk, and prints the sentences of chunk 3 that name a Ca2+-pathway
or PCP-core term, so a reader can judge whether a structurally empty answer was correct.
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
CHUNKS = [("1", "3547 genesdiss Genes", 63691), ("2", "Diaphanous 1 (DIA1) DIA1", 66006),
          ("3", "These fracture calluses also demonstrated", 75671),
          ("4", "New ligand binding function of human Cerberus", 27330)]
TERMS = ["Ca2+", "calcium", "CaMKII", "PKC", "calcineurin", "NFAT", "PLC", "Ror2", "Vangl", "Celsr",
         "Prickle", "Dvl", "JNK", "RhoA", "ROCK", "Rac1", "DAAM1", "Frizzled", "Wnt5a", "Wnt11"]
for name, start, length in CHUNKS:
    i = text.find(start)
    chunk = text[i:i + length] if i >= 0 else ""
    counts = {t: len(re.findall(re.escape(t), chunk)) for t in TERMS}
    print(f"chunk {name}: found_at={i} chars={len(chunk)} | " + " ".join(f"{t}={c}" for t, c in counts.items() if c))
    if name == "3" and chunk:
        hits = [s for s in re.split(r"(?<=[.!?])\s+", chunk)
                if re.search(r"CaMKII|calcineurin|NFAT|PKC|Ca2\+|Vangl|Celsr|Prickle|DAAM1|JNK|RhoA", s)]
        print(f"  chunk 3 sentences naming a Ca2+/PCP mechanism term: {len(hits)}")
        for s in hits[:12]:
            print("   -", s[:260])
        tail = chunk[-600:].replace("\n", " ")
        print("  chunk 3 ends:", tail[-300:])
