"""Conservative T-107 cost bound. T-105 recorded NO token usage, so this is built
from measurable inputs and stated as an UPPER bound, not a prediction."""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
RUN = Path(sys.argv[1])

PROMPT_PER_M = 0.0868
COMPL_PER_M  = 0.1736

srcs = sorted(glob.glob(str(RUN / "papers" / "*" / "01_source_text.txt")))
by_paper = {}
for s in srcs:
    paper = Path(s).parents[0].name
    by_paper.setdefault(paper, os.path.getsize(s))
tot_chars = sum(by_paper.values())
print(f"papers with cached source text : {len(by_paper)}")
print(f"total source characters        : {tot_chars:,}")
print(f"mean per paper                 : {tot_chars//max(len(by_paper),1):,}")

CHARS_PER_TOK = 4.0
tok_per_paper = (tot_chars / max(len(by_paper),1)) / CHARS_PER_TOK
print(f"~tokens per paper full text    : {tok_per_paper:,.0f}  (at {CHARS_PER_TOK} chars/token)")

LEGS = 20
print()
print("UPPER-BOUND SCENARIOS -- prompt volume expressed as full-text-equivalent passes per leg")
print(f"{'passes/leg':>11} {'prompt Mtok':>12} {'compl Mtok':>11} {'prompt $':>9} {'compl $':>8} {'TOTAL $':>9}")
for passes in (20, 40, 60, 80, 120, 200):
    p_tok = passes * tok_per_paper * LEGS
    c_tok = p_tok * 0.10          # completions conservatively 10% of prompt volume
    p_cost = p_tok/1e6 * PROMPT_PER_M
    c_cost = c_tok/1e6 * COMPL_PER_M
    print(f"{passes:>11} {p_tok/1e6:>12.2f} {c_tok/1e6:>11.2f} {p_cost:>9.2f} {c_cost:>8.2f} {p_cost+c_cost:>9.2f}")

print()
print("Break-even against the $5 ceiling:")
lo, hi = 1, 100000
for _ in range(60):
    mid = (lo+hi)/2
    p_tok = mid * tok_per_paper * LEGS
    total = p_tok/1e6*PROMPT_PER_M + (p_tok*0.10)/1e6*COMPL_PER_M
    if total < 5.0: lo = mid
    else: hi = mid
print(f"  $5 is reached at ~{lo:,.0f} full-text-equivalent passes per leg")
print(f"  i.e. ~{lo*tok_per_paper/1e6:,.1f}M prompt tokens per leg, ~{lo*tok_per_paper*LEGS/1e6:,.0f}M across the run")
