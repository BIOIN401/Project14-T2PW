"""Variant of the ORCH-755 merged payload with every '+' removed from the channel names and Ca++ -> Ca2+.

Used to ask: does the compound row named like the channel appear at BASE too, with no '+' involved?
"""
import json
import sys

text = open(sys.argv[1], encoding="utf-8").read()
text = text.replace("voltage-gated Ca ++ channels", "voltage-gated calcium channels")
text = text.replace("voltage-gated Ca++ channels", "voltage-gated calcium channel")
text = text.replace('"Ca++"', '"Ca2+"')
json.loads(text)
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(text)
