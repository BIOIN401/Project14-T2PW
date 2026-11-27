"""Added pwc-ids that h1 had and h2 dropped (or gained), from cc9 outputs. usage: addeddiff.py OLD.json NEW.json"""
import json, sys
k = lambda p: {(a["file"], a["id"], a["name"], a["pwc_id"]) for a in json.load(open(p, encoding="utf-8"))["added"]}
a, b = k(sys.argv[1]), k(sys.argv[2])
for x in sorted(a - b): print("dropped", x)
for x in sorted(b - a): print("gained", x)
