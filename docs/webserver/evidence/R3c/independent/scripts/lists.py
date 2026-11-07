import json,sys
d=json.load(open(sys.argv[1]))
print(" ".join(a for a in d["command"] if a.startswith("tests/")))
