import sys
p = "C:/t/fp/src/t2pw/pwml/writer.py"
s = open(p, encoding="utf-8").read()
a = 'if layout_version() == "v2":'
b = 'if False and layout_version() == "v2":  # BITE-PROOF TEMP'
if sys.argv[1] == "on":
    assert s.count(a) == 1; s = s.replace(a, b)
else:
    assert s.count(b) == 1; s = s.replace(b, a)
open(p, "w", encoding="utf-8").write(s)
