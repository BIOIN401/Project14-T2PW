"""Mark the source-based guess in section 6 as disproved by the probe."""
t = "C:/t/lane-pwdraw/docs/pwml_output/DRAW-PATHWHIZ.md"
s = open(t, encoding="utf-8").read()
old = "which is nil for an unknown id, so the browser throws;"
assert s.count(old) == 1
s = s.replace(old, "which is nil for an unknown id, so the browser throws (a guess from the source; the probe below disproves it);")
open(t, "w", encoding="utf-8", newline="\n").write(s)
print("marked")
