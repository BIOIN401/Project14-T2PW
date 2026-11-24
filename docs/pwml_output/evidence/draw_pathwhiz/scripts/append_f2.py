"""Append the F2 tests to tests/test_pwml_layout.py once."""
s = open("C:/t/lane-pwdraw-work/scripts/f2_tests.py.txt", encoding="utf-8").read()
t = "C:/t/lane-pwdraw/tests/test_pwml_layout.py"
cur = open(t, encoding="utf-8").read()
assert "test_no_parent_claimed_when_two" not in cur
open(t, "a", encoding="utf-8", newline="\n").write(s)
print("appended")
