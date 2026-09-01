import io
p = r"C:\t\rev106\docs\pwml_recovery_sprint\evidence\rev106_f152_scenarios.py"
s = io.open(p, encoding="utf-8").read()
marker = ' "green_with_failure_prose":'
i = s.index(marker)
j = s.index('"plain_green"', i)
new = ''' "green_with_failure_prose": \'\'\'
import warnings
def test_ok():
    warnings.warn(UserWarning("historical note: 7 failed and 3 errors in run 2026-08-01"))
    assert True
\'\'\',
 "red_plus_prose": \'\'\'
import warnings
def test_bad():
    warnings.warn(UserWarning("upstream reported 9 failed and 3 errors"))
    assert 1 == 2
def test_ok():
    assert True
\'\'\',
 '''
s = s[:i] + new + s[j:]
old2 = '    elif name == "genuine_red":'
new2 = ('    elif name == "red_plus_prose":\n'
        "        ok = t['failed'] == 1 and t['passed'] == 1 and te == 0 and not aborts(t, p.returncode)\n"
        '        print(f"    EXPECT tip: SHARPEST CASE -- a genuine red whose output ALSO says '
        "'3 errors': failed=1, passed=1, errors=0, no abort -> {'OK' if ok else 'VIOLATED'}\")\n"
        '    elif name == "genuine_red":')
assert s.count(old2) == 1
s = s.replace(old2, new2, 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched ok")
