"""F4: correct the header comment of final2.sh (work copy; the repo evidence copy is updated in the docs commit)."""
p = "C:/t/lane-pwdraw-work/scripts/final2.sh"
s = open(p, encoding="utf-8").read()
old = ("# One heavy-lock hold on HEAD 4b701677: focused tests, the 97 + 8 replays (after4), content check, suite2, then the\n"
       "# PathWhiz after arm only (the base arm of final.sh is reused: same base files).\n")
new = ("# One heavy-lock hold on HEAD 4b701677: focused tests, the 97 + 8 replays (after4), content check, suite2, then the\n"
       "# PathWhiz A/B with both arms imported fresh (base into pw/base2, after into pw/after2; final.sh's run lost its\n"
       "# showcase imports to CRLF paths), then the stack goes down (volumes kept).\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("fixed")
