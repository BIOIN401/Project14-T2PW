"""Point the copied showcase harness at compose project t2pw-pwdraw (db 3357, app 3054, own bundle volume)."""
H = "C:/t/lane-pwdraw-work/harness/"
for f, subs in (("run_import.py", [('PROJECT = "t2pw-show"', 'PROJECT = "t2pw-pwdraw"'),
                                   ('"http://localhost:3014"', '"http://localhost:3054"'),
                                   ('BUNDLE_VOL = "t2pw-show_bundle"', 'BUNDLE_VOL = "t2pw-pwdraw_bundle"')]),
                ("docker-compose.yml", [("Project name: t2pw-show", "Project name: t2pw-pwdraw"), ('"3321:3306"', '"3357:3306"'),
                                        ('"3014:3000"', '"3054:3000"'), ("name: t2pw-show_bundle", "name: t2pw-pwdraw_bundle")])):
    s = open(H + f, encoding="utf-8", newline="").read()
    for a, b in subs:
        assert s.count(a) == 1, (f, a)
        s = s.replace(a, b)
    open(H + f, "w", encoding="utf-8", newline="").write(s)
print("ok")
