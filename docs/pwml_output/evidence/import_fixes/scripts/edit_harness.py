H = "C:/t/pwml-fix-work/harness/"
p = H + "run_import.py"
s = open(p, encoding="utf-8", newline="").read()
for a, b in [('PROJECT = "t2pw-pwi"', 'PROJECT = "t2pw-fix"'), ('"http://localhost:3010"', '"http://localhost:3034"'),
             ('r"C:/Users/Angad/Desktop/SummerBIOIN/smpdb")', 'r"C:/t/pwml-fix-work/smpdb-run")'),
             ('BUNDLE_VOL = "t2pw-pwi_bundle" ', 'BUNDLE_VOL = "t2pw-fix_bundle" '),
             ('OWNER_BUNDLE_VOL = "smpdb_bundle_cache"', 'OWNER_BUNDLE_VOL = "t2pw-pwi_bundle"  # copied read-only, as the showcase did')]:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
p = H + "docker-compose.yml"
s = open(p, encoding="utf-8", newline="").read()
for a, b in [("Project name: t2pw-pwi", "Project name: t2pw-fix"), ('"3317:3306"', '"3337:3306"'), ('"3010:3000"', '"3034:3000"'),
             ("name: t2pw-pwi_bundle", "name: t2pw-fix_bundle")]:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
