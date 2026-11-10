import re
p = open("C:/t/ev-dedup/pwml_dup.py", encoding="utf-8").read()
p = p.replace('''    root = ET.parse(path).getroot()''', '''    txt = open(path, encoding="utf-8", errors="replace").read()
    txt = re.sub(r"<(/?)\s*[A-Za-z_][\w.-]*:", r"<\1", txt)          # drop tag prefixes (no ns decl in these files)
    txt = re.sub(r"(\s)[A-Za-z_][\w.-]*:([A-Za-z_][\w.-]*=)", r"\1\2", txt)
    txt = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", txt)
    txt = re.sub(r"&(?!(amp|lt|gt|apos|quot|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", txt)
    root = ET.fromstring(txt.encode("utf-8"))''')
open("C:/t/ev-dedup/pwml_dup.py","w",encoding="utf-8").write(p)
