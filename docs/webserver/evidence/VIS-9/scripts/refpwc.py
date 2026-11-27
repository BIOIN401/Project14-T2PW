import sys, re
n = m = 0; bad = []; nopwc = 0
for p in sys.argv[1:]:
    t = open(p, encoding="utf-8").read()
    for blk in re.findall(r"<compound>(.*?)</compound>", t, re.S):
        i = re.search(r"<id[^>]*>(\d+)</id>", blk); w = re.search(r"<pwc-id>([^<]+)</pwc-id>", blk)
        nm = re.search(r"<name>([^<]*)</name>", blk)
        if i and w:
            n += 1
            if w.group(1) != "PW_C%06d" % int(i.group(1)):
                m += 1; bad.append((p[-13:], i.group(1), w.group(1), nm and nm.group(1)))
        elif i:
            nopwc += 1
print("ref compounds with id+pwc", n, "mismatch", m, "without pwc", nopwc, bad[:10])
