p = 'C:/t/corr-assess/scripts/a1_check.py'
raw = open(p, encoding='utf-8', newline='').read()
crlf = '\r\n' in raw
s = raw.replace('\r\n', '\n')
a1 = '''            if pl:
                return sum(p["x"] + p["w"] / 2 for p in pl) / len(pl)'''
b1 = '''            if pl:
                return (sum(p["x"] + p["w"] / 2 for p in pl) / len(pl), sum(p["y"] + p["h"] / 2 for p in pl) / len(pl))'''
a2 = '''            return sum(c["x"] + c["w"] / 2 for c in cs) / len(cs)
        return None'''
b2 = '''            return (sum(c["x"] + c["w"] / 2 for c in cs) / len(cs), sum(c["y"] + c["h"] / 2 for c in cs) / len(cs))
        return None'''
a3 = '''                xa, xb = rx_x(P["reactions"][a]), rx_x(P["reactions"][b])
                if xa is None or xb is None:
                    continue
                prec += 1
                if xa >= xb:'''
b3 = '''                pa, pb = rx_x(P["reactions"][a]), rx_x(P["reactions"][b])
                if pa is None or pb is None:
                    continue
                if n(nm.get(("Compound", c), "")) in CURRENCY:
                    continue
                xa, xb = pa[0], pb[0]
                prec += 1
                # reading order = left to right; a wrap to a later row (>= 100 px lower) also counts as forward
                if not (xb > xa or pb[1] > pa[1] + 100):'''
for a, b in ((a1, b1), (a2, b2), (a3, b3)):
    assert a in s, a[:60]
    s = s.replace(a, b)
if crlf:
    s = s.replace('\n', '\r\n')
open(p, 'w', encoding='utf-8', newline='').write(s)
print('patched')
