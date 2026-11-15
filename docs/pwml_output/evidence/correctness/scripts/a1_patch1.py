p = 'C:/t/corr-assess/scripts/a1_check.py'
s = open(p, encoding='utf-8').read()
s = s.replace('''    el = collections.defaultdict(lambda: {"l": [], "r": []})
    for e in db["reaction_elements"]:
        el[e["reaction_id"]]["l" if "Left" in e["type"] else "r"].append(n(e["name"]))
    enz = collections.defaultdict(list)
    for e in db["reaction_enzymes"]:
        enz[e["reaction_id"]].append(n(e["complex_name"]))''', '''    na = lambda s: re.sub(r"[^\x00-\x7f]", "?", n(s))  # mysql client returned non-ascii as '?'
    pw_sigs = [tuple(frozenset(na(x) for x in grp) for grp in sg) for sg in pw_sigs]
    el = collections.defaultdict(lambda: {"l": [], "r": []})
    for e in db["reaction_elements"]:
        el[e["reaction_id"]]["l" if "Left" in e["type"] else "r"].append(na(e["name"]))
    enz = collections.defaultdict(list)
    for e in db["reaction_enzymes"]:
        enz[e["reaction_id"]].append(na(e["complex_name"]))''')
s = s.replace('dtw = [(n(e["name"]), bs.get', 'dtw = [(na(e["name"]), bs.get')
s = s.replace('pww = [(a, n(states', 'pww = [(na(a), n(states')
open(p, 'w', encoding='utf-8').write(s)
