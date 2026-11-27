import json, sys
a = json.load(open("C:/t/vis-pw8-work/metrics/m_b0_all.json")); b = json.load(open(sys.argv[1]))
print("| pathway | canvas before | canvas after | aspect | label px at 1100 wide | visible edges: total length | crossings | interaction edges: total / longest |")
print("|---|---|---|---|---|---|---|---|")
for k in a:
    x, y = a[k], b[k]
    print(f"| {k} | {x['w']}x{x['h']} | {y['w']}x{y['h']} | {x['aspect']} -> {y['aspect']} | {x['label_px_1100']} -> {y['label_px_1100']} | {x['edge_total']} -> {y['edge_total']} | {x['edge_cross']} -> {y['edge_cross']} | {x['int_total']}/{x['int_max']} -> {y['int_total']}/{y['int_max']} |")
