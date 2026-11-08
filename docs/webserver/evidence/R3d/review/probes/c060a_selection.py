"""Apply the selection rule of c060a (imported from the tip test file) to base APP/1440 and tip EXT/790."""
import ast, importlib.util, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "C:/t/ws-R3d/src")
spec = importlib.util.spec_from_file_location("c060a_t", "C:/t/ws-R3d/tests/test_c060a_requested_pathway_wiring.py")
m = importlib.util.module_from_spec(spec)
sys.modules["c060a_t"] = m
spec.loader.exec_module(m)
BASE = "C:/t/ws-R3d-base/src/t2pw/app/streamlit_app.py"
TIP = "C:/t/ws-R3d/src/t2pw/service/extraction.py"
GATE = m.GATE_MODULE

def select(path, floor):
    tree = ast.parse(open(path, encoding="utf-8").read())
    calls = m._merge_calls(tree)
    sites = [n for n in calls if n.lineno >= floor]
    print(path, "floor", floor, "all merge statements", [n.lineno for n in calls], "selected", [n.lineno for n in sites])
    out = []
    for i, s in enumerate(sites):
        lower = sites[i - 1].lineno if i else floor - 1
        imps = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == GATE and lower < n.lineno < s.lineno]
        out.append((ast.unparse(s), [ast.unparse(x) for x in imps], s.lineno, [x.lineno for x in imps]))
    return out
b = select(BASE, 1440)
t = select(TIP, 790)
assert len(b) == len(t) == 2, (len(b), len(t))
for i in range(2):
    print("site", i, "base lines", b[i][2], b[i][3], "tip lines", t[i][2], t[i][3])
    print("  statement identical:", b[i][0] == t[i][0], "| gate imports identical:", b[i][1] == t[i][1])
    print("  ", t[i][0]); print("  ", t[i][1])
tree = ast.parse(open(TIP, encoding="utf-8").read())
print("every EXT import of the gate module:", [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == GATE])
