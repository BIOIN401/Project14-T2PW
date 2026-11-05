"""R2 verbatim proof (scratch). Prints every line that differs between the moved
blocks at base and their copies in rag_orchestration.py, plus the app-side diff."""
import difflib
import subprocess
from pathlib import Path

ROOT = Path("C:/t/ws-R2")
BASE = "4a41b91c"
base_app = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:src/t2pw/app/streamlit_app.py"],
                          capture_output=True, check=True).stdout.decode("utf-8").replace("\r\n", "\n")
bl = base_app.split("\n")
block_a = "\n".join(bl[184:251])
block_b = "\n".join(bl[434:1527])
rag = (ROOT / "src/t2pw/service/rag_orchestration.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
assert block_a in rag, "block A not verbatim"
print("block A (lines 185-251): verbatim, byte-identical")
start = rag.index("def maybe_run_rag(")
moved_b = rag[start:].rstrip("\n")
d = list(difflib.unified_diff(block_b.split("\n"), moved_b.split("\n"), "base:435-1527", "rag_orchestration", n=0, lineterm=""))
print("block B diff:")
print("\n".join(d))
app = (ROOT / "src/t2pw/app/streamlit_app.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
expected = "\n".join(bl[:184] + bl[253:434] + bl[1529:])
d2 = list(difflib.unified_diff(expected.split("\n"), app.split("\n"), "base-minus-moved", "app-tip", n=0, lineterm=""))
print("app diff vs base-minus-moved-ranges:")
print("\n".join(d2))
