import sys
import t2pw.service.extraction  # noqa: F401
mods = sorted(n for n in sys.modules if n == "t2pw" or n.startswith("t2pw."))
print(len(mods))
print(mods)
