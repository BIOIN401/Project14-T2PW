"""Dev check (not the gate): call the review-round tests against the renderer in <src tree>. usage: failfirst.py <tree>"""
import importlib.util
import os
import sys
import traceback

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
spec = importlib.util.spec_from_file_location("tvd", "C:/t/lane-draw/tests/test_vision_diagram.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
print("renderer from", T.dg.__file__)
cases = [("double " + n, lambda n=n, d=d: T.test_double_membrane_only_for_real_envelopes(n, d))
         for n, d in (("nucleus", True), ("nucleoid", False), ("nucleolus", False))]
cases += [("cargo", T.test_transported_currency_keeps_its_link_to_the_reactions_that_use_it),
          ("chained", T.test_chained_transports_in_one_membrane_keep_flow_order)]
for name, fn in cases:
    try:
        fn()
        print("PASS", name)
    except Exception:  # noqa: BLE001
        print("FAIL", name, traceback.format_exc().strip().splitlines()[-1][:200])
