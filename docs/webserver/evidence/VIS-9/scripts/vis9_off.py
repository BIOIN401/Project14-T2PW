"""pytest plugin for the VIS-9 A/B only: switch the new pwc-id fallback off (the writer then behaves as b9534f68)."""
import t2pw.pwml.writer as _w

_w._mapped_compound_pwc_id = lambda *a, **k: None
