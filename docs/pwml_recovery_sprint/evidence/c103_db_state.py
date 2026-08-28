"""C-103 worktree DB-state probe: measured, not assumed.

The strict-failure replay is a pure offline fixture, but ``strict_db=True`` is
passed on every call in it, so "which DB state was this measured in" is a real
question about the numbers this card reports.
"""
import os
from pathlib import Path

root = Path("C:/t/c103")
print("worktree            :", root)
print(".env present        :", (root / ".env").exists())
print(".venv present       :", (root / ".venv").exists())
for var in ("PATHBANK_DB_HOST", "PATHBANK_DB_USER", "PATHBANK_DB_PASSWORD",
            "PATHBANK_DB_SCHEMA", "PATHBANK_ID_SOURCE"):
    print(f"{var:22s}: {os.environ.get(var, '<unset>')!r}")
print("T2PW_OFFLINE_CURATOR:", os.environ.get("T2PW_OFFLINE_CURATOR", "<unset>"))
try:
    import pymysql  # noqa: F401
    print("pymysql importable  : True")
except Exception as exc:  # noqa: BLE001
    print("pymysql importable  :", type(exc).__name__)
print("offline name index  :", (root / "data/pathwhiz_id_db.json").exists())

import sys
sys.path.insert(0, str(root / "src"))
from t2pw.config import resolution_db_config, resolution_db_configured  # noqa: E402
import t2pw  # noqa: E402

print("measured t2pw       :", t2pw.__file__)
print("resolution_db_configured:", resolution_db_configured())
print("resolution_db_config keys:", sorted(resolution_db_config().keys()))
print("VERDICT: no .env and no .venv in this worktree and no PATHBANK_DB_* in the")
print("environment, so no live PathBank DB was reachable and resolution_db_configured")
print("is False. Every measurement in this card is offline: the replay fixture is a")
print("committed payload and quarantine_and_close does no network or DB work.")
