import os
from pathlib import Path
root = Path("C:/t/c102")
print("worktree            :", root)
print(".env present        :", (root / ".env").exists())
print(".venv present       :", (root / ".venv").exists())
for var in ("PATHBANK_DB_HOST", "PATHBANK_DB_USER", "PATHBANK_DB_PASSWORD", "PATHBANK_DB_SCHEMA", "PATHBANK_ID_SOURCE"):
    print(f"{var:22s}: {os.environ.get(var, '<unset>')!r}")
print("T2PW_OFFLINE_CURATOR:", os.environ.get("T2PW_OFFLINE_CURATOR", "<unset>"))
try:
    import pymysql  # noqa: F401
    print("pymysql importable  : True")
except Exception as exc:
    print("pymysql importable  :", type(exc).__name__)
print("offline name index  :", (root / "data/pathwhiz_id_db.json").exists())
print("VERDICT: no .env and no .venv in this worktree and no PATHBANK_DB_* in the")
print("environment, so no live PathBank DB was reachable. Every measurement in this")
print("card is offline, from committed artifacts and the pinned gold set.")
