"""C-056a: is the resolution DB actually reachable, measured FROM INSIDE the process?

A ``qb`` cohort that cannot reach the PathBank resolution DB is **not a pass** -- it is
``DB_UNAVAILABLE`` and must be reported as *not run*. Env vars being present is not
evidence: ``.env`` may name a host that is down. So this opens a real connection through
the PRODUCTION class (``mapping.map_ids.PathBankDbResolver``), never a restatement of its
settings, and runs one trivial query.

Exit 0 = reachable. Exit 4 = ``DB_UNAVAILABLE``. Secrets are never printed: only the
host/schema shape and a boolean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from t2pw.config import resolution_db_config, resolution_db_configured  # noqa: E402
from t2pw.mapping.map_ids import PathBankDbResolver  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import t2pw

    cfg = resolution_db_config()
    out = {
        "task": "C-056a",
        "purpose": "verify resolution-DB reachability from inside the process before qb",
        "t2pw_file": t2pw.__file__,
        "configured": resolution_db_configured(),
        "host_present": bool(cfg.get("host")),
        "user_present": bool(cfg.get("user")),
        "schema": cfg.get("schema", ""),
        "driver_importable": False,
        "connected": False,
        "query_ok": False,
        "error": "",
    }

    resolver = PathBankDbResolver.from_env()
    if resolver is None:
        out["error"] = "from_env returned None: host/user absent"
    else:
        out["driver_importable"] = resolver.available()
        try:
            if resolver._ensure_connection():
                out["connected"] = True
                with resolver._conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    out["query_ok"] = bool(cur.fetchone())
            else:
                out["error"] = str(resolver.last_error or "connection refused")
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            resolver.close()

    out["verdict"] = "DB_AVAILABLE" if out["query_ok"] else "DB_UNAVAILABLE"
    dest = Path(args.out).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"t2pw.__file__ = {t2pw.__file__}")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["query_ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
