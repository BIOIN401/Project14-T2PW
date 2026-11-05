"""Record/replay of the pipeline's external calls (R2.5, new capability). Test-only; no ``src`` change.

A run of the app touches the outside world at three boundaries. Each is patched at ONE place that every
caller already routes through, so no importer's bound name is missed:

* **LLM** -- ``t2pw.llm.client._client.chat.completions.create``, patched on the *instance*. ``chat``,
  ``chat_detailed`` and ``chat_with_tools`` read the module global ``_client`` at call time, and
  ``t2pw.stoich.agent`` binds that same object (``from t2pw.llm.client import _client``), so the one
  instance attribute reaches every importer, including the ones that bound ``chat`` by name.
* **HTTP** -- ``requests.Session.request`` (class level). ``map_ids.HttpClient``, the per-thread enrichment
  clients, ``stoich.agent._SESSION`` and bare ``requests.get`` all end there.
* **PathBank DB** -- ``map_ids.PathBankDbResolver._query``, the only method that touches the connection.

``record`` calls through and saves request -> response pairs; ``replay`` answers from the fixture, raises
:class:`ReplayMiss` on anything unrecorded (and lists it in ``misses``), and refuses every non-loopback
socket connect and DNS lookup (listed in ``refused``) as ``tests/test_orch768_replay_corpus.py`` does.
A recording may RESUME from an interrupted segment's JSONL transcript (``prior``/``sink``): the prior
answers are served first and only what lies beyond them goes live (see ``harness.MAX_SEGMENTS``).

**Keying** (see :data:`KEYING`): sha256 of a canonical JSON of what decides the answer.

* LLM: every ``create`` kwarg EXCEPT ``model``, ``timeout``, ``extra_headers``, ``extra_query`` and
  ``extra_body`` -- i.e. messages, temperature, max_tokens, response_format, tools, tool_choice. The key is
  model-agnostic; the model actually answering is kept in the stored response only.
* HTTP: method, the fully prepared URL (``params`` merged in, in the caller's order) and the body
  (``data``/``json``). Headers, timeouts, ``verify`` and the session are ignored. The VALUES of the query
  parameters in :data:`SECRET_PARAMS` are replaced with ``<redacted>`` in the key and in storage.
* DB: the SQL text and its params, exactly.

The same key may be asked several times (identical draws at temperature 0, retries after a recorded
HTTP error): answers are served in recorded order, and asking once more than recorded is a miss.

**No secrets are stored**: no request headers at all, ``Set-Cookie`` dropped from response headers, the
secret query parameters redacted, DB error strings redacted of the connection settings. The recorder
also refuses to write a fixture containing any configured secret value (:func:`assert_no_secret_values`);
``tests/test_equivalence_app_vs_service.py`` scans the committed fixture for secret shapes.
"""

from __future__ import annotations

import base64
import datetime as _dt
import decimal
import hashlib
import json
import os
import re
import socket
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

KEYING = {
    "llm": "sha256 of create() kwargs minus model/timeout/extra_headers/extra_query/extra_body",
    "http": "sha256 of method + prepared URL (secret param values redacted) + body; headers ignored",
    "db": "sha256 of PathBankDbResolver._query(sql, params), exact",
    "repeats": "one key may hold several answers, served in recorded order; one more ask is a miss",
}
IGNORED_LLM_KWARGS = ("model", "timeout", "extra_headers", "extra_query", "extra_body")
SECRET_PARAMS = frozenset({"api_key", "apikey", "key", "token", "access_token", "email", "tool"})
DROPPED_RESPONSE_HEADERS = frozenset({"set-cookie", "cookie", "authorization", "proxy-authorization"})
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})

#: Environment the runs are made under. Every variable with one of these prefixes is cleared first, so
#: neither the operator's shell nor a ``.env`` can steer a replay; then the fixture's pinned values go in.
ENV_PREFIXES = ("T2PW_", "RAG_", "LLM_", "OPENROUTER_", "PATHBANK_", "NCBI_", "LMSTUDIO_", "LOCAL_")
#: Never pinned into a fixture. Record mode takes them from the primary checkout's ``.env``; replay mode
#: gets the placeholders below (the DB placeholders only make ``PathBankDbResolver.from_env`` build a
#: resolver, as it did when recording; ``_query`` is replayed so nothing connects).
SECRET_ENV = frozenset({
    "OPENROUTER_API_KEY", "PATHBANK_DB_HOST", "PATHBANK_DB_USER", "PATHBANK_DB_PASSWORD",
    "NCBI_EMAIL", "NCBI_TOOL", "NCBI_API_KEY", "RAG_EMBEDDING_API_KEY", "LOCAL_API_KEY",
})
REPLAY_PLACEHOLDERS = {
    "OPENROUTER_API_KEY": "sk-or-" + "replay-placeholder-not-a-key",  # client.py checks the prefix
    "PATHBANK_DB_HOST": "pathbank-db.replay.invalid",
    "PATHBANK_DB_USER": "replay",
    "PATHBANK_DB_PASSWORD": "replay",
}
#: Forced in record AND replay: the offline flags of TEST_MATRIX, and RAG off (its acquisition uses
#: urllib against PubMed and an embedding endpoint, neither of which this cassette covers).
FORCED_ENV = {"T2PW_OFFLINE_CURATOR": "1", "T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0", "RAG_ENABLED": "0"}
#: Replay only: ``client.py`` sleeps this long after every completion. Timing only, never a decision.
REPLAY_ONLY_ENV = {"LLM_CALL_SPACING": "0"}


class ReplayMiss(RuntimeError):
    """A request the fixture has no (more) answers for."""


def _sha(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    """Plain JSON for keying: pydantic models (an earlier reply re-sent as a message) via model_dump."""

    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot key a {type(value).__name__}")


def llm_key(kwargs: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """``(key, keyed request)`` for one ``chat.completions.create(**kwargs)``."""

    request = {k: jsonable(v) for k, v in sorted(kwargs.items()) if k not in IGNORED_LLM_KWARGS}
    return _sha(request), request


def pinned_env_from(dotenv_values: Mapping[str, Optional[str]]) -> Dict[str, str]:
    """The behaviour-relevant, non-secret part of a ``.env``, plus the forced flags."""

    pinned = {k: str(v) for k, v in dotenv_values.items()
              if v is not None and k.startswith(ENV_PREFIXES) and k not in SECRET_ENV and not k.startswith("RAG_")}
    pinned.update(FORCED_ENV)
    return dict(sorted(pinned.items()))


def apply_env(mp: Any, pinned: Mapping[str, str], secrets: Mapping[str, str]) -> None:
    """Clear every prefixed variable, then set ``pinned`` + ``secrets``. Also disables ``.env`` loading,
    so ``t2pw.llm.client`` (``override=True``) and ``t2pw.config`` cannot overwrite what was pinned."""

    import dotenv

    for name in list(os.environ):
        if name.startswith(ENV_PREFIXES):
            mp.delenv(name)
    for name, value in {**pinned, **secrets}.items():
        mp.setenv(name, value)
    mp.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


_SECRET_PARAM_RE = re.compile(r"(?i)\b(" + "|".join(sorted(SECRET_PARAMS)) + r")=([^&\s'\"]+)")


def redact_url(text: str) -> str:
    """Replace the value of every :data:`SECRET_PARAMS` query parameter, in a URL or in any message."""

    return _SECRET_PARAM_RE.sub(r"\1=<redacted>", text)


def _body(data: Any, json_body: Any) -> Any:
    if json_body is not None:
        return {"json": jsonable(json_body)}
    if data is None or data == {} or data == b"" or data == "":
        return None
    if isinstance(data, Mapping):
        return {"form": sorted((str(k), str(v)) for k, v in data.items())}
    if isinstance(data, (list, tuple)):
        return {"form": [list(map(str, item)) for item in data]}
    if isinstance(data, bytes):
        return {"bytes": base64.b64encode(data).decode("ascii")}
    return {"text": str(data)}


def _encode_db(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return {"__t": "decimal", "v": str(value)}
    if isinstance(value, _dt.datetime):
        return {"__t": "datetime", "v": value.isoformat()}
    if isinstance(value, _dt.date):
        return {"__t": "date", "v": value.isoformat()}
    if isinstance(value, (bytes, bytearray)):
        return {"__t": "bytes", "v": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(k): _encode_db(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_db(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot record a DB value of type {type(value).__name__}")


def _decode_db(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__t", "v"}:
        kind, raw = value["__t"], value["v"]
        return {"decimal": decimal.Decimal, "datetime": _dt.datetime.fromisoformat,
                "date": _dt.date.fromisoformat, "bytes": base64.b64decode}[kind](raw)
    if isinstance(value, dict):
        return {k: _decode_db(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_db(v) for v in value]
    return value


class Cassette:
    """One arm's view of the fixture. ``mode`` is ``"record"`` or ``"replay"``; cursors start at zero."""

    def __init__(self, mode: str, entries: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                 redact: Iterable[str] = (), prior: Optional["Cassette"] = None,
                 sink: Optional[Path] = None) -> None:
        assert mode in ("record", "replay"), mode
        assert prior is None or mode == "record", "only a recording resumes from a prior segment"
        self.mode = mode
        #: Record mode only: an interrupted earlier segment. Its answers are served first, in order; only
        #: what it does not hold goes live. Every answer, served or live, lands in THIS transcript.
        self.prior = prior
        self._sink = open(sink, "a", encoding="utf-8") if sink else None
        self.entries: Dict[str, List[Dict[str, Any]]] = {"llm": [], "http": [], "db": []}
        self._index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for kind, rows in (entries or {}).items():
            for row in rows:
                self._index[(kind, row["key"])].append(len(self.entries[kind]))
                self.entries[kind].append(row)
        self._cursor: Dict[Tuple[str, str], int] = defaultdict(int)
        self._served: set = set()
        self._lock = threading.Lock()
        self._redact = [s for s in redact if s and len(s) >= 4]
        self.misses: List[Dict[str, Any]] = []
        self.refused: List[str] = []
        self.live_llm_errors: List[str] = []

    # -- bookkeeping ---------------------------------------------------------------------------
    def _scrub(self, text: str) -> str:
        for secret in self._redact:
            text = text.replace(secret, "<redacted>")
        return text

    def _record(self, kind: str, key: str, request: Any, answer: Dict[str, Any]) -> None:
        row = {"key": key, "request": request, **answer}
        with self._lock:
            self.entries[kind].append(row)
            if self._sink is not None:  # flushed per answer: a killed segment keeps everything it got
                self._sink.write(json.dumps({"kind": kind, **row}, ensure_ascii=False) + "\n")
                self._sink.flush()

    def _take(self, kind: str, key: str) -> Tuple[Optional[Dict[str, Any]], int, int]:
        with self._lock:
            slots = self._index.get((kind, key), [])
            n = self._cursor[(kind, key)]
            if n < len(slots):
                self._cursor[(kind, key)] = n + 1
                self._served.add((kind, slots[n]))
                return self.entries[kind][slots[n]], n, len(slots)
            return None, n, len(slots)

    def _lookup(self, kind: str, key: str, request: Any) -> Optional[Dict[str, Any]]:
        """Replay: the next recorded answer or a loud :class:`ReplayMiss`. Record: the prior segment's next
        answer (re-logged into this transcript), or ``None`` meaning "call the real service"."""

        if self.mode == "record":
            row = self.prior._take(kind, key)[0] if self.prior is not None else None
            if row is not None:
                self._record(kind, key, request, {k: v for k, v in row.items() if k not in ("key", "request")})
            return row
        row, n, recorded = self._take(kind, key)
        if row is not None:
            return row
        miss = {"kind": kind, "key": key, "asked": n + 1, "recorded": recorded,
                "request": json.dumps(request, ensure_ascii=False)[:600]}
        with self._lock:
            self.misses.append(miss)
        raise ReplayMiss(f"unrecorded {kind} request (ask #{n + 1}, {recorded} recorded): {miss['request']}")

    def unused(self) -> List[Tuple[str, int]]:
        return [(kind, i) for kind, rows in self.entries.items() for i in range(len(rows)) if (kind, i) not in self._served]

    def counts(self) -> Dict[str, int]:
        return {kind: len(rows) for kind, rows in self.entries.items()}

    # -- the three boundaries ------------------------------------------------------------------
    def install(self, mp: Any) -> None:
        """Patch the boundaries on ``mp`` (a ``pytest.MonkeyPatch``). Imports ``t2pw.llm.client``."""

        import requests
        from openai.types.chat import ChatCompletion

        import t2pw.llm.client as client
        from t2pw.mapping import map_ids

        completions = client._client.chat.completions
        real_create = completions.create
        real_request = requests.Session.request
        real_query = map_ids.PathBankDbResolver._query
        cassette = self

        def create(*args: Any, **kwargs: Any) -> Any:
            assert not args, "chat.completions.create is keyword-only"
            key, request = llm_key(kwargs)
            row = cassette._lookup("llm", key, request)
            if row is not None:
                return ChatCompletion.model_validate(row["response"])
            try:
                resp = real_create(**kwargs)
            except Exception as exc:  # noqa: BLE001 -- noted, then the client's own retry runs
                cassette.live_llm_errors.append(f"{type(exc).__name__}: {cassette._scrub(str(exc))[:200]}")
                raise
            cassette._record("llm", key, request, {"response": resp.model_dump(mode="json")})
            return resp

        def request(session: Any, method: str, url: str, params: Any = None, data: Any = None,
                    headers: Any = None, *args: Any, json: Any = None, **kwargs: Any) -> Any:
            prepared = requests.Request(method.upper(), url, params=params).prepare().url or url
            req = {"method": method.upper(), "url": redact_url(prepared), "body": _body(data, json)}
            key = _sha(req)
            row = cassette._lookup("http", key, req)
            if row is None:
                try:
                    resp = real_request(session, method, url, params, data, headers, *args, json=json, **kwargs)
                except Exception as exc:  # noqa: BLE001 -- replayed as the same exception type
                    cassette._record("http", key, req, {"error": {"type": type(exc).__name__,
                                                                   "message": cassette._scrub(redact_url(str(exc)))[:300]}})
                    raise
                content = resp.content
                try:
                    body = {"text": content.decode("utf-8")}
                except UnicodeDecodeError:
                    body = {"b64": base64.b64encode(content).decode("ascii")}
                cassette._record("http", key, req, {"response": {
                    "status": resp.status_code, "reason": resp.reason, "url": redact_url(resp.url),
                    "encoding": resp.encoding,
                    "headers": {k: v for k, v in resp.headers.items() if k.lower() not in DROPPED_RESPONSE_HEADERS},
                    **body}})
                return resp
            if "error" in row:
                exc_type = getattr(requests.exceptions, row["error"]["type"], requests.exceptions.RequestException)
                raise exc_type(row["error"]["message"])
            saved = row["response"]
            resp = requests.Response()
            resp.status_code = saved["status"]
            resp.reason = saved["reason"]
            resp.url = saved["url"]
            resp.encoding = saved["encoding"]
            resp.headers = requests.structures.CaseInsensitiveDict(saved["headers"])
            resp._content = saved["text"].encode("utf-8") if "text" in saved else base64.b64decode(saved["b64"])
            resp._content_consumed = True
            resp.elapsed = _dt.timedelta(0)
            resp.request = requests.Request(method.upper(), saved["url"]).prepare()
            return resp

        def _query(resolver: Any, sql: str, params: Any) -> List[Dict[str, Any]]:
            req = {"sql": sql, "params": _encode_db(list(params or ()))}
            key = _sha(req)
            row = cassette._lookup("db", key, req)
            if row is None:
                before = resolver.last_error
                rows = real_query(resolver, sql, params)
                error = resolver.last_error if resolver.last_error != before else None
                cassette._record("db", key, req, {"rows": _encode_db(rows),
                                                  "error": cassette._scrub(str(error)) if error else None})
                return rows
            if row.get("error"):
                resolver.last_error = row["error"]
            return _decode_db(row["rows"])

        mp.setattr(completions, "create", create)
        mp.setattr(requests.Session, "request", request)
        mp.setattr(map_ids.PathBankDbResolver, "_query", _query)
        if self.mode == "replay":
            self._block_sockets(mp)

    def _block_sockets(self, mp: Any) -> None:
        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
        real_create_connection, real_getaddrinfo = socket.create_connection, socket.getaddrinfo
        cassette = self

        def _refuse(target: Any) -> None:
            cassette.refused.append(repr(target)[:120])
            raise OSError(f"network refused by the equivalence replay: {target!r}")

        def _host(address: Any) -> str:
            return str(address[0] if isinstance(address, tuple) and address else address)

        def connect(sock: Any, address: Any) -> Any:
            if isinstance(address, tuple) and _host(address) not in LOOPBACK:
                _refuse(address)
            return real_connect(sock, address)

        def connect_ex(sock: Any, address: Any) -> Any:
            if isinstance(address, tuple) and _host(address) not in LOOPBACK:
                _refuse(address)
            return real_connect_ex(sock, address)

        def create_connection(address: Any, *a: Any, **k: Any) -> Any:
            if _host(address) not in LOOPBACK:
                _refuse(address)
            return real_create_connection(address, *a, **k)

        def getaddrinfo(host: Any, *a: Any, **k: Any) -> Any:
            if host is not None and str(host) not in LOOPBACK:
                _refuse(host)
            return real_getaddrinfo(host, *a, **k)

        mp.setattr(socket.socket, "connect", connect)
        mp.setattr(socket.socket, "connect_ex", connect_ex)
        mp.setattr(socket, "create_connection", create_connection)
        mp.setattr(socket, "getaddrinfo", getaddrinfo)


def load_transcript(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """A segment's JSONL transcript as fixture ``entries``. A torn last line (the segment was killed
    mid-write) is dropped."""

    entries: Dict[str, List[Dict[str, Any]]] = {"llm": [], "http": [], "db": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        entries[row.pop("kind")].append(row)
    return entries


def assert_no_secret_values(text: str, secrets: Mapping[str, str]) -> None:
    """Refuse a fixture that contains any configured secret VALUE. Names only in the error, never values."""

    leaked = sorted(name for name, value in secrets.items() if value and len(value) >= 4 and value in text)
    if leaked:
        raise SystemExit(f"refusing to write the fixture: it contains the value of {', '.join(leaked)}")
