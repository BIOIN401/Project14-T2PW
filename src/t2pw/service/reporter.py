"""How service code reports to a user without importing a UI toolkit (R2).

The service functions moved out of ``streamlit_app.py`` used to call ``st.status``,
``st.warning`` and friends directly, and wrote ``st.session_state``. They now take a
``reporter`` and a ``state`` mapping instead:

* :class:`StreamlitReporter` wraps the ``streamlit`` module it is *given* (this module
  never imports it) and forwards every call unchanged, so the app's UI is what it was.
* :class:`LoggingReporter` writes the same messages to a logger, for headless use.

``reporter=None`` / ``state=None`` resolve through :func:`default_reporter` /
:func:`default_state`. The app installs its Streamlit pair with
:func:`install_app_defaults` when it is imported, so a caller that reaches the service
through the app module without passing either keeps exactly today's behaviour. With no
app imported the defaults are a :class:`LoggingReporter` and a fresh ``dict``.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import (Any, Callable, ContextManager, Dict, Iterator, MutableMapping, Optional,
                    Protocol)


class Progress(Protocol):
    """A live progress line: ``st.status``'s container, or a logged stand-in."""

    def write(self, body: Any) -> Any: ...

    def update(self, *, label: Optional[str] = None, state: Optional[str] = None,
               **kwargs: Any) -> Any: ...


class Reporter(Protocol):
    """What the service may say to a user. ``state`` is ``running|complete|error``."""

    def info(self, body: Any) -> Any: ...

    def warning(self, body: Any) -> Any: ...

    def error(self, body: Any) -> Any: ...

    def json(self, body: Any) -> Any: ...

    def progress(self, label: str, **kwargs: Any) -> Progress: ...

    # R3a: the input-assembly messages (st.success / caption / text and the
    # st.spinner / st.expander context managers).
    def success(self, body: Any) -> Any: ...

    def caption(self, body: Any) -> Any: ...

    def text(self, body: Any) -> Any: ...

    def spinner(self, text: str, **kwargs: Any) -> ContextManager[Any]: ...

    def expander(self, label: str, **kwargs: Any) -> ContextManager[Any]: ...

    # R3c: the Stage-1 attempts listing (``render_attempts``: st.markdown / st.code).
    def markdown(self, body: Any) -> Any: ...

    def code(self, body: Any, **kwargs: Any) -> Any: ...


class StreamlitReporter:
    """Forwards each call to the given ``streamlit`` module, arguments untouched."""

    def __init__(self, st: Any) -> None:
        self._st = st

    def info(self, body: Any) -> Any:
        return self._st.info(body)

    def warning(self, body: Any) -> Any:
        return self._st.warning(body)

    def error(self, body: Any) -> Any:
        return self._st.error(body)

    def json(self, body: Any) -> Any:
        return self._st.json(body)

    def progress(self, label: str, **kwargs: Any) -> Any:
        # ``st.status`` returns the container whose ``write`` / ``update`` the caller uses.
        return self._st.status(label, **kwargs)

    def success(self, body: Any) -> Any:
        return self._st.success(body)

    def caption(self, body: Any) -> Any:
        return self._st.caption(body)

    def text(self, body: Any) -> Any:
        return self._st.text(body)

    def spinner(self, text: str, **kwargs: Any) -> Any:
        return self._st.spinner(text, **kwargs)

    def expander(self, label: str, **kwargs: Any) -> Any:
        return self._st.expander(label, **kwargs)

    def markdown(self, body: Any) -> Any:
        return self._st.markdown(body)

    def code(self, body: Any, **kwargs: Any) -> Any:
        return self._st.code(body, **kwargs)


class _LoggedProgress:
    def __init__(self, logger: logging.Logger, label: str) -> None:
        self._logger, self.label, self.state = logger, label, "running"
        self._logger.info("%s", label)

    def write(self, body: Any) -> None:
        self._logger.info("  %s", body)

    def update(self, *, label: Optional[str] = None, state: Optional[str] = None,
               **_kwargs: Any) -> None:
        if label is not None:
            self.label = label
        if state is not None:
            self.state = state
        level = logging.ERROR if self.state == "error" else logging.INFO
        self._logger.log(level, "%s [%s]", self.label, self.state)


class LoggingReporter:
    """Headless reporter: every message goes to ``logger`` at a matching level."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger("t2pw.service.reporter")

    def info(self, body: Any) -> None:
        self._logger.info("%s", body)

    def warning(self, body: Any) -> None:
        self._logger.warning("%s", body)

    def error(self, body: Any) -> None:
        self._logger.error("%s", body)

    def json(self, body: Any) -> None:
        self._logger.info("%s", json.dumps(body, indent=2, ensure_ascii=False, default=str))

    def progress(self, label: str, **_kwargs: Any) -> _LoggedProgress:
        return _LoggedProgress(self._logger, label)

    def success(self, body: Any) -> None:
        self._logger.info("%s", body)

    def caption(self, body: Any) -> None:
        self._logger.info("%s", body)

    def text(self, body: Any) -> None:
        self._logger.info("%s", body)

    @contextlib.contextmanager
    def spinner(self, text: str, **_kwargs: Any) -> Iterator[None]:
        self._logger.info("%s", text)
        yield

    @contextlib.contextmanager
    def expander(self, label: str, **_kwargs: Any) -> Iterator[None]:
        self._logger.info("%s", label)
        yield

    def markdown(self, body: Any) -> None:
        self._logger.info("%s", body)

    def code(self, body: Any, **_kwargs: Any) -> None:
        self._logger.info("%s", body)


_APP_DEFAULTS: Dict[str, Callable[[], Any]] = {}


def install_app_defaults(*, reporter: Callable[[], Reporter],
                         state: Callable[[], MutableMapping[str, Any]]) -> None:
    """Called by the app at import: what ``reporter=None`` / ``state=None`` resolve to."""

    _APP_DEFAULTS.update(reporter=reporter, state=state)


def default_reporter() -> Reporter:
    factory = _APP_DEFAULTS.get("reporter")
    return factory() if factory is not None else LoggingReporter()


def default_state() -> MutableMapping[str, Any]:
    factory = _APP_DEFAULTS.get("state")
    return factory() if factory is not None else {}
