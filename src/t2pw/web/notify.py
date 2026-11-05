"""Job-finished email (W8). Stand-alone: the worker or a route calls :func:`notify_job`.

SMTP settings come from the environment: ``T2PW_SMTP_HOST``, ``T2PW_SMTP_PORT``
(default 587), ``T2PW_SMTP_USER``, ``T2PW_SMTP_PASSWORD``, ``T2PW_SMTP_FROM`` and
``T2PW_SMTP_STARTTLS`` (default on; port 465 uses implicit TLS instead). With no host or
no from-address, :func:`notify_job` logs one line and returns False. It never raises.

The message is fixed text plus the job link: no user-supplied text reaches a header or
the body, so it cannot be used for header or content injection. The recipient address
and the link are validated first. Neither the address, the password nor any exception
text is logged (SMTP errors can echo the address); only the exception class name is.
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
import ssl
from email.message import EmailMessage
from typing import Mapping, Optional

log = logging.getLogger("t2pw.web.notify")

#: Seconds for connect and every SMTP command, so a dead mail server cannot hold the worker.
SMTP_TIMEOUT_S = 10

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+")
_URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]{1,500}")

SUBJECTS = {
    "finished": "Your T2PW pathway job has finished",
    "failed": "Your T2PW pathway job did not complete",
}
BODIES = {
    "finished": "Your T2PW pathway job has finished.\n\nOpen it here:\n{link}\n",
    "failed": ("Your T2PW pathway job did not complete.\n\n"
               "See what happened, and try again, here:\n{link}\n"),
}


def is_valid_email(addr: object) -> bool:
    """Trust-boundary check for the optional email field (also use it in the form)."""
    return (isinstance(addr, str) and len(addr) <= 254 and not addr.startswith(".")
            and ".." not in addr.split("@")[0] and _EMAIL_RE.fullmatch(addr) is not None)


def notify_job(to_addr: Optional[str], job_url: str, *, outcome: str = "finished",
               base_url: Optional[str] = None,
               env: Optional[Mapping[str, str]] = None) -> bool:
    """Email ``to_addr`` the job link; True only if the server accepted the message.

    When ``base_url`` is given, ``job_url`` must start with it (``base_url + "/"``), so a
    link built from anything but our own site is refused.
    ``outcome`` is ``"finished"`` or ``"failed"`` (whether to send for failures is the
    caller's decision). Returns False, never raises, when email is off, the address or
    link is invalid, or sending fails.
    """
    env = os.environ if env is None else env
    host, sender = env.get("T2PW_SMTP_HOST", "").strip(), env.get("T2PW_SMTP_FROM", "").strip()
    if not to_addr:
        return False
    if not host or not sender:
        log.info("job email skipped: SMTP is not configured")
        return False
    if outcome not in SUBJECTS or not is_valid_email(to_addr) or not is_valid_email(sender) \
            or not isinstance(job_url, str) or not _URL_RE.fullmatch(job_url) \
            or (base_url and not job_url.startswith(base_url.rstrip("/") + "/")):
        log.warning("job email skipped: invalid recipient, sender, link or outcome")
        return False
    try:
        port = int(env.get("T2PW_SMTP_PORT", "") or 587)
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = sender, to_addr, SUBJECTS[outcome]
        msg.set_content(BODIES[outcome].format(link=job_url))
        starttls = env.get("T2PW_SMTP_STARTTLS", "1").strip().lower() not in ("0", "false", "no", "off")
        ctx = ssl.create_default_context()
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT_S, context=ctx)
        else:
            server = smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_S)
        with server:
            if port != 465 and starttls:
                server.starttls(context=ctx)
            if env.get("T2PW_SMTP_USER"):
                server.login(env["T2PW_SMTP_USER"], env.get("T2PW_SMTP_PASSWORD", ""))
            server.send_message(msg)
        return True
    except Exception as exc:  # never raise into the worker; the class name only, see above
        log.warning("job email failed: %s", type(exc).__name__)
        return False
