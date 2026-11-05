"""W8 notify: job-finished email.

NEW ACCEPTANCE TEST (W8, new capability): the module is new; no network, no real email.
"""

import logging

import pytest

from t2pw.web import notify

URL = "https://t2pw.example.org/jobs/abc123"
ENV = {"T2PW_SMTP_HOST": "smtp.example.org", "T2PW_SMTP_FROM": "t2pw@example.org",
       "T2PW_SMTP_USER": "mailuser", "T2PW_SMTP_PASSWORD": "hunter2-SECRET"}


class FakeSMTP:
    instances = []
    fail = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout, self.ctx = host, port, timeout, context
        self.calls, self.sent = [], []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, pw):
        self.calls.append(("login", user, pw))

    def send_message(self, msg):
        if FakeSMTP.fail:
            raise FakeSMTP.fail
        self.sent.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances, FakeSMTP.fail = [], None
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notify.smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def test_unset_smtp_is_a_logged_noop_and_never_raises(smtp, caplog):
    caplog.set_level(logging.DEBUG)
    assert notify.notify_job("user@example.org", URL, env={}) is False
    assert notify.notify_job("user@example.org", URL, env={"T2PW_SMTP_HOST": "h"}) is False  # no from
    assert smtp.instances == []
    assert "not configured" in caplog.text and "user@example.org" not in caplog.text
    assert notify.notify_job(None, URL, env=ENV) is False and smtp.instances == []   # no address given


def test_message_is_built_correctly_with_timeout_starttls_and_login(smtp):
    assert notify.notify_job("User.Name+tag@Example.org", URL, env=ENV) is True
    (s,) = smtp.instances
    assert (s.host, s.port, s.timeout) == ("smtp.example.org", 587, notify.SMTP_TIMEOUT_S)
    assert 0 < s.timeout <= 30
    assert s.calls == ["starttls", ("login", "mailuser", "hunter2-SECRET")]
    (m,) = s.sent
    assert (m["To"], m["From"]) == ("User.Name+tag@Example.org", "t2pw@example.org")
    assert m["Subject"] == notify.SUBJECTS["finished"]
    assert m.get_content() == notify.BODIES["finished"].format(link=URL)


def test_failed_outcome_port_465_and_no_starttls(smtp):
    assert notify.notify_job("a@example.org", URL, outcome="failed",
                             env={**ENV, "T2PW_SMTP_PORT": "465"})
    s = smtp.instances[-1]
    assert s.port == 465 and "starttls" not in s.calls
    assert s.sent[0]["Subject"] == notify.SUBJECTS["failed"]
    assert notify.notify_job("a@example.org", URL,
                             env={**ENV, "T2PW_SMTP_STARTTLS": "0", "T2PW_SMTP_USER": ""})
    assert smtp.instances[-1].calls == []                    # no TLS, no login: as configured
    assert notify.notify_job("a@example.org", URL, outcome="bogus", env=ENV) is False


@pytest.mark.parametrize("addr", [
    "a@example.org\r\nBcc: evil@example.org", "a@example.org\nSubject: x", "a b@example.org",
    "a@example.org, b@example.org", "<a@example.org>", "no-at-sign", "a@localhost", "@example.org",
    "a@@example.org", ".a@example.org", "a..b@example.org", "a@-example.org",
    "x" * 300 + "@example.org", "a@example.org\x00", "é@example.org"])
def test_invalid_addresses_are_rejected_before_any_connection(smtp, addr):
    assert notify.is_valid_email(addr) is False
    assert notify.notify_job(addr, URL, env=ENV) is False
    assert smtp.instances == []


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "https://x.org/a b", "https://x.org/\r\nBcc: e@x.org", "ftp://x.org/j",
    "https://x.org/" + "a" * 600, "", "https://x.org/j\n"])
def test_invalid_links_are_rejected(smtp, url):
    assert notify.notify_job("a@example.org", url, env=ENV) is False and smtp.instances == []


def test_only_fixed_text_and_the_link_reach_headers_and_body(smtp):
    link = URL + "?x=1&y=%0d%0aBcc:e"
    assert notify.notify_job("a@example.org", link, env=ENV)
    (m,) = smtp.instances[0].sent
    assert set(m.keys()) == {"From", "To", "Subject", "Content-Type", "MIME-Version",
                             "Content-Transfer-Encoding"}
    assert m.get_content() == notify.BODIES["finished"].format(link=link)


def test_send_failures_return_false_and_log_nothing_secret(smtp, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG)
    smtp.fail = RuntimeError("550 rejected user@example.org password hunter2-SECRET /srv/t2pw/x")
    assert notify.notify_job("user@example.org", URL, env=ENV) is False
    smtp.fail = None
    for bad in (TimeoutError("timed out smtp.example.org"), OSError("refused user@example.org")):
        def boom(*a, _bad=bad, **k):
            raise _bad
        monkeypatch.setattr(notify.smtplib, "SMTP", boom)
        assert notify.notify_job("user@example.org", URL, env=ENV) is False
    assert "RuntimeError" in caplog.text and "TimeoutError" in caplog.text
    for secret in ("user@example.org", "hunter2-SECRET", "mailuser", "/srv/t2pw", "smtp.example.org"):
        assert secret not in caplog.text


def test_success_and_skip_logs_hold_no_secrets(smtp, caplog):
    caplog.set_level(logging.DEBUG)
    notify.notify_job("user@example.org", URL, env=ENV)
    notify.notify_job("bad\naddr", URL, env=ENV)
    notify.notify_job("user@example.org", "javascript:x", env=ENV)
    for secret in ("user@example.org", "hunter2-SECRET", "mailuser", "bad"):
        assert secret not in caplog.text


def test_default_env_is_os_environ(smtp, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    assert notify.notify_job("a@example.org", URL) is True
    monkeypatch.delenv("T2PW_SMTP_HOST")
    assert notify.notify_job("a@example.org", URL) is False


# ======================= review round 1 (F4, F5) =======================
def test_f4_base_url_pins_the_link_to_our_own_site(smtp):
    assert notify.notify_job("a@example.org", URL, base_url="https://t2pw.example.org", env=ENV)
    assert notify.notify_job("a@example.org", URL, base_url="https://t2pw.example.org/", env=ENV)
    n = len(smtp.instances)
    for base in ("https://evil.example.net", "https://t2pw.example.org.evil.net",
                 "http://t2pw.example.org"):
        assert notify.notify_job("a@example.org", URL, base_url=base, env=ENV) is False
    assert notify.notify_job("a@example.org", "https://t2pw.example.org.evil.net/x",
                             base_url="https://t2pw.example.org", env=ENV) is False
    assert len(smtp.instances) == n
    assert notify.notify_job("a@example.org", URL, env=ENV)          # base_url is optional


def test_f5_starttls_failure_sends_nothing_and_never_logs_in(smtp, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    def bad_starttls(self, context=None):
        raise notify.smtplib.SMTPException("STARTTLS refused for user@example.org")

    monkeypatch.setattr(FakeSMTP, "starttls", bad_starttls)
    assert notify.notify_job("user@example.org", URL, env=ENV) is False
    (s,) = smtp.instances
    assert s.calls == [] and s.sent == []                  # no login call, nothing sent
    assert "SMTPException" in caplog.text
    for secret in ("user@example.org", "hunter2-SECRET", "mailuser"):
        assert secret not in caplog.text
