"""Throttle spacing (fake clock), retries, User-Agent, cache hits and negative caching. No network."""

import json

import pytest

from t2pw.clean_model.fill.cache import Cache, CachedFetcher, request_key
from t2pw.clean_model.fill.http import USER_AGENT, FetchError, Throttle, UrllibFetcher


def test_user_agent_is_generic_without_email():
    assert "@" not in USER_AGENT and USER_AGENT.startswith("t2pw-clean-model/")


def test_throttle_spaces_requests_per_host():
    t = [100.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    th = Throttle(0.34, clock=lambda: t[0], sleep=sleep)
    assert th.wait("a") == 0.0
    assert th.wait("b") == 0.0  # a different host is independent
    t[0] += 0.1
    assert th.wait("a") == pytest.approx(0.24)
    t[0] += 1.0
    assert th.wait("a") == 0.0
    assert len(slept) == 1
    stamps = []  # ten back-to-back requests to one host never exceed 3 per second
    th2 = Throttle(0.34, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    for _ in range(10):
        th2.wait("h")
        stamps.append(t[0])
    assert all(b - a >= 0.34 - 1e-9 for a, b in zip(stamps, stamps[1:]))


def _fetcher(responses, sleeps):
    it = iter(responses)
    seen = []

    def opener(url, headers, timeout):
        seen.append((url, headers))
        return next(it)

    return UrllibFetcher(Throttle(0, sleep=lambda s: None), retries=3, backoff=1.0, opener=opener,
                         sleep=sleeps.append), seen


def test_retry_with_backoff_on_429_and_5xx_then_success():
    sleeps = []
    f, seen = _fetcher([(429, {}, b""), (503, {}, b""), (200, {}, b'{"a": 1}')], sleeps)
    assert f.get_json("https://x.org/p", {"q": "1"}) == {"a": 1}
    assert sleeps == [1.0, 2.0]
    assert seen[0][0] == "https://x.org/p?q=1" and seen[0][1]["User-Agent"] == USER_AGENT


def test_retry_after_header_is_honoured_and_failure_raises():
    sleeps = []
    f, _ = _fetcher([(429, {"Retry-After": "5"}, b"")] + [(500, {}, b"")] * 3, sleeps)
    with pytest.raises(FetchError):
        f.get_json("https://x.org/p")
    assert sleeps[0] == 5.0


def test_404_is_none_and_other_4xx_raises():
    f, _ = _fetcher([(404, {}, b"")], [])
    assert f.get_json("https://x.org/p") is None
    f, _ = _fetcher([(400, {}, b"")], [])
    with pytest.raises(FetchError):
        f.get_text("https://x.org/p")


class _Counting:
    def __init__(self, value):
        self.value, self.n = value, 0

    def get_json(self, url, params=None):
        self.n += 1
        return self.value

    get_text = get_json


def test_cache_hit_and_negative_cache(tmp_path):
    inner = _Counting({"k": 1})
    cf = CachedFetcher(inner, Cache(str(tmp_path / "c")), "2026-11-10")
    assert cf.get_json("https://x.org/a", {"p": 1}) == {"k": 1}
    assert cf.get_json("https://x.org/a", {"p": 1}) == {"k": 1}
    assert (inner.n, cf.requests, cf.hits) == (1, 1, 1)
    cf2 = CachedFetcher(_Counting(None), Cache(str(tmp_path / "c")), "2026-11-10")  # new object, same folder: disk
    assert cf2.get_json("https://x.org/a", {"p": 1}) == {"k": 1} and cf2.requests == 0
    neg = _Counting(None)
    cf3 = CachedFetcher(neg, Cache(str(tmp_path / "c")), "2026-11-10")
    assert cf3.get_json("https://x.org/missing") is None and cf3.get_json("https://x.org/missing") is None
    assert neg.n == 1 and cf3.hits == 1


def test_cache_entry_stores_retrieval_date_and_kinds_have_distinct_keys(tmp_path):
    CachedFetcher(_Counting("t"), Cache(str(tmp_path)), "2026-11-10").get_text("https://x.org/a")
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text(encoding="utf-8"))["retrieved"] == "2026-11-10"
    assert request_key("json", "u", None) != request_key("text", "u", None)


def test_default_cache_folder_is_not_the_data_caches():
    from t2pw.clean_model.fill.cache import DEFAULT_DIR
    d = DEFAULT_DIR.replace("\\", "/")
    assert d.endswith("clean_model_cache") and "/data" not in d


def test_200_with_non_json_or_non_utf8_body_raises_fetcherror():
    f, _ = _fetcher([(200, {}, b"<html>oops</html>")], [])
    with pytest.raises(FetchError):
        f.get_json("https://x.org/p")
    f, _ = _fetcher([(200, {}, b"\xff\xfe\x00")], [])
    with pytest.raises(FetchError):
        f.get_text("https://x.org/p")
