"""D-W19a: T2PW_FINAL_PAYLOAD is on unless explicitly 0/false/no/off; T2PW_TRANSPORT_SIDES stays default off."""
import pytest

from t2pw.service.final_payload import final_payload_enabled


def test_unset_means_on(monkeypatch):
    monkeypatch.delenv("T2PW_FINAL_PAYLOAD")
    assert final_payload_enabled() is True


@pytest.mark.parametrize("v", ["", "   ", "1", "true", "YES", " True ", "on", "anything"])
def test_on_values(monkeypatch, v):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", v)
    assert final_payload_enabled() is True


@pytest.mark.parametrize("v", ["0", "false", "FALSE", "No", "off", "OFF", " 0 ", "  Off\t"])
def test_off_values(monkeypatch, v):
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", v)
    assert final_payload_enabled() is False


def test_the_suite_default_pin_is_off():
    # tests/conftest.py pins =0 so flag-off byte-identity tests keep running the old path
    assert final_payload_enabled() is False
