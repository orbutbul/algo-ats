"""Risk flags, account guard, idempotent ids, and the Alpaca payload."""

import json

import pytest

from live_signals.alpaca_broker import AlpacaPaperBroker, PAPER_BASE_URL, order_payload
from live_signals.broker import PAPER_ACCOUNT_ID, InMemoryBroker, assert_submission_allowed
from live_signals.config import load_config_from_mapping
from live_signals.errors import AccountGuardError, BrokerError, ConfigError, LiveTradingDisabled
from live_signals.ids import client_order_id
from live_signals.models import OrderRequest, Quote, new_book
from live_signals.risk import kill_switch_active, step_risk
from live_signals.state import StateStore, dump_book, load_book
from live_signals.tests.conftest import et


def test_daily_loss_and_drawdown_and_reset(cfg):
    start = new_book(15_000).risk
    stepped = step_risk(start, 15_000, et(2026, 11, 9, 11, 0), cfg, reset_requested=False)
    losing = step_risk(stepped, 14_700, et(2026, 11, 9, 12, 0), cfg, reset_requested=False)
    assert losing.day_halted
    # Next calendar day clears the daily halt, not a same-day bounce.
    bounce = step_risk(losing, 15_000, et(2026, 11, 9, 13, 0), cfg, reset_requested=False)
    assert bounce.day_halted
    nxt = step_risk(losing, 15_000, et(2026, 11, 10, 10, 0), cfg, reset_requested=False)
    assert not nxt.day_halted

    deep = step_risk(stepped, 13_000, et(2026, 11, 9, 12, 0), cfg, reset_requested=False)
    assert deep.drawdown_halted
    still = step_risk(deep, 13_200, et(2026, 11, 9, 13, 0), cfg, reset_requested=False)
    assert still.drawdown_halted
    cleared = step_risk(deep, 13_000, et(2026, 11, 9, 13, 0), cfg, reset_requested=True)
    assert not cleared.drawdown_halted
    assert cleared.peak_equity == 13_000


def test_kill_switch_file_and_env(cfg, tmp_path, monkeypatch):
    path = tmp_path / "KILL"
    assert not kill_switch_active(cfg, str(path), env={})
    path.write_text("stop")
    assert kill_switch_active(cfg, str(path), env={})
    monkeypatch.delenv(cfg.kill_env, raising=False)
    assert kill_switch_active(cfg, str(tmp_path / "absent")) is False
    monkeypatch.setenv(cfg.kill_env, "1")
    assert kill_switch_active(cfg, str(tmp_path / "absent")) is True
    monkeypatch.setenv(cfg.kill_env, "true")
    assert kill_switch_active(cfg, str(tmp_path / "absent")) is False


def test_account_guard_blocks_the_other_paper_account_and_live_mode():
    assert PAPER_ACCOUNT_ID == "PA32ZMSEI"
    assert_submission_allowed("paper", "PA32ZMSEI", "paper")
    with pytest.raises(AccountGuardError):
        assert_submission_allowed("paper", "PA3V3A1X0IX6", "paper")
    with pytest.raises(AccountGuardError):
        assert_submission_allowed("paper", "PA32ZMSEI", "live")
    with pytest.raises(LiveTradingDisabled):
        assert_submission_allowed("live", "PA32ZMSEI", "paper")
    with pytest.raises(LiveTradingDisabled):
        assert_submission_allowed("log_only", "PA32ZMSEI", "paper")


def test_config_rejects_live_mode():
    from pathlib import Path
    raw = json.loads(Path("live_signals/default_config.json").read_text())
    raw["mode"] = "live"
    with pytest.raises(ConfigError):
        load_config_from_mapping(raw)


def test_client_order_id_is_stable_and_distinct():
    first = client_order_id("ww", "entry", "aapl", "sig", 1, 0)
    assert first == client_order_id("ww", "entry", "AAPL", "sig", 1, 0)
    assert first != client_order_id("ww", "entry", "AAPL", "sig", 2, 0)
    assert first.startswith("ww-")
    assert len(first) <= 48


def test_in_memory_fill_is_idempotent():
    broker = InMemoryBroker()
    broker.set_quotes({"AAPL": Quote(100, 100, 100)})
    order = OrderRequest("ww-abc", "AAPL", "buy", 5, 100.0, False)
    first = broker.submit_order(order)
    second = broker.submit_order(order)
    assert first.status == "filled" and second.status == "duplicate"
    assert broker.positions["AAPL"].qty == 5
    assert broker.get_account().equity == pytest.approx(15_000)


def test_alpaca_payload_is_limit_only_and_secrets_are_scrubbed():
    order = OrderRequest("ww-abc", "AAPL", "buy", 3, 100.5, True)
    payload = order_payload(order)
    assert payload["type"] == "limit"
    assert payload["qty"] == "3"
    assert payload["time_in_force"] == "day"
    assert payload["extended_hours"] is True
    assert "notional" not in payload
    assert "trail_price" not in payload
    assert "order_class" not in payload

    secret = "SUPERSECRETKEYVALUE"
    key = "AKIATESTKEYVALUE123"
    def transport(method, url, body):
        raise RuntimeError(f"401 {secret} header {key} {url}")
    broker = AlpacaPaperBroker(api_key=key, api_secret=secret, transport=transport)
    with pytest.raises(BrokerError) as caught:
        broker.get_account()
    assert secret not in str(caught.value)
    assert key not in str(caught.value)
    assert secret not in repr(broker)
    with pytest.raises(BrokerError):
        AlpacaPaperBroker(api_key=key, api_secret=secret, transport=transport, base_url="https://api.alpaca.markets")
    assert PAPER_BASE_URL == "https://paper-api.alpaca.markets"


def test_alpaca_refuses_the_wrong_account_before_posting():
    posted = []
    def transport(method, url, body):
        if url.endswith("/v2/account"):
            return {"account_number": "PA3V3A1X0IX6", "equity": "15000", "cash": "15000"}
        posted.append(url)
        return {}
    broker = AlpacaPaperBroker(api_key="k", api_secret="s", transport=transport)
    order = OrderRequest("ww-abc", "AAPL", "buy", 1, 100.0, False)
    with pytest.raises(AccountGuardError):
        broker.submit_order(order)
    assert posted == []


def test_state_roundtrip(tmp_path):
    store = StateStore(tmp_path / "book.db", 15_000)
    book = new_book(15_000)
    store.save(book)
    assert store.load().cash == 15_000
    again = load_book(dump_book(book))
    assert again.risk.peak_equity == 15_000
    store.close()
