"""Engine cycles: log-only simulation, paper guard, halts, scale-in, reprice."""

import csv
import json
from datetime import timedelta

import pytest

from live_signals.broker import InMemoryBroker
from live_signals.config import config_with, default_config
from live_signals.engine import Engine
from live_signals.errors import AccountGuardError
from live_signals.journal import DecisionLog
from live_signals.models import MarketSnapshot, OrderRequest, PositionState, PriceBar, Quote, new_book
from live_signals.notify import MemoryNotifier
from live_signals.state import StateStore
from live_signals.tests.conftest import et, flat_bar, quote, signal, snapshot
from live_signals.__main__ import main, run_loop
from live_signals.sources import JsonFileMarketSource, JsonFileSignalSource

NOW = et(2026, 11, 9, 11, 0)
SECRET = "LIVE_SIGNALS_TEST_SECRET_DO_NOT_LEAK"


def _engine(tmp_path, mode="log_only", broker=None):
    cfg = default_config() if mode == "log_only" else config_with(mode="paper")
    state = StateStore(tmp_path / "book.db", cfg.reference_equity)
    journal = DecisionLog(tmp_path / "decisions.csv")
    notifier = MemoryNotifier()
    engine = Engine(
        cfg, state, journal, broker=broker, notifier=notifier,
        kill_file=str(tmp_path / "KILL"), reset_file=str(tmp_path / "RESET"),
    )
    return engine


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_log_only_enters_once_then_stops_out_and_never_calls_a_broker(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", SECRET)
    monkeypatch.setenv("ALPACA_API_KEY", SECRET + "KEY")
    broker = InMemoryBroker()
    engine = _engine(tmp_path, broker=broker)
    market = snapshot(atr=4.0)
    decisions = engine.run_once(NOW, [signal(now=NOW)], market)
    takes = [row for row in decisions if row.action == "take"]
    # Target is 13 shares; RTH sends 60% (8) and holds 5 for a scale-in.
    assert len(takes) == 1 and takes[0].qty == 8
    assert takes[0].details["target_qty"] == 13
    assert takes[0].details["extended_hours"] is False
    assert takes[0].details["order_type"] == "limit"
    assert isinstance(takes[0].qty, int)
    book = engine.state.load()
    assert book.positions["AAPL"].qty == 8
    assert book.positions["AAPL"].stop == pytest.approx(94.0)
    assert broker.calls == []

    again = engine.run_once(NOW + timedelta(minutes=10), [signal(now=NOW)], market)
    assert any(row.reason == "duplicate" for row in again)
    assert engine.state.load().positions["AAPL"].qty == 8

    stopped = snapshot(atr=4.0, q=quote(94), bar=PriceBar(94, 94, 94))
    exits = engine.run_once(NOW + timedelta(minutes=20), [], stopped)
    assert any(row.reason == "stop" and row.action == "exit" for row in exits)
    assert "AAPL" not in engine.state.load().positions
    assert engine.state.load().cash == pytest.approx(15_000 - 8 * 6)

    text = (tmp_path / "decisions.csv").read_text()
    blob = text + json.dumps(engine.notifier.events)
    assert SECRET not in blob
    assert broker.calls == []


def test_single_reddit_seventy_is_skipped(tmp_path):
    from live_signals.models import SignalSource
    engine = _engine(tmp_path)
    reddit = (SignalSource("https://www.reddit.com/r/stocks/googl", "reddit"),)
    sig = signal(ticker="GOOGL", score=70, sources=reddit, signal_id="googl", now=NOW)
    decisions = engine.run_once(NOW, [sig], snapshot(ticker="GOOGL"))
    assert decisions[0].action == "skip"
    assert decisions[0].reason == "single_social_unconfirmed"
    assert engine.state.load().positions == {}


def test_premarket_order_is_extended_hours_limit(tmp_path):
    engine = _engine(tmp_path)
    when = et(2026, 11, 9, 8, 0)
    decisions = engine.run_once(when, [signal(now=when)], snapshot(atr=4.0))
    take = next(row for row in decisions if row.action == "take")
    assert take.details["extended_hours"] is True
    assert take.details["time_in_force"] == "day"
    assert take.details["order_type"] == "limit"
    assert take.details["target_qty"] == 7  # 0.55 session multiplier, high vol
    assert take.qty == 4  # 60% of that target; the rest is the premarket scale-in clip


def test_overnight_is_one_clip(tmp_path):
    engine = _engine(tmp_path)
    when = et(2026, 11, 9, 21, 0)
    decisions = engine.run_once(when, [signal(now=when, score=90)], snapshot(atr=4.0))
    take = next(row for row in decisions if row.action == "take")
    assert take.details["extended_hours"] is True
    assert take.details["tranche2_qty"] == 0
    assert take.qty == 4


def test_scale_in_after_a_favorable_move(tmp_path):
    engine = _engine(tmp_path)
    first = engine.run_once(NOW, [signal(now=NOW)], snapshot(atr=1.2))
    take = next(row for row in first if row.action == "take")
    assert take.qty == 11 and take.details["target_qty"] == 18
    later = NOW + timedelta(minutes=10)
    moved = snapshot(price=100.42, atr=1.2)
    second = engine.run_once(later, [signal(now=NOW)], moved)
    assert any(row.reason == "scale_in" and row.qty == 7 for row in second)
    assert engine.state.load().positions["AAPL"].qty == 18


def test_kill_switch_flattens_and_blocks_entries(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    engine.run_once(NOW, [signal(now=NOW)], snapshot(atr=4.0))
    (tmp_path / "KILL").write_text("1")
    later = NOW + timedelta(minutes=10)
    decisions = engine.run_once(later, [signal(now=later, signal_id="later")], snapshot(atr=4.0))
    assert any(row.reason == "kill_switch" and row.action == "exit" for row in decisions)
    assert any(row.reason == "kill_switch" and row.action == "skip" for row in decisions)
    assert engine.state.load().positions == {}
    # Env flag, exact value 1, also halts a fresh book.
    monkeypatch.setenv("LIVE_SIGNALS_KILL", "1")
    fresh = _engine(tmp_path / "other")
    skipped = fresh.run_once(NOW, [signal(now=NOW)], snapshot())
    assert skipped[0].reason == "kill_switch"
    assert fresh.state.load().positions == {}


def test_daily_loss_halts_new_risk(tmp_path):
    engine = _engine(tmp_path)
    engine.run_once(NOW, [signal(now=NOW)], snapshot(atr=4.0))
    crushed = snapshot(atr=4.0, q=quote(70), bar=flat_bar(70))
    later = NOW + timedelta(minutes=30)
    halted = engine.run_once(later, [signal(now=later, signal_id="next")], crushed)
    assert any(row.reason == "daily_loss_halt" and row.action == "exit" for row in halted)
    assert any(row.reason == "daily_loss_halt" and row.action == "skip" for row in halted)
    assert engine.state.load().positions == {}
    assert engine.state.load().risk.day_halted
    # Same day, price recovered: still halted.
    recovered = engine.run_once(later + timedelta(minutes=10), [signal(now=later, signal_id="back")], snapshot())
    assert any(row.reason == "daily_loss_halt" for row in recovered)
    assert engine.state.load().positions == {}


def test_drawdown_breaker_and_reset(tmp_path):
    engine = _engine(tmp_path)
    book = new_book(15_000).evolve(cash=5_000, positions={
        "AAPL": PositionState(
            ticker="AAPL", side="long", qty=100, entry_price=100, entry_time=NOW,
            entry_score=90, signal_id="seed", initial_stop=90, stop=90, atr=4, r=10,
            watermark=100, trail_armed=False, tp1_done=False, tp2_done=False,
            target_qty=100, tranche2_qty=0, tranche2_deadline=None, tranche2_sent=True,
            adopted=False, entry_session="rth", mfe_r=0, basis_qty=100,
        ),
    })
    engine.state.save(book)
    # Equity = 5000 + 100 * 80 = 13000, drawdown from 15000 is more than 8%.
    market = snapshot(q=quote(80), bar=flat_bar(80))
    halted = engine.run_once(NOW, [signal(now=NOW, signal_id="new")], market)
    assert any(row.reason == "drawdown_halt" for row in halted)
    assert engine.state.load().risk.drawdown_halted
    assert "AAPL" not in engine.state.load().positions

    # A reset rebases the peak and lets the next cycle trade again.
    engine.state.save(engine.state.load().evolve(
        risk=engine.state.load().risk.evolve(drawdown_halted=True, peak_equity=15_000),
        cash=13_000, positions={},
    ))
    (tmp_path / "RESET").write_text("go")
    resumed = engine.run_once(NOW + timedelta(minutes=10), [signal(now=NOW, signal_id="after-reset")], snapshot())
    assert any(row.action == "take" for row in resumed)
    assert not engine.state.load().risk.drawdown_halted
    assert not (tmp_path / "RESET").exists()


def test_paper_submits_on_the_right_account_and_refuses_the_wrong_one(tmp_path):
    good = InMemoryBroker()
    engine = _engine(tmp_path, mode="paper", broker=good)
    engine.run_once(NOW, [signal(now=NOW)], snapshot(atr=4.0))
    assert any(call.startswith("submit:") for call in good.calls)
    assert good.positions["AAPL"].qty == 8
    submits = [call for call in good.calls if call.startswith("submit:")]
    engine.run_once(NOW + timedelta(minutes=10), [signal(now=NOW)], snapshot(atr=4.0))
    assert [call for call in good.calls if call.startswith("submit:")] == submits

    bad = InMemoryBroker(account_id="PA3V3A1X0IX6")
    guarded = _engine(tmp_path / "bad", mode="paper", broker=bad)
    with pytest.raises(AccountGuardError):
        guarded.run_once(NOW, [signal(now=NOW)], snapshot())
    assert bad.calls == []
    rows = _rows(tmp_path / "bad" / "decisions.csv")
    assert rows[0]["reason"] == "account_guard"


def test_reconcile_adopts_qty_and_cancels_only_our_orphans(tmp_path):
    broker = InMemoryBroker()
    from live_signals.broker import _MemOrder
    from live_signals.models import BrokerPosition, VolSnapshot
    broker.positions["AAPL"] = BrokerPosition("AAPL", 5, 100.0)
    broker.positions["MSFT"] = BrokerPosition("MSFT", 4, 100.0)
    # An order this engine did not create, and one that belongs to someone else.
    broker._orders["ww-orphan"] = _MemOrder(OrderRequest("ww-orphan", "MSFT", "buy", 1, 50.0, False), "open", None)
    broker._orders["rh-keep"] = _MemOrder(OrderRequest("rh-keep", "NVDA", "buy", 1, 50.0, False), "open", None)

    engine = _engine(tmp_path, mode="paper", broker=broker)
    local = new_book(15_000).evolve(positions={
        "MSFT": PositionState(
            ticker="MSFT", side="long", qty=10, entry_price=100, entry_time=NOW,
            entry_score=90, signal_id="msft", initial_stop=90, stop=90, atr=4, r=10,
            watermark=100, trail_armed=False, tp1_done=False, tp2_done=False,
            target_qty=10, tranche2_qty=0, tranche2_deadline=None, tranche2_sent=True,
            adopted=False, entry_session="rth", mfe_r=0, basis_qty=10,
        ),
    })
    engine.state.save(local)
    market = MarketSnapshot(
        quotes={"AAPL": quote(100), "MSFT": quote(100), "NVDA": quote(100)},
        vols={
            "AAPL": VolSnapshot(4, 100),
            "MSFT": VolSnapshot(4, 100),
        },
    )
    decisions = engine.run_once(NOW, [], market)
    assert any(row.reason == "adopted" and row.ticker == "AAPL" for row in decisions)
    assert any(row.reason == "qty_reconciled" and row.ticker == "MSFT" for row in decisions)
    book = engine.state.load()
    assert book.positions["AAPL"].adopted and book.positions["AAPL"].qty == 5
    assert book.positions["MSFT"].qty == 4
    assert any(call == "cancel:ww-orphan" for call in broker.calls)
    assert not any("rh-keep" in call for call in broker.calls)
    # Adopted names are not managed, so nothing is submitted for them.
    assert not any(call.startswith("submit:") for call in broker.calls)


def test_resting_entry_reprices_then_expires(tmp_path):
    engine = _engine(tmp_path)
    wide = snapshot(q=Quote(100.0, 100.20, 100.10))
    first = engine.run_once(NOW, [signal(now=NOW)], wide)
    assert any(row.action == "take" for row in first)
    book = engine.state.load()
    assert book.positions == {}
    assert len(book.working) == 1
    order = next(iter(book.working.values()))
    assert order.limit_price == pytest.approx(100.15, abs=0.02)

    later = NOW + timedelta(seconds=121)
    moved = snapshot(q=Quote(100.05, 100.16, 100.10))
    second = engine.run_once(later, [], moved)
    assert any(row.reason == "reprice" for row in second)
    repriced = next(iter(engine.state.load().working.values()))
    assert repriced.replaces == 1
    assert repriced.limit_price != order.limit_price

    # Ask stays above the repriced limit so the order does not fill before it expires.
    still_resting = snapshot(q=Quote(100.05, 100.25, 100.15))
    third = engine.run_once(later + timedelta(seconds=121), [], still_resting)
    assert any(row.reason == "entry_expired" for row in third)
    assert engine.state.load().working == {}
    assert engine.state.load().positions == {}


def test_cli_is_log_only(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    from datetime import datetime, timezone
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    signals = tmp_path / "signals.json"
    market = tmp_path / "market.json"
    signals.write_text(json.dumps({
        "signals": [{
            "ticker": "AAPL", "score": 90, "sentiment": "bullish",
            "headline": "h", "rationale": "r",
            "timestamp": fresh, "id": "cli-1",
            "sources": [
                {"url": "https://www.reuters.com/a", "type": "news"},
                {"url": "https://www.bloomberg.com/b", "type": "news"},
                {"url": "https://www.wsj.com/c", "type": "news"},
                {"url": "https://www.ft.com/d", "type": "news"},
            ],
        }],
    }))
    market.write_text(json.dumps({
        "quotes": {"AAPL": {"bid": 100, "ask": 100, "last": 100}},
        "vols": {"AAPL": {"atr": 4, "prev_rth_close": 100}},
    }))
    # 16:00 UTC is 11:00 ET in November.
    code = main([
        "--signals", str(signals), "--market", str(market), "--once",
        "--state", str(tmp_path / "state.db"), "--log", str(tmp_path / "log.csv"),
        "--kill-file", str(tmp_path / "KILL"), "--reset-file", str(tmp_path / "RESET"),
    ])
    assert code == 0
    rows = _rows(tmp_path / "log.csv")
    assert rows[0]["action"] == "take" and rows[0]["mode"] == "log_only"
    sleeps = []
    engine = _engine(tmp_path / "loop")
    run_loop(
        engine, JsonFileSignalSource(signals), JsonFileMarketSource(market),
        sleep=lambda seconds: sleeps.append(seconds),
        clock=lambda: NOW,
        max_cycles=2,
    )
    assert sleeps == [engine.config.poll_seconds]
