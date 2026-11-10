"""Entry pricing, session clock, and extended-hours order shape."""

from datetime import date, time, timedelta

import pytest

from live_signals.models import OrderRequest, Quote, VolSnapshot
from live_signals.pricing import market_skip_reason, price_entry, review_resting_entry, split_tranches
from live_signals.sessions import (
    classify_session,
    must_flatten_for_clock,
    session_hours_between,
)
from live_signals.tests.conftest import et


def test_session_clock():
    holidays = {date(2026, 11, 16)}
    assert classify_session(et(2026, 11, 9, 11, 0)) == "rth"
    assert classify_session(et(2026, 11, 9, 8, 0)) == "pre"
    assert classify_session(et(2026, 11, 9, 16, 30)) == "after"
    assert classify_session(et(2026, 11, 9, 21, 0)) == "overnight"
    assert classify_session(et(2026, 11, 10, 2, 0)) == "overnight"
    assert classify_session(et(2026, 11, 8, 12, 0)) == "closed"  # Sunday afternoon
    assert classify_session(et(2026, 11, 8, 21, 0)) == "overnight"  # Sunday night, Monday open
    assert classify_session(et(2026, 11, 13, 21, 0)) == "closed"  # Friday night
    assert classify_session(et(2026, 11, 14, 12, 0)) == "closed"
    assert classify_session(et(2026, 11, 15, 21, 0), holidays) == "closed"  # Monday holiday
    early = {date(2026, 11, 12): time(13, 0)}
    assert classify_session(et(2026, 11, 12, 12, 0), early_closes=early) == "rth"
    assert classify_session(et(2026, 11, 12, 13, 30), early_closes=early) == "after"
    assert classify_session(et(2026, 11, 12, 17, 30), early_closes=early) == "closed"
    assert classify_session(et(2026, 11, 12, 21, 0), early_closes=early) == "overnight"


def test_session_hours_skip_the_weekend():
    start = et(2026, 11, 9, 10, 0)
    assert session_hours_between(start, et(2026, 11, 9, 12, 0)) == pytest.approx(2.0)
    assert session_hours_between(start, et(2026, 11, 9, 18, 0)) == pytest.approx(8.0)


def test_hold_policy():
    # Thursday 15:50, score under 85, flatten. Score 85 may stay if Friday is a normal day.
    assert must_flatten_for_clock(
        et(2026, 11, 12, 15, 50), 80, overnight_min_score=85, minutes_before_close=10,
    ) == "eod_flat"
    assert must_flatten_for_clock(
        et(2026, 11, 12, 15, 49), 80, overnight_min_score=85, minutes_before_close=10,
    ) is None
    assert must_flatten_for_clock(
        et(2026, 11, 12, 15, 50), 85, overnight_min_score=85, minutes_before_close=10,
    ) is None
    assert must_flatten_for_clock(
        et(2026, 11, 13, 15, 50), 95, overnight_min_score=85, minutes_before_close=10,
    ) == "weekend_flat"
    assert must_flatten_for_clock(
        et(2026, 11, 12, 15, 50), 95, overnight_min_score=85, minutes_before_close=10,
        holidays={date(2026, 11, 13)},
    ) == "eod_flat"
    # Friday 00:30 is still Thursday night's overnight, not the weekend flatten.
    assert must_flatten_for_clock(
        et(2026, 11, 13, 0, 30), 90, overnight_min_score=85, minutes_before_close=10,
    ) is None


def test_limit_price_respects_spread_and_chase(cfg):
    tight = Quote(100.0, 100.0, 100.0)
    rth = price_entry("buy", tight, "rth", cfg)
    assert rth.ok and rth.extended_hours is False and rth.time_in_force == "day"
    assert rth.order_type == "limit"
    pre = price_entry("buy", tight, "pre", cfg)
    after = price_entry("sell", tight, "after", cfg)
    overnight = price_entry("buy", tight, "overnight", cfg)
    assert pre.extended_hours and after.extended_hours and overnight.extended_hours
    assert price_entry("buy", tight, "closed", cfg).reason == "session_closed"

    wide = Quote(100.0, 100.50, 100.25)
    assert price_entry("buy", wide, "rth", cfg).reason == "spread_too_wide"

    # Wide enough to rest, tight enough to be legal. Chase is capped at the original mid.
    book = Quote(100.0, 100.20, 100.10)
    priced = price_entry("buy", book, "rth", cfg)
    assert priced.ok
    assert priced.limit_price == pytest.approx(100.15, abs=0.02)
    assert priced.limit_price <= 100.10 * (1 + cfg.max_chase["rth"]) + 0.01
    later = Quote(100.40, 100.55, 100.50)
    capped = price_entry("buy", later, "rth", cfg, decision_mid=priced.mid)
    assert capped.limit_price <= priced.mid * (1 + cfg.max_chase["rth"]) + 0.011


def test_skip_halt_gap_and_borrow(cfg):
    halted = Quote(100, 100.05, 100, halted=True)
    assert market_skip_reason("long", halted, VolSnapshot(2, 100), cfg) == "halted"
    unborrowable = Quote(100, 100.05, 100, shortable=False, easy_to_borrow=False)
    assert market_skip_reason("short", unborrowable, VolSnapshot(2, 100), cfg) == "not_shortable"
    # 1.25 * ATR of $2 is $2.50. A $3 extension is too far; $2 is still allowed.
    assert market_skip_reason("long", Quote(100, 100.05, 103), VolSnapshot(2, 100), cfg) == "gap_already_extended"
    assert market_skip_reason("long", Quote(100, 100.05, 102), VolSnapshot(2, 100), cfg) is None


def test_resting_order_reprices_once_then_cancels(cfg):
    quote = Quote(100.05, 100.16, 100.10)
    assert review_resting_entry(
        age_seconds=10, replaces=0, order_session="rth", current_session="rth",
        side="buy", quote=quote, decision_mid=100.10, current_limit=100.15, cfg=cfg,
    ) == "hold"
    action = review_resting_entry(
        age_seconds=121, replaces=0, order_session="rth", current_session="rth",
        side="buy", quote=quote, decision_mid=100.10, current_limit=100.15, cfg=cfg,
    )
    assert isinstance(action, float) and action != pytest.approx(100.15)
    assert review_resting_entry(
        age_seconds=121, replaces=1, order_session="rth", current_session="rth",
        side="buy", quote=quote, decision_mid=100.10, current_limit=action, cfg=cfg,
    ) == "cancel"
    assert review_resting_entry(
        age_seconds=10, replaces=0, order_session="pre", current_session="rth",
        side="buy", quote=quote, decision_mid=100.10, current_limit=100.15, cfg=cfg,
    ) == "cancel"


def test_scale_in_only_in_rth_and_premarket(cfg):
    assert split_tranches(18, "rth", cfg) == (11, 7)
    assert split_tranches(18, "overnight", cfg) == (18, 0)
    assert split_tranches(1, "rth", cfg) == (1, 0)


def test_order_request_rejects_market_and_fractional_qty():
    with pytest.raises(ValueError):
        OrderRequest("ww-1", "AAPL", "buy", 1, 100.0, False, order_type="market")
    with pytest.raises(ValueError):
        OrderRequest("ww-1", "AAPL", "buy", 1.5, 100.0, True)  # type: ignore[arg-type]


def test_whole_share_tranches_are_ints(cfg):
    first, second = split_tranches(18, "rth", cfg)
    assert isinstance(first, int) and isinstance(second, int)
    assert first + second == 18
