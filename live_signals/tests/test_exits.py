"""Stops, trailing, scale-out, time, clock, events, and signal exits."""

from datetime import timedelta

from live_signals.exits import evaluate_exit
from live_signals.models import PositionState, PriceBar, ScheduledEvent, SignalSource
from live_signals.tests.conftest import et, quote, signal

ENTRY = et(2026, 11, 9, 10, 0)


def _pos(**overrides) -> PositionState:
    data = dict(
        ticker="AAPL", side="long", qty=10, entry_price=100.0, entry_time=ENTRY,
        entry_score=90.0, signal_id="sig", initial_stop=94.0, stop=94.0, atr=4.0, r=6.0,
        watermark=100.0, trail_armed=False, tp1_done=False, tp2_done=False, target_qty=10,
        tranche2_qty=0, tranche2_deadline=None, tranche2_sent=True, adopted=False,
        entry_session="rth", mfe_r=0.0, basis_qty=10,
    )
    data.update(overrides)
    return PositionState(**data)


def _exit(cfg, pos, bar, now, signals=None, events=(), **kwargs):
    return evaluate_exit(
        pos, bar, now, signals or [], events, cfg,
        last_price=kwargs.pop("last", bar.close), **kwargs,
    )


def test_stop_and_short_stop(cfg):
    hit = _exit(cfg, _pos(), PriceBar(100, 94, 94), et(2026, 11, 9, 12, 0))
    assert hit.action == "exit" and hit.reason == "stop" and hit.qty == 10
    held = _exit(cfg, _pos(), PriceBar(101, 94.5, 100), et(2026, 11, 9, 12, 0))
    assert held.action == "hold"
    short = _pos(side="short", initial_stop=106, stop=106, watermark=100)
    cover = _exit(cfg, short, PriceBar(106, 99, 106), et(2026, 11, 9, 12, 0))
    assert cover.reason == "stop" and cover.qty == 10


def test_trail_arms_then_ratchets_and_never_loosens(cfg):
    # +1R is $6. The bar's low stays above the original stop, so this bar
    # arms the trail for the NEXT bar instead of stopping itself out.
    first = _exit(cfg, _pos(), PriceBar(106, 100, 105), et(2026, 11, 9, 12, 0))
    assert first.action == "hold" and first.position.trail_armed
    assert first.position.stop == 102  # watermark 106 minus 1 ATR
    second = _exit(cfg, first.position, PriceBar(104, 102, 103), et(2026, 11, 9, 12, 10))
    assert second.reason == "trail_stop" and second.qty == 10
    # A lower high does not pull the stop back down.
    quiet = _exit(cfg, first.position, PriceBar(104, 103, 103.5), et(2026, 11, 9, 12, 10))
    assert quiet.action == "hold" and quiet.position.stop == 102


def test_scale_out_levels(cfg):
    tp1 = _exit(cfg, _pos(), PriceBar(109, 100, 109), et(2026, 11, 9, 13, 0))
    assert tp1.reason == "take_profit" and tp1.qty == 4 and tp1.set_tp1 and not tp1.set_tp2
    both = _exit(cfg, _pos(), PriceBar(115, 100, 115), et(2026, 11, 9, 13, 0))
    assert both.qty == 7 and both.set_tp1 and both.set_tp2
    small = _exit(cfg, _pos(qty=2, basis_qty=2), PriceBar(112, 100, 112), et(2026, 11, 9, 13, 0))
    assert small.reason == "take_profit" and small.qty == 2


def test_time_stop_and_alpha_decay(cfg):
    faded = _exit(cfg, _pos(mfe_r=0.2), PriceBar(100, 99, 100), et(2026, 11, 9, 18, 0))
    assert faded.reason == "alpha_decay"
    worked = _exit(cfg, _pos(mfe_r=0.4), PriceBar(100, 99, 100), et(2026, 11, 9, 18, 0))
    assert worked.reason != "alpha_decay"
    old = _pos(entry_time=et(2026, 11, 11, 12, 0), entry_score=90, mfe_r=1.0)
    stopped = _exit(cfg, old, PriceBar(101, 100, 101), et(2026, 11, 13, 0, 30))
    assert stopped.reason == "time_stop"


def test_clock_flattens(cfg):
    # Entry the same morning, so a 36-hour time stop is not what these assert.
    thu = et(2026, 11, 12, 10, 0)
    fri = et(2026, 11, 13, 10, 0)
    assert _exit(cfg, _pos(entry_score=80, entry_time=thu), PriceBar(100, 99, 100), et(2026, 11, 12, 15, 50)).reason == "eod_flat"
    assert _exit(cfg, _pos(entry_score=85, entry_time=thu), PriceBar(100, 99, 100), et(2026, 11, 12, 15, 50)).action == "hold"
    assert _exit(cfg, _pos(entry_score=95, entry_time=fri), PriceBar(100, 99, 100), et(2026, 11, 13, 15, 50)).reason == "weekend_flat"
    pre = _pos(entry_score=80, entry_session="pre", entry_price=100)
    assert _exit(cfg, pre, PriceBar(99, 98, 99), et(2026, 11, 9, 9, 25)).reason == "preopen_flat"
    assert _exit(cfg, pre, PriceBar(101, 100.5, 101), et(2026, 11, 9, 9, 25)).action == "hold"
    assert _exit(cfg, _pos(entry_score=80), PriceBar(99, 98, 99), et(2026, 11, 9, 9, 25)).action == "hold"


def test_events(cfg):
    earnings = (ScheduledEvent("AAPL earnings", et(2026, 11, 9, 16, 0), "earnings", "AAPL"),)
    assert _exit(cfg, _pos(), PriceBar(100, 99, 100), et(2026, 11, 9, 13, 0), events=earnings).reason == "earnings_flat"
    assert _exit(cfg, _pos(), PriceBar(100, 99, 100), et(2026, 11, 9, 8, 0), events=earnings).action == "hold"
    cpi = (ScheduledEvent("CPI", et(2026, 11, 9, 10, 0), "cpi", None),)
    assert _exit(cfg, _pos(entry_score=80), PriceBar(100, 99, 100), et(2026, 11, 9, 9, 50), events=cpi).reason == "macro_flat"
    tight = _exit(cfg, _pos(entry_score=90), PriceBar(100, 99, 100), et(2026, 11, 9, 9, 50), events=cpi)
    assert tight.action == "hold" and tight.reason == "macro_tighten"
    assert tight.position.stop == 95.5  # entry - 0.75R


def test_signal_reversal_requires_a_qualified_opposite(cfg):
    now = et(2026, 11, 9, 12, 0)
    reddit = (SignalSource("https://www.reddit.com/r/stocks/x", "reddit"),)
    noise = signal(score=70, sentiment="bearish", sources=reddit, signal_id="reddit", now=now)
    assert _exit(cfg, _pos(entry_score=80), PriceBar(100, 99, 100), now, [noise]).action == "hold"
    real = signal(score=90, sentiment="bearish", signal_id="news", now=now)
    assert _exit(cfg, _pos(entry_score=80), PriceBar(100, 99, 100), now, [real]).reason == "signal_reversed"
    # Weaker than the position: not an exit.
    weaker = signal(score=75, sentiment="bearish", signal_id="weaker", now=now)
    assert _exit(cfg, _pos(entry_score=90), PriceBar(100, 99, 100), now, [weaker]).action == "hold"
    neutral = signal(score=75, sentiment="neutral", signal_id="flat", now=now)
    assert _exit(cfg, _pos(), PriceBar(100, 99, 100), now, [neutral]).reason == "signal_invalidated"
    stale = signal(score=90, sentiment="bearish", signal_id="old", now=now, minutes_ago=90)
    assert _exit(cfg, _pos(entry_score=70), PriceBar(100, 99, 100), now, [stale]).action == "hold"
    # keep the import used when a test wants a quote-shaped last
    assert quote(100).last == 100
    assert timedelta(minutes=1).total_seconds() == 60
