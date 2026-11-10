"""Shared builders for live_signals tests. No network."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from live_signals.config import default_config
from live_signals.models import (
    MarketSnapshot,
    NewsSignal,
    PriceBar,
    Quote,
    SignalSource,
    VolSnapshot,
)

ET = ZoneInfo("America/New_York")


def et(year, month, day, hour, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET)


@pytest.fixture
def cfg():
    return default_config()


def quote(price=100.0, **overrides) -> Quote:
    data = {"bid": price, "ask": price, "last": price, "halted": False,
            "shortable": True, "easy_to_borrow": True}
    data.update(overrides)
    return Quote(**data)


def four_news() -> tuple[SignalSource, ...]:
    return (
        SignalSource("https://www.reuters.com/a", "news"),
        SignalSource("https://www.bloomberg.com/b", "news"),
        SignalSource("https://www.wsj.com/c", "news"),
        SignalSource("https://www.ft.com/d", "news"),
    )


def signal(
    ticker="AAPL",
    score=90,
    sentiment="bullish",
    sources=None,
    minutes_ago=5,
    signal_id="sig-1",
    now=None,
    headline="headline",
) -> NewsSignal:
    now = now or et(2026, 11, 9, 11, 0)
    return NewsSignal(
        ticker=ticker,
        score=score,
        sentiment=sentiment,
        sources=four_news() if sources is None else sources,
        headline=headline,
        rationale="because",
        timestamp=now - timedelta(minutes=minutes_ago),
        signal_id=signal_id,
    )


def snapshot(ticker="AAPL", price=100.0, atr=4.0, sector=None, bar=None, **extra) -> MarketSnapshot:
    vol = VolSnapshot(atr=atr, prev_rth_close=extra.pop("prev_close", price), sector=sector)
    data = dict(
        quotes={ticker: extra.pop("q", quote(price))},
        vols={ticker: vol},
        bars={} if bar is None else {ticker: bar},
    )
    data.update(extra)
    return MarketSnapshot(**data)


def flat_bar(price) -> PriceBar:
    return PriceBar(price, price, price)
