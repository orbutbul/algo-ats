"""Market-data parsers. Transports are fakes; nothing here opens a socket."""

from datetime import date, datetime, timezone

import pytest

from live_signals.errors import BrokerError
from live_signals.marketdata import (
    AlpacaMarketClient,
    average_true_range,
    calendar_from_rows,
    enrich_market,
    load_market_snapshot,
    snapshot_from_alpaca,
    snapshot_from_yahoo,
)
from live_signals.models import MarketSnapshot, Quote, VolSnapshot
from live_signals.sectors import sector_of


def _bars(n=20, start=100.0):
    rows = []
    for index in range(n):
        close = start + index
        rows.append({"h": close + 1, "l": close - 1, "c": close, "v": 1000 + index * 10})
    return rows


def test_atr_and_yahoo_blocks_shorts_without_a_locate():
    bars = _bars()
    highs = [row["h"] for row in bars]
    lows = [row["l"] for row in bars]
    closes = [row["c"] for row in bars]
    assert average_true_range(highs, lows, closes) > 0
    snap = snapshot_from_yahoo({"AAPL": [
        {"high": row["h"], "low": row["l"], "close": row["c"], "volume": row["v"]} for row in bars
    ]})
    assert snap.quotes["AAPL"].last == bars[-1]["c"]
    assert snap.quotes["AAPL"].easy_to_borrow is False
    assert snap.vols["AAPL"].atr > 0
    assert snap.confirmations["AAPL"].rel_volume > 0


def test_alpaca_payload_builds_quote_atr_and_calendar():
    snaps = {
        "AAPL": {
            "latestQuote": {"bp": 100.0, "ap": 100.10},
            "latestTrade": {"p": 100.05},
            "prevDailyBar": {"c": 99.0, "v": 1000},
        }
    }
    bars = {"AAPL": _bars()}
    assets = {"AAPL": {"shortable": True, "easy_to_borrow": True, "status": "active"}}
    calendar = [
        {"date": "2026-11-09", "open": "09:30", "close": "16:00"},
        {"date": "2026-11-10", "open": "09:30", "close": "16:00"},
        {"date": "2026-11-12", "open": "09:30", "close": "13:00"},
    ]
    # 2026-11-11 is a Wednesday missing from the calendar, so it is a holiday.
    snap = snapshot_from_alpaca(snaps, bars, assets, calendar, today=date(2026, 11, 9))
    assert snap.quotes["AAPL"].bid == 100
    assert snap.quotes["AAPL"].easy_to_borrow is True
    assert snap.vols["AAPL"].prev_rth_close == 99
    holidays, early = calendar_from_rows(calendar, date(2026, 11, 9), date(2026, 11, 13))
    assert date(2026, 11, 11) in holidays
    assert early[date(2026, 11, 12)].hour == 13
    assert date(2026, 11, 11) in snap.holidays


def test_alpaca_failure_falls_back_to_yahoo_and_scrubs_secrets():
    secret = "MARKETSECRETVALUE"

    class Boom:
        def snapshot(self, tickers, now):
            raise BrokerError(f"401 {secret}")

    def yahoo(ticker):
        return [{"high": 11, "low": 9, "close": 10, "volume": 100},
                {"high": 12, "low": 10, "close": 11, "volume": 200}]

    now = datetime(2026, 11, 9, 16, 0, tzinfo=timezone.utc)
    snap, source = load_market_snapshot(["AAPL"], now, alpaca=Boom(), yahoo_fetch=yahoo)
    assert source == "yahoo"
    assert snap.quotes["AAPL"].last == 11

    def transport(method, url, body):
        raise RuntimeError(f"nope {secret}")

    client = AlpacaMarketClient(api_key="KEY123456", api_secret=secret, transport=transport)
    with pytest.raises(BrokerError) as caught:
        client.snapshot(["AAPL"], now)
    assert secret not in str(caught.value)
    assert "KEY123456" not in str(caught.value)


def test_enrich_does_not_fetch_when_the_file_is_complete():
    def boom(tickers, now):
        raise AssertionError("fetch")

    market = MarketSnapshot(
        quotes={"AAPL": Quote(100, 100, 100)},
        vols={"AAPL": VolSnapshot(4, 100)},
    )
    now = datetime(2026, 11, 9, 16, 0, tzinfo=timezone.utc)
    assert enrich_market(market, ["AAPL"], now, loader=boom) is market


def test_static_sector_map():
    assert sector_of("aapl") == "Technology"
    assert sector_of("JPM") == "Financials"
    assert sector_of("ZZZZ") is None
