"""Quotes, ATR, borrow flags, and the session calendar.

Execution role: build a `MarketSnapshot` without the engine calling the
network itself. Alpaca paper keys are tried first (data API for quotes and
daily bars, paper trading API for shortable/easy-to-borrow and the market
calendar). If that call fails, daily bars from Yahoo Finance are used for
last price and ATR. Yahoo has no NBBO and no borrow flag, so the fallback
quote is the last trade printed on both sides of the spread and shorts are
treated as not easy to borrow.

Alpaca's corporate-actions feed is splits, dividends, and mergers. It does
not publish earnings timestamps or CPI/FOMC. Those stay on the injected
event list. Tests pass a transport or a bar list and never open a socket.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta

from live_signals.alpaca_broker import PAPER_BASE_URL, scrub_text
from live_signals.errors import BrokerError
from live_signals.models import Confirmation, MarketSnapshot, Quote, VolSnapshot

DATA_BASE = "https://data.alpaca.markets"
_ATR_PERIOD = 14


def average_true_range(highs: list[float], lows: list[float], closes: list[float], period: int = _ATR_PERIOD) -> float | None:
    """Simple average of true range over the last `period` completed steps."""
    if len(closes) < 2 or not (len(highs) == len(lows) == len(closes)):
        return None
    true_ranges: list[float] = []
    for index in range(1, len(closes)):
        high, low, prev = highs[index], lows[index], closes[index - 1]
        true_ranges.append(max(high - low, abs(high - prev), abs(low - prev)))
    window = true_ranges[-period:]
    if not window:
        return None
    return sum(window) / len(window)


def relative_volume(volumes: list[float]) -> float | None:
    """Latest bar's volume divided by the mean of the bars before it."""
    if len(volumes) < 2 or volumes[-1] < 0:
        return None
    baseline = volumes[:-1]
    mean = sum(baseline) / len(baseline)
    if mean <= 0:
        return None
    return volumes[-1] / mean


def calendar_from_rows(rows: list[dict], start: date, end: date) -> tuple[frozenset[date], dict[date, time]]:
    """Weekdays missing from an Alpaca `/v2/calendar` payload are full holidays.

    A session that closes before 16:00 ET is an early close. Weekends are not
    listed as holidays; the session clock already treats them as closed.
    """
    sessions: dict[date, time] = {}
    for row in rows:
        day = date.fromisoformat(str(row["date"]))
        close = str(row.get("close") or "16:00")
        hour, minute = (int(part) for part in close.split(":")[:2])
        sessions[day] = time(hour, minute)
    holidays: set[date] = set()
    early: dict[date, time] = {}
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            close = sessions.get(cursor)
            if close is None:
                holidays.add(cursor)
            elif close < time(16, 0):
                early[cursor] = close
        cursor += timedelta(days=1)
    return frozenset(holidays), early


def _bars_to_vol(highs, lows, closes, volumes, prev_close: float | None) -> VolSnapshot | None:
    atr = average_true_range(highs, lows, closes)
    if atr is None or atr <= 0 or not closes:
        return None
    return VolSnapshot(atr=atr, prev_rth_close=prev_close if prev_close and prev_close > 0 else closes[-1])


def snapshot_from_alpaca(
    snapshots: dict,
    bars_by_symbol: dict,
    assets_by_symbol: dict,
    calendar_rows: list[dict] | None = None,
    *,
    today: date | None = None,
) -> MarketSnapshot:
    """Turn Alpaca JSON into a snapshot. Unknown symbols are omitted."""
    quotes: dict[str, Quote] = {}
    vols: dict[str, VolSnapshot] = {}
    confirmations: dict[str, Confirmation] = {}
    for symbol, payload in snapshots.items():
        ticker = symbol.upper()
        quote_row = payload.get("latestQuote") or {}
        trade = payload.get("latestTrade") or {}
        bid = float(quote_row.get("bp") or 0)
        ask = float(quote_row.get("ap") or 0)
        last = float(trade.get("p") or payload.get("dailyBar", {}).get("c") or 0)
        if last <= 0:
            continue
        if bid <= 0 or ask <= 0:
            bid = ask = last
        asset = assets_by_symbol.get(ticker) or assets_by_symbol.get(symbol) or {}
        quotes[ticker] = Quote(
            bid=bid,
            ask=ask,
            last=last,
            halted=str(asset.get("status") or "active").lower() != "active" and bool(asset),
            shortable=bool(asset.get("shortable", True)),
            easy_to_borrow=bool(asset.get("easy_to_borrow", False)),
        )
        daily = bars_by_symbol.get(ticker) or bars_by_symbol.get(symbol) or []
        highs = [float(bar["h"]) for bar in daily]
        lows = [float(bar["l"]) for bar in daily]
        closes = [float(bar["c"]) for bar in daily]
        prev = payload.get("prevDailyBar") or {}
        prev_close = float(prev.get("c") or 0) or None
        volumes = [float(bar.get("v") or 0) for bar in daily]
        vol = _bars_to_vol(highs, lows, closes, volumes, prev_close)
        if vol is not None:
            vols[ticker] = vol
        rel = relative_volume(volumes)
        if rel is not None and prev_close and prev_close > 0:
            confirmations[ticker] = Confirmation((last - prev_close) / prev_close, rel)
    holidays: frozenset[date] = frozenset()
    early: dict[date, time] = {}
    if calendar_rows and today is not None:
        holidays, early = calendar_from_rows(calendar_rows, today - timedelta(days=10), today + timedelta(days=21))
    return MarketSnapshot(
        quotes=quotes, vols=vols, confirmations=confirmations, holidays=holidays, early_closes=early,
    )


def snapshot_from_yahoo(bars_by_symbol: dict[str, list[dict]]) -> MarketSnapshot:
    """Last trade from daily bars. No bid/ask and no locate, so shorts are blocked.

    Each bar needs high, low, close, and volume. The last bar is treated as
    the latest print.
    """
    quotes: dict[str, Quote] = {}
    vols: dict[str, VolSnapshot] = {}
    confirmations: dict[str, Confirmation] = {}
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        ticker = symbol.upper()
        last = float(bars[-1]["close"])
        if last <= 0:
            continue
        quotes[ticker] = Quote(
            bid=last, ask=last, last=last, shortable=True, easy_to_borrow=False,
        )
        prev = float(bars[-2]["close"]) if len(bars) > 1 else last
        volumes = [float(bar.get("volume") or 0) for bar in bars]
        vol = _bars_to_vol(
            [float(bar["high"]) for bar in bars],
            [float(bar["low"]) for bar in bars],
            [float(bar["close"]) for bar in bars],
            volumes,
            prev,
        )
        if vol is not None:
            vols[ticker] = vol
        rel = relative_volume(volumes)
        if rel is not None and prev > 0:
            confirmations[ticker] = Confirmation((last - prev) / prev, rel)
    return MarketSnapshot(quotes=quotes, vols=vols, confirmations=confirmations)


def merge_snapshots(primary: MarketSnapshot, extra: MarketSnapshot) -> MarketSnapshot:
    """`primary` wins on a field it already has. Used so a local file overrides a feed."""
    return MarketSnapshot(
        quotes={**extra.quotes, **primary.quotes},
        vols={**extra.vols, **primary.vols},
        bars={**extra.bars, **primary.bars},
        confirmations={**extra.confirmations, **primary.confirmations},
        events=primary.events or extra.events,
        correlations={**extra.correlations, **primary.correlations},
        holidays=primary.holidays or extra.holidays,
        early_closes={**extra.early_closes, **primary.early_closes},
    )


class AlpacaMarketClient:
    """Paper-key client. `transport(method, url, body)` is the only IO."""

    def __init__(self, *, api_key: str, api_secret: str, transport) -> None:
        if not api_key or not api_secret:
            raise BrokerError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")
        self._api_key = api_key
        self._api_secret = api_secret
        self._transport = transport

    def __repr__(self) -> str:
        return "AlpacaMarketClient(paper-data)"

    def snapshot(self, tickers: list[str], now: datetime) -> MarketSnapshot:
        symbols = ",".join(ticker.upper() for ticker in tickers)
        snaps = self._call("GET", f"{DATA_BASE}/v2/stocks/snapshots?symbols={symbols}", None)
        bars = self._call(
            "GET",
            f"{DATA_BASE}/v2/stocks/bars?symbols={symbols}&timeframe=1Day&limit=30",
            None,
        )
        assets = {}
        for ticker in tickers:
            try:
                assets[ticker.upper()] = self._call("GET", f"{PAPER_BASE_URL}/v2/assets/{ticker.upper()}", None)
            except BrokerError:
                assets[ticker.upper()] = {}
        calendar_rows = None
        try:
            start = (now.date() - timedelta(days=10)).isoformat()
            end = (now.date() + timedelta(days=21)).isoformat()
            calendar_rows = self._call("GET", f"{PAPER_BASE_URL}/v2/calendar?start={start}&end={end}", None)
        except BrokerError:
            calendar_rows = None
        bar_map = bars.get("bars", bars) if isinstance(bars, dict) else {}
        if not isinstance(snaps, dict):
            raise BrokerError("alpaca snapshot response was not an object")
        return snapshot_from_alpaca(
            snaps,
            bar_map if isinstance(bar_map, dict) else {},
            assets,
            calendar_rows if isinstance(calendar_rows, list) else None,
            today=now.date(),
        )

    def _call(self, method: str, url: str, body: dict | None):
        try:
            return self._transport(method, url, body)
        except BrokerError as exc:
            raise BrokerError(scrub_text(str(exc), [self._api_key, self._api_secret])) from None
        except Exception as exc:
            raise BrokerError(scrub_text(str(exc), [self._api_key, self._api_secret])) from None


def yahoo_bars(ticker: str) -> list[dict]:
    """Daily bars from yfinance. Imported only when this function runs."""
    import yfinance as yf

    frame = yf.Ticker(ticker).history(period="2mo", interval="1d", auto_adjust=False)
    rows = []
    for _index, row in frame.iterrows():
        rows.append({
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        })
    return rows


def load_market_snapshot(
    tickers: list[str],
    now: datetime,
    *,
    alpaca: AlpacaMarketClient | None = None,
    yahoo_fetch=None,
) -> tuple[MarketSnapshot, str]:
    """Alpaca first, then Yahoo. Returns the snapshot and which source filled it.

    Pass `alpaca` or `yahoo_fetch` in tests. With neither, keys in the
    environment build an Alpaca client and Yahoo is the fallback.
    """
    client = alpaca if alpaca is not None else _alpaca_from_env()
    if client is not None:
        try:
            return client.snapshot(tickers, now), "alpaca"
        except BrokerError:
            pass
    fetch = yahoo_fetch if yahoo_fetch is not None else yahoo_bars
    bars = {}
    for ticker in tickers:
        try:
            bars[ticker] = fetch(ticker)
        except Exception as exc:
            raise BrokerError(f"market data failed for {ticker}: {type(exc).__name__}") from None
    return snapshot_from_yahoo(bars), "yahoo"


def _alpaca_from_env() -> AlpacaMarketClient | None:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET") or ""
    if not key or not secret:
        return None

    def transport(method: str, url: str, body: dict | None):
        import requests

        try:
            response = requests.request(
                method, url, json=body, timeout=15,
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except Exception as exc:
            raise BrokerError(scrub_text(str(exc), [key, secret])) from None

    return AlpacaMarketClient(api_key=key, api_secret=secret, transport=transport)


def enrich_market(market: MarketSnapshot, tickers: list[str], now: datetime, loader=None) -> MarketSnapshot:
    """Fill quotes and ATR for tickers the local snapshot does not already have.

    A complete snapshot is returned unchanged, so a test file never triggers a fetch.
    """
    missing = [ticker for ticker in dict.fromkeys(tickers) if ticker not in market.quotes or ticker not in market.vols]
    if not missing:
        return market
    if loader is None:
        def loader(symbols, moment):
            snapshot, _source = load_market_snapshot(symbols, moment)
            return snapshot
    extra = loader(missing, now)
    return merge_snapshots(market, extra)
