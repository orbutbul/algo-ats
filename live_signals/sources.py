"""Signal and market inputs.

Execution role: the engine polls these. `JsonFileSignalSource` re-reads a file
Wirewatch (or a person) can overwrite. `WirewatchPushSource` is a stub for a
future push or HTTP feed; it does not open a socket. Market data is the same
idea: a JSON file the caller refreshes. This package does not fetch quotes.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

from live_signals.models import (
    Confirmation,
    MarketSnapshot,
    NewsSignal,
    PriceBar,
    Quote,
    ScheduledEvent,
    VolSnapshot,
    parse_timestamp,
    parse_wirewatch_payload,
)


class JsonFileSignalSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, now: datetime) -> list[NewsSignal]:
        del now
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("signals", [])
        return [parse_wirewatch_payload(row) for row in rows]


class WirewatchPushSource:
    """Placeholder for a Wirewatch push or HTTP feed.

    Wirewatch's code is not in this repo. When it can POST signals or expose
    an endpoint, implement `fetch` behind this class. Until then it returns
    nothing and does not make a network call.
    """

    def fetch(self, now: datetime) -> list[NewsSignal]:
        del now
        return []


class JsonFileMarketSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, now: datetime) -> MarketSnapshot:
        del now
        if not self.path.exists():
            return MarketSnapshot()
        return market_from_payload(json.loads(self.path.read_text()))


def market_from_payload(payload: dict) -> MarketSnapshot:
    quotes = {
        ticker: Quote(
            bid=float(item["bid"]),
            ask=float(item["ask"]),
            last=float(item["last"]),
            halted=bool(item.get("halted", False)),
            shortable=bool(item.get("shortable", True)),
            easy_to_borrow=bool(item.get("easy_to_borrow", True)),
        )
        for ticker, item in (payload.get("quotes") or {}).items()
    }
    vols = {
        ticker: VolSnapshot(
            atr=float(item["atr"]),
            prev_rth_close=float(item["prev_rth_close"]),
            sector=item.get("sector"),
        )
        for ticker, item in (payload.get("vols") or {}).items()
    }
    bars = {
        ticker: PriceBar(high=float(item["high"]), low=float(item["low"]), close=float(item["close"]))
        for ticker, item in (payload.get("bars") or {}).items()
    }
    confirmations = {
        ticker: Confirmation(
            return_since_signal=float(item["return_since_signal"]),
            rel_volume=float(item["rel_volume"]),
        )
        for ticker, item in (payload.get("confirmations") or {}).items()
    }
    events = []
    for item in payload.get("events") or []:
        events.append(
            ScheduledEvent(
                name=str(item.get("name") or item.get("kind") or "event"),
                at=parse_timestamp(str(item["at"])),
                kind=str(item.get("kind") or "macro"),
                ticker=item.get("ticker"),
            )
        )
    correlations = {}
    for item in payload.get("correlations") or []:
        pair = (str(item["a"]).upper(), str(item["b"]).upper())
        correlations[pair] = float(item["corr"])
    holidays = frozenset(date.fromisoformat(item) for item in payload.get("holidays") or [])
    early = {}
    for key, value in (payload.get("early_closes") or {}).items():
        hour, minute, second = (int(part) for part in str(value).split(":"))
        early[date.fromisoformat(key)] = time(hour, minute, second)
    return MarketSnapshot(
        quotes=quotes,
        vols=vols,
        bars=bars,
        confirmations=confirmations,
        events=tuple(events),
        correlations=correlations,
        holidays=holidays,
        early_closes=early,
    )
