"""SQLite persistence for one atomic book document.

Execution role: the cycle commits the whole book (positions, resting orders,
seen signals, take history, risk flags) as a single JSON document inside
SQLite. One transaction means a crash cannot save a new position without its
stop. The decision log is a separate append-only CSV.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from pathlib import Path

from live_signals.models import (
    Book,
    FillPlan,
    PositionState,
    RiskState,
    TakeRecord,
    WorkingOrder,
    new_book,
)


def _encode(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _dump_dataclass(value)
    if isinstance(value, datetime):
        return {"__dt__": value.isoformat()}
    if isinstance(value, date):
        return {"__date__": value.isoformat()}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict) and set(value) == {"__dt__"}:
        text = value["__dt__"]
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    if isinstance(value, dict) and set(value) == {"__date__"}:
        return date.fromisoformat(value["__date__"])
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _dump_dataclass(obj) -> dict:
    payload = {}
    for item in fields(obj):
        payload[item.name] = _encode(getattr(obj, item.name))
    return payload


def _load_dataclass(cls, payload: dict):
    data = {key: _decode(value) for key, value in payload.items()}
    return cls(**data)


def dump_book(book: Book) -> dict:
    return {
        "cash": book.cash,
        "positions": {ticker: _dump_dataclass(pos) for ticker, pos in book.positions.items()},
        "working": {cid: _dump_dataclass(order) for cid, order in book.working.items()},
        "seen": sorted(book.seen),
        "takes": [_dump_dataclass(item) for item in book.takes],
        "risk": _dump_dataclass(book.risk),
    }


def load_book(payload: dict) -> Book:
    positions = {
        ticker: _load_dataclass(PositionState, item) for ticker, item in payload.get("positions", {}).items()
    }
    working_raw = {}
    for cid, item in payload.get("working", {}).items():
        decoded = _decode(item)
        effect = FillPlan(**decoded.pop("effect"))
        working_raw[cid] = WorkingOrder(**decoded, effect=effect)
    takes = tuple(_load_dataclass(TakeRecord, item) for item in payload.get("takes", []))
    risk = _load_dataclass(RiskState, payload["risk"])
    return Book(
        cash=float(payload["cash"]),
        positions=positions,
        working=working_raw,
        seen=frozenset(payload.get("seen", [])),
        takes=takes,
        risk=risk,
    )


class StateStore:
    def __init__(self, path: str | Path, reference_equity: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._reference_equity = reference_equity
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (name TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        self._conn.commit()

    def load(self) -> Book:
        row = self._conn.execute("SELECT body FROM documents WHERE name = 'book'").fetchone()
        if row is None:
            return new_book(self._reference_equity)
        return load_book(json.loads(row[0]))

    def save(self, book: Book) -> None:
        body = json.dumps(dump_book(book))
        self._conn.execute(
            "INSERT INTO documents (name, body) VALUES ('book', ?) "
            "ON CONFLICT(name) DO UPDATE SET body = excluded.body",
            (body,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
