"""Data contracts for the live-signal engine.

Execution role: Wirewatch, the broker, and the planner talk only through these
types. Nothing here performs IO. A Wirewatch payload is adapted by
`parse_wirewatch_payload`; the rest of the package never sees Wirewatch's keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
from typing import Any, Mapping

from live_signals.errors import ConfigError


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ConfigError(f"unparseable timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ConfigError("timestamp must include a timezone offset")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class SignalSource:
    url: str
    source_type: str


@dataclass(frozen=True)
class NewsSignal:
    ticker: str
    score: float
    sentiment: str
    sources: tuple[SignalSource, ...]
    headline: str
    rationale: str
    timestamp: datetime
    signal_id: str

    @property
    def direction(self) -> str:
        sentiment = self.sentiment.lower()
        if sentiment == "bullish":
            return "long"
        if sentiment == "bearish":
            return "short"
        if sentiment == "neutral":
            return "neutral"
        return ""


def parse_wirewatch_payload(payload: Mapping[str, Any]) -> NewsSignal:
    """Map one Wirewatch JSON object into a `NewsSignal`.

    Expected keys: ticker, score, sentiment, sources (url + type or
    source_type), headline, rationale, timestamp, and id or signal_id.
    Extra keys are ignored. This is the only function that knows Wirewatch's
    field names; it does not fetch anything.
    """
    if not isinstance(payload, Mapping):
        raise ConfigError("wirewatch payload must be an object")
    sources_raw = payload.get("sources") or []
    sources: list[SignalSource] = []
    for item in sources_raw:
        if not isinstance(item, Mapping):
            raise ConfigError("each source must be an object")
        source_type = item.get("source_type", item.get("type", ""))
        sources.append(SignalSource(url=str(item.get("url", "")), source_type=str(source_type)))
    timestamp = parse_timestamp(str(payload["timestamp"]))
    ticker = str(payload["ticker"]).upper().strip()
    score = float(payload["score"])
    sentiment = str(payload["sentiment"]).lower().strip()
    headline = str(payload.get("headline") or "")
    rationale = str(payload.get("rationale") or "")
    signal_id = str(payload.get("id") or payload.get("signal_id") or "")
    if not signal_id:
        raw = f"{ticker}|{timestamp.isoformat()}|{headline}|{score}|{sentiment}"
        signal_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return NewsSignal(
        ticker=ticker,
        score=score,
        sentiment=sentiment,
        sources=tuple(sources),
        headline=headline,
        rationale=rationale,
        timestamp=timestamp,
        signal_id=signal_id,
    )


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    last: float
    halted: bool = False
    shortable: bool = True
    easy_to_borrow: bool = True


@dataclass(frozen=True)
class VolSnapshot:
    atr: float
    prev_rth_close: float
    sector: str | None = None


@dataclass(frozen=True)
class PriceBar:
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Confirmation:
    """Price/volume check supplied by the caller. No network inside the engine.

    `return_since_signal` is the raw price return (positive when price is up).
    `rel_volume` is recent volume divided by a baseline such as a 20-day average.
    """

    return_since_signal: float
    rel_volume: float


@dataclass(frozen=True)
class ScheduledEvent:
    name: str
    at: datetime
    kind: str
    ticker: str | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    quotes: dict[str, Quote] = field(default_factory=dict)
    vols: dict[str, VolSnapshot] = field(default_factory=dict)
    bars: dict[str, PriceBar] = field(default_factory=dict)
    confirmations: dict[str, Confirmation] = field(default_factory=dict)
    events: tuple[ScheduledEvent, ...] = ()
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    holidays: frozenset[date] = field(default_factory=frozenset)
    early_closes: dict[date, time] = field(default_factory=dict)


@dataclass(frozen=True)
class Qualification:
    outcome: str
    reason: str
    direction: str
    credibility: float
    terminal: bool


@dataclass(frozen=True)
class SizeResult:
    qty: int
    conviction: float
    credibility: float
    session_mult: float
    stop_distance: float
    risk_dollars: float
    risk_pct: float
    notional: float
    binding: str


@dataclass(frozen=True)
class PriceDecision:
    ok: bool
    reason: str
    limit_price: float | None
    extended_hours: bool
    time_in_force: str
    order_type: str
    mid: float | None


@dataclass(frozen=True)
class Exposure:
    ticker: str
    side: str
    qty: int
    mark: float
    sector: str | None = None

    @property
    def notional(self) -> float:
        return abs(self.qty * self.mark)


@dataclass(frozen=True)
class OrderRequest:
    client_order_id: str
    ticker: str
    side: str
    qty: int
    limit_price: float
    extended_hours: bool
    time_in_force: str = "day"
    order_type: str = "limit"

    def __post_init__(self) -> None:
        if self.order_type != "limit":
            raise ValueError("only limit orders are supported")
        if self.time_in_force not in ("day", "gtc"):
            raise ValueError("time_in_force must be day or gtc")
        if self.extended_hours and self.time_in_force not in ("day", "gtc"):
            raise ValueError("extended hours requires day or gtc")
        if isinstance(self.qty, bool) or not isinstance(self.qty, int) or self.qty < 1:
            raise ValueError("qty must be a positive whole number of shares")
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be positive")


@dataclass(frozen=True)
class FillPlan:
    """Bookkeeping for a fill. Never serialized into a broker payload."""

    purpose: str
    direction: str = ""
    score: float = 0.0
    atr: float = 0.0
    signal_id: str = ""
    target_qty: int = 0
    tranche2_qty: int = 0
    entry_session: str = ""
    credibility: float = 0.0
    set_tp1: bool = False
    set_tp2: bool = False
    reserve_risk: float = 0.0
    count_trade: int = 0
    tranche2_deadline: datetime | None = None
    exit_attempt: int = 0


@dataclass(frozen=True)
class PositionState:
    ticker: str
    side: str
    qty: int
    entry_price: float
    entry_time: datetime
    entry_score: float
    signal_id: str
    initial_stop: float
    stop: float
    atr: float
    r: float
    watermark: float
    trail_armed: bool
    tp1_done: bool
    tp2_done: bool
    target_qty: int
    tranche2_qty: int
    tranche2_deadline: datetime | None
    tranche2_sent: bool
    adopted: bool
    entry_session: str
    mfe_r: float
    exit_attempt: int = 0
    credibility: float = 0.0
    basis_qty: int = 0

    def evolve(self, **changes: Any) -> PositionState:
        return replace(self, **changes)


@dataclass(frozen=True)
class WorkingOrder:
    client_order_id: str
    ticker: str
    side: str
    qty: int
    limit_price: float
    submitted_at: datetime
    decision_mid: float
    replaces: int
    session: str
    purpose: str
    signal_id: str
    tranche: int
    effect: FillPlan

    def evolve(self, **changes: Any) -> WorkingOrder:
        return replace(self, **changes)


@dataclass(frozen=True)
class TakeRecord:
    ticker: str
    direction: str
    score: float
    at: datetime
    signal_id: str


@dataclass(frozen=True)
class RiskState:
    peak_equity: float
    drawdown_halted: bool
    day: date | None
    day_start_equity: float
    day_halted: bool
    trades_today: int
    new_risk_today: float

    def evolve(self, **changes: Any) -> RiskState:
        return replace(self, **changes)


@dataclass(frozen=True)
class Book:
    cash: float
    positions: dict[str, PositionState]
    working: dict[str, WorkingOrder]
    seen: frozenset[str]
    takes: tuple[TakeRecord, ...]
    risk: RiskState

    def evolve(self, **changes: Any) -> Book:
        return replace(self, **changes)


def new_book(equity: float) -> Book:
    return Book(
        cash=float(equity),
        positions={},
        working={},
        seen=frozenset(),
        takes=(),
        risk=RiskState(
            peak_equity=float(equity),
            drawdown_halted=False,
            day=None,
            day_start_equity=float(equity),
            day_halted=False,
            trades_today=0,
            new_risk_today=0.0,
        ),
    )


@dataclass(frozen=True)
class Intent:
    action: str
    reason: str
    order: OrderRequest | None = None
    client_order_id: str = ""
    new_limit: float | None = None
    working: WorkingOrder | None = None
    effect: FillPlan | None = None


@dataclass(frozen=True)
class Decision:
    ts: datetime
    ticker: str
    signal_id: str
    score: float | None
    sentiment: str
    action: str
    reason: str
    qty: int
    limit_price: float | None
    session: str
    mode: str
    details: dict[str, Any]


@dataclass(frozen=True)
class BrokerPosition:
    ticker: str
    qty: int
    avg_price: float


@dataclass(frozen=True)
class BrokerOrder:
    client_order_id: str
    ticker: str
    side: str
    qty: int
    limit_price: float | None
    status: str


@dataclass(frozen=True)
class AccountView:
    account_id: str
    account_type: str
    equity: float
    cash: float
    positions: dict[str, BrokerPosition]
    open_orders: dict[str, BrokerOrder]


@dataclass(frozen=True)
class SubmitResult:
    status: str
    fill_price: float | None = None
    fill_qty: int | None = None


def make_decision(
    ts: datetime,
    *,
    ticker: str,
    action: str,
    reason: str,
    session: str,
    mode: str,
    signal_id: str = "",
    score: float | None = None,
    sentiment: str = "",
    qty: int = 0,
    limit_price: float | None = None,
    details: dict | None = None,
) -> Decision:
    return Decision(
        ts=ts,
        ticker=ticker,
        signal_id=signal_id,
        score=score,
        sentiment=sentiment,
        action=action,
        reason=reason,
        qty=qty,
        limit_price=limit_price,
        session=session,
        mode=mode,
        details={} if details is None else details,
    )
