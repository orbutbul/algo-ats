"""Broker boundary.

Execution role: the engine talks to this protocol and never to Alpaca
directly. `InMemoryBroker` is the test double and the log-only book's
stand-in is not even this class — log-only never calls a broker. Paper
submission has to pass `assert_submission_allowed` first, which accepts only
mode `paper`, account type `paper`, and account number PA32ZMSEI (the
Wirewatch paper account, not the ATS account).
"""

from __future__ import annotations

from typing import Protocol

from live_signals.errors import AccountGuardError, LiveTradingDisabled
from live_signals.models import (
    AccountView,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    Quote,
    SubmitResult,
)
from live_signals.pricing import marketable_fill

# Wirewatch paper account ("Grok Recs"). Not a config knob.
PAPER_ACCOUNT_ID = "PA32ZMSEI"


def assert_submission_allowed(mode: str, account_id: str, account_type: str) -> None:
    """Raise before any cancel, replace, or submit.

    `log_only` and `live` both fail. Live trading is not a mode this package
    implements; the check is here so a caller cannot sneak it in by bypassing
    config validation.
    """
    if mode != "paper":
        raise LiveTradingDisabled(
            "order submission requires mode=paper; live trading is hard-disabled"
        )
    if account_type != "paper":
        raise AccountGuardError("refusing orders: account_type must be paper")
    if account_id != PAPER_ACCOUNT_ID:
        raise AccountGuardError(
            f"refusing orders: account {account_id} is not {PAPER_ACCOUNT_ID}"
        )


class Broker(Protocol):
    def get_account(self) -> AccountView: ...

    def submit_order(self, order: OrderRequest) -> SubmitResult: ...

    def cancel_order(self, client_order_id: str) -> None: ...

    def replace_order(self, client_order_id: str, limit_price: float) -> SubmitResult: ...

    def mark_prices(self, marks: dict[str, float]) -> None: ...

    def set_quotes(self, quotes: dict[str, Quote]) -> None: ...


class _MemOrder:
    def __init__(self, order: OrderRequest, status: str, fill_price: float | None) -> None:
        self.order = order
        self.status = status
        self.fill_price = fill_price


class InMemoryBroker:
    """Fills marketable limits against the quotes last passed to `set_quotes`.

    Duplicate `client_order_id`s return the original result and do not change
    positions a second time. Equity is cash plus signed qty marked at the
    last price, falling back to average price.
    """

    def __init__(
        self,
        account_id: str = PAPER_ACCOUNT_ID,
        equity: float = 15_000.0,
        account_type: str = "paper",
    ) -> None:
        self.account_id = account_id
        self.account_type = account_type
        self.cash = float(equity)
        self.positions: dict[str, BrokerPosition] = {}
        self._orders: dict[str, _MemOrder] = {}
        self._marks: dict[str, float] = {}
        self._quotes: dict[str, Quote] = {}
        self.calls: list[str] = []

    def mark_prices(self, marks: dict[str, float]) -> None:
        self._marks.update(marks)

    def set_quotes(self, quotes: dict[str, Quote]) -> None:
        self._quotes = dict(quotes)
        for ticker, quote in quotes.items():
            self._marks[ticker] = quote.last

    def get_account(self) -> AccountView:
        equity = self.cash
        for position in self.positions.values():
            mark = self._marks.get(position.ticker, position.avg_price)
            equity += position.qty * mark
        open_orders = {
            cid: BrokerOrder(
                client_order_id=cid,
                ticker=item.order.ticker,
                side=item.order.side,
                qty=item.order.qty,
                limit_price=item.order.limit_price,
                status=item.status,
            )
            for cid, item in self._orders.items()
            if item.status == "open"
        }
        return AccountView(
            account_id=self.account_id,
            account_type=self.account_type,
            equity=equity,
            cash=self.cash,
            positions=dict(self.positions),
            open_orders=open_orders,
        )

    def submit_order(self, order: OrderRequest) -> SubmitResult:
        self.calls.append(f"submit:{order.client_order_id}")
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            qty = existing.order.qty if existing.status == "filled" else None
            return SubmitResult("duplicate", existing.fill_price, qty)
        quote = self._quotes.get(order.ticker)
        fill = marketable_fill(order.side, order.limit_price, quote) if quote is not None else None
        if fill is None:
            self._orders[order.client_order_id] = _MemOrder(order, "open", None)
            return SubmitResult("open")
        self._apply_fill(order, fill)
        self._orders[order.client_order_id] = _MemOrder(order, "filled", fill)
        return SubmitResult("filled", fill, order.qty)

    def cancel_order(self, client_order_id: str) -> None:
        self.calls.append(f"cancel:{client_order_id}")
        item = self._orders.get(client_order_id)
        if item is not None and item.status == "open":
            item.status = "canceled"

    def replace_order(self, client_order_id: str, limit_price: float) -> SubmitResult:
        self.calls.append(f"replace:{client_order_id}")
        item = self._orders.get(client_order_id)
        if item is None or item.status != "open":
            return SubmitResult("duplicate")
        updated = OrderRequest(
            client_order_id=item.order.client_order_id,
            ticker=item.order.ticker,
            side=item.order.side,
            qty=item.order.qty,
            limit_price=limit_price,
            extended_hours=item.order.extended_hours,
            time_in_force=item.order.time_in_force,
            order_type="limit",
        )
        quote = self._quotes.get(updated.ticker)
        fill = marketable_fill(updated.side, limit_price, quote) if quote is not None else None
        if fill is None:
            item.order = updated
            return SubmitResult("open")
        self._apply_fill(updated, fill)
        item.order = updated
        item.status = "filled"
        item.fill_price = fill
        return SubmitResult("filled", fill, updated.qty)

    def _apply_fill(self, order: OrderRequest, fill: float) -> None:
        current = self.positions.get(order.ticker)
        signed = current.qty if current else 0
        avg = current.avg_price if current else fill
        delta = order.qty if order.side == "buy" else -order.qty
        new_qty = signed + delta
        if order.side == "buy":
            self.cash -= order.qty * fill
        else:
            self.cash += order.qty * fill
        if new_qty == 0:
            self.positions.pop(order.ticker, None)
            return
        if signed == 0 or (signed > 0) == (new_qty > 0):
            if abs(new_qty) > abs(signed):
                avg = (abs(signed) * avg + order.qty * fill) / abs(new_qty)
            # A reduction keeps the old average.
        else:
            avg = fill
        self.positions[order.ticker] = BrokerPosition(order.ticker, new_qty, avg)
