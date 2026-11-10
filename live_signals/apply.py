"""Apply a cycle plan to the book.

Execution role: turn intents plus fill results into positions, cash, and
resting orders. Log-only and paper share this function. The broker, if any,
has already been called; this module does not call one. Entry fills are what
start the cooldown, so a resting order that never fills does not block a
later signal once it is cancelled.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from live_signals.config import Config
from live_signals.models import (
    Book,
    FillPlan,
    Intent,
    MarketSnapshot,
    PositionState,
    SubmitResult,
    TakeRecord,
    WorkingOrder,
)
from live_signals.pricing import marketable_fill
from live_signals.sizing import stop_distance


def _open_position(
    effect: FillPlan,
    ticker: str,
    qty: int,
    fill: float,
    now: datetime,
    cfg: Config,
) -> PositionState:
    distance = stop_distance(fill, effect.atr, cfg)
    if effect.direction == "long":
        stop = fill - distance
    else:
        stop = fill + distance
    return PositionState(
        ticker=ticker,
        side=effect.direction,
        qty=qty,
        entry_price=fill,
        entry_time=now,
        entry_score=effect.score,
        signal_id=effect.signal_id,
        initial_stop=stop,
        stop=stop,
        atr=effect.atr,
        r=distance,
        watermark=fill,
        trail_armed=False,
        tp1_done=False,
        tp2_done=False,
        target_qty=effect.target_qty or qty,
        tranche2_qty=effect.tranche2_qty,
        tranche2_deadline=effect.tranche2_deadline,
        tranche2_sent=False,
        adopted=False,
        entry_session=effect.entry_session,
        mfe_r=0.0,
        exit_attempt=0,
        credibility=effect.credibility,
        basis_qty=qty,
    )


def apply_fill(
    cash: float,
    positions: dict[str, PositionState],
    *,
    ticker: str,
    side: str,
    qty: int,
    fill: float,
    effect: FillPlan,
    now: datetime,
    cfg: Config,
) -> tuple[float, dict[str, PositionState]]:
    """Apply one fill. `side` is the order side (buy/sell), not the position side."""
    positions = dict(positions)
    existing = positions.get(ticker)
    if effect.purpose == "exit":
        if existing is None:
            return cash, positions
        close_qty = min(qty, existing.qty)
        if existing.side == "long":
            cash += close_qty * fill
        else:
            cash -= close_qty * fill
        left = existing.qty - close_qty
        if left <= 0:
            positions.pop(ticker, None)
        else:
            positions[ticker] = existing.evolve(
                qty=left,
                tp1_done=existing.tp1_done or effect.set_tp1,
                tp2_done=existing.tp2_done or effect.set_tp2,
            )
        return cash, positions

    if effect.direction == "long":
        cash -= qty * fill
    else:
        cash += qty * fill

    if existing is None:
        positions[ticker] = _open_position(effect, ticker, qty, fill, now, cfg)
        return cash, positions

    new_qty = existing.qty + qty
    basis = existing.basis_qty or existing.qty
    if not existing.tp1_done and not existing.tp2_done:
        basis = new_qty
    updated = existing.evolve(qty=new_qty, basis_qty=basis, target_qty=max(existing.target_qty, effect.target_qty or new_qty))
    if effect.purpose == "scale_in":
        updated = updated.evolve(tranche2_sent=True, tranche2_qty=0, tranche2_deadline=None)
    elif effect.purpose == "entry":
        updated = updated.evolve(
            tranche2_qty=effect.tranche2_qty,
            tranche2_deadline=effect.tranche2_deadline,
            tranche2_sent=False,
        )
    positions[ticker] = updated
    return cash, positions


def apply_plan(
    book: Book,
    *,
    position_edits: dict[str, PositionState],
    drop_working: list[str],
    replace_working: dict[str, WorkingOrder],
    seen_add: list[str],
    intents: list[Intent],
    results: dict[str, SubmitResult],
    now: datetime,
    cfg: Config,
    risk_state,
) -> Book:
    positions = {**book.positions, **position_edits}
    for ticker, edited in position_edits.items():
        # A full exit removes the position on fill. Keep the edit until then
        # so a resting exit still carries the ratcheted stop.
        positions[ticker] = edited
    working = dict(book.working)
    for cid in drop_working:
        working.pop(cid, None)
    working.update(replace_working)
    cash = book.cash
    seen = set(book.seen)
    seen.update(seen_add)
    takes = list(book.takes)

    for intent in intents:
        if intent.action != "submit" or intent.order is None or intent.effect is None:
            continue
        result = results.get(intent.order.client_order_id)
        if result is None or result.status == "duplicate":
            continue
        if result.status == "open":
            if intent.working is not None:
                working[intent.order.client_order_id] = intent.working
            continue
        if result.status != "filled" or result.fill_price is None:
            continue
        fill_qty = result.fill_qty or intent.order.qty
        cash, positions = apply_fill(
            cash,
            positions,
            ticker=intent.order.ticker,
            side=intent.order.side,
            qty=fill_qty,
            fill=result.fill_price,
            effect=intent.effect,
            now=now,
            cfg=cfg,
        )
        working.pop(intent.order.client_order_id, None)
        if intent.effect.purpose in ("entry", "scale_in") and intent.effect.signal_id:
            seen.add(intent.effect.signal_id)
        if intent.effect.purpose == "entry" and intent.effect.direction in ("long", "short"):
            takes.append(
                TakeRecord(
                    ticker=intent.order.ticker,
                    direction=intent.effect.direction,
                    score=intent.effect.score,
                    at=now,
                    signal_id=intent.effect.signal_id,
                )
            )

    cutoff = now - timedelta(hours=24)
    takes = [item for item in takes if item.at >= cutoff]
    return Book(
        cash=cash,
        positions=positions,
        working=working,
        seen=frozenset(seen),
        takes=tuple(takes),
        risk=risk_state,
    )


def simulate_results(intents: list[Intent], market: MarketSnapshot) -> dict[str, SubmitResult]:
    """Fill marketable limits against the snapshot. Unfilled limits rest."""
    results: dict[str, SubmitResult] = {}
    for intent in intents:
        if intent.action != "submit" or intent.order is None:
            continue
        quote = market.quotes.get(intent.order.ticker)
        if quote is None:
            results[intent.order.client_order_id] = SubmitResult("open")
            continue
        fill = marketable_fill(intent.order.side, intent.order.limit_price, quote)
        if fill is None:
            results[intent.order.client_order_id] = SubmitResult("open")
        else:
            results[intent.order.client_order_id] = SubmitResult("filled", fill, intent.order.qty)
    return results


def resolve_resting(book: Book, market: MarketSnapshot, now: datetime, cfg: Config) -> Book:
    """Fill resting orders that have become marketable since the last cycle.

    Used by log-only, which has no broker to report the fill. Risk was
    reserved when the order was submitted, so this does not change the budget.
    """
    positions = dict(book.positions)
    working = dict(book.working)
    cash = book.cash
    seen = set(book.seen)
    takes = list(book.takes)
    for cid, order in list(working.items()):
        quote = market.quotes.get(order.ticker)
        if quote is None:
            continue
        fill = marketable_fill(order.side, order.limit_price, quote)
        if fill is None:
            continue
        cash, positions = apply_fill(
            cash,
            positions,
            ticker=order.ticker,
            side=order.side,
            qty=order.qty,
            fill=fill,
            effect=order.effect,
            now=now,
            cfg=cfg,
        )
        working.pop(cid, None)
        if order.purpose in ("entry", "scale_in") and order.signal_id:
            seen.add(order.signal_id)
        if order.purpose == "entry" and order.effect.direction in ("long", "short"):
            takes.append(
                TakeRecord(
                    ticker=order.ticker,
                    direction=order.effect.direction,
                    score=order.effect.score,
                    at=now,
                    signal_id=order.signal_id,
                )
            )
    return book.evolve(
        cash=cash,
        positions=positions,
        working=working,
        seen=frozenset(seen),
        takes=tuple(takes),
    )


def mark_equity(book: Book, quotes) -> float:
    equity = book.cash
    for position in book.positions.values():
        quote = quotes.get(position.ticker)
        price = quote.last if quote is not None else position.entry_price
        if position.side == "long":
            equity += position.qty * price
        else:
            equity -= position.qty * price
    return equity
