"""Startup and per-cycle reconciliation against a broker snapshot.

Execution role: paper mode treats the broker as the source of truth for
quantities and cash. Local state keeps the stops and signal scores the broker
does not know. Orphan orders tagged with our client-id prefix are cancelled.
Orders without that prefix are left alone (they belong to the morning batch
or to a human). Positions the broker has and we do not are adopted and, by
default, not managed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from live_signals.apply import apply_fill
from live_signals.config import Config
from live_signals.models import (
    AccountView,
    Book,
    MarketSnapshot,
    PositionState,
    TakeRecord,
    make_decision,
)
from live_signals.sizing import stop_distance


@dataclass(frozen=True)
class SyncResult:
    book: Book
    decisions: list
    cancel_ids: list[str]


def _adopt(ticker: str, qty: int, avg: float, now: datetime, cfg: Config, market: MarketSnapshot) -> PositionState:
    vol = market.vols.get(ticker)
    atr = vol.atr if vol is not None and vol.atr > 0 else max(avg * cfg.min_stop_pct, 0.01)
    distance = stop_distance(avg, atr, cfg)
    side = "long" if qty > 0 else "short"
    stop = avg - distance if side == "long" else avg + distance
    shares = abs(qty)
    return PositionState(
        ticker=ticker,
        side=side,
        qty=shares,
        entry_price=avg,
        entry_time=now,
        entry_score=0.0,
        signal_id="adopted",
        initial_stop=stop,
        stop=stop,
        atr=atr,
        r=distance,
        watermark=avg,
        trail_armed=False,
        tp1_done=False,
        tp2_done=False,
        target_qty=shares,
        tranche2_qty=0,
        tranche2_deadline=None,
        tranche2_sent=True,
        adopted=True,
        entry_session="rth",
        mfe_r=0.0,
        basis_qty=shares,
    )


def sync_broker(
    book: Book,
    account: AccountView,
    now: datetime,
    market: MarketSnapshot,
    cfg: Config,
    *,
    session: str,
    mode: str,
) -> SyncResult:
    decisions = []
    cancels: list[str] = []
    positions = dict(book.positions)
    working = dict(book.working)
    seen = set(book.seen)
    takes = list(book.takes)
    risk_used = book.risk.new_risk_today
    trades = book.risk.trades_today
    open_ids = set(account.open_orders)
    prefix = cfg.order_id_prefix + "-"

    for cid, order in account.open_orders.items():
        if cid.startswith(prefix) and cid not in working:
            cancels.append(cid)
            decisions.append(
                make_decision(
                    now, ticker=order.ticker, action="cancel", reason="orphan_order",
                    session=session, mode=mode, signal_id=cid, qty=order.qty,
                    limit_price=order.limit_price,
                )
            )

    for cid, order in list(working.items()):
        if cid in open_ids:
            continue
        broker_pos = account.positions.get(order.ticker)
        broker_qty = abs(broker_pos.qty) if broker_pos is not None else 0
        local = positions.get(order.ticker)
        local_qty = local.qty if local is not None else 0
        if order.purpose in ("entry", "scale_in") and broker_qty > local_qty:
            fill = broker_pos.avg_price if broker_pos is not None and broker_pos.avg_price > 0 else order.limit_price
            _, positions = apply_fill(
                book.cash, positions, ticker=order.ticker, side=order.side,
                qty=broker_qty - local_qty, fill=fill, effect=order.effect, now=now, cfg=cfg,
            )
            if order.signal_id:
                seen.add(order.signal_id)
            if order.purpose == "entry" and order.effect.direction in ("long", "short"):
                takes.append(
                    TakeRecord(
                        ticker=order.ticker, direction=order.effect.direction,
                        score=order.effect.score, at=now, signal_id=order.signal_id,
                    )
                )
            decisions.append(
                make_decision(
                    now, ticker=order.ticker, action="take", reason="filled_at_broker",
                    session=session, mode=mode, signal_id=order.signal_id, qty=broker_qty - local_qty,
                    limit_price=fill,
                )
            )
        elif order.purpose == "exit" and local is not None and broker_qty < local_qty:
            if broker_qty == 0:
                positions.pop(order.ticker, None)
            else:
                positions[order.ticker] = local.evolve(
                    qty=broker_qty,
                    tp1_done=local.tp1_done or order.effect.set_tp1,
                    tp2_done=local.tp2_done or order.effect.set_tp2,
                )
            decisions.append(
                make_decision(
                    now, ticker=order.ticker, action="exit", reason="exit_filled_at_broker",
                    session=session, mode=mode, signal_id=order.signal_id, qty=local_qty - broker_qty,
                )
            )
        else:
            if order.purpose == "entry":
                risk_used = max(0.0, risk_used - order.effect.reserve_risk)
                trades = max(0, trades - order.effect.count_trade)
            elif order.purpose == "scale_in":
                risk_used = max(0.0, risk_used - order.effect.reserve_risk)
                if order.ticker in positions:
                    positions[order.ticker] = positions[order.ticker].evolve(tranche2_qty=0, tranche2_sent=True)
            decisions.append(
                make_decision(
                    now, ticker=order.ticker, action="cancel", reason="broker_order_gone",
                    session=session, mode=mode, signal_id=order.signal_id, qty=order.qty,
                )
            )
        working.pop(cid, None)

    for ticker, position in list(positions.items()):
        broker_pos = account.positions.get(ticker)
        broker_qty_signed = broker_pos.qty if broker_pos is not None else 0
        if broker_qty_signed == 0:
            positions.pop(ticker, None)
            decisions.append(
                make_decision(
                    now, ticker=ticker, action="exit", reason="broker_flat",
                    session=session, mode=mode, signal_id=position.signal_id, qty=position.qty,
                )
            )
            continue
        side = "long" if broker_qty_signed > 0 else "short"
        shares = abs(broker_qty_signed)
        if side != position.side:
            positions[ticker] = _adopt(ticker, broker_qty_signed, broker_pos.avg_price, now, cfg, market)
            decisions.append(
                make_decision(
                    now, ticker=ticker, action="update", reason="side_reconciled",
                    session=session, mode=mode, qty=shares,
                )
            )
        elif shares != position.qty:
            positions[ticker] = position.evolve(qty=shares)
            decisions.append(
                make_decision(
                    now, ticker=ticker, action="update", reason="qty_reconciled",
                    session=session, mode=mode, signal_id=position.signal_id, qty=shares,
                    details={"previous_qty": position.qty},
                )
            )

    for ticker, broker_pos in account.positions.items():
        if broker_pos.qty == 0 or ticker in positions:
            continue
        positions[ticker] = _adopt(ticker, broker_pos.qty, broker_pos.avg_price, now, cfg, market)
        decisions.append(
            make_decision(
                now, ticker=ticker, action="update", reason="adopted",
                session=session, mode=mode, signal_id="adopted", qty=abs(broker_pos.qty),
                limit_price=broker_pos.avg_price,
            )
        )

    risk = book.risk.evolve(new_risk_today=risk_used, trades_today=trades)
    return SyncResult(
        book=Book(
            cash=account.cash,
            positions=positions,
            working=working,
            seen=frozenset(seen),
            takes=tuple(takes),
            risk=risk,
        ),
        decisions=decisions,
        cancel_ids=cancels,
    )
