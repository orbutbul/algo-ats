"""One planning cycle.

Execution role: pure function from a book, a clock, signals, and a market
snapshot to decisions and order intents. It does not touch a broker, a file,
or the clock. The engine applies the result.

Signal rows are always produced (taken or skipped, with a reason). Exit rows
are produced when a position is reduced or flattened. A halt (kill switch,
daily loss, drawdown) flattens everything, including adopted positions, and
skips every signal with the halt reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from live_signals.config import Config
from live_signals.events import entry_event_block
from live_signals.exits import evaluate_exit
from live_signals.ids import client_order_id
from live_signals.models import (
    Book,
    Exposure,
    FillPlan,
    Intent,
    MarketSnapshot,
    NewsSignal,
    OrderRequest,
    PositionState,
    PriceBar,
    TakeRecord,
    WorkingOrder,
    make_decision,
)
from live_signals.pricing import (
    exit_limit,
    market_skip_reason,
    price_entry,
    review_resting_entry,
    split_tranches,
)
from live_signals.qualify import ambiguous_ids, qualify_signal
from live_signals.risk import step_risk
from live_signals.sessions import classify_session, must_flatten_for_clock
from live_signals.sizing import size_target_with_cluster, stop_distance


@dataclass
class CyclePlan:
    decisions: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    position_edits: dict = field(default_factory=dict)
    drop_working: list = field(default_factory=list)
    replace_working: dict = field(default_factory=dict)
    seen_add: list = field(default_factory=list)
    risk: object | None = None


def _bar_for(ticker: str, market: MarketSnapshot) -> PriceBar | None:
    if ticker in market.bars:
        return market.bars[ticker]
    quote = market.quotes.get(ticker)
    if quote is None or quote.last <= 0:
        return None
    return PriceBar(high=quote.last, low=quote.last, close=quote.last)


def _order_side(direction: str) -> str:
    return "buy" if direction == "long" else "sell"


def _close_side(position: PositionState) -> str:
    return "sell" if position.side == "long" else "buy"


def _exposures(positions: dict[str, PositionState], market: MarketSnapshot, skip: set[str]) -> list[Exposure]:
    rows: list[Exposure] = []
    for ticker, position in positions.items():
        if ticker in skip or position.qty <= 0:
            continue
        quote = market.quotes.get(ticker)
        mark = quote.last if quote is not None and quote.last > 0 else position.entry_price
        sector = market.vols[ticker].sector if ticker in market.vols else None
        rows.append(Exposure(ticker, position.side, position.qty, mark, sector))
    return rows


def _exit_intent(
    position: PositionState,
    qty: int,
    reason: str,
    now: datetime,
    session: str,
    mode: str,
    market: MarketSnapshot,
    cfg: Config,
    *,
    set_tp1: bool = False,
    set_tp2: bool = False,
) -> tuple[Intent | None, object]:
    quote = market.quotes.get(position.ticker)
    if quote is None or qty < 1:
        return None, make_decision(
            now, ticker=position.ticker, action="exit", reason="exit_deferred_no_quote",
            session=session, mode=mode, signal_id=position.signal_id, score=position.entry_score,
            qty=0,
        )
    side = _close_side(position)
    limit = exit_limit(side, quote)
    if limit is None:
        return None, make_decision(
            now, ticker=position.ticker, action="exit", reason="exit_deferred_no_quote",
            session=session, mode=mode, signal_id=position.signal_id, qty=0,
        )
    oid = client_order_id(
        cfg.order_id_prefix, f"exit-{reason}", position.ticker, position.signal_id or position.ticker,
        0, position.exit_attempt,
    )
    order = OrderRequest(
        client_order_id=oid, ticker=position.ticker, side=side, qty=qty,
        limit_price=limit, extended_hours=session != "rth" and session != "closed",
        time_in_force="day", order_type="limit",
    )
    effect = FillPlan(
        purpose="exit", signal_id=position.signal_id, set_tp1=set_tp1, set_tp2=set_tp2,
        exit_attempt=position.exit_attempt,
    )
    working = WorkingOrder(
        client_order_id=oid, ticker=position.ticker, side=side, qty=qty, limit_price=limit,
        submitted_at=now, decision_mid=(quote.bid + quote.ask) / 2, replaces=0, session=session,
        purpose="exit", signal_id=position.signal_id, tranche=0, effect=effect,
    )
    intent = Intent("submit", reason, order=order, client_order_id=oid, working=working, effect=effect)
    decision = make_decision(
        now, ticker=position.ticker, action="exit", reason=reason, session=session, mode=mode,
        signal_id=position.signal_id, score=position.entry_score, qty=qty, limit_price=limit,
        details={"extended_hours": order.extended_hours, "order_type": "limit", "time_in_force": "day"},
    )
    return intent, decision


def plan_cycle(
    book: Book,
    now: datetime,
    signals: list[NewsSignal],
    market: MarketSnapshot,
    cfg: Config,
    *,
    kill: bool,
    reset_requested: bool,
    equity: float,
    mode: str,
) -> CyclePlan:
    session = classify_session(now, set(market.holidays), market.early_closes)
    risk = step_risk(book.risk, equity, now, cfg, reset_requested=reset_requested)
    halt = "kill_switch" if kill else None
    if halt is None and risk.day_halted:
        halt = "daily_loss_halt"
    if halt is None and risk.drawdown_halted:
        halt = "drawdown_halt"

    plan = CyclePlan(risk=risk)
    positions = dict(book.positions)
    risk_used = risk.new_risk_today
    trades = risk.trades_today

    def edit(position: PositionState) -> None:
        plan.position_edits[position.ticker] = position
        positions[position.ticker] = position

    # Resting orders. A session change or a second timeout cancels; one reprice
    # is allowed and it stays inside the original chase cap.
    for order in list(book.working.values()):
        quote = market.quotes.get(order.ticker)
        if halt and order.purpose in ("entry", "scale_in"):
            plan.drop_working.append(order.client_order_id)
            plan.intents.append(Intent("cancel", halt, client_order_id=order.client_order_id))
            risk_used = max(0.0, risk_used - order.effect.reserve_risk)
            trades = max(0, trades - order.effect.count_trade)
            continue
        if order.purpose == "exit" and halt:
            continue
        age = (now - order.submitted_at).total_seconds()
        if order.purpose == "exit":
            timeout = cfg.cancel_replace_seconds.get(order.session, cfg.cancel_replace_seconds["rth"])
            if session != order.session or age >= timeout:
                plan.drop_working.append(order.client_order_id)
                plan.intents.append(Intent("cancel", "exit_expired", client_order_id=order.client_order_id))
                if order.ticker in positions:
                    edit(positions[order.ticker].evolve(exit_attempt=positions[order.ticker].exit_attempt + 1))
            continue
        action = review_resting_entry(
            age_seconds=age,
            replaces=order.replaces,
            order_session=order.session,
            current_session=session,
            side=order.side,
            quote=quote,
            decision_mid=order.decision_mid,
            current_limit=order.limit_price,
            cfg=cfg,
        )
        if action == "hold":
            continue
        if action == "cancel":
            plan.drop_working.append(order.client_order_id)
            plan.intents.append(Intent("cancel", "entry_expired", client_order_id=order.client_order_id))
            risk_used = max(0.0, risk_used - order.effect.reserve_risk)
            trades = max(0, trades - order.effect.count_trade)
            if order.purpose == "scale_in" and order.ticker in positions:
                edit(positions[order.ticker].evolve(tranche2_qty=0, tranche2_sent=True))
            plan.decisions.append(
                make_decision(
                    now, ticker=order.ticker, action="cancel", reason="entry_expired",
                    session=session, mode=mode, signal_id=order.signal_id, qty=order.qty,
                )
            )
            continue
        new_limit = float(action)
        replaced = order.evolve(limit_price=new_limit, replaces=order.replaces + 1, submitted_at=now)
        plan.replace_working[order.client_order_id] = replaced
        plan.intents.append(
            Intent("replace", "reprice", client_order_id=order.client_order_id, new_limit=new_limit)
        )
        plan.decisions.append(
            make_decision(
                now, ticker=order.ticker, action="update", reason="reprice",
                session=session, mode=mode, signal_id=order.signal_id, qty=order.qty,
                limit_price=new_limit,
            )
        )

    live_exit = {
        order.ticker
        for order in book.working.values()
        if order.purpose == "exit" and order.client_order_id not in plan.drop_working
    }
    exiting_full: set[str] = set()
    signal_exit_score: dict[str, float] = {}

    for ticker, position in list(positions.items()):
        if position.adopted and not cfg.manage_adopted_positions and halt is None:
            continue
        if ticker in live_exit:
            continue
        bar = _bar_for(ticker, market)
        if halt:
            intent, decision = _exit_intent(position, position.qty, halt, now, session, mode, market, cfg)
            plan.decisions.append(decision)
            if intent is not None:
                plan.intents.append(intent)
                exiting_full.add(ticker)
            continue
        if bar is None:
            continue
        outcome = evaluate_exit(
            position, bar, now, signals, market.events, cfg,
            holidays=set(market.holidays), early_closes=market.early_closes,
            last_price=market.quotes[ticker].last if ticker in market.quotes else bar.close,
            confirmations=market.confirmations,
        )
        edit(outcome.position)
        if outcome.reason == "macro_tighten":
            plan.decisions.append(
                make_decision(
                    now, ticker=ticker, action="update", reason="macro_tighten",
                    session=session, mode=mode, signal_id=position.signal_id,
                    score=position.entry_score, details={"stop": outcome.position.stop},
                )
            )
        if outcome.action != "exit":
            continue
        intent, decision = _exit_intent(
            outcome.position, outcome.qty, outcome.reason, now, session, mode, market, cfg,
            set_tp1=outcome.set_tp1, set_tp2=outcome.set_tp2,
        )
        plan.decisions.append(decision)
        if intent is not None:
            plan.intents.append(intent)
            if outcome.qty >= position.qty:
                exiting_full.add(ticker)
                if outcome.reason == "signal_reversed":
                    signal_exit_score[ticker] = position.entry_score

    if halt is None:
        _plan_scale_ins(
            positions, plan, now, session, mode, market, cfg, equity, exiting_full,
            risk_used_box=[risk_used],
        )
        risk_used = plan.risk_used

    if halt is not None:
        for signal in signals:
            plan.decisions.append(_skip(signal, now, session, mode, halt, {}))
        plan.risk = risk.evolve(new_risk_today=risk_used, trades_today=trades)
        return plan

    _plan_entries(
        book, positions, plan, now, signals, market, cfg, session, mode, equity,
        exiting_full, signal_exit_score, risk_used, trades,
    )
    return plan


def _skip(signal: NewsSignal, now, session, mode, reason, details) -> object:
    return make_decision(
        now, ticker=signal.ticker, action="skip", reason=reason, session=session, mode=mode,
        signal_id=signal.signal_id, score=signal.score, sentiment=signal.sentiment,
        details={"headline": signal.headline[:180], **details},
    )


def _plan_scale_ins(positions, plan, now, session, mode, market, cfg, equity, exiting_full, risk_used_box):
    risk_used = risk_used_box[0]
    for ticker, position in list(positions.items()):
        if ticker in exiting_full or position.tranche2_qty < 1 or position.tranche2_sent:
            continue
        if position.tranche2_deadline is not None and now > position.tranche2_deadline:
            # The second tranche was never reserved; only a submitted order holds budget.
            updated = position.evolve(tranche2_qty=0, tranche2_sent=True, tranche2_deadline=None)
            plan.position_edits[ticker] = updated
            positions[ticker] = updated
            plan.decisions.append(
                make_decision(
                    now, ticker=ticker, action="skip", reason="scale_in_expired",
                    session=session, mode=mode, signal_id=position.signal_id, score=position.entry_score,
                )
            )
            continue
        if session not in cfg.scale_in_sessions:
            continue
        quote = market.quotes.get(ticker)
        if quote is None:
            continue
        sign = 1.0 if position.side == "long" else -1.0
        moved = (quote.last - position.entry_price) * sign
        if moved < cfg.scale_in_atr * position.atr:
            continue
        side = _order_side(position.side)
        priced = price_entry(side, quote, session, cfg)
        if not priced.ok or priced.limit_price is None:
            continue
        reserve = position.tranche2_qty * position.r
        if risk_used + reserve > cfg.daily_new_risk_pct * equity + 1e-9:
            updated = position.evolve(tranche2_qty=0, tranche2_sent=True)
            plan.position_edits[ticker] = updated
            positions[ticker] = updated
            plan.decisions.append(
                make_decision(
                    now, ticker=ticker, action="skip", reason="scale_in_budget",
                    session=session, mode=mode, signal_id=position.signal_id,
                )
            )
            continue
        oid = client_order_id(cfg.order_id_prefix, "scale_in", ticker, position.signal_id, 2, 0)
        order = OrderRequest(
            client_order_id=oid, ticker=ticker, side=side, qty=position.tranche2_qty,
            limit_price=priced.limit_price, extended_hours=priced.extended_hours,
            time_in_force="day", order_type="limit",
        )
        effect = FillPlan(
            purpose="scale_in", direction=position.side, score=position.entry_score,
            atr=position.atr, signal_id=position.signal_id, target_qty=position.target_qty,
            reserve_risk=reserve, entry_session=position.entry_session,
            credibility=position.credibility,
        )
        working = WorkingOrder(
            client_order_id=oid, ticker=ticker, side=side, qty=position.tranche2_qty,
            limit_price=priced.limit_price, submitted_at=now,
            decision_mid=priced.mid or (quote.bid + quote.ask) / 2,
            replaces=0, session=session, purpose="scale_in", signal_id=position.signal_id,
            tranche=2, effect=effect,
        )
        updated = position.evolve(tranche2_sent=True)
        plan.position_edits[ticker] = updated
        positions[ticker] = updated
        risk_used += reserve
        plan.intents.append(Intent("submit", "scale_in", order=order, working=working, effect=effect))
        plan.decisions.append(
            make_decision(
                now, ticker=ticker, action="take", reason="scale_in", session=session, mode=mode,
                signal_id=position.signal_id, score=position.entry_score, qty=position.tranche2_qty,
                limit_price=priced.limit_price,
                details={"extended_hours": priced.extended_hours, "order_type": "limit"},
            )
        )
    plan.risk_used = risk_used
    risk_used_box[0] = risk_used


def _plan_entries(
    book, positions, plan, now, signals, market, cfg, session, mode, equity,
    exiting_full, signal_exit_score, risk_used, trades,
):
    banned = ambiguous_ids(signals, now, cfg, market.confirmations)
    ordered = sorted(signals, key=lambda signal: (-signal.score, signal.ticker, signal.signal_id))
    cycle_takes = list(book.takes)
    accepted_tickers: set[str] = set()
    working_ids = {order.signal_id for order in book.working.values()}
    working_entry_tickers = {
        order.ticker
        for order in book.working.values()
        if order.purpose in ("entry", "scale_in") and order.client_order_id not in plan.drop_working
    }

    for signal in ordered:
        base_details: dict = {}
        qual_seen = (
            signal.signal_id in book.seen
            or signal.signal_id in working_ids
            or signal.signal_id in plan.seen_add
        )
        if signal.ticker in working_entry_tickers or signal.ticker in accepted_tickers:
            plan.decisions.append(_skip(signal, now, session, mode, "working_order", base_details))
            continue

        held = positions.get(signal.ticker)
        if held is not None and signal.ticker in exiting_full:
            held_for_qual = None
        else:
            held_for_qual = held
        if held_for_qual is not None and held_for_qual.adopted and not cfg.manage_adopted_positions:
            plan.decisions.append(_skip(signal, now, session, mode, "adopted_unmanaged", base_details))
            continue

        takes = tuple(
            item for item in cycle_takes
            if not (item.ticker == signal.ticker and signal.ticker in signal_exit_score)
        )
        qual = qualify_signal(
            signal, now, cfg,
            confirmation=market.confirmations.get(signal.ticker),
            already_seen=qual_seen,
            takes=takes,
            held_direction=None if held_for_qual is None else held_for_qual.side,
            held_score=None if held_for_qual is None else held_for_qual.entry_score,
            ambiguous=signal.signal_id in banned,
        )
        if qual.terminal:
            plan.seen_add.append(signal.signal_id)
        if qual.outcome == "reject":
            plan.decisions.append(_skip(
                signal, now, session, mode, qual.reason,
                {"credibility": qual.credibility},
            ))
            continue

        if signal.ticker in signal_exit_score:
            needed = signal_exit_score[signal.ticker] + cfg.conflict_margin
            if signal.score < cfg.flip_min_score or signal.score < needed:
                plan.decisions.append(_skip(signal, now, session, mode, "conflict_no_flip", {}))
                continue

        clock = must_flatten_for_clock(
            now, signal.score,
            overnight_min_score=cfg.overnight_min_score,
            minutes_before_close=cfg.flatten_before_close_minutes,
            holidays=set(market.holidays), early_closes=market.early_closes,
        )
        if session == "closed":
            plan.decisions.append(_skip(signal, now, session, mode, "session_closed", {}))
            continue
        if clock:
            reason = "weekend_flat" if clock == "weekend_flat" else "after_flatten_cutoff"
            plan.decisions.append(_skip(signal, now, session, mode, reason, {}))
            continue

        blocked = entry_event_block(now, signal.ticker, market.events, cfg)
        if blocked:
            plan.decisions.append(_skip(signal, now, session, mode, blocked, {}))
            continue

        quote = market.quotes.get(signal.ticker)
        vol = market.vols.get(signal.ticker)
        skip = market_skip_reason(qual.direction, quote, vol, cfg)
        if skip:
            plan.decisions.append(_skip(signal, now, session, mode, skip, {}))
            continue
        if vol is None or vol.atr <= 0:
            plan.decisions.append(_skip(signal, now, session, mode, "no_volatility", {}))
            continue

        open_names = {
            ticker for ticker, position in positions.items()
            if position.qty > 0 and ticker not in exiting_full
        }
        open_names |= working_entry_tickers
        open_names |= accepted_tickers
        if signal.ticker not in open_names and len(open_names) >= cfg.max_positions:
            plan.decisions.append(_skip(signal, now, session, mode, "max_positions", {}))
            continue

        exposures = _exposures(positions, market, {signal.ticker} | exiting_full)
        sized = size_target_with_cluster(
            score=signal.score, credibility=qual.credibility, session=session, equity=equity,
            price=quote.last, atr=vol.atr, direction=qual.direction, ticker=signal.ticker,
            sector=vol.sector, exposures=exposures, correlations=market.correlations,
            trades_today=trades, new_risk_today=risk_used, cfg=cfg,
        )
        if sized.qty < 1:
            plan.decisions.append(_skip(signal, now, session, mode, sized.binding, {
                "conviction": sized.conviction, "credibility": qual.credibility,
            }))
            continue
        held_qty = held_for_qual.qty if held_for_qual is not None and held_for_qual.side == qual.direction else 0
        additional = sized.qty - held_qty
        if additional < 1:
            plan.decisions.append(_skip(signal, now, session, mode, "already_sized", {
                "target_qty": sized.qty, "held_qty": held_qty,
            }))
            continue

        first, second = split_tranches(additional, session, cfg)
        side = _order_side(qual.direction)
        priced = price_entry(side, quote, session, cfg)
        if not priced.ok or priced.limit_price is None or priced.mid is None:
            plan.decisions.append(_skip(signal, now, session, mode, priced.reason or "bad_quote", {}))
            continue

        distance = stop_distance(quote.last, vol.atr, cfg)
        reserve = first * distance
        oid = client_order_id(cfg.order_id_prefix, "entry", signal.ticker, signal.signal_id, 1, 0)
        order = OrderRequest(
            client_order_id=oid, ticker=signal.ticker, side=side, qty=first,
            limit_price=priced.limit_price, extended_hours=priced.extended_hours,
            time_in_force=priced.time_in_force, order_type="limit",
        )
        deadline = now + timedelta(minutes=cfg.scale_in_window_minutes) if second else None
        effect = FillPlan(
            purpose="entry", direction=qual.direction, score=signal.score, atr=vol.atr,
            signal_id=signal.signal_id, target_qty=(held_qty + additional),
            tranche2_qty=second, entry_session=session, credibility=qual.credibility,
            reserve_risk=reserve, count_trade=1, tranche2_deadline=deadline,
        )
        working = WorkingOrder(
            client_order_id=oid, ticker=signal.ticker, side=side, qty=first,
            limit_price=priced.limit_price, submitted_at=now, decision_mid=priced.mid,
            replaces=0, session=session, purpose="entry", signal_id=signal.signal_id,
            tranche=1, effect=effect,
        )
        reason = "flip" if signal.ticker in signal_exit_score else qual.reason
        plan.intents.append(Intent("submit", reason, order=order, working=working, effect=effect))
        plan.decisions.append(
            make_decision(
                now, ticker=signal.ticker, action="take", reason=reason, session=session, mode=mode,
                signal_id=signal.signal_id, score=signal.score, sentiment=signal.sentiment,
                qty=first, limit_price=priced.limit_price,
                details={
                    "headline": signal.headline[:180],
                    "conviction": round(sized.conviction, 6),
                    "credibility": qual.credibility,
                    "binding": sized.binding,
                    "target_qty": held_qty + additional,
                    "tranche2_qty": second,
                    "stop_distance": distance,
                    "extended_hours": priced.extended_hours,
                    "time_in_force": priced.time_in_force,
                    "order_type": "limit",
                    "session_mult": sized.session_mult,
                },
            )
        )
        risk_used += reserve
        trades += 1
        accepted_tickers.add(signal.ticker)
        cycle_takes.append(
            TakeRecord(signal.ticker, qual.direction, signal.score, now, signal.signal_id)
        )

    plan.risk = plan.risk.evolve(new_risk_today=risk_used, trades_today=trades)
