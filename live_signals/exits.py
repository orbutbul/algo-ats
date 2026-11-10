"""Exit decisions.

Execution role: given a position, the latest bar, the clock, and any fresh
signals, decide whether to hold, tighten the engine-managed stop, or sell
(cover) some shares. Broker trailing stops are not used. The trail lives
here because it has to work in extended hours, where Alpaca will not trigger
a trailing stop.

Intrabar order is conservative. The existing stop is checked against the bar
before the watermark is updated, so a bar that makes a new high and a new low
cannot trail itself into a stop that did not exist at the open of the bar.
A stop on that bar wins over a take-profit on the same bar.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from live_signals.config import Config
from live_signals.events import position_event_action, tightened_stop
from live_signals.models import Confirmation, NewsSignal, PositionState, PriceBar, ScheduledEvent
from live_signals.pricing import round_shares
from live_signals.qualify import is_confirmed, source_block_reason, source_facts
from live_signals.sessions import must_flatten_for_clock, session_hours_between


class ExitDecision:
    """`action` is `hold` or `exit`.

    A macro tighten is a hold whose stop moved and whose reason is
    `macro_tighten`. Take-profit exits set `set_tp1` / `set_tp2` so the fill
    can remember the scale-out; the flags are not set until the fill happens.
    """

    def __init__(
        self,
        action: str,
        reason: str,
        qty: int,
        position: PositionState,
        *,
        set_tp1: bool = False,
        set_tp2: bool = False,
    ) -> None:
        self.action = action
        self.reason = reason
        self.qty = qty
        self.position = position
        self.set_tp1 = set_tp1
        self.set_tp2 = set_tp2


def _stop_hit(position: PositionState, bar: PriceBar) -> bool:
    if position.side == "long":
        return bar.low <= position.stop + 1e-9
    return bar.high + 1e-9 >= position.stop


def _favorable(position: PositionState, price: float) -> float:
    if position.r <= 0:
        return 0.0
    if position.side == "long":
        return (price - position.entry_price) / position.r
    return (position.entry_price - price) / position.r


def _trailed(position: PositionState) -> bool:
    if position.side == "long":
        return position.stop > position.initial_stop + 1e-6
    return position.stop < position.initial_stop - 1e-6


def _reached(position: PositionState, bar: PriceBar, multiple: float) -> bool:
    if position.side == "long":
        return bar.high + 1e-9 >= position.entry_price + multiple * position.r
    return bar.low <= position.entry_price - multiple * position.r + 1e-9


def _signal_exit(
    position: PositionState,
    signals: list[NewsSignal],
    now: datetime,
    cfg: Config,
    confirmations: dict[str, Confirmation] | None,
) -> str | None:
    """Flatten on a fresh neutral print, or on an opposite print that would itself qualify.

    The opposite print has to clear the same source gate as a new entry, so a
    single unconfirmed Reddit post cannot knock out a position. It also has
    to score at least as high as the position (and at least `opposite_exit_score`).
    A weaker opposite print is a conflict, not an exit; qualify rejects it.
    """
    fresh_cutoff = timedelta(minutes=cfg.freshness_minutes)
    for signal in signals:
        if signal.ticker != position.ticker:
            continue
        age = now - signal.timestamp
        if age < -timedelta(minutes=cfg.clock_skew_minutes) or age > fresh_cutoff:
            continue
        # A fresh print under hold_until_score closes the position whether or
        # not the wording is still bullish. Above that, the position stays.
        if signal.score < cfg.hold_until_score:
            return "score_below_hold"
        if signal.direction == "neutral":
            continue
        if signal.direction not in ("long", "short") or signal.direction == position.side:
            continue
        n_domains, institutional = source_facts(signal, cfg)
        confirmation = None if confirmations is None else confirmations.get(signal.ticker)
        confirmed = is_confirmed(confirmation, signal.direction, cfg)
        if source_block_reason(n_domains, institutional, confirmed, signal.score, cfg):
            continue
        if signal.score >= max(cfg.opposite_exit_score, position.entry_score):
            return "signal_reversed"
    return None


def _tight_trail(position: PositionState, cfg: Config) -> bool:
    """Score at or below the overnight threshold uses a 2% trail from the high."""
    return position.entry_score <= cfg.overnight_min_score


def _update_trail(position: PositionState, bar: PriceBar, cfg: Config) -> PositionState:
    if position.side == "long":
        extreme = bar.high
        mfe = max(position.mfe_r, _favorable(position, extreme))
        watermark = max(position.watermark, extreme)
        armed = position.trail_armed or mfe >= cfg.trail_arm_r or _tight_trail(position, cfg)
        stop = position.stop
        if _tight_trail(position, cfg):
            stop = max(stop, watermark * (1 - cfg.low_score_trail_pct))
        elif armed and position.atr > 0:
            distance = max(cfg.trail_atr_mult * position.atr, cfg.trail_min_fraction_of_r * position.r)
            stop = max(stop, watermark - distance)
        return position.evolve(mfe_r=mfe, watermark=watermark, trail_armed=armed, stop=stop)
    extreme = bar.low
    mfe = max(position.mfe_r, _favorable(position, extreme))
    watermark = min(position.watermark, extreme)
    armed = position.trail_armed or mfe >= cfg.trail_arm_r or _tight_trail(position, cfg)
    stop = position.stop
    if _tight_trail(position, cfg):
        stop = min(stop, watermark * (1 + cfg.low_score_trail_pct))
    elif armed and position.atr > 0:
        distance = max(cfg.trail_atr_mult * position.atr, cfg.trail_min_fraction_of_r * position.r)
        stop = min(stop, watermark + distance)
    return position.evolve(mfe_r=mfe, watermark=watermark, trail_armed=armed, stop=stop)


def _slice(basis: int, fraction: float, current: int) -> int:
    """Shares to sell for one take-profit level, leaving at least one."""
    if current < 2 or basis < 1:
        return 0
    raw = max(1, round_shares(basis * fraction))
    return min(raw, current - 1)


def evaluate_exit(
    position: PositionState,
    bar: PriceBar,
    now: datetime,
    signals: list[NewsSignal],
    events: tuple[ScheduledEvent, ...],
    cfg: Config,
    *,
    holidays: set | frozenset | None = None,
    early_closes: dict | None = None,
    last_price: float | None = None,
    confirmations: dict[str, Confirmation] | None = None,
) -> ExitDecision:
    holidays = holidays or set()
    if _stop_hit(position, bar):
        reason = "trail_stop" if _trailed(position) else "stop"
        return ExitDecision("exit", reason, position.qty, position)

    clock = must_flatten_for_clock(
        now,
        position.entry_score,
        overnight_min_score=cfg.overnight_min_score,
        minutes_before_close=cfg.flatten_before_close_minutes,
        holidays=holidays,
        early_closes=early_closes,
    )
    if clock:
        return ExitDecision("exit", clock, position.qty, position)

    event_action = position_event_action(now, position, events, cfg)
    if event_action in ("earnings_flat", "macro_flat"):
        return ExitDecision("exit", event_action, position.qty, position)

    signal_reason = _signal_exit(position, signals, now, cfg, confirmations)
    if signal_reason:
        return ExitDecision("exit", signal_reason, position.qty, position)

    extreme = bar.high if position.side == "long" else bar.low
    mfe_now = max(position.mfe_r, _favorable(position, extreme))
    if cfg.time_exits_enabled:
        # Off by default. A position stays open until a fresh score is under
        # hold_until_score, or a stop / daily-loss halt fires.
        wall_hours = (now - position.entry_time).total_seconds() / 3600
        if wall_hours >= cfg.max_hold_hours:
            return ExitDecision("exit", "time_stop", position.qty, position.evolve(mfe_r=mfe_now))
        open_hours = session_hours_between(position.entry_time, now, holidays, early_closes)
        if open_hours + 1e-6 >= cfg.alpha_decay_hours and mfe_now < cfg.alpha_decay_min_mfe_r:
            return ExitDecision("exit", "alpha_decay", position.qty, position.evolve(mfe_r=mfe_now))

    updated = _update_trail(position, bar, cfg)
    reason = "hold"
    if event_action == "macro_tighten":
        tightened = tightened_stop(updated, cfg)
        if abs(tightened - updated.stop) > 1e-9:
            updated = updated.evolve(stop=tightened)
            reason = "macro_tighten"

    basis = position.basis_qty or position.qty
    if position.qty < cfg.scale_out_min_qty:
        if _favorable(updated, extreme) >= cfg.single_target_r:
            return ExitDecision("exit", "take_profit", position.qty, updated)
        return ExitDecision("hold", reason, 0, updated)

    if _reached(updated, bar, cfg.tp2_r) and not position.tp2_done:
        first = 0 if position.tp1_done else _slice(basis, cfg.tp1_fraction, position.qty)
        remaining = position.qty - first
        second = _slice(basis, cfg.tp2_fraction, remaining) if remaining >= 2 else 0
        total = first + second
        if total >= 1:
            return ExitDecision(
                "exit", "take_profit", total, updated,
                set_tp1=not position.tp1_done, set_tp2=True,
            )
    if _reached(updated, bar, cfg.tp1_r) and not position.tp1_done:
        qty = _slice(basis, cfg.tp1_fraction, position.qty)
        if qty >= 1:
            return ExitDecision("exit", "take_profit", qty, updated, set_tp1=True)
    return ExitDecision("hold", reason, 0, updated)
