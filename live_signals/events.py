"""Earnings and macro-event gates.

Execution role: block new risk around a scheduled print, and tell an open
position whether to flatten or to tighten. The calendar is supplied by the
caller. CPI and FOMC are both `macro` (kind may also be the literal `cpi` or
`fomc`). This module does not download a calendar.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from live_signals.config import Config
from live_signals.models import PositionState, ScheduledEvent

_MACRO = frozenset({"macro", "cpi", "fomc"})


def _kind(event: ScheduledEvent) -> str:
    return event.kind.lower().strip()


def entry_event_block(
    now: datetime,
    ticker: str,
    events: tuple[ScheduledEvent, ...],
    cfg: Config,
) -> str | None:
    for event in events:
        kind = _kind(event)
        if kind == "earnings" and (event.ticker or "").upper() == ticker.upper():
            start = event.at - timedelta(hours=cfg.earnings_entry_blackout_before_hours)
            end = event.at + timedelta(hours=cfg.earnings_entry_blackout_after_hours)
            if start <= now <= end:
                return "earnings_blackout"
        if kind in _MACRO:
            start = event.at - timedelta(minutes=cfg.macro_entry_blackout_before_minutes)
            end = event.at + timedelta(minutes=cfg.macro_entry_blackout_after_minutes)
            if start <= now <= end:
                return "macro_blackout"
    return None


def position_event_action(
    now: datetime,
    position: PositionState,
    events: tuple[ScheduledEvent, ...],
    cfg: Config,
) -> str | None:
    """`earnings_flat`, `macro_flat`, `macro_tighten`, or None.

    Earnings flatten wins over a macro tighten when both are active. A macro
    position at or above `macro_hold_min_score` is held with a tighter stop
    instead of being flattened. New entries are still blocked in that window.
    """
    earnings = False
    macro_flat = False
    macro_tighten = False
    for event in events:
        kind = _kind(event)
        if kind == "earnings" and (event.ticker or "").upper() == position.ticker.upper():
            start = event.at - timedelta(hours=cfg.earnings_flatten_before_hours)
            end = event.at + timedelta(minutes=cfg.earnings_flatten_after_minutes)
            if start <= now <= end:
                earnings = True
        if kind in _MACRO:
            if event.at - timedelta(minutes=cfg.macro_flatten_before_minutes) <= now <= event.at:
                if position.entry_score < cfg.macro_hold_min_score:
                    macro_flat = True
            if event.at - timedelta(minutes=cfg.macro_flatten_before_minutes) <= now <= event.at + timedelta(
                minutes=cfg.macro_entry_blackout_after_minutes
            ):
                if position.entry_score >= cfg.macro_hold_min_score:
                    macro_tighten = True
    if earnings:
        return "earnings_flat"
    if macro_flat:
        return "macro_flat"
    if macro_tighten:
        return "macro_tighten"
    return None


def tightened_stop(position: PositionState, cfg: Config) -> float:
    """Pull the stop in to `macro_tighten_r` * R, and never loosen it."""
    if position.side == "long":
        tight = position.entry_price - cfg.macro_tighten_r * position.r
        return max(position.stop, tight)
    tight = position.entry_price + cfg.macro_tighten_r * position.r
    return min(position.stop, tight)
