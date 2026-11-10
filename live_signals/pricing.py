"""Entry pricing and skip checks that depend on the quote.

Execution role: build a limit price from the NBBO, or refuse the entry when
the spread, halt, borrow, or already-realized gap makes the print untradable.
Extended hours are limit-only, whole-share, time-in-force day. The engine
never emits a market, bracket, notional, or broker trailing-stop order.
Chase is measured from the decision mid captured when the order was first
priced, so a cancel/replace cannot follow a spike.
"""

from __future__ import annotations

import math

from live_signals.config import Config
from live_signals.models import PriceDecision, Quote, VolSnapshot

_OPEN = ("rth", "pre", "after", "overnight")


def round_penny(price: float) -> float:
    return math.floor(price * 100 + 0.5) / 100.0


def round_shares(value: float) -> int:
    return int(math.floor(value + 0.5 + 1e-9))


def extended_hours_for(session: str) -> bool:
    return session in ("pre", "after", "overnight")


def split_tranches(target: int, session: str, cfg: Config) -> tuple[int, int]:
    """First clip and optional second clip. Outside RTH and premarket, one clip."""
    if target <= 1 or session not in cfg.scale_in_sessions:
        return target, 0
    first = max(1, round_shares(target * cfg.scale_in_first_fraction))
    first = min(first, target)
    return first, target - first


def market_skip_reason(
    direction: str,
    quote: Quote | None,
    vol: VolSnapshot | None,
    cfg: Config,
) -> str | None:
    if quote is None:
        return "no_quote"
    if quote.halted:
        return "halted"
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid or quote.last <= 0:
        return "bad_quote"
    if direction == "short" and (not quote.shortable or not quote.easy_to_borrow):
        return "not_shortable"
    if vol is not None and vol.atr > 0 and vol.prev_rth_close > 0:
        sign = 1.0 if direction == "long" else -1.0
        extension = (quote.last - vol.prev_rth_close) * sign
        if extension > cfg.max_gap_atr * vol.atr:
            return "gap_already_extended"
    return None


def price_entry(
    side: str,
    quote: Quote,
    session: str,
    cfg: Config,
    decision_mid: float | None = None,
) -> PriceDecision:
    """Limit price for an entry.

    Buy: min(ask, max(mid * (1 + offset), bid)), then capped at
    decision_mid * (1 + max_chase).
    Sell: the mirror, floored at decision_mid * (1 - max_chase).
    `decision_mid` defaults to the current mid. Pass the original mid when
    replacing, so the cap does not walk up with the quote.
    """
    refused = PriceDecision(False, "", None, extended_hours_for(session), "day", "limit", None)
    if session not in _OPEN:
        return PriceDecision(False, "session_closed", None, False, "day", "limit", None)
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        return PriceDecision(False, "bad_quote", None, extended_hours_for(session), "day", "limit", None)
    mid = (quote.bid + quote.ask) / 2
    if mid <= 0:
        return PriceDecision(False, "bad_quote", None, extended_hours_for(session), "day", "limit", None)
    spread = (quote.ask - quote.bid) / mid
    if spread > cfg.max_spread_pct[session]:
        return PriceDecision(False, "spread_too_wide", None, extended_hours_for(session), "day", "limit", mid)

    anchor = decision_mid if decision_mid is not None else mid
    offset = cfg.entry_offset[session]
    chase = cfg.max_chase[session]
    if side == "buy":
        generous = mid * (1 + offset)
        limit = min(quote.ask, max(generous, quote.bid))
        limit = min(limit, anchor * (1 + chase))
    elif side == "sell":
        generous = mid * (1 - offset)
        limit = max(quote.bid, min(generous, quote.ask))
        limit = max(limit, anchor * (1 - chase))
    else:
        return PriceDecision(False, "bad_side", None, extended_hours_for(session), "day", "limit", mid)
    if limit <= 0:
        return PriceDecision(False, "bad_quote", None, extended_hours_for(session), "day", "limit", mid)
    return PriceDecision(
        ok=True,
        reason="priced",
        limit_price=round_penny(limit),
        extended_hours=extended_hours_for(session),
        time_in_force="day",
        order_type="limit",
        mid=mid,
    )


def marketable_fill(side: str, limit: float, quote: Quote) -> float | None:
    """Fill price if a limit is marketable against the quote, else None.

    Buys fill at the ask when the limit is at or through it. Sells fill at
    the bid. A halted name does not fill.
    """
    if quote.halted or quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        return None
    if side == "buy" and limit + 1e-9 >= quote.ask:
        return quote.ask
    if side == "sell" and limit - 1e-9 <= quote.bid:
        return quote.bid
    return None


def exit_limit(order_side: str, quote: Quote) -> float | None:
    """Marketable limit at the touch. Exits do not skip a wide spread.

    Getting flat matters more than saving the spread. Entries still refuse
    wide markets; exits cross to the bid (selling) or ask (covering).
    """
    if order_side == "sell" and quote.bid > 0:
        return round_penny(quote.bid)
    if order_side == "buy" and quote.ask > 0:
        return round_penny(quote.ask)
    return None


def review_resting_entry(
    *,
    age_seconds: float,
    replaces: int,
    order_session: str,
    current_session: str,
    side: str,
    quote: Quote | None,
    decision_mid: float,
    current_limit: float,
    cfg: Config,
) -> str | float:
    """Return 'hold', 'cancel', or a new limit price.

    One replace is allowed, and only inside the original chase cap. A session
    change cancels: a premarket day-order must not be silently re-parameterized
    as a regular-hours order by this engine (the broker may keep it; we pull it).
    """
    if current_session != order_session or current_session not in _OPEN:
        return "cancel"
    timeout = cfg.cancel_replace_seconds[order_session]
    if age_seconds < timeout:
        return "hold"
    if replaces >= cfg.max_replaces or quote is None:
        return "cancel"
    priced = price_entry(side, quote, current_session, cfg, decision_mid=decision_mid)
    if not priced.ok or priced.limit_price is None:
        return "cancel"
    if abs(priced.limit_price - current_limit) < 0.009:
        return "cancel"
    return priced.limit_price
