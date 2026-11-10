"""Position sizing.

Execution role: turn a qualified score into a whole-share quantity. The size
is a volatility budget (dollars at risk if the initial stop is hit), scaled by
a logistic conviction curve, a credibility multiplier, and a session-liquidity
multiplier, then cut by the hard caps. Quarter-Kelly is a ceiling, not the
target: with the default 0.60% base risk it does not bind.

The function sizes the *target* position against the rest of the book, as if
any current shares in this ticker were absent. The planner orders
`target - shares_already_held`.
"""

from __future__ import annotations

import math

from live_signals.config import Config
from live_signals.models import Exposure, SizeResult


def conviction(score: float, cfg: Config) -> float:
    """Logistic conviction on [min_score, score_hi], pinned to the floor at the threshold.

    Let L(s) = 1 / (1 + exp(-k * (s - mid))).
    t(s) = (L(s) - L(lo)) / (L(hi) - L(lo))
    conviction(s) = 0                         if s < lo
                   = 1                         if s >= hi
                   = floor + (1 - floor) * t   otherwise

    Defaults: k = 0.18, mid = 82, lo = 70, hi = 95, floor = 0.15.
    The curve is convex from the threshold through the midpoint (a 70 is a
    small position, an 80 is several times larger) and flattens into 95 so a
    perfect score cannot size past the caps by itself.
    """
    if score < cfg.conviction_score_lo:
        return 0.0
    if score >= cfg.conviction_score_hi:
        return 1.0

    def logistic(point: float) -> float:
        return 1.0 / (1.0 + math.exp(-cfg.conviction_k * (point - cfg.conviction_mid)))

    lo = logistic(cfg.conviction_score_lo)
    hi = logistic(cfg.conviction_score_hi)
    span = hi - lo
    if span <= 0:
        return cfg.conviction_floor
    t = (logistic(score) - lo) / span
    t = min(1.0, max(0.0, t))
    return cfg.conviction_floor + (1.0 - cfg.conviction_floor) * t


def kelly_risk_cap(score_conviction: float, cfg: Config) -> float:
    """Quarter-Kelly risk fraction, hard-capped at `max_kelly_risk_pct`.

    p = p_lo + p_span * conviction
    full_kelly = p - (1 - p) / b
    cap = min(kelly_fraction * max(full_kelly, 0), max_kelly_risk_pct)

    p is an assumed win rate, not a fitted probability. It exists so a future
    edit that raises `base_risk_pct` still cannot put more than 1% of equity
    at risk on one trade. At the default base of 0.60% the vol budget is the
    tighter number.
    """
    p = cfg.kelly_p_lo + cfg.kelly_p_span * score_conviction
    full = p - (1.0 - p) / cfg.kelly_payoff_b
    fractional = cfg.kelly_fraction * max(full, 0.0)
    return min(fractional, cfg.max_kelly_risk_pct)


def stop_distance(price: float, atr: float, cfg: Config) -> float:
    """Initial risk per share: max(atr_stop_mult * ATR, min_stop_pct * price)."""
    return max(cfg.atr_stop_mult * atr, cfg.min_stop_pct * price)


def _floor_shares(value: float) -> int:
    if value <= 0:
        return 0
    return int(math.floor(value + 1e-9))


def _corr(correlations: dict[tuple[str, str], float], left: str, right: str) -> float | None:
    if (left, right) in correlations:
        return correlations[(left, right)]
    return correlations.get((right, left))


def size_target(
    *,
    score: float,
    credibility: float,
    session: str,
    equity: float,
    price: float,
    atr: float,
    direction: str,
    ticker: str,
    exposures: list[Exposure],
    correlations: dict[tuple[str, str], float],
    trades_today: int,
    new_risk_today: float,
    cfg: Config,
) -> SizeResult:
    """Whole-share target for a new position. `exposures` must exclude `ticker`."""
    zero = SizeResult(0, 0.0, credibility, 0.0, 0.0, 0.0, 0.0, 0.0, "zero")
    if session == "closed" or price < 1 or equity <= 0 or atr <= 0 or direction not in ("long", "short"):
        binding = "session_closed" if session == "closed" else "bad_market"
        if price < 1:
            binding = "price_below_one_dollar"
        return SizeResult(0, conviction(score, cfg), credibility, cfg.session_risk_mult.get(session, 0.0),
                          0.0, 0.0, 0.0, 0.0, binding)
    if trades_today >= cfg.max_trades_per_day:
        return SizeResult(0, conviction(score, cfg), credibility, cfg.session_risk_mult[session],
                          0.0, 0.0, 0.0, 0.0, "max_trades")

    score_conviction = conviction(score, cfg)
    if score_conviction <= 0:
        return SizeResult(0, 0.0, credibility, cfg.session_risk_mult[session], 0.0, 0.0, 0.0, 0.0,
                          "below_threshold")

    session_mult = float(cfg.session_risk_mult[session])
    distance = stop_distance(price, atr, cfg)
    vol_risk_pct = cfg.base_risk_pct * score_conviction * credibility * session_mult
    kelly_cap = kelly_risk_cap(score_conviction, cfg)
    risk_pct = min(vol_risk_pct, kelly_cap)
    risk_dollars = risk_pct * equity
    remaining = cfg.daily_new_risk_pct * equity - new_risk_today
    if remaining <= 0:
        return SizeResult(0, score_conviction, credibility, session_mult, distance, 0.0, 0.0, 0.0,
                          "daily_risk_budget")
    budget_dollars = min(risk_dollars, remaining)

    candidates: list[tuple[int, str]] = []
    qty_risk = _floor_shares(budget_dollars / distance)
    binding = "kelly" if kelly_cap < vol_risk_pct - 1e-12 else "risk_budget"
    if budget_dollars < risk_dollars - 1e-9:
        binding = "daily_risk_budget"
    candidates.append((qty_risk, binding))

    position_room = cfg.session_position_cap[session] * equity
    candidates.append((_floor_shares(position_room / price), "position_cap"))

    long_notional = sum(item.notional for item in exposures if item.side == "long")
    short_notional = sum(item.notional for item in exposures if item.side == "short")
    gross_now = long_notional + short_notional
    gross_room = cfg.gross_exposure_cap * equity - gross_now
    candidates.append((_floor_shares(gross_room / price), "gross"))

    if direction == "long":
        net_room = cfg.net_exposure_cap * equity - long_notional + short_notional
    else:
        net_room = cfg.net_exposure_cap * equity - short_notional + long_notional
    candidates.append((_floor_shares(net_room / price), "net"))

    qty, bound = min(candidates, key=lambda item: (item[0], item[1]))
    if qty < 1:
        return SizeResult(0, score_conviction, credibility, session_mult, distance, 0.0, risk_pct, 0.0, bound)
    notional = qty * price
    return SizeResult(
        qty=qty,
        conviction=score_conviction,
        credibility=credibility,
        session_mult=session_mult,
        stop_distance=distance,
        risk_dollars=qty * distance,
        risk_pct=risk_pct,
        notional=notional,
        binding=bound,
    )


def size_target_with_cluster(
    *,
    score: float,
    credibility: float,
    session: str,
    equity: float,
    price: float,
    atr: float,
    direction: str,
    ticker: str,
    sector: str | None,
    exposures: list[Exposure],
    correlations: dict[tuple[str, str], float],
    trades_today: int,
    new_risk_today: float,
    cfg: Config,
) -> SizeResult:
    """`size_target` plus sector and correlation-cluster caps.

    Names with no sector are exempt from the sector cap (missing metadata must
    not freeze the book). A pair with no correlation entry is exempt from the
    cluster cap for the same reason. When a correlation is present and at or
    above `corr_threshold`, the names share a notional cap.
    """
    base = size_target(
        score=score,
        credibility=credibility,
        session=session,
        equity=equity,
        price=price,
        atr=atr,
        direction=direction,
        ticker=ticker,
        exposures=exposures,
        correlations=correlations,
        trades_today=trades_today,
        new_risk_today=new_risk_today,
        cfg=cfg,
    )
    if base.qty < 1 or price <= 0:
        return base

    limits: list[tuple[int, str]] = [(base.qty, base.binding)]
    if sector:
        used = sum(item.notional for item in exposures if item.sector == sector)
        room = cfg.sector_cap * equity - used
        limits.append((_floor_shares(room / price), "sector"))

    if correlations:
        used = 0.0
        linked = False
        for item in exposures:
            value = _corr(correlations, ticker, item.ticker)
            if value is not None and value >= cfg.corr_threshold:
                used += item.notional
                linked = True
        if linked:
            room = cfg.corr_cluster_cap * equity - used
            limits.append((_floor_shares(room / price), "cluster"))

    qty, bound = min(limits, key=lambda item: item[0])
    if qty < 1:
        return SizeResult(
            0, base.conviction, base.credibility, base.session_mult, base.stop_distance,
            0.0, base.risk_pct, 0.0, bound,
        )
    return SizeResult(
        qty=qty,
        conviction=base.conviction,
        credibility=base.credibility,
        session_mult=base.session_mult,
        stop_distance=base.stop_distance,
        risk_dollars=qty * base.stop_distance,
        risk_pct=base.risk_pct,
        notional=qty * price,
        binding=bound,
    )
