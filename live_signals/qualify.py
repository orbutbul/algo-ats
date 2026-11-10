"""Signal qualification.

Execution role: decide whether a Wirewatch signal is allowed to open or add
risk, and the credibility multiplier that sizing will apply. Neutral never
trades. A single social source (the failure mode of one Reddit post at score
70) needs either a much higher score or price-and-volume confirmation.
The function is pure: time, confirmation, and recent takes are arguments.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlparse

from live_signals.config import Config
from live_signals.models import Confirmation, NewsSignal, Qualification, TakeRecord

# Reasons that will not change if the same payload is seen again this window.
TERMINAL_REASONS = frozenset(
    {
        "neutral",
        "invalid_sentiment",
        "score_below_min",
        "stale",
        "future_timestamp",
        "no_sources",
        "single_social_unconfirmed",
        "social_needs_confirmation",
        "thin_single_source",
        "cooldown",
        "ambiguous_conflict",
        "duplicate",
    }
)


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def source_facts(signal: NewsSignal, cfg: Config) -> tuple[int, bool]:
    """Return (distinct domains, has_institutional_source).

    A source type that is neither institutional nor social counts as social.
    That is the stricter bucket: an unlabeled scrape cannot clear the news bar.
    """
    institutional = {item.lower() for item in cfg.institutional_source_types}
    domains: set[str] = set()
    has_institutional = False
    for source in signal.sources:
        kind = source.source_type.lower().strip()
        if kind in institutional:
            has_institutional = True
        host = _host(source.url)
        domains.add(host or f"unknown:{kind or 'blank'}")
    return len(domains), has_institutional


def is_confirmed(confirmation: Confirmation | None, direction: str, cfg: Config) -> bool:
    if confirmation is None or direction not in ("long", "short"):
        return False
    signed = confirmation.return_since_signal if direction == "long" else -confirmation.return_since_signal
    return signed >= cfg.confirm_min_return and confirmation.rel_volume >= cfg.confirm_min_rel_volume


def credibility_multiplier(
    n_domains: int,
    has_institutional: bool,
    confirmed: bool,
    cfg: Config,
) -> float:
    """Piecewise diversity score in [cred_min, cred_max].

    c = cred_base + cred_per_extra_domain * (n - 1)
        + cred_institutional_bonus if any news/filing/wire/press source
        + cred_confirmation_bonus if price and volume confirmed
    Defaults: one social source is 0.45, one news source is 0.60, two news
    sources are 0.78, and three confirmed news sources saturate at 1.00.
    """
    if n_domains <= 0:
        return 0.0
    value = cfg.cred_base + cfg.cred_per_extra_domain * (n_domains - 1)
    if has_institutional:
        value += cfg.cred_institutional_bonus
    if confirmed:
        value += cfg.cred_confirmation_bonus
    return min(cfg.cred_max, max(cfg.cred_min, value))


def source_block_reason(
    n_domains: int,
    has_institutional: bool,
    confirmed: bool,
    score: float,
    cfg: Config,
) -> str | None:
    """Return a reject reason when the source mix is too thin for `score`."""
    if n_domains <= 0:
        return "no_sources"
    if not has_institutional and n_domains == 1:
        if score >= cfg.single_social_min_score:
            return None
        if confirmed and score >= cfg.single_social_confirmed_min_score:
            return None
        return "single_social_unconfirmed"
    if not has_institutional:
        if score >= cfg.multi_social_min_score:
            return None
        if confirmed and score >= cfg.min_score:
            return None
        return "social_needs_confirmation"
    if n_domains == 1:
        if score >= cfg.single_news_min_score:
            return None
        return "thin_single_source"
    if score >= cfg.min_score:
        return None
    return "score_below_min"


def _reject(reason: str, direction: str = "", credibility: float = 0.0) -> Qualification:
    return Qualification(
        outcome="reject",
        reason=reason,
        direction=direction,
        credibility=credibility,
        terminal=reason in TERMINAL_REASONS,
    )


def qualify_signal(
    signal: NewsSignal,
    now: datetime,
    cfg: Config,
    *,
    confirmation: Confirmation | None = None,
    already_seen: bool = False,
    takes: tuple[TakeRecord, ...] = (),
    held_direction: str | None = None,
    held_score: float | None = None,
    ambiguous: bool = False,
) -> Qualification:
    """Qualify one signal against freshness, source mix, cooldown, and conflicts.

    `held_*` describes an open position that will still be on after this
    cycle's non-signal exits. Historical `takes` cover the cooldown and the
    conflict window after a position has already been closed.
    """
    direction = signal.direction
    if direction == "":
        return _reject("invalid_sentiment")
    if direction == "neutral" or signal.sentiment.lower() == "neutral":
        return _reject("neutral", "neutral")
    if already_seen:
        return _reject("duplicate", direction)
    if ambiguous:
        return _reject("ambiguous_conflict", direction)

    age = now - signal.timestamp
    if age < -timedelta(minutes=cfg.clock_skew_minutes):
        return _reject("future_timestamp", direction)
    if age > timedelta(minutes=cfg.freshness_minutes):
        return _reject("stale", direction)
    if signal.score < cfg.min_score:
        return _reject("score_below_min", direction)

    n_domains, institutional = source_facts(signal, cfg)
    confirmed = is_confirmed(confirmation, direction, cfg)
    blocked = source_block_reason(n_domains, institutional, confirmed, signal.score, cfg)
    cred = credibility_multiplier(n_domains, institutional, confirmed, cfg)
    if blocked:
        return _reject(blocked, direction, cred)

    incumbent_direction = held_direction
    incumbent_score = held_score
    incumbent_at: datetime | None = None
    window = timedelta(minutes=cfg.conflict_window_minutes)
    cooldown = timedelta(minutes=cfg.cooldown_minutes)
    relevant = [item for item in takes if item.ticker == signal.ticker and now - item.at <= window]
    if incumbent_direction is None and relevant:
        latest = max(relevant, key=lambda item: item.at)
        incumbent_direction = latest.direction
        incumbent_score = latest.score
        incumbent_at = latest.at
    elif incumbent_direction is not None:
        incumbent_at = now

    if incumbent_direction == direction and incumbent_score is not None:
        reference_time = incumbent_at or now
        in_cooldown = (now - reference_time) <= cooldown or held_direction == direction
        upgraded = signal.score >= incumbent_score + cfg.score_upgrade_delta
        if in_cooldown and not upgraded:
            return _reject("cooldown", direction, cred)
        if upgraded and (held_direction == direction or in_cooldown):
            return Qualification("upgrade", "score_upgrade", direction, cred, False)

    if (
        incumbent_direction
        and incumbent_direction != direction
        and incumbent_score is not None
    ):
        clears = signal.score >= incumbent_score + cfg.conflict_margin and signal.score >= cfg.flip_min_score
        if not clears:
            return Qualification("reject", "conflict", direction, cred, False)
        return Qualification("flip", "flip", direction, cred, False)

    return Qualification("enter", "qualified", direction, cred, False)


def ambiguous_ids(signals: tuple[NewsSignal, ...] | list[NewsSignal], now: datetime, cfg: Config,
                  confirmations: dict[str, Confirmation]) -> set[str]:
    """Signal ids to drop when the best bullish and bearish scores on a ticker tie.

    A tie would otherwise enter on sort order and immediately flatten on the
    opposite print. Both are skipped as `ambiguous_conflict`.
    """
    grouped: dict[str, list[NewsSignal]] = {}
    for signal in signals:
        grouped.setdefault(signal.ticker, []).append(signal)
    banned: set[str] = set()
    for ticker, group in grouped.items():
        best: dict[str, float] = {}
        for signal in group:
            if signal.direction not in ("long", "short"):
                continue
            if signal.score < cfg.min_score:
                continue
            age = now - signal.timestamp
            if age < -timedelta(minutes=cfg.clock_skew_minutes) or age > timedelta(minutes=cfg.freshness_minutes):
                continue
            n_domains, institutional = source_facts(signal, cfg)
            confirmed = is_confirmed(confirmations.get(ticker), signal.direction, cfg)
            if source_block_reason(n_domains, institutional, confirmed, signal.score, cfg):
                continue
            best[signal.direction] = max(best.get(signal.direction, 0.0), signal.score)
        if "long" in best and "short" in best and best["long"] == best["short"]:
            banned.update(signal.signal_id for signal in group)
    return banned
