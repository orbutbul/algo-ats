"""US equity session clock for the live-signal engine.

Execution role: classify a timestamp into overnight / pre / RTH / after-hours /
closed, and measure how long a position has been inside a tradable session.
Holiday handling is an injected set of dates so tests stay deterministic and
this package does not call a calendar service. Overnight (20:00–04:00 ET,
Sunday evening through Friday morning) is closed on the night before a
holiday and from Friday 20:00 through Sunday 20:00.

Early-close days are an injected map of date → RTH end. After-hours then runs
for four hours after that end, capped at 20:00 ET. That is an assumption, not
an exchange feed; see DESIGN.md.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

OVERNIGHT = "overnight"
PRE = "pre"
RTH = "rth"
AFTER = "after"
CLOSED = "closed"

_STEP = timedelta(minutes=5)


def to_et(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return moment.astimezone(ET)


def _rth_end(day: date, early_closes: dict[date, time] | None) -> time:
    if early_closes and day in early_closes:
        return early_closes[day]
    return time(16, 0)


def _after_end(day: date, early_closes: dict[date, time] | None) -> time:
    end = datetime.combine(day, _rth_end(day, early_closes)) + timedelta(hours=4)
    return min(end.time(), time(20, 0))


def classify_session(
    moment: datetime,
    holidays: set[date] | frozenset[date] | None = None,
    early_closes: dict[date, time] | None = None,
) -> str:
    """Return the session name at `moment`."""
    holidays = holidays or set()
    et = to_et(moment)
    day = et.date()
    clock = et.timetz().replace(tzinfo=None)
    weekday = et.weekday()

    if weekday == 5:
        return CLOSED
    if weekday == 6 and clock < time(20, 0):
        return CLOSED
    if weekday == 4 and clock >= time(20, 0):
        return CLOSED

    if clock >= time(20, 0) or clock < time(4, 0):
        belongs = day + timedelta(days=1) if clock >= time(20, 0) else day
        if belongs.weekday() >= 5 or belongs in holidays:
            return CLOSED
        return OVERNIGHT

    if weekday >= 5 or day in holidays:
        return CLOSED

    rth_end = _rth_end(day, early_closes)
    after_end = _after_end(day, early_closes)
    if time(4, 0) <= clock < time(9, 30):
        return PRE
    if time(9, 30) <= clock < rth_end:
        return RTH
    if rth_end <= clock < after_end:
        return AFTER
    return CLOSED


def flatten_deadline(
    moment: datetime,
    early_closes: dict[date, time] | None,
    minutes_before_close: float,
) -> time:
    et = to_et(moment)
    close = datetime.combine(et.date(), _rth_end(et.date(), early_closes))
    deadline = close - timedelta(minutes=minutes_before_close)
    return deadline.time()


def upcoming_overnight_closed(
    moment: datetime,
    holidays: set[date] | frozenset[date] | None = None,
) -> bool:
    """True when the overnight that starts at 20:00 on this calendar date will not run.

    Friday and Saturday have none. Sunday's 20:00 session belongs to Monday
    and is closed only if Monday is a holiday. Mon–Thu evenings are closed
    when the next calendar day is a holiday (the night before a holiday).
    """
    holidays = holidays or set()
    et = to_et(moment)
    weekday = et.weekday()
    if weekday == 4 or weekday == 5:
        return True
    if weekday == 6:
        return (et.date() + timedelta(days=1)) in holidays
    belongs = et.date() + timedelta(days=1)
    return belongs.weekday() >= 5 or belongs in holidays


def must_flatten_for_clock(
    moment: datetime,
    entry_score: float,
    *,
    overnight_min_score: float,
    minutes_before_close: float,
    holidays: set[date] | frozenset[date] | None = None,
    early_closes: dict[date, time] | None = None,
) -> str | None:
    """Return `weekend_flat` or `eod_flat` when a position should not be held, else None.

    Friday at the flatten deadline, Saturday, and Sunday before 20:00 are
    always flat. Other days: a score below `overnight_min_score`, or a night
    whose overnight session is closed, is flattened from the deadline onward.
    """
    holidays = holidays or set()
    et = to_et(moment)
    clock = et.timetz().replace(tzinfo=None)
    weekday = et.weekday()
    session = classify_session(moment, holidays, early_closes)
    low = entry_score < overnight_min_score
    deadline = flatten_deadline(moment, early_closes, minutes_before_close)

    if weekday == 5 or (weekday == 6 and clock < time(20, 0)):
        return "weekend_flat"
    if weekday == 4 and clock >= deadline:
        return "weekend_flat"

    # From the flatten deadline until 20:00, drop names that cannot ride the
    # overnight. A high score keeps going when that overnight will actually open.
    if weekday < 4 and deadline <= clock < time(20, 0):
        if low or upcoming_overnight_closed(moment, holidays):
            return "eod_flat"

    if session == OVERNIGHT and low:
        return "eod_flat"
    if session == OVERNIGHT and weekday == 6 and (et.date() + timedelta(days=1)) in holidays:
        return "eod_flat"

    if session == CLOSED:
        # Early-close days have a gap between the short after-hours and 20:00.
        # A high-score name can sit through that gap when overnight will open.
        if weekday < 4 and time(16, 0) <= clock < time(20, 0):
            if low or upcoming_overnight_closed(moment, holidays):
                return "eod_flat"
            return None
        return "weekend_flat"
    return None


def session_hours_between(
    start: datetime,
    end: datetime,
    holidays: set[date] | frozenset[date] | None = None,
    early_closes: dict[date, time] | None = None,
) -> float:
    """Hours between two timestamps that fall inside a non-closed session.

    Walks in five-minute steps and classifies the midpoint. Closed time
    (weekends, holidays, the gap after an early-close after-hours) does not
    count. Used for alpha decay, which should not fire just because the
    position sat through a Sunday.
    """
    if end <= start:
        return 0.0
    holidays = holidays or set()
    cursor = start
    hours = 0.0
    while cursor < end:
        nxt = min(cursor + _STEP, end)
        midpoint = cursor + (nxt - cursor) / 2
        if classify_session(midpoint, holidays, early_closes) != CLOSED:
            hours += (nxt - cursor).total_seconds() / 3600
        cursor = nxt
    return hours
