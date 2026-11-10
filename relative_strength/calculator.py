"""
relative_strength/calculator.py — IBD/Minervini-inspired relative-strength proxy.

Research calculator only. It turns a panel of daily session closes into a
cross-sectional 1–99 percentile and, separately, a price-ratio line versus a
benchmark. It does not read DuckDB, fetch prices, or place orders. Callers
pass prices in (see relative_strength/io.py for the opt-in read-only adapter).

This is a labeled transparent proxy, not official IBD RS. IBD's methodology
and universe are proprietary and were updated in 2026; they cannot be
replicated from public descriptions. The returns below are overlapping
cumulative session returns, not four independent quarters.

Method
------
On one as-of NYSE session t, with a positive close P:

    R_n = P_t / P_(t-n) - 1
    S   = 0.40*R_63 + 0.20*R_126 + 0.20*R_189 + 0.20*R_252

n is 63, 126, 189, and 252 NYSE sessions, not rows of a gappy series.
Sessions after t are ignored. A missing print is never forward-filled.

Eligible symbols (all four anchors present, valid prices, gap tolerance met)
are ranked on S on that same date. Ties share the average rank. The proxy is

    rs_proxy = half-up round of 1 + (n_eligible - average_rank) / (n_eligible - 1) * 98

clipped to 1..99, where average rank 1 is the highest score. The percentile
is withheld when fewer than min_universe symbols are eligible (default 30)
or when n_eligible < 2 (a singleton is not a percentile). broad_market_rating
is true only for a stocks or etfs universe with at least 1000 eligible names.
A watchlist percentile is never reported as a broad-market rating.

Stock and ETF universes are ranked separately when asset_class is supplied
(equities vs etfs). Crypto is excluded by default. Symbols with no class are
not silently treated as stocks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

# Same quote-suffix rule as utils._tracked_symbols. Kept here so this package
# does not import utils (that import loads brokerage and HTTP clients).
CRYPTO_TICKER_RE = re.compile(r'^.{2,}(USDT|BUSD|BTC|ETH|BNB)$')

LOOKBACKS: tuple[int, ...] = (63, 126, 189, 252)
WEIGHTS: dict[int, float] = {63: 0.40, 126: 0.20, 189: 0.20, 252: 0.20}
MIN_UNIVERSE_FOR_PERCENTILE = 30
BROAD_MARKET_MIN = 1000
MAX_MISSING_SESSIONS = 5
REQUIRED_SESSIONS = 252 + 1  # as-of session plus the 252-session anchor
# Long enough that an NYSE calendar always contains REQUIRED_SESSIONS.
CALENDAR_LOOKBACK_CALENDAR_DAYS = 420

METHOD_NAME = 'ibd_minervini_inspired_rs_proxy'
RATING_LABEL = 'transparent_proxy_not_official_ibd_rs'
SESSION_CALENDAR = 'NYSE'
RS_LINE_NOTE = (
    'close divided by benchmark close on the same NYSE session; '
    'not the 1-99 rs_proxy percentile; missing benchmark prints are left empty'
)

RATING_COLUMNS = [
    'as_of_session',
    'as_of_requested',
    'ticker',
    'asset_class',
    'universe',
    'r_63',
    'r_126',
    'r_189',
    'r_252',
    'raw_score',
    'rs_proxy',
    'eligible',
    'eligible_count',
    'status',
    'reason',
    'reason_detail',
    'percentile_withheld_reason',
    'missing_sessions_in_window',
    'first_session',
    'last_session',
    'price_source',
    'adjustment',
    'adjustment_note',
    'session_calendar',
    'benchmark_symbol',
    'benchmark_available',
    'benchmark_note',
    'rating_label',
    'broad_market_rating',
    'run_warnings',
]

RS_LINE_COLUMNS = [
    'session_date',
    'ticker',
    'close',
    'benchmark_symbol',
    'benchmark_close',
    'rs_line',
    'price_source',
    'adjustment',
    'note',
]

_DATE_ALIASES = ('date', 'datetime', 'session_date')
_TICKER_ALIASES = ('ticker', 'symbol')
_CLOSE_ALIASES = ('close',)
_CLASS_ALIASES = ('asset_class', 'assetclass')

_CLASS_MAP = {
    'equity': 'stocks',
    'equities': 'stocks',
    'stock': 'stocks',
    'stocks': 'stocks',
    'etf': 'etfs',
    'etfs': 'etfs',
    'fund': 'etfs',
    'crypto': 'cryptos',
    'cryptos': 'cryptos',
    'cryptocurrency': 'cryptos',
}


class PriceSchemaError(ValueError):
    """Supplied prices are not one daily close per session."""


@dataclass
class RelativeStrengthResult:
    """Ratings table, optional RS-line table, and a JSON-ready summary."""

    ratings: pd.DataFrame
    rs_line: pd.DataFrame
    summary: dict


def percentile_from_average_rank(average_rank: float, n_eligible: int) -> int:
    """Map an average rank (1 = best) to an integer in 1..99.

    Half-up rounding, not Python's bankers' round. n_eligible must be >= 2.
    """
    if n_eligible < 2:
        raise ValueError('percentile is undefined for fewer than 2 eligible symbols')
    raw = 1.0 + (n_eligible - average_rank) / (n_eligible - 1) * 98.0
    rounded = int(math.floor(raw + 0.5))
    return min(99, max(1, rounded))


def average_ranks(scores: dict[str, float]) -> dict[str, float]:
    """Average rank on exact score equality. 1 is the highest score.

    Ticker order is only used to walk the list; it does not split ties.
    """
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        # 1-based ranks i+1 .. j, averaged.
        avg = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg
        i = j
    return ranks


def nyse_sessions(start: date, end: date) -> list[date]:
    """NYSE session dates in [start, end], inclusive. Local calendar, no network."""
    if end < start:
        return []
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [ts.date() for ts in schedule.index]


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise PriceSchemaError(f'invalid as-of date: {value!r}')
    return ts.date()


def _session_date(value) -> date:
    """Calendar date of a daily bar.

    This repo stores yfinance daily bars at UTC midnight as a session label.
    Converting that instant to America/New_York would shift the session back
    one day, so midnight timestamps keep their UTC calendar date. A non-midnight
    timestamp is intraday and is rejected rather than guessed into a session.
    """
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        return value
    else:
        ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise PriceSchemaError('price frame contains a missing date')
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    if ts.hour or ts.minute or ts.second or ts.nanosecond:
        raise PriceSchemaError(
            f'intraday timestamp {value!s} is not a daily session close. '
            'Pass one close per NYSE session (a date or a midnight bar). '
            '1-minute tables ohlcv_1min_airflow, massive_1min, and robinhood_1min '
            'are not resampled by this calculator.'
        )
    return ts.date()


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _canonical_class(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    key = str(value).strip().lower()
    if not key or key == 'nan':
        return None
    return _CLASS_MAP.get(key)


def _normalize_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return columns ticker, session_date, close, asset_class_raw.

    Does not drop non-session dates; the caller counts those against the
    NYSE calendar. Raises PriceSchemaError on intraday bars or bad schema.
    """
    if not isinstance(prices, pd.DataFrame):
        raise PriceSchemaError('prices must be a pandas DataFrame')
    warnings: list[str] = []
    date_col = _pick_column(list(prices.columns), _DATE_ALIASES)
    ticker_col = _pick_column(list(prices.columns), _TICKER_ALIASES)
    close_col = _pick_column(list(prices.columns), _CLOSE_ALIASES)
    class_col = _pick_column(list(prices.columns), _CLASS_ALIASES)
    missing = [
        name for name, col in (('date', date_col), ('ticker', ticker_col), ('close', close_col)) if col is None
    ]
    if missing:
        raise PriceSchemaError(
            'price frame needs columns date (or datetime/session_date), ticker (or symbol), '
            f'and close. Missing: {", ".join(missing)}.'
        )
    out = pd.DataFrame({
        'ticker': prices[ticker_col].map(lambda v: '' if pd.isna(v) else str(v).strip().upper()),
        'session_date': [_session_date(v) for v in prices[date_col].tolist()],
        'close': pd.to_numeric(prices[close_col], errors='coerce'),
    })
    if class_col is None:
        out['asset_class_raw'] = None
    else:
        out['asset_class_raw'] = prices[class_col].tolist()
    blank = out['ticker'] == ''
    if blank.any():
        warnings.append(f'dropped {int(blank.sum())} rows with a blank ticker')
        out = out.loc[~blank].copy()
    return out, warnings


def _positive(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _series_map(
    frame: pd.DataFrame,
) -> tuple[dict[str, dict[date, float]], dict[str, str], dict[str, set[date]]]:
    """Collapse to one close per ticker and session.

    Identical duplicate closes are kept once. Distinct closes on the same
    date are recorded and later disqualify the symbol only if that date is
    an NYSE session still inside the as-of window. A weekend double print
    must not void an otherwise clean session history.
    """
    by_ticker: dict[str, dict[date, float]] = {}
    raw_class: dict[str, set] = {}
    conflicts: dict[str, set[date]] = {}
    for row in frame.itertuples(index=False):
        ticker = row.ticker
        slot = by_ticker.setdefault(ticker, {})
        raw_class.setdefault(ticker, set()).add(row.asset_class_raw)
        close = None if pd.isna(row.close) else float(row.close)
        prev = slot.get(row.session_date, _MISSING)
        if prev is _MISSING or prev is None:
            slot[row.session_date] = close
        elif close is None:
            continue
        elif prev != close:
            conflicts.setdefault(ticker, set()).add(row.session_date)
            slot[row.session_date] = close
    classes: dict[str, str] = {}
    for ticker, values in raw_class.items():
        canonical = {_canonical_class(v) for v in values}
        canonical.discard(None)
        if len(canonical) > 1:
            classes[ticker] = 'conflict'
        elif len(canonical) == 1:
            classes[ticker] = canonical.pop()
    return by_ticker, classes, conflicts


def _active_duplicate_tickers(
    conflicts: dict[str, set[date]],
    by_ticker: dict[str, dict[date, float]],
) -> set[str]:
    return {
        ticker
        for ticker, days in conflicts.items()
        if any(day in by_ticker.get(ticker, {}) for day in days)
    }


_MISSING = object()


def _apply_class_overrides(classes: dict[str, str], override: dict[str, str] | None) -> dict[str, str]:
    if not override:
        return classes
    merged = dict(classes)
    for ticker, value in override.items():
        key = str(ticker).strip().upper()
        canonical = _canonical_class(value)
        if canonical is None:
            merged[key] = 'unknown'
        else:
            merged[key] = canonical
    return merged


def _is_crypto(ticker: str, asset_class: str | None) -> bool:
    if asset_class == 'cryptos':
        return True
    # An explicit stock/ETF label wins over the suffix guess so a classified
    # name is not dropped. Unclassified tickers still use the suffix rule.
    if asset_class in ('stocks', 'etfs'):
        return False
    return CRYPTO_TICKER_RE.search(ticker) is not None


def _exclusion(ticker: str, asset_class: str | None, universe: str) -> str | None:
    if asset_class == 'conflict':
        return 'conflicting_asset_class'
    if asset_class == 'unknown':
        return 'unclassified'
    if _is_crypto(ticker, asset_class):
        return 'crypto_excluded'
    if universe == 'unseparated':
        return None
    if universe == 'stocks':
        if asset_class == 'stocks':
            return None
        if asset_class == 'etfs':
            return 'outside_universe'
        return 'unclassified'
    if universe == 'etfs':
        if asset_class == 'etfs':
            return None
        if asset_class == 'stocks':
            return 'outside_universe'
        return 'unclassified'
    raise PriceSchemaError("universe must be 'stocks', 'etfs', or 'unseparated'")


def _weighted_score(returns: dict[int, float]) -> float:
    return sum(WEIGHTS[n] * returns[n] for n in LOOKBACKS)


def _evaluate_symbol(
    series: dict[date, float],
    sessions: list[date],
    anchors: dict[int, date] | None,
    window: list[date],
    as_of_session: date,
    duplicate: bool,
    max_missing_sessions: int,
) -> dict:
    """Price-quality decision for one symbol already inside the universe."""
    valid_dates = [d for d, value in series.items() if _positive(value)]
    first = min(valid_dates) if valid_dates else None
    last = max(valid_dates) if valid_dates else None
    as_of_close = series.get(as_of_session)
    returns: dict[int, float] = {}

    def pack(reason: str, detail: str, missing: int | None) -> dict:
        return {
            'reason': reason,
            'reason_detail': detail,
            'returns': returns,
            'raw_score': None,
            'eligible': False,
            'missing_sessions_in_window': missing,
            'first_session': first,
            'last_session': last,
        }

    if duplicate:
        return pack(
            'duplicate_session_prints',
            'more than one distinct close on the same NYSE session; no close was chosen',
            None,
        )

    if not _positive(as_of_close):
        # NaN / missing is a hole. Zero, negative, and non-finite are a bad print.
        if as_of_close is not None and not (isinstance(as_of_close, float) and math.isnan(as_of_close)):
            return pack(
                'invalid_price',
                f'close on as-of session {as_of_session.isoformat()} is {as_of_close}; '
                'non-positive or non-finite prices are rejected and not replaced',
                None,
            )
        if last is None:
            return pack(
                'missing_as_of',
                f'no close on as-of session {as_of_session.isoformat()}',
                None,
            )
        return pack(
            'stale',
            f'last valid session {last.isoformat()} is before as-of session {as_of_session.isoformat()}; '
            'the older close was not forward-filled',
            None,
        )

    if anchors is None:
        return pack(
            'short_history',
            f'reference calendar has {len(sessions)} NYSE sessions through {as_of_session.isoformat()}; '
            f'{REQUIRED_SESSIONS} are required (as-of plus a 252-session lookback)',
            None,
        )

    problems: list[tuple[str, int, date]] = []
    for n in LOOKBACKS:
        anchor = anchors[n]
        value = series.get(anchor)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if first is None or first > anchor:
                problems.append(('short_history', n, anchor))
            else:
                problems.append(('missing_price', n, anchor))
            continue
        if not _positive(value):
            problems.append(('invalid_price', n, anchor))
            continue
        returns[n] = as_of_close / value - 1.0

    missing = sum(1 for day in window if not _positive(series.get(day)))
    if any(kind == 'invalid_price' for kind, _, _ in problems):
        bits = [
            f'lookback {n} anchor {day.isoformat()} close is invalid'
            for kind, n, day in problems if kind == 'invalid_price'
        ]
        return pack('invalid_price', '; '.join(bits), missing)
    if any(kind == 'short_history' for kind, _, _ in problems):
        bits = [
            f'lookback {n} needs {day.isoformat()} but first valid session is '
            f'{first.isoformat() if first else "none"}'
            for kind, n, day in problems if kind == 'short_history'
        ]
        return pack(
            'short_history',
            '; '.join(bits) + f'; need {REQUIRED_SESSIONS} sessions, anchors are not row counts',
            missing,
        )
    if any(kind == 'missing_price' for kind, _, _ in problems):
        bits = [
            f'no close on {day.isoformat()} for lookback {n}'
            for kind, n, day in problems if kind == 'missing_price'
        ]
        return pack('missing_price', '; '.join(bits) + '; the missing print was not forward-filled', missing)
    if missing > max_missing_sessions:
        return pack(
            'gappy',
            f'{missing} missing or invalid prints in the {len(window)}-session window '
            f'(max {max_missing_sessions}); lookbacks stay on NYSE dates and the score is withheld',
            missing,
        )
    return {
        'reason': 'ok',
        'reason_detail': (
            f'all four overlapping lookbacks present; missing sessions in window={missing}'
        ),
        'returns': returns,
        'raw_score': _weighted_score(returns),
        'eligible': True,
        'missing_sessions_in_window': missing,
        'first_session': first,
        'last_session': last,
    }


def _exclusion_detail(reason: str, universe: str) -> str:
    if reason == 'crypto_excluded':
        return 'crypto is excluded from the price-RS universe by default'
    if reason == 'outside_universe':
        return f'asset class is outside the requested {universe} universe; universes are not mixed'
    if reason == 'unclassified':
        return (
            f'no stock/ETF asset class for the requested {universe} universe; '
            'unlabeled names are not treated as stocks'
        )
    if reason == 'conflicting_asset_class':
        return 'ticker has more than one asset class in the input'
    return reason


def _method_summary(min_universe: int, max_missing_sessions: int) -> dict:
    return {
        'name': METHOD_NAME,
        'method_version': 1,
        'official_ibd_rs': False,
        'description': (
            'Transparent IBD/Minervini-inspired proxy. Not official IBD RS. '
            'Overlapping cumulative session returns, not four independent quarters '
            'and not a proprietary formula.'
        ),
        'lookbacks_trading_sessions': list(LOOKBACKS),
        'weights': {str(n): WEIGHTS[n] for n in LOOKBACKS},
        'return_definition': 'R_n = P_t / P_(t-n) - 1 using closes on the NYSE session calendar',
        'rank_policy': (
            'Eligible symbols on the same as-of session only. '
            'Average rank of exact score ties; ticker does not break ties. Rank 1 is the highest raw score.'
        ),
        'percentile_policy': (
            'rs_proxy = half-up round of 1 + (n_eligible - average_rank) / (n_eligible - 1) * 98, '
            'clipped to 1..99. Withheld when n_eligible < min_universe or n_eligible < 2.'
        ),
        'singleton_policy': 'n_eligible < 2 withholds rs_proxy; one name is not a percentile',
        'small_universe_policy': (
            f'n_eligible < min_universe (default {MIN_UNIVERSE_FOR_PERCENTILE}) withholds rs_proxy. '
            'Raw returns and raw_score are still reported. A small cross-section is not a broad-market rating.'
        ),
        'broad_market_policy': (
            f'broad_market_rating is true only when universe is stocks or etfs and '
            f'n_eligible >= {BROAD_MARKET_MIN}. unseparated universes are never broad-market ratings.'
        ),
        'gap_policy': (
            'Lookbacks are NYSE session dates, not per-symbol row counts. '
            f'More than {max_missing_sessions} missing or invalid prints inside the '
            f'{REQUIRED_SESSIONS}-session window marks the symbol gappy. Missing prints are not forward-filled.'
        ),
        'future_data_policy': 'Sessions after the as-of session are ignored',
        'as_of_completeness': 'Not verified. Pass the last completed session; a same-day bar may be partial.',
        'min_universe': min_universe,
        'max_missing_sessions': max_missing_sessions,
        'required_sessions': REQUIRED_SESSIONS,
    }


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.DataFrame(columns=RATING_COLUMNS)
    line = pd.DataFrame(columns=RS_LINE_COLUMNS)
    return ratings, line


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def compute_relative_strength(
    prices: pd.DataFrame,
    as_of: date | datetime | str,
    *,
    universe: str = 'stocks',
    asset_class: dict[str, str] | None = None,
    benchmark: str = 'SPY',
    benchmark_prices: pd.DataFrame | None = None,
    price_source: str = 'supplied',
    adjustment: str = 'unknown',
    adjustment_note: str | None = None,
    min_universe: int = MIN_UNIVERSE_FOR_PERCENTILE,
    max_missing_sessions: int = MAX_MISSING_SESSIONS,
    include_rs_line: bool = True,
) -> RelativeStrengthResult:
    """Rank ``prices`` on one as-of date.

    ``prices`` is not modified. Required columns are date, ticker, and close.
    Optional ``asset_class`` values are equities/etfs/cryptos (or stocks/etfs).
    ``adjustment`` is recorded, not inferred: use ``split``, ``split_and_dividend``,
    or ``unknown``. Unknown is not reported as split-adjusted.
    """
    if universe not in ('stocks', 'etfs', 'unseparated'):
        raise PriceSchemaError("universe must be 'stocks', 'etfs', or 'unseparated'")
    if adjustment not in ('split', 'split_and_dividend', 'unknown'):
        raise PriceSchemaError("adjustment must be 'split', 'split_and_dividend', or 'unknown'")
    if min_universe < 1:
        raise PriceSchemaError('min_universe must be >= 1')
    if max_missing_sessions < 0:
        raise PriceSchemaError('max_missing_sessions must be >= 0')

    # Copy before any column work so a caller's frame, including its index, stays put.
    prices = prices.copy(deep=True)
    if benchmark_prices is not None:
        benchmark_prices = benchmark_prices.copy(deep=True)

    as_of_requested = _as_date(as_of)
    benchmark_symbol = str(benchmark).strip().upper()
    note = adjustment_note or _default_adjustment_note(adjustment)
    warnings: list[str] = []
    effective_min = max(min_universe, 2)
    if min_universe < 2:
        warnings.append('min_universe_raised_to_2: a singleton percentile is undefined')

    if prices.empty:
        warnings.append('no_prices')
        summary = _base_summary(
            as_of_requested=as_of_requested,
            as_of_session=None,
            stale=False,
            universe=universe,
            eligible_count=0,
            input_rows=0,
            ignored_non_session_rows=0,
            price_source=price_source,
            adjustment=adjustment,
            adjustment_note=note,
            benchmark_symbol=benchmark_symbol,
            benchmark_available=False,
            benchmark_note=f'benchmark {benchmark_symbol} was not evaluated because no prices were supplied',
            broad_market=False,
            warnings=warnings,
            anchors=None,
            min_universe=effective_min,
            max_missing_sessions=max_missing_sessions,
            rs_line_rows=0,
        )
        ratings, line = _empty_frames()
        return RelativeStrengthResult(ratings, line, summary)

    frame, norm_warnings = _normalize_prices(prices)
    warnings.extend(norm_warnings)
    by_ticker, classes, conflicts = _series_map(frame)
    classes = _apply_class_overrides(classes, asset_class)

    benchmark_series: dict[date, float] = {}
    benchmark_conflicts: set[date] = set()
    if benchmark_symbol in by_ticker:
        benchmark_series = dict(by_ticker[benchmark_symbol])
        benchmark_conflicts = set(conflicts.get(benchmark_symbol, set()))
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_frame, bench_warnings = _normalize_prices(benchmark_prices)
        warnings.extend(f'benchmark input: {item}' for item in bench_warnings)
        bench_map, _, bench_conflicts = _series_map(bench_frame)
        if benchmark_symbol in bench_map:
            benchmark_series = bench_map[benchmark_symbol]
            benchmark_conflicts = set(bench_conflicts.get(benchmark_symbol, set()))
        else:
            warnings.append(
                f'benchmark file has no rows for {benchmark_symbol}; falling back to the price input if present'
            )

    session_dates = sorted({day for series in by_ticker.values() for day in series})
    if not session_dates:
        warnings.append('no_prices')
        summary = _base_summary(
            as_of_requested=as_of_requested,
            as_of_session=None,
            stale=False,
            universe=universe,
            eligible_count=0,
            input_rows=int(len(frame)),
            ignored_non_session_rows=0,
            price_source=price_source,
            adjustment=adjustment,
            adjustment_note=note,
            benchmark_symbol=benchmark_symbol,
            benchmark_available=False,
            benchmark_note=f'benchmark {benchmark_symbol} unavailable',
            broad_market=False,
            warnings=warnings,
            anchors=None,
            min_universe=effective_min,
            max_missing_sessions=max_missing_sessions,
            rs_line_rows=0,
        )
        ratings, line = _empty_frames()
        return RelativeStrengthResult(ratings, line, summary)

    span_start = min(session_dates)
    span_end = max(max(session_dates), as_of_requested)
    known_sessions = set(nyse_sessions(span_start, span_end))
    ignored = int(sum(day not in known_sessions for day in frame['session_date']))
    for series in list(by_ticker.values()) + [benchmark_series]:
        for day in list(series):
            if day not in known_sessions:
                series.pop(day, None)

    usable_dates = sorted({day for series in by_ticker.values() for day in series if day <= as_of_requested})
    expected = [day for day in nyse_sessions(span_start, as_of_requested)]
    expected_session = expected[-1] if expected else None
    last_data = usable_dates[-1] if usable_dates else None

    stale = False
    as_of_session: date | None
    if expected_session is None or last_data is None:
        as_of_session = None
        warnings.append(
            f'as_of_before_data: no NYSE session on or before {as_of_requested.isoformat()} has a price'
        )
    elif last_data < expected_session:
        stale = True
        as_of_session = last_data
        warnings.append(
            'stale_as_of: requested NYSE session '
            f'{expected_session.isoformat()} but the latest price session is {last_data.isoformat()}. '
            'Ratings use the latest session actually present. No quote was forward-filled.'
        )
    else:
        as_of_session = expected_session

    # Drop prints after the session actually used, including prints after a stale snap.
    if as_of_session is not None:
        for series in by_ticker.values():
            for day in list(series):
                if day > as_of_session:
                    series.pop(day)
        for day in list(benchmark_series):
            if day > as_of_session:
                benchmark_series.pop(day)
    if benchmark_conflicts and any(day in benchmark_series for day in benchmark_conflicts):
        warnings.append(
            f'benchmark {benchmark_symbol} has more than one distinct close on a session; RS line is withheld'
        )
        benchmark_series = {}

    sessions: list[date] = []
    anchors: dict[int, date] | None = None
    window: list[date] = []
    if as_of_session is not None:
        cal_start = as_of_session - timedelta(days=CALENDAR_LOOKBACK_CALENDAR_DAYS)
        sessions = nyse_sessions(cal_start, as_of_session)
        if not sessions or sessions[-1] != as_of_session:
            raise RuntimeError(f'NYSE calendar did not include as-of session {as_of_session.isoformat()}')
        if len(sessions) >= REQUIRED_SESSIONS:
            anchors = {n: sessions[-1 - n] for n in LOOKBACKS}
            window = sessions[-REQUIRED_SESSIONS:]
        else:
            warnings.append(
                f'short_calendar: {len(sessions)} NYSE sessions through {as_of_session.isoformat()}; '
                f'{REQUIRED_SESSIONS} required'
            )

    labeled = [ticker for ticker in by_ticker if classes.get(ticker) in ('stocks', 'etfs', 'cryptos')]
    if universe in ('stocks', 'etfs') and by_ticker and not labeled:
        warnings.append(
            'classification_unavailable: no asset_class metadata was supplied, so no symbol was labeled '
            f'{universe}. Pass asset_class or run with universe=unseparated. '
            'Unlabeled names were not ranked as stocks or ETFs.'
        )
    elif universe in ('stocks', 'etfs'):
        unlabeled = [
            ticker for ticker in by_ticker
            if _exclusion(ticker, classes.get(ticker), universe) == 'unclassified'
        ]
        if unlabeled:
            warnings.append(
                f'partial_classification: {len(unlabeled)} symbols have no stock/ETF class and were excluded '
                f'from the {universe} universe'
            )
    if universe == 'unseparated':
        warnings.append(
            'stock_etf_not_separated: stocks and ETFs are ranked together. '
            'This is not a stock-only or ETF-only rating and is not a broad-market rating.'
        )

    crypto_n = sum(1 for ticker in by_ticker if _is_crypto(ticker, classes.get(ticker)))
    if crypto_n:
        warnings.append(f'crypto_excluded: {crypto_n} symbols were left out of the ranking')

    duplicate_tickers = _active_duplicate_tickers(conflicts, by_ticker)
    evaluations: dict[str, dict] = {}
    for ticker in sorted(by_ticker):
        klass = classes.get(ticker)
        excluded = _exclusion(ticker, klass, universe)
        if excluded is not None or as_of_session is None:
            reason = excluded or 'missing_as_of'
            detail = (
                _exclusion_detail(excluded, universe)
                if excluded is not None
                else f'no NYSE session on or before {as_of_requested.isoformat()} has a price'
            )
            evaluations[ticker] = {
                'reason': reason,
                'reason_detail': detail,
                'returns': {},
                'raw_score': None,
                'eligible': False,
                'missing_sessions_in_window': None,
                'first_session': None,
                'last_session': None,
                'asset_class': klass,
            }
            continue
        evaluated = _evaluate_symbol(
            by_ticker[ticker],
            sessions,
            anchors,
            window,
            as_of_session,
            ticker in duplicate_tickers,
            max_missing_sessions,
        )
        evaluated['asset_class'] = klass
        evaluations[ticker] = evaluated

    eligible_scores = {
        ticker: info['raw_score']
        for ticker, info in evaluations.items()
        if info['eligible'] and info['raw_score'] is not None
    }
    eligible_count = len(eligible_scores)
    withheld = None
    proxy: dict[str, int] = {}
    if eligible_count >= effective_min:
        for ticker, rank in average_ranks(eligible_scores).items():
            proxy[ticker] = percentile_from_average_rank(rank, eligible_count)
    elif eligible_count == 0:
        warnings.append('no_eligible_symbols')
        withheld = 'no_eligible_symbols'
    else:
        withheld = (
            f'universe_too_small: eligible_count={eligible_count} < min_universe={effective_min}. '
            'rs_proxy was withheld. This is not a broad-market rating.'
        )
        warnings.append(withheld)

    broad = bool(
        universe in ('stocks', 'etfs')
        and eligible_count >= BROAD_MARKET_MIN
        and eligible_count >= effective_min
    )
    if eligible_count >= effective_min and not broad:
        if universe == 'unseparated':
            warnings.append(
                'not_broad_market: universe is unseparated, so rs_proxy is only a cross-section of the names supplied'
            )
        else:
            warnings.append(
                f'narrow_universe: eligible_count={eligible_count} < {BROAD_MARKET_MIN}. '
                'rs_proxy is a cross-section of this universe, not a broad-market rating.'
            )

    benchmark_available = _benchmark_usable(benchmark_series, as_of_session)
    if benchmark_available:
        benchmark_note = (
            f'benchmark {benchmark_symbol} uses the same adjustment label ({adjustment}). '
            'rs_line is the close ratio and is not rs_proxy. '
            'A missing benchmark print does not block the percentile.'
        )
    else:
        benchmark_note = (
            f'benchmark {benchmark_symbol} is not available with a positive close on the as-of session. '
            'Percentile ratings were still computed. No RS line was filled in.'
        )
        if include_rs_line:
            warnings.append(
                f'benchmark_unavailable: {benchmark_symbol} missing or invalid on the as-of session; rs_line omitted'
            )

    # Deduplicate warnings while keeping order, now that the full set exists.
    warnings = list(dict.fromkeys(warnings))
    warning_text = ' | '.join(warnings)

    rows = []
    for ticker in sorted(evaluations):
        info = evaluations[ticker]
        returns = info['returns']
        rows.append({
            'as_of_session': _iso(as_of_session),
            'as_of_requested': as_of_requested.isoformat(),
            'ticker': ticker,
            'asset_class': info.get('asset_class'),
            'universe': universe,
            'r_63': returns.get(63),
            'r_126': returns.get(126),
            'r_189': returns.get(189),
            'r_252': returns.get(252),
            'raw_score': info['raw_score'],
            'rs_proxy': proxy.get(ticker),
            'eligible': bool(info['eligible']),
            'eligible_count': eligible_count,
            'status': 'eligible' if info['eligible'] else 'ineligible',
            'reason': info['reason'],
            'reason_detail': info['reason_detail'],
            'percentile_withheld_reason': withheld if info['eligible'] and ticker not in proxy else None,
            'missing_sessions_in_window': info['missing_sessions_in_window'],
            'first_session': _iso(info['first_session']),
            'last_session': _iso(info['last_session']),
            'price_source': price_source,
            'adjustment': adjustment,
            'adjustment_note': note,
            'session_calendar': SESSION_CALENDAR,
            'benchmark_symbol': benchmark_symbol,
            'benchmark_available': benchmark_available,
            'benchmark_note': benchmark_note,
            'rating_label': RATING_LABEL,
            'broad_market_rating': broad,
            'run_warnings': warning_text,
        })

    ratings = pd.DataFrame(rows, columns=RATING_COLUMNS)
    if not ratings.empty:
        ratings = ratings.sort_values(
            by=['rs_proxy', 'raw_score', 'ticker'],
            ascending=[False, False, True],
            na_position='last',
        ).reset_index(drop=True)
        ratings['rs_proxy'] = ratings['rs_proxy'].astype('Int64')
        ratings['eligible_count'] = ratings['eligible_count'].astype('Int64')
        ratings['missing_sessions_in_window'] = ratings['missing_sessions_in_window'].astype('Int64')

    rs_line = _build_rs_line(
        by_ticker,
        benchmark_series,
        sessions,
        benchmark_symbol,
        benchmark_available and include_rs_line,
        price_source,
        adjustment,
    )
    summary = _base_summary(
        as_of_requested=as_of_requested,
        as_of_session=as_of_session,
        stale=stale,
        universe=universe,
        eligible_count=eligible_count,
        input_rows=int(len(frame)),
        ignored_non_session_rows=ignored,
        price_source=price_source,
        adjustment=adjustment,
        adjustment_note=note,
        benchmark_symbol=benchmark_symbol,
        benchmark_available=benchmark_available,
        benchmark_note=benchmark_note,
        broad_market=broad,
        warnings=warnings,
        anchors=anchors,
        min_universe=effective_min,
        max_missing_sessions=max_missing_sessions,
        rs_line_rows=int(len(rs_line)),
    )
    return RelativeStrengthResult(ratings=ratings, rs_line=rs_line, summary=summary)


def _benchmark_usable(series: dict[date, float], as_of_session: date | None) -> bool:
    if as_of_session is None:
        return False
    return _positive(series.get(as_of_session))


def _build_rs_line(
    by_ticker: dict[str, dict[date, float]],
    benchmark_series: dict[date, float],
    sessions: list[date],
    benchmark_symbol: str,
    enabled: bool,
    price_source: str,
    adjustment: str,
) -> pd.DataFrame:
    if not enabled:
        return pd.DataFrame(columns=RS_LINE_COLUMNS)
    rows = []
    for ticker in sorted(by_ticker):
        if ticker == benchmark_symbol:
            continue
        series = by_ticker[ticker]
        for day in sessions:
            close = series.get(day)
            if not _positive(close):
                # No symbol print this session: do not emit a forward-filled row.
                continue
            bench = benchmark_series.get(day)
            if _positive(bench):
                bench_out = bench
                ratio = close / bench
            else:
                bench_out = None
                ratio = None
            rows.append({
                'session_date': day.isoformat(),
                'ticker': ticker,
                'close': close,
                'benchmark_symbol': benchmark_symbol,
                'benchmark_close': bench_out,
                'rs_line': ratio,
                'price_source': price_source,
                'adjustment': adjustment,
                'note': RS_LINE_NOTE,
            })
    return pd.DataFrame(rows, columns=RS_LINE_COLUMNS)


def _default_adjustment_note(adjustment: str) -> str:
    if adjustment == 'split':
        return (
            'Caller labeled these closes split-adjusted only, not dividend-adjusted. '
            'The calculator does not verify that label and does not call them raw closes.'
        )
    if adjustment == 'split_and_dividend':
        return (
            'Caller labeled these closes adjusted for splits and dividends. '
            'The calculator does not verify that label.'
        )
    return (
        'Adjustment policy is unknown. These closes are not labeled split-adjusted or dividend-adjusted, '
        'and they are not described as raw or as official IBD prices.'
    )


def _base_summary(
    *,
    as_of_requested: date,
    as_of_session: date | None,
    stale: bool,
    universe: str,
    eligible_count: int,
    input_rows: int,
    ignored_non_session_rows: int,
    price_source: str,
    adjustment: str,
    adjustment_note: str,
    benchmark_symbol: str,
    benchmark_available: bool,
    benchmark_note: str,
    broad_market: bool,
    warnings: list[str],
    anchors: dict[int, date] | None,
    min_universe: int,
    max_missing_sessions: int,
    rs_line_rows: int,
) -> dict:
    return {
        'method': _method_summary(min_universe, max_missing_sessions),
        'as_of_requested': as_of_requested.isoformat(),
        'as_of_session': _iso(as_of_session),
        'stale': stale,
        'universe': universe,
        'eligible_count': eligible_count,
        'input_rows': input_rows,
        'ignored_non_session_rows': ignored_non_session_rows,
        'price_source': price_source,
        'adjustment': adjustment,
        'adjustment_note': adjustment_note,
        'session_calendar': SESSION_CALENDAR,
        'lookback_anchor_sessions': (
            {str(n): day.isoformat() for n, day in anchors.items()} if anchors else None
        ),
        'benchmark_symbol': benchmark_symbol,
        'benchmark_available': benchmark_available,
        'benchmark_note': benchmark_note,
        'rs_line_definition': RS_LINE_NOTE,
        'rs_line_rows': rs_line_rows,
        'rating_label': RATING_LABEL,
        'broad_market_rating': broad_market,
        'warnings': warnings,
    }
