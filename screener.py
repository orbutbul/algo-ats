"""
screener.py -- point-in-time cross-sectional screener for backtests.

Overview
--------
The OHLCV parquet is long-format: MultiIndex (datetime, ticker), one row per
bar per ticker. That shape is right for storage but wrong for screening --
"which tickers pass condition X at each bar" is a cross-sectional question,
answered one column-vector-per-timestamp at a time. So the first step is
always `load_fields()`, which pivots to wide: one DataFrame per field
(open/high/low/close/volume), each shaped (time x ticker).

Once fields are wide, a screening condition is just a boolean DataFrame of
that same (time x ticker) shape -- e.g. `close > close.rolling(20).mean()`.
Conditions compose with plain `&` / `|`. This is efficient because every op
is columnar across the whole universe at once; there is no per-ticker loop.

Point-in-time correctness falls out of this for free, with one caveat:
  - Safe by construction: any condition built only from backward-looking ops
    (`.rolling()`, `.ewm()`, `.shift()`, cross-sectional rank/z-score computed
    from the same row) has row t depend only on rows <= t.
  - NOT covered automatically: survivorship bias. If your ticker universe is
    "today's" list of tickers applied back over history, delisted/renamed
    names are silently missing from past screens. `load_fields()` takes
    whatever tickers exist in the parquet at each point in time (a ticker
    with no rows before it started trading just has no data there, which is
    correct) -- but if the ticker *list itself* was chosen using
    survivorship-biased criteria (e.g. "current S&P 500 members"), that bias
    is in the input, not something this module can detect or fix.

Two ways to use this module:
  1. Call `load_fields()` yourself, build conditions with plain pandas
     (rolling means, `dollar_volume()`, `cs_rank()`, `cs_zscore()`, ...), and
     combine them into a bool mask by hand. Most flexible -- use this for
     anything not covered by `Screener`'s helpers.
  2. Wrap the fields in a `Screener` for the common flow: AND/OR a few
     conditions together, then ask who passed at a given timestamp, or take
     the top-N by some ranking signal at each bar. Use `Screener.iter_chunks()`
     instead of `from_parquet()` once the source is the 1-minute file and the
     full wide pivot no longer fits in memory.

The result of a screen is always a (time x ticker) bool DataFrame -- which is
exactly the shape vbt expects for `entries`/`exits`, so it plugs directly into
`vbt.Portfolio.from_signals(close, entries=screen, ...)`.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Points at the hourly file for now -- swap to ohlcv_1min.parquet once that
# file's schema/backfill is finalized (see utils.OHLCV_PATH for the same
# caveat on the 1-min file).
DEFAULT_PATH = Path(__file__).parent / 'data' / 'ohlcv_1hour.parquet'
OHLCV_FIELDS = ('open', 'high', 'low', 'close', 'volume')


def _as_utc_ns(ts) -> pd.Timestamp:
    """
    Normalizes a start/end bound to tz-aware UTC, ns-precision -- matching
    the parquet's `datetime` column dtype exactly. Needed because a bare
    `pd.Timestamp('2025-01-01')` defaults to naive, second-precision under
    pandas 2.x, which pyarrow's row-group filter can't compare against a
    tz[UTC]/ns column (raises ArrowNotImplementedError instead of coercing).
    """
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    return ts.as_unit('ns')


def load_fields(
    path: Path | str = DEFAULT_PATH,
    tickers: list[str] | None = None,
    start=None,
    end=None,
    fields: tuple[str, ...] = OHLCV_FIELDS,
) -> dict[str, pd.DataFrame]:
    """
    Reads the long-format OHLCV parquet and pivots it to wide.

    Returns {field: DataFrame} where each DataFrame is shaped
    (datetime index x ticker columns), e.g. fields['close'].loc[ts, 'AAPL'].

    tickers/start/end are pushed down as parquet row-group filters (not
    applied after a full read), so scoping to a sub-universe or date range
    keeps memory down -- important once `path` points at the 1-minute file.
    """
    filters = []
    if tickers is not None:
        filters.append(('ticker', 'in', list(tickers)))
    if start is not None:
        filters.append(('datetime', '>=', _as_utc_ns(start)))
    if end is not None:
        filters.append(('datetime', '<=', _as_utc_ns(end)))

    df = pd.read_parquet(path, columns=list(fields), filters=filters or None)
    return {field: df[field].unstack('ticker') for field in fields}


def dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling average traded value (price x volume) -- a standard liquidity filter."""
    return (close * volume).rolling(window).mean()


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional percentile rank (0-1) of each row, i.e. rank across
    tickers at each timestamp -- not across time. NaNs (ticker not yet
    listed, no data yet) are excluded from the ranking rather than ranked.
    """
    return x.rank(axis=1, pct=True)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score of each row: (x - row mean) / row std."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std.replace(0, pd.NA), axis=0)


class Screener:
    """
    Holds a set of wide (time x ticker) field matrices and answers
    "who passes" / "who's top-N" questions against them.

    Not a strategy and not aware of positions/orders -- it only ever
    produces bool or ranking DataFrames shaped (time x ticker). Feed the
    result into vbt (`entries=screen`) or into strategy/utils.py's
    weights_to_decisions() for the live-trading side.
    """

    def __init__(self, fields: dict[str, pd.DataFrame]):
        self.fields = fields

    @classmethod
    def from_parquet(cls, path: Path | str = DEFAULT_PATH, **load_kwargs) -> 'Screener':
        return cls(load_fields(path, **load_kwargs))

    def __getitem__(self, field: str) -> pd.DataFrame:
        return self.fields[field]

    def screen(self, *conditions: pd.DataFrame, combine: str = 'and') -> pd.DataFrame:
        """
        Combines one or more (time x ticker) bool DataFrames into a single
        mask. `combine='and'` (default) requires every condition to pass;
        `combine='or'` requires at least one.
        """
        if combine not in ('and', 'or'):
            raise ValueError("combine must be 'and' or 'or'")
        result = conditions[0]
        for cond in conditions[1:]:
            result = (result & cond) if combine == 'and' else (result | cond)
        return result.fillna(False)

    def passing(self, mask: pd.DataFrame, timestamp) -> list[str]:
        """Tickers where `mask` is True at the given timestamp."""
        row = mask.loc[pd.Timestamp(timestamp)]
        return row.index[row].tolist()

    def top_n(self, signal: pd.DataFrame, n: int, ascending: bool = False) -> pd.DataFrame:
        """
        Bool mask, True for the top `n` tickers by `signal` at each
        timestamp (ties broken by column order). NaNs never qualify.
        """
        ranks = signal.rank(axis=1, method='first', ascending=ascending, na_option='keep')
        return (ranks <= n).fillna(False) & signal.notna()

    def iter_chunks(
        self,
        path: Path | str = DEFAULT_PATH,
        freq: str = 'YS',
        **load_kwargs,
    ) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, dict[str, pd.DataFrame]]]:
        """
        Yields (chunk_start, chunk_end, fields) for successive date ranges
        of width `freq` (e.g. 'YS' = per calendar year, 'MS' = per month),
        each pivoted to wide just like `load_fields()`. Use this instead of
        `from_parquet()`/`load_fields()` when the source is large enough
        (e.g. the eventual 1-minute file) that one wide pivot of the whole
        history won't fit in memory.

        Rolling indicators that need lookback history spanning a chunk
        boundary won't see it here -- pad `chunk_start` backwards by your
        max window when computing conditions per chunk, and drop the pad
        before concatenating results.
        """
        pf = pq.ParquetFile(path)
        min_ts = pd.Timestamp(pf.metadata.row_group(0).column(5).statistics.min)
        max_ts = pd.Timestamp(pf.metadata.row_group(pf.metadata.num_row_groups - 1).column(5).statistics.max)
        # date_range anchors to freq boundaries (e.g. Jan 1 for 'YS'), which
        # would silently skip the partial period between min_ts and the
        # first boundary -- prepend/append the true min/max explicitly.
        bounds = pd.date_range(min_ts.floor('D'), max_ts.ceil('D'), freq=freq)
        if bounds.empty or bounds[0] > min_ts:
            bounds = bounds.insert(0, min_ts)
        if bounds[-1] <= max_ts:
            bounds = bounds.append(pd.DatetimeIndex([max_ts + pd.Timedelta(seconds=1)]))

        for chunk_start, chunk_end in zip(bounds[:-1], bounds[1:]):
            fields = load_fields(path, start=chunk_start, end=chunk_end - pd.Timedelta(microseconds=1), **load_kwargs)
            if fields[next(iter(fields))].empty:
                continue
            yield chunk_start, chunk_end, fields
