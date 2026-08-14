import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import requests
import json
from datetime import datetime, timezone
import time
from pathlib import Path
from vectorbtpro import *
from tradingview_screener import Query, col
from binance.client import Client as BinanceClient
import pandas_market_calendars as mcal
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- Paths ---
# Anchored to this file's location (not the caller's cwd) so imports from
# research/ notebooks, whose Jupyter kernel cwd is the notebook's own
# directory rather than the project root, still resolve correctly.
OHLCV_PATH = Path(__file__).parent / 'data' / 'ohlcv_1min.parquet'   # 1-minute OHLCV bars for all asset classes
OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']
BATCH_SIZE = 100                                # tickers per yfinance download call


def get_data(tickers=None, start=None, end=None):
    """
    Reads the OHLCV parquet and returns {ticker: DataFrame} with Title-cased columns.
    All filters are optional — omit to load everything.
    """
    df = pd.read_parquet(OHLCV_PATH)
    if tickers is not None:
        tickers = [tickers] if isinstance(tickers, str) else tickers
        df = df[df.index.get_level_values('ticker').isin(tickers)]
    if start is not None:
        df = df[df.index.get_level_values('datetime') >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index.get_level_values('datetime') <= pd.Timestamp(end)]

    return {
        ticker: group.droplevel('ticker').rename(columns=str.title)
        for ticker, group in df.groupby(level='ticker')
    }


_DUCKDB_FREQ_ALIASES = {
    'hourly': 'h',
    'daily': 'D',
    'weekly': 'W',
    'monthly': 'ME',
}

# Bucket expressions for pushing freq resampling into SQL instead of pulling raw
# 1-min rows and resampling in pandas — see get_data_duckdb's SQL-pushdown branch.
# Only covers the 4 canonical freq names above (not arbitrary pandas offset
# aliases like '15min'), matched to pandas' own bin-label convention so the
# pushdown path and the pandas-resample path agree: 'hourly'/'daily' both label
# at bin start; 'weekly' matches pandas' default W-SUN (label = the bin's
# Sunday); 'monthly' matches 'ME' (label = month end).
#
# `datetime AT TIME ZONE 'UTC'` is required before date_trunc: this connection's
# session TimeZone is America/New_York (verified via current_setting('TimeZone')),
# and date_trunc on a TIMESTAMPTZ truncates in the *session* timezone, not UTC —
# for 'day'/'week'/'month' buckets (not whole-hour multiples of the NY/UTC
# offset) that silently shifts bucket boundaries by the UTC offset (verified:
# without this, daily buckets landed on 04:00 UTC instead of 00:00 UTC, 4hrs off
# from the pandas-resample path's labels). 'hourly' is unaffected in practice
# (a whole-hour offset commutes with hour-truncation) but included for
# consistency/robustness to a future non-whole-hour session timezone.
_DUCKDB_BUCKET_SQL = {
    'hourly': "date_trunc('hour', datetime AT TIME ZONE 'UTC')",
    'daily': "date_trunc('day', datetime AT TIME ZONE 'UTC')",
    'weekly': "date_trunc('week', datetime AT TIME ZONE 'UTC') + INTERVAL 6 DAY",
    'monthly': "date_trunc('month', datetime AT TIME ZONE 'UTC') + INTERVAL 1 MONTH - INTERVAL 1 DAY",
}

# Above this many tickers (or no ticker filter at all), pulling raw 1-min rows
# and resampling in pandas is not just slow but can exhaust memory (verified:
# a full 2,314-ticker daily pull over ~9 months tried to materialize ~155M raw
# rows and raised MemoryError). SQL-side aggregation does the same pull in
# single-digit seconds. Ticker-filtered calls below this threshold keep using
# the existing pandas-resample path, which is already fast and well-exercised.
_DUCKDB_SQL_PUSHDOWN_TICKER_THRESHOLD = 300


def get_data_duckdb(tickers=None, start=None, end=None, freq=None, asdf=False, asvbt=False):
    """
    Same shape and params as get_data(), but sourced from
    data/ohlcv.duckdb::massive_1min (the Massive equities/ETF backfill —
    NOT the full ohlcv_1min.parquet universe; crypto isn't in this table)
    instead of a parquet file. Filters are pushed down into SQL rather than
    applied in pandas after loading, so this never reads the whole table
    into memory — only the matching rows come across.

    Opens a fresh read-only connection per call rather than holding one
    open, since data/ohlcv.duckdb is a single-writer file: the Massive
    backfill (extraction/ohlcv_massive.py) holds an exclusive read-write
    connection for hours at a time while running, and DuckDB does not allow
    even a read-only connection to open concurrently with it (verified: it
    raises duckdb.IOException, not a silent/partial read). Callers should
    expect this to raise while a backfill is actively running and simply
    retry later — see the raised error message.

    freq: resample the raw 1-min bars before returning. One of
    'hourly'/'daily'/'weekly'/'monthly', or any pandas offset alias (e.g. '15min').
    OHLC is aggregated as first/max/min/last, volume summed, per-ticker so bars
    never blend across symbols; empty bins (e.g. weekends) are dropped. None
    (default) returns the raw 1-min bars.

    When freq is one of the 4 named aliases above AND tickers is None or covers
    more than _DUCKDB_SQL_PUSHDOWN_TICKER_THRESHOLD symbols, the aggregation is
    pushed down into SQL (GROUP BY a date_trunc bucket) instead of pulling raw
    1-min rows and resampling in pandas — pulling+resampling a full-universe
    daily bar set in pandas tries to materialize hundreds of millions of raw
    rows first and can exhaust memory (verified). Smaller ticker-filtered calls
    keep using the pandas-resample path, which is already fast. Both paths
    produce the same bin boundaries for 'hourly'/'daily'; 'weekly'/'monthly' are
    label-matched to pandas' 'W'/'ME' convention but lower-confidence since nothing
    in this codebase exercises them at the moment.

    asdf: if True, return a single DataFrame with a (datetime, ticker) MultiIndex
    instead of a {ticker: DataFrame} dict — convenient for cross-sectional ops
    (e.g. `.xs(ticker, level='ticker')` or `.loc[:, 'Close'].unstack('ticker')`).

    asvbt: if True, return a `vbt.Data` instance (built via `vbt.Data.from_data`
    on the {ticker: DataFrame} dict) instead of a plain dict/DataFrame — plugs
    straight into vbt indicators/portfolios (`data.get('Close')`, `data.hlc3`,
    `vbt.Portfolio.from_signals(data, ...)`, etc). Mutually exclusive with asdf.
    """
    if asdf and asvbt:
        raise ValueError('asdf and asvbt are mutually exclusive — pick one return shape')

    import duckdb
    tickers = [tickers] if isinstance(tickers, str) else tickers

    def _to_utc_ts(x):
        # datetime is stored as TIMESTAMPTZ — a naive Timestamp param compares
        # incorrectly against it (silently matches nothing), so always localize
        # or convert to UTC before binding.
        ts = pd.Timestamp(x)
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')

    where = []
    params = []
    if tickers is not None:
        placeholders = ', '.join('?' * len(tickers))
        where.append(f'ticker IN ({placeholders})')
        params.extend(tickers)
    if start is not None:
        where.append('datetime >= ?')
        params.append(_to_utc_ts(start))
    if end is not None:
        where.append('datetime <= ?')
        params.append(_to_utc_ts(end))
    where_clause = f"WHERE {' AND '.join(where)}" if where else ''

    use_sql_pushdown = (
        freq in _DUCKDB_BUCKET_SQL
        and (tickers is None or len(tickers) > _DUCKDB_SQL_PUSHDOWN_TICKER_THRESHOLD)
    )

    if use_sql_pushdown:
        bucket = _DUCKDB_BUCKET_SQL[freq]
        query = f'''
            SELECT
                ticker,
                {bucket} AS datetime,
                arg_min(open, datetime) AS open,
                max(high) AS high,
                min(low) AS low,
                arg_max(close, datetime) AS close,
                sum(volume) AS volume
            FROM massive_1min
            {where_clause}
            GROUP BY ticker, {bucket}
            ORDER BY datetime
        '''
    else:
        query = f'SELECT * FROM massive_1min {where_clause} ORDER BY datetime'

    try:
        con = duckdb.connect(str(Path(__file__).parent / 'data' / 'ohlcv.duckdb'), read_only=True)
    except duckdb.IOException as e:
        raise RuntimeError(
            'Could not open data/ohlcv.duckdb read-only — it is likely locked by a '
            'currently-running Massive backfill (extraction/ohlcv_massive.py), which '
            'holds an exclusive read-write connection while fetching. Wait for it to '
            'finish (or pause it) and try again.'
        ) from e

    try:
        df = con.execute(query, params).df()
    finally:
        con.close()

    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index(['datetime', 'ticker'])

    per_ticker = {
        ticker: group.droplevel('ticker').rename(columns=str.title)
        for ticker, group in df.groupby(level='ticker')
    }

    if freq is not None and not use_sql_pushdown:
        rule = _DUCKDB_FREQ_ALIASES.get(freq, freq)
        agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        per_ticker = {
            ticker: frame.resample(rule).agg(agg).dropna(subset=['Close'])
            for ticker, frame in per_ticker.items()
        }

    if asvbt:
        return vbt.Data.from_data(per_ticker)

    if not asdf:
        return per_ticker

    return pd.concat(per_ticker, names=['ticker']).swaplevel().sort_index()



def _tracked_symbols() -> dict:
    """
    Reads ohlcv_1min.parquet and returns unique tickers bucketed by asset class,
    so symbols that drop out of the screener are still maintained.
    Returns {'cryptos': [...], 'equities': [...], 'etfs': [...]} or empty lists if the file doesn't exist.
    """
    if not OHLCV_PATH.exists():
        return {'cryptos': [], 'equities': [], 'etfs': []}

    tickers = pd.read_parquet(OHLCV_PATH, columns=[]).index.get_level_values('ticker').unique().tolist()

    # Crypto tickers on BinanceUS end with USDT/BUSD/BTC/ETH/BNB and are at least
    # 6 chars (2+ char base asset + 3-4 char quote asset) — this excludes short
    # equity tickers that happen to end in the same letters (e.g. 'ABNB').
    crypto_pattern = re.compile(r'^.{2,}(USDT|BUSD|BTC|ETH|BNB)$')
    cryptos = [t for t in tickers if crypto_pattern.search(t)]
    non_cryptos = [t for t in tickers if not crypto_pattern.search(t)]

    # Split non-cryptos into ETFs vs equities using yfinance quoteType in bulk isn't feasible,
    # so bucket everything non-crypto as equities — ETF distinction only matters for fundamentals,
    # and the screener results will correctly label them there.
    return {'cryptos': cryptos, 'equities': non_cryptos, 'etfs': []}


def screen_symbols():
    """
    Returns {'cryptos': [...], 'equities': [...], 'etfs': [...]}.
    Pulls the top symbols for each asset class from the TradingView screener,
    then merges in any previously tracked symbols from ohlcv_1min.parquet.
    """
    # Top 50 BinanceUS cryptos by 24h volume
    resp = requests.post(
        'https://scanner.tradingview.com/crypto/scan',
        headers={'User-Agent': 'Mozilla/5.0'},
        json={
            "filter": [
                {"left": "exchange", "operation": "equal", "right": "BINANCEUS"}
            ],
            "columns": ["name", "volume"],
            "sort": {"sortBy": "volume", "sortOrder": "desc"},
            "range": [0, 50]
        }
    )
    cryptos = [row['d'][0] for row in resp.json()['data']]
    # Drop bare-USD-quoted pairs (e.g. 'DOGEUSD', quoteAsset='USD') — they get
    # misclassified as equities by _tracked_symbols() since they don't end in
    # USDT/BUSD/BTC/ETH/BNB, and Alpaca (equities/ETFs) can't fetch them.
    # Checked against Binance.US's actual quoteAsset rather than a suffix
    # guess, since e.g. 'SHIBUSD' (base 'SHIB', quote 'USD') coincidentally
    # ends in the substring 'BUSD'.
    quote_by_symbol = {s['symbol']: s['quoteAsset'] for s in BinanceClient(tld='us').get_exchange_info()['symbols']}
    cryptos = [c for c in cryptos if quote_by_symbol.get(c) != 'USD']

    # Top 2000 US equities by market cap, requiring min 100k avg daily volume
    _, equities_df = (Query()
        .set_markets('america')
        .select('name', 'average_volume_60d_calc', 'market_cap_basic')
        .where(
            col('average_volume_60d_calc') >= 100_000,
            col('type') == 'stock',
            col('is_primary') == True,
        )
        .order_by('market_cap_basic', ascending=False)
        .limit(2000)
        .get_scanner_data()
    )
    equities = equities_df['name'].tolist()
    # TradingView uses '.' for share classes (e.g. BRK.B); yfinance expects '-'
    equities = [t.replace('.', '-').strip().upper() for t in equities]

    # Top 200 ETFs by AUM, filtered to $500M+ AUM and 100k+ avg volume
    etf_resp = requests.post(
        'https://scanner.tradingview.com/america/scan',
        headers={'User-Agent': 'Mozilla/5.0'},
        json={
            "filter": [
                {"left": "type", "operation": "equal", "right": "fund"},
                {"left": "typespecs", "operation": "has", "right": ["etf"]},
                {"left": "average_volume_60d_calc", "operation": "greater", "right": 100_000},
                {"left": "aum", "operation": "greater", "right": 500_000_000},
            ],
            "columns": ["name", "aum", "average_volume_60d_calc"],
            "sort": {"sortBy": "aum", "sortOrder": "desc"},
            "range": [0, 200]
        }
    )
    etfs = [row['d'][0] for row in etf_resp.json()['data']]

    tracked = _tracked_symbols()
    etf_set = set(etfs)
    return {
        'cryptos':  list(dict.fromkeys(cryptos  + tracked['cryptos'])),
        'equities': list(dict.fromkeys(equities + [t for t in tracked['equities'] if t not in etf_set])),
        'etfs':     list(dict.fromkeys(etfs)),
    }


def _compute_allocations_single(
    signals: pd.DataFrame,
    method: str,
    top_k: int | None,
    long_only: bool,
) -> pd.DataFrame:
    s = signals.fillna(0.0)

    if long_only:
        eligible = s.where(s > 0)
    else:
        eligible = s.abs().where(s != 0)

    if top_k is not None:
        eligible = eligible.where(
            eligible.rank(axis=1, ascending=False, method='first') <= top_k
        )

    if method == 'equal':
        weights = eligible.notna().astype(float)
    elif method == 'signal':
        weights = eligible.fillna(0.0)
    elif method == 'rank':
        weights = eligible.rank(axis=1, method='first', na_option='keep').fillna(0.0)
    elif method == 'inverse_rank':
        weights = (1.0 / eligible.rank(axis=1, method='first', ascending=False, na_option='keep')).fillna(0.0)

    row_sums = weights.sum(axis=1)
    weights = weights.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    if not long_only:
        weights *= np.sign(s)

    return weights


def compute_allocations(
    signals: pd.DataFrame,
    method: str | list[str] = "signal",
    top_k: int | list[int] | None = None,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Convert per-asset signal strengths into portfolio weight allocations.

    Parameters
    ----------
    signals : pd.DataFrame
        Rows = timestamps, columns = assets, values = signal strengths.
        NaN values are treated as no signal (zero allocation).
    method : str or list of str
        One of {'equal', 'signal', 'rank', 'inverse_rank'}, or a list of them
        to sweep. When a list, the result has a 'method' MultiIndex level.
    top_k : int, list of int, or None
        Restrict allocation to the top-k assets by signal strength.
        Pass a list to sweep; the result gains a 'top_k' MultiIndex level.
        If None, all assets with positive signal are eligible.
    long_only : bool
        If True (default), negative signals are floored to zero.
        If False, absolute signal value is used for sizing; sign is preserved
        and weights are normalised by the sum of absolute weights.

    Returns
    -------
    pd.DataFrame
        Flat if both method and top_k are scalars (rows sum to 1).
        MultiIndex columns when either is a list:
          - top_k list only  → levels ['top_k', asset]
          - method list only → levels ['method', asset]
          - both lists       → levels ['method', 'top_k', asset]
    """
    valid_methods = {"equal", "signal", "rank", "inverse_rank"}

    multi_method = isinstance(method, list)
    multi_top_k = isinstance(top_k, list)

    methods = method if multi_method else [method]
    top_ks = top_k if multi_top_k else [top_k]

    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{m}'")
    for k in top_ks:
        if k is not None and k < 1:
            raise ValueError("top_k must be >= 1")

    if not multi_method and not multi_top_k:
        return _compute_allocations_single(signals, methods[0], top_ks[0], long_only)

    combos = [
        ((m, k) if (multi_method and multi_top_k) else (m if multi_method else k), m, k)
        for m in methods for k in top_ks
    ]

    def _run(args):
        key, m, k = args
        return key, _compute_allocations_single(signals, m, k, long_only)

    with ThreadPoolExecutor(max_workers=len(combos)) as ex:
        frames = dict(ex.map(_run, combos))

    result = pd.concat(frames, axis=1)

    existing_names = list(signals.columns.names)
    if multi_method and multi_top_k:
        result.columns.names = ['method', 'top_k'] + existing_names
    elif multi_method:
        result.columns.names = ['method'] + existing_names
    else:
        result.columns.names = ['top_k'] + existing_names

    return result

def compact_view(alloc, combined=True):
    def top_positions(row):
        held = row[row > 0].sort_values(ascending=False)
        idx = range(1, len(held) + 1)
        tickers = pd.Series(held.index, index=idx)
        weights = pd.Series(held.values, index=idx)
        return tickers, weights

    t, w = zip(*alloc.apply(top_positions, axis=1))
    tickers = pd.DataFrame(list(t), index=alloc.index)
    weights = pd.DataFrame(list(w), index=alloc.index)
    if combined:
            return tickers + ' (' + weights.map(lambda x: f'{x:.1%}' if pd.notna(x) else '') + ')'
    return tickers, weights