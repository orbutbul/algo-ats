"""Read-only daily OHLCV from data/ohlcv.duckdb::ohlcv_max_daily."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OHLCV_DUCKDB_PATH = REPO_ROOT / 'data' / 'ohlcv.duckdb'
OHLCV_MAX_TABLE = 'ohlcv_max_daily'


def _connect_readonly(path: Path):
    import duckdb

    if not path.exists():
        raise FileNotFoundError(
            f'No DuckDB at {path}. Expected table {OHLCV_MAX_TABLE}. '
            'Create it with: python -m vcp seed --tickers NVDA,AMD '
            'or extraction.ohlcv.get_max_data([...]).'
        )
    return duckdb.connect(str(path), read_only=True)


def load_daily_ohlcv(
    ticker: str,
    path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return a single-ticker daily OHLCV frame indexed by datetime (UTC)."""
    path = Path(path) if path is not None else OHLCV_DUCKDB_PATH
    ticker = ticker.strip().upper()
    con = _connect_readonly(path)
    try:
        tables = [row[0] for row in con.execute('SHOW TABLES').fetchall()]
        if OHLCV_MAX_TABLE not in tables:
            raise LookupError(
                f'{path} has no table {OHLCV_MAX_TABLE}. Tables present: {tables or "(none)"}.'
            )
        sql = f'''
            SELECT datetime, open, high, low, close, volume
            FROM {OHLCV_MAX_TABLE}
            WHERE ticker = ?
        '''
        params: list = [ticker]
        if start:
            sql += ' AND datetime >= ?'
            params.append(start)
        if end:
            sql += ' AND datetime <= ?'
            params.append(end)
        sql += ' ORDER BY datetime'
        frame = con.execute(sql, params).df()
    finally:
        con.close()

    if frame.empty:
        raise LookupError(f'No rows for {ticker} in {path}::{OHLCV_MAX_TABLE}')

    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
    frame = frame.set_index('datetime').sort_index()
    frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return frame


def list_tickers(path: Path | None = None) -> list[str]:
    path = Path(path) if path is not None else OHLCV_DUCKDB_PATH
    con = _connect_readonly(path)
    try:
        rows = con.execute(f'SELECT DISTINCT ticker FROM {OHLCV_MAX_TABLE} ORDER BY 1').fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _ohlcv_fields_from_frame(part: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single-ticker OHLCV frame to lower-case open/high/low/close/volume."""
    out = part.copy()
    if isinstance(out.columns, pd.MultiIndex):
        fields = {'open', 'high', 'low', 'close', 'adj close', 'volume'}
        chosen = None
        for level, name in enumerate(out.columns.names or []):
            if name is not None and str(name).lower() == 'price':
                chosen = level
                break
        if chosen is None:
            for level in range(out.columns.nlevels):
                vals = {str(v).lower() for v in out.columns.get_level_values(level)}
                if vals & fields:
                    chosen = level
                    break
        if chosen is None:
            chosen = out.columns.nlevels - 1
        out.columns = out.columns.get_level_values(chosen)
    out.columns = [str(c).strip().lower().replace('adj close', 'close') for c in out.columns]
    return out


def _frames_from_yfinance(raw: pd.DataFrame, tickers: list[str]) -> list[pd.DataFrame]:
    """Split a yfinance download into per-ticker flat OHLCV frames."""
    frames: list[pd.DataFrame] = []
    if raw.empty:
        return frames

    if not isinstance(raw.columns, pd.MultiIndex):
        if len(tickers) != 1:
            raise RuntimeError(f'Expected MultiIndex columns for tickers {tickers}, got {list(raw.columns)}')
        part = _ohlcv_fields_from_frame(raw)
        part['ticker'] = tickers[0]
        frames.append(part.rename_axis('datetime').reset_index())
        return frames

    level0 = {str(v).upper() for v in raw.columns.get_level_values(0)}
    # (Ticker, Price) when group_by='ticker'
    if any(t in level0 for t in tickers):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                # yfinance sometimes lowercases unexpectedly; try casefold match.
                matches = [v for v in raw.columns.get_level_values(0) if str(v).upper() == t]
                if not matches:
                    continue
                key = matches[0]
            else:
                key = t
            part = _ohlcv_fields_from_frame(raw.xs(key, axis=1, level=0))
            part['ticker'] = t
            frames.append(part.rename_axis('datetime').reset_index())
        return frames

    # (Price, Ticker)
    for t in tickers:
        keys = list(raw.columns.get_level_values(-1))
        if t in keys:
            key = t
        else:
            matches = [v for v in keys if str(v).upper() == t]
            if not matches:
                continue
            key = matches[0]
        part = _ohlcv_fields_from_frame(raw.xs(key, axis=1, level=-1))
        part['ticker'] = t
        frames.append(part.rename_axis('datetime').reset_index())
    return frames


def seed_daily_ohlcv(
    tickers: list[str],
    path: Path | None = None,
    period: str = 'max',
) -> Path:
    """Download yfinance daily bars into ohlcv_max_daily (creates DB if needed).

    Same adjustment policy as extraction.ohlcv.get_max_data (auto_adjust=True).
    Write path — used to bootstrap a local research DB when data/ is empty.
    """
    import duckdb
    import yfinance as yf

    path = Path(path) if path is not None else OHLCV_DUCKDB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError('No tickers to seed')

    raw = yf.download(
        tickers=tickers,
        interval='1d',
        auto_adjust=True,
        progress=False,
        threads=True,
        period=period,
        group_by='ticker',
    )
    if raw.empty:
        raise RuntimeError(f'yfinance returned no data for {tickers}')

    frames = _frames_from_yfinance(raw, tickers)
    if not frames:
        raise RuntimeError(f'Could not parse yfinance payload for {tickers}')

    new_data = pd.concat(frames, ignore_index=True)
    # yfinance may name the index column Date instead of datetime after reset_index.
    if 'datetime' not in new_data.columns:
        for cand in ('date', 'Date', 'index'):
            if cand in new_data.columns:
                new_data = new_data.rename(columns={cand: 'datetime'})
                break
    keep = ['datetime', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    for col in keep:
        if col not in new_data.columns:
            raise RuntimeError(f'seed missing column {col}: {list(new_data.columns)}')
    new_data = new_data[keep].dropna(subset=['close'])
    new_data['datetime'] = pd.to_datetime(new_data['datetime'], utc=True)

    con = duckdb.connect(str(path))
    try:
        con.execute(f'''
            CREATE TABLE IF NOT EXISTS {OHLCV_MAX_TABLE} (
                datetime TIMESTAMPTZ NOT NULL,
                ticker   VARCHAR NOT NULL,
                open     DOUBLE,
                high     DOUBLE,
                low      DOUBLE,
                close    DOUBLE,
                volume   DOUBLE
            )
        ''')
        con.register('_new_data', new_data)
        con.execute(f'''
            DELETE FROM {OHLCV_MAX_TABLE}
            WHERE (datetime, ticker) IN (SELECT datetime, ticker FROM _new_data)
        ''')
        con.execute(f'INSERT INTO {OHLCV_MAX_TABLE} SELECT * FROM _new_data')
        con.unregister('_new_data')
    finally:
        con.close()
    return path
