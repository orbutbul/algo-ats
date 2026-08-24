"""
extraction/ohlcv_robinhood.py — resumable historical backfill of 1-min
equity/ETF OHLCV bars from Robinhood (via the Robinhood MCP trading server,
reached directly through robinhood_client.RobinhoodMCPClient, bypassing
Claude Code). Checkpointed into a DuckDB staging table
(data/ohlcv.duckdb::robinhood_1min), resumable via a per-ticker progress
ledger. Run manually: `python -m extraction.ohlcv_robinhood`.

Mirrors extraction/ohlcv_massive.py's backfill pattern (progress ledger,
checkpointed delete-then-insert flushes, per-ticker error tracking) with a
few deliberate differences:
  - get_equity_historicals batches up to 10 symbols per call (Massive is
    one ticker per call), and its ~5,000-bar cap applies per symbol
    regardless of batch size (confirmed empirically), so requests are
    chunked across both symbols (RH_BATCH_SIZE) and date ranges
    (RH_CHUNK_DAYS).
  - The default end date is resolved at call time to "yesterday", not
    frozen as a module-level constant — Massive's hardcoded DEFAULT_END
    goes stale the day after it's written; this doesn't.
  - Pulls bounds='24_5' (full 24h, Mon-Fri) rather than the tool's default
    'regular' (RTH-only) bounds, so pre/post-market and overnight bars are
    captured too. This roughly quadruples bar density per day (~1,440
    bars/weekday vs. ~390 for regular hours, confirmed empirically), which
    is why RH_CHUNK_DAYS is much smaller than a regular-hours backfill
    would need to stay under the per-symbol bar cap.
  - Progress records store the `bounds` they were fetched with, and a
    ticker is only treated as already-done if its recorded bounds matches
    RH_BOUNDS -- so switching RH_BOUNDS later (e.g. back to 'regular')
    automatically re-fetches everything under the new setting instead of
    silently skipping tickers done under the old one.

Does NOT touch crypto or the daily incremental Alpaca pull in
extraction/ohlcv.py — this only backfills equity/ETF history from
Robinhood. No --promote step: this only populates the robinhood_1min
DuckDB table and stops there.
"""

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd

from extraction.ohlcv import OHLCV_COLS
from robinhood_client import RobinhoodMCPClient
from utils import _tracked_symbols

DUCKDB_PATH = Path(__file__).parent.parent / 'data' / 'ohlcv.duckdb'
STAGING_TABLE = 'robinhood_1min'
PROGRESS_PATH = Path(__file__).parent.parent / 'data' / 'ohlcv_robinhood_progress.json'
RH_BATCH_SIZE = 10        # symbols per get_equity_historicals call (MCP tool's hard cap)
RH_BOUNDS = '24_5'        # full 24h Mon-Fri, not just regular trading hours
RH_CHUNK_DAYS = 3         # calendar days per request window -- at bounds='24_5' that's up
                           # to ~4,320 minute bars/symbol (confirmed empirically: 4 calendar
                           # days = 5,760 bars was rejected, 3 = ~4,320 was not), safely under
                           # the empirically-confirmed 5,000-bar-per-symbol cap
RETRY_ATTEMPTS = 5
RETRY_BACKOFF_BASE = 15   # seconds; same schedule as ohlcv_massive._get(): 15 * (attempt + 1)
CHECKPOINT_EVERY = 25     # tickers per DuckDB flush, same meaning as ohlcv_massive.py

# Robinhood's share-class ticker convention (dot vs. hyphen, e.g. 'BF-A') is
# unverified against Alpaca's, so tickers are passed through unchanged rather
# than routed through extraction.ohlcv._to_alpaca_symbol.


class _ToolError(Exception):
    """
    Raised when the MCP tool call itself reports isError=true (e.g. an
    invalid/missing symbol -> 404). This is deterministic, not transient --
    retrying it wastes the full backoff schedule for no benefit, so it's
    raised immediately rather than going through the retry loop below.
    """
    pass


def _call_historicals(client: RobinhoodMCPClient, symbols: list[str], start: date, end: date,
                       retries: int = RETRY_ATTEMPTS) -> dict:
    """
    Calls get_equity_historicals for a batch of symbols over [start, end],
    retrying with a 15*(attempt+1)s backoff on transient failures. Returns
    the tool's already-parsed structuredContent payload (shape:
    {'data': {'results': [{'symbol', 'interval', 'bounds', 'bars'}, ...]}}).
    """
    arguments = {
        'symbols': symbols,
        'start_time': f'{start.isoformat()}T00:00:00Z',
        'end_time': f'{end.isoformat()}T23:59:59Z',
        'interval': 'minute',
        'bounds': RH_BOUNDS,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.call_tool('get_equity_historicals', arguments)
            result = resp['result']
            if result.get('isError'):
                text = (result.get('content') or [{}])[0].get('text', 'unknown tool error')
                raise _ToolError(text)
            return result['structuredContent']
        except _ToolError:
            raise
        except Exception as e:
            last_err = e
            wait = RETRY_BACKOFF_BASE * (attempt + 1)
            print(f'  historicals call failed ({e}) -- retrying in {wait}s')
            time.sleep(wait)
    raise RuntimeError(f'exceeded retries fetching {symbols} {start}..{end}: {last_err}')


def _bars_to_frame(payload: dict) -> pd.DataFrame:
    """
    Flattens a get_equity_historicals payload into a DataFrame indexed
    (datetime, ticker) with OHLCV_COLS. Drops interpolated (gap-fill) bars
    per the tool's own guidance that they carry no new information.
    """
    rows = []
    for result in payload.get('data', {}).get('results', []):
        symbol = result['symbol']
        for bar in result.get('bars', []):
            if bar.get('interpolated'):
                continue
            rows.append({
                'datetime': bar['begins_at'],
                'ticker': symbol,
                'open': float(bar['open_price']),
                'high': float(bar['high_price']),
                'low': float(bar['low_price']),
                'close': float(bar['close_price']),
                'volume': float(bar['volume']),
            })

    if not rows:
        empty = pd.DataFrame(columns=OHLCV_COLS)
        empty.index = pd.MultiIndex.from_arrays([[], []], names=['datetime', 'ticker'])
        return empty

    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    return df.set_index(['datetime', 'ticker'])[OHLCV_COLS].sort_index()


def _date_chunks(start: date, end: date, chunk_days: int):
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, default=str))


def _connect() -> duckdb.DuckDBPyConnection:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (
            datetime TIMESTAMPTZ NOT NULL,
            ticker   VARCHAR NOT NULL,
            open     DOUBLE,
            high     DOUBLE,
            low      DOUBLE,
            close    DOUBLE,
            volume   DOUBLE
        )
    """)
    con.execute(f'CREATE INDEX IF NOT EXISTS idx_{STAGING_TABLE}_ticker ON {STAGING_TABLE}(ticker)')
    return con


def build_robinhood_ohlcv(tickers: list[str] | None = None, start: date | None = None,
                           end: date | None = None, force: bool = False,
                           checkpoint_every: int = CHECKPOINT_EVERY) -> None:
    """
    Backfills 1-min bars for `tickers` (default: tracked equities + ETFs)
    from Robinhood, batching up to RH_BATCH_SIZE symbols and RH_CHUNK_DAYS
    days per call. Resumable: skips tickers already marked 'done' in
    PROGRESS_PATH unless force=True.

    end defaults to yesterday, start to 90 days before end -- both resolved
    fresh on every call, not frozen at import time.
    """
    end = end or (date.today() - timedelta(days=1))
    start = start or (end - timedelta(days=90))

    if tickers is None:
        tracked = _tracked_symbols()
        tickers = sorted(set(tracked['equities']) | set(tracked['etfs']))

    progress = _load_progress()

    def _is_done(ticker: str) -> bool:
        rec = progress.get(ticker, {})
        return rec.get('status') == 'done' and rec.get('bounds') == RH_BOUNDS

    remaining = [t for t in tickers if force or not _is_done(t)]
    print(f'{len(tickers) - len(remaining)}/{len(tickers)} already done at bounds={RH_BOUNDS}, {len(remaining)} remaining')

    client = RobinhoodMCPClient()
    con = _connect()
    pending_frames: list[pd.DataFrame] = []
    pending_tickers: list[str] = []

    def _flush() -> None:
        nonlocal pending_frames, pending_tickers
        if not pending_tickers:
            return
        non_empty = [f for f in pending_frames if len(f)]
        if non_empty:
            new_data = pd.concat(non_empty).reset_index()
            data_tickers = new_data['ticker'].unique().tolist()
            # Delete-then-insert only the tickers with actual new rows, so a
            # re-run (or --force) never duplicates a ticker's rows -- no read
            # of existing data required.
            placeholders = ', '.join('?' * len(data_tickers))
            con.execute(f'DELETE FROM {STAGING_TABLE} WHERE ticker IN ({placeholders})', data_tickers)
            con.register('_new_data', new_data)
            con.execute(f'INSERT INTO {STAGING_TABLE} SELECT * FROM _new_data')
            con.unregister('_new_data')
        n_rows = con.execute(f'SELECT COUNT(*) FROM {STAGING_TABLE}').fetchone()[0]
        # Persist progress only after any new data is safely on disk, so a
        # crash between flush and progress-save can't mark a ticker 'done'
        # for data that isn't actually there.
        _save_progress(progress)
        print(f'  checkpoint: {len(pending_tickers)} tickers flushed, {n_rows:,} rows in {DUCKDB_PATH.name}::{STAGING_TABLE}')
        pending_frames = []
        pending_tickers = []

    n_batches = (len(remaining) + RH_BATCH_SIZE - 1) // RH_BATCH_SIZE
    for bi in range(0, len(remaining), RH_BATCH_SIZE):
        batch = remaining[bi:bi + RH_BATCH_SIZE]
        batch_num = bi // RH_BATCH_SIZE + 1
        try:
            chunk_frames = []
            for chunk_start, chunk_end in _date_chunks(start, end, RH_CHUNK_DAYS):
                payload = _call_historicals(client, batch, chunk_start, chunk_end)
                chunk_frames.append(_bars_to_frame(payload))
            batch_df = pd.concat(chunk_frames) if chunk_frames else pd.DataFrame()
        except Exception as e:
            for ticker in batch:
                progress[ticker] = {
                    'status': 'error', 'n_bars': 0, 'first': None, 'last': None,
                    'error': str(e), 'bounds': RH_BOUNDS,
                    'fetched_at': datetime.now(timezone.utc).isoformat(),
                }
            print(f'[batch {batch_num}/{n_batches}] {batch}: FAILED -- {e}')
            pending_tickers.extend(batch)
            if len(pending_tickers) >= checkpoint_every or bi + RH_BATCH_SIZE >= len(remaining):
                _flush()
            continue

        batch_tickers_present = set(batch_df.index.get_level_values('ticker')) if len(batch_df) else set()
        for ticker in batch:
            if ticker in batch_tickers_present:
                ticker_df = batch_df.xs(ticker, level='ticker', drop_level=False)
                dates = ticker_df.index.get_level_values('datetime')
                progress[ticker] = {
                    'status': 'done', 'n_bars': len(ticker_df),
                    'first': dates.min().isoformat(), 'last': dates.max().isoformat(),
                    'error': None, 'bounds': RH_BOUNDS,
                    'fetched_at': datetime.now(timezone.utc).isoformat(),
                }
            else:
                # Robinhood returned no bars for this ticker/window -- no
                # distinct "not found" signal exists in a batch response the
                # way Massive's HTTP 404 gave one, so just record zero bars.
                progress[ticker] = {
                    'status': 'done', 'n_bars': 0, 'first': None, 'last': None,
                    'error': None, 'bounds': RH_BOUNDS,
                    'fetched_at': datetime.now(timezone.utc).isoformat(),
                }
        print(f'[batch {batch_num}/{n_batches}] {batch}: {len(batch_df)} bars')

        pending_frames.append(batch_df)
        pending_tickers.extend(batch)
        if len(pending_tickers) >= checkpoint_every or bi + RH_BATCH_SIZE >= len(remaining):
            _flush()

    con.close()

    n_done = sum(1 for v in progress.values() if v['status'] == 'done')
    n_error = sum(1 for v in progress.values() if v['status'] == 'error')
    print(f'Done. {n_done} succeeded, {n_error} failed. See {PROGRESS_PATH} for per-ticker detail.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Backfill 1-min equity/ETF OHLCV from Robinhood (resumable).')
    p.add_argument('--start', type=date.fromisoformat, default=None,
                    help='Defaults to 90 days before --end.')
    p.add_argument('--end', type=date.fromisoformat, default=None,
                    help='Defaults to yesterday.')
    p.add_argument('--force', action='store_true', help='Re-fetch tickers already marked done.')
    args = p.parse_args()

    build_robinhood_ohlcv(start=args.start, end=args.end, force=args.force)
