"""
extraction/ohlcv_robinhood.py — 1-min equity/ETF OHLCV bars from Robinhood
(via the Robinhood MCP trading server, reached directly through
robinhood_client.RobinhoodMCPClient, bypassing Claude Code), written into
data/ohlcv.duckdb::robinhood_1min. Two entry points:

  - build_robinhood_ohlcv() — resumable historical backfill over an
    arbitrary date range, checkpointed with a per-ticker progress ledger
    (PROGRESS_PATH). Run manually: `python -m extraction.ohlcv_robinhood`.
  - download_daily_robinhood() — daily incremental pull of just the most
    recent day (plus catch-up if a run was missed), tracked with a single
    last-run date (LAST_RUN_PATH) instead of the per-ticker ledger, since a
    daily task must always refetch the target day rather than skip tickers
    already marked done. Run manually: `python -m extraction.ohlcv_robinhood
    --daily`; wired into airflow/dags/daily_run_dag.py for the scheduled run.

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
LAST_RUN_PATH = Path(__file__).parent.parent / 'data' / 'ohlcv_robinhood_last_run.txt'
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


def _read_last_run() -> date | None:
    if not LAST_RUN_PATH.exists():
        return None
    return date.fromisoformat(LAST_RUN_PATH.read_text().strip())


def _write_last_run(d: date) -> None:
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(d.isoformat())


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


def _massive_last_date() -> date | None:
    """
    Returns the most recent date in ohlcv.duckdb::massive_1min (None if the
    DB/table doesn't exist or is empty) -- used as build_robinhood_ohlcv()'s
    default start date, so the Robinhood backfill picks up right where the
    Massive backfill's coverage ends rather than an arbitrary fixed lookback.
    """
    if not DUCKDB_PATH.exists():
        return None
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute('SHOW TABLES').fetchall()}
        if 'massive_1min' not in tables:
            return None
        row = con.execute('SELECT MAX(datetime) FROM massive_1min').fetchone()
    finally:
        con.close()
    return row[0].date() if row and row[0] is not None else None


def build_robinhood_ohlcv(tickers: list[str] | None = None, start: date | None = None,
                           end: date | None = None, force: bool = False,
                           checkpoint_every: int = CHECKPOINT_EVERY) -> None:
    """
    Backfills 1-min bars for `tickers` (default: tracked equities + ETFs)
    from Robinhood, batching up to RH_BATCH_SIZE symbols and RH_CHUNK_DAYS
    days per call. Resumable: skips tickers already marked 'done' in
    PROGRESS_PATH unless force=True.

    end defaults to yesterday. start defaults to massive_1min's last date
    (extending that backfill's coverage forward instead of leaving a gap
    between it and the daily Robinhood pull), falling back to 90 days
    before end if massive_1min doesn't exist or is empty. Both resolved
    fresh on every call, not frozen at import time.
    """
    end = end or (date.today() - timedelta(days=1))
    if start is None:
        start = _massive_last_date() or (end - timedelta(days=90))

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


def download_daily_robinhood(tickers: list[str] | None = None) -> None:
    """
    Pulls today's day of 1-min bars (bounds=RH_BOUNDS) for `tickers`
    (default: tracked equities + ETFs) into data/ohlcv.duckdb::robinhood_1min,
    deleting-then-reinserting only the exact (datetime, ticker) rows being
    written -- same dedup pattern as extraction.ohlcv.download_daily_1min --
    so re-running the same day is always safe.

    end defaults to *today*, not yesterday: this is meant to be triggered
    at/after that day's market close (matching download_daily_1min's
    rolling-24h-to-now window, which for the same reason ends up covering
    the session that just closed), so "today" is the session it should
    capture. Called on a non-trading day (weekend/holiday) it just returns
    0 bars -- harmless, but the caller (see airflow/dags/daily_run_dag.py)
    should still gate on validation.is_market_day() to skip the wasted
    calls, same as the Alpaca pull did.

    Unlike build_robinhood_ohlcv(), this always (re)fetches the target range
    rather than skipping tickers already marked done: a daily task needs
    fresh data every run, not a one-time resumable backfill, so it tracks
    progress with a single last-run date (LAST_RUN_PATH) instead of the
    per-ticker PROGRESS_PATH ledger used by the backfill.

    If the last successful run was more than a day ago (e.g. the scheduler
    was paused), backfills every missed day up through today instead of
    just today alone, chunked via RH_CHUNK_DAYS/_date_chunks like the
    backfill. One batch's failure (after _call_historicals' own retries) is
    logged and skipped rather than aborting the run -- a bad symbol or a
    transient outage doesn't block the other tickers, and gets picked up
    again on the next scheduled run since the last-run marker still
    advances (see below).
    """
    end = date.today()
    last_run = _read_last_run()
    start = min(last_run + timedelta(days=1), end) if last_run else end

    if tickers is None:
        tracked = _tracked_symbols()
        tickers = sorted(set(tracked['equities']) | set(tracked['etfs']))

    print(f'Pulling {start} .. {end} for {len(tickers)} tickers (bounds={RH_BOUNDS})')

    client = RobinhoodMCPClient()
    con = _connect()
    n_ok, n_failed = 0, 0

    n_batches = (len(tickers) + RH_BATCH_SIZE - 1) // RH_BATCH_SIZE
    for bi in range(0, len(tickers), RH_BATCH_SIZE):
        batch = tickers[bi:bi + RH_BATCH_SIZE]
        batch_num = bi // RH_BATCH_SIZE + 1
        try:
            chunk_frames = [
                _bars_to_frame(_call_historicals(client, batch, chunk_start, chunk_end))
                for chunk_start, chunk_end in _date_chunks(start, end, RH_CHUNK_DAYS)
            ]
            batch_df = pd.concat(chunk_frames) if chunk_frames else pd.DataFrame()
        except Exception as e:
            print(f'[batch {batch_num}/{n_batches}] {batch}: FAILED -- {e}')
            n_failed += len(batch)
            continue

        if len(batch_df):
            new_data = batch_df.reset_index()
            con.register('_new_data', new_data)
            con.execute(f'''
                DELETE FROM {STAGING_TABLE}
                WHERE (datetime, ticker) IN (SELECT datetime, ticker FROM _new_data)
            ''')
            con.execute(f'INSERT INTO {STAGING_TABLE} SELECT * FROM _new_data')
            con.unregister('_new_data')
        n_ok += len(batch)
        print(f'[batch {batch_num}/{n_batches}] {batch}: {len(batch_df)} bars')

    n_rows = con.execute(f'SELECT COUNT(*) FROM {STAGING_TABLE}').fetchone()[0]
    con.close()

    # Advance the last-run marker even if some batches failed -- otherwise a
    # single bad symbol would wedge every future run into re-attempting the
    # same window forever. Failed tickers just get picked up again on the
    # next scheduled run, same as a transient miss in download_daily_1min.
    _write_last_run(end)
    print(f'Done. {n_ok} tickers ok, {n_failed} failed, {n_rows:,} total rows in {DUCKDB_PATH.name}::{STAGING_TABLE}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Backfill or daily-pull 1-min equity/ETF OHLCV from Robinhood.')
    p.add_argument('--daily', action='store_true',
                    help='Run download_daily_robinhood() (last day / missed-day catch-up) instead of the resumable backfill.')
    p.add_argument('--start', type=date.fromisoformat, default=None,
                    help='Backfill mode only. Defaults to massive_1min\'s last date '
                         '(90 days before --end if that table is missing/empty).')
    p.add_argument('--end', type=date.fromisoformat, default=None,
                    help='Backfill mode only. Defaults to yesterday.')
    p.add_argument('--force', action='store_true',
                    help='Backfill mode only. Re-fetch tickers already marked done.')
    args = p.parse_args()

    if args.daily:
        download_daily_robinhood()
    else:
        build_robinhood_ohlcv(start=args.start, end=args.end, force=args.force)
