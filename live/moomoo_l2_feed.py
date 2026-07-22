"""
live/moomoo_l2_feed.py — long-running process that subscribes to moomoo
Level 2 order book pushes for a given ticker list and persists snapshots to
Parquet under data/moomoo_l2/<date>/.

Requires OpenD running and logged in (see live/moomoo_l2.py's module
docstring for setup). Run directly:

    python -m live.moomoo_l2_feed AAPL,NVDA,MSFT

or import run_feed(tickers) to embed it elsewhere.

Design notes (see conversation history / benchmarking in
scratch_setup/moomoo_l2_size_test.py for the numbers behind these choices):

- moomoo pushes a full order-book snapshot on every book change, not a
  delta — for 100 actively-traded tickers at full depth that's ~2.4M
  rows/min, ~2.8GB/day. Writing every push is wasteful for most strategy
  use cases, so this feed instead keeps only the *latest* pushed snapshot
  per ticker in memory (cheap, just an overwrite) and samples it once per
  SNAPSHOT_INTERVAL_S on a timer, which is the actual write cadence.

- Depth is adaptively bucketed: moomoo's push depth varies per ticker (thin
  names may show ~10 levels, liquid ones can show 40-60). Tickers already
  at or under MAX_LEVELS are stored raw/unmodified. Tickers with deeper
  books are compressed into MAX_LEVELS price buckets (equal-width across
  that ticker's pushed price range, sizes summed per bucket) rather than
  applying one fixed cent-width to every ticker, since a fixed width would
  either be too coarse for cheap stocks or too fine for expensive ones.

- Each flush writes one *complete, independent* Parquet file (named by
  flush timestamp) rather than appending row groups to one growing
  per-day file. A single open-and-append file is corrupted beyond
  recovery if the process dies before its footer is written (only
  close() writes it) — verified empirically: a hard-killed run here left
  an unreadable "Parquet magic bytes not found" file, losing the whole
  day, not just the unflushed tail. Many small self-contained files means
  a crash only ever loses the current in-flight buffer (bounded by
  FLUSH_INTERVAL_S). Read a day back with
  pd.read_parquet('data/moomoo_l2/<date>/') — pyarrow reads a directory
  of Parquet files as one dataset.
"""

import logging
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from moomoo import OpenQuoteContext, OrderBookHandlerBase, RET_OK, SubType

OPEND_HOST = '127.0.0.1'
OPEND_PORT = 11111

MAX_LEVELS = 10          # per side; deeper books get bucketed down to this
SNAPSHOT_INTERVAL_S = 1.0
FLUSH_INTERVAL_S = 30.0

DATA_DIR = Path('data/moomoo_l2')
PID_PATH = Path('data/moomoo_l2_feed.pid')
LOG_DIR = Path('logs')

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / 'moomoo_l2_feed.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('moomoo_l2_feed')

ROW_SCHEMA = pa.schema([
    ('timestamp', pa.timestamp('ns')),
    ('ticker', pa.string()),
    ('side', pa.string()),
    ('level', pa.int32()),
    ('price', pa.float64()),
    ('size', pa.float64()),
])


def _bucket_levels(levels: list, max_levels: int) -> list[tuple[float, float]]:
    """Returns [(price, size), ...], at most max_levels entries. Levels at
    or under max_levels pass through unchanged; deeper books are aggregated
    into max_levels equal-width price buckets with sizes summed per bucket."""
    if len(levels) <= max_levels:
        return [(lvl[0], lvl[1]) for lvl in levels]

    prices = [lvl[0] for lvl in levels]
    lo, hi = min(prices), max(prices)
    width = (hi - lo) / max_levels if hi > lo else 1.0
    bucket_sizes = [0.0] * max_levels
    for price, size, *_ in levels:
        idx = min(int((price - lo) / width), max_levels - 1)
        bucket_sizes[idx] += size
    return [(lo + width * (i + 0.5), bucket_sizes[i]) for i in range(max_levels)]


class _LatestBookStore:
    """Thread-safe holder for the most recent push per ticker — the
    snapshot timer reads from this, the push handler only ever overwrites
    it, so neither side blocks on the other for long."""

    def __init__(self):
        self._lock = threading.Lock()
        self._books: dict[str, dict] = {}

    def update(self, code: str, data: dict) -> None:
        with self._lock:
            self._books[code] = data

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._books)


class _ParquetDayWriter:
    """Writes each flush as its own complete Parquet file under a
    per-day subdirectory (data/moomoo_l2/<date>/<HHMMSS_ffffff>.parquet).
    No open file handle is held between flushes, so there's nothing to
    corrupt on a hard kill — see module docstring for why that matters."""

    def __init__(self, out_dir: Path):
        self._out_dir = out_dir

    def write(self, df: pd.DataFrame) -> None:
        now = pd.Timestamp.now()
        day_dir = self._out_dir / now.date().isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f'{now.strftime("%H%M%S_%f")}.parquet'
        table = pa.Table.from_pandas(df, schema=ROW_SCHEMA, preserve_index=False)
        pq.write_table(table, path, compression='zstd')

    def close(self) -> None:
        pass  # no persistent handle to release


def run_feed(tickers: list[str], out_dir: str = str(DATA_DIR),
             max_levels: int = MAX_LEVELS,
             snapshot_interval: float = SNAPSHOT_INTERVAL_S,
             flush_interval: float = FLUSH_INTERVAL_S) -> None:
    codes = [f'US.{t}' for t in tickers]
    store = _LatestBookStore()

    class _Handler(OrderBookHandlerBase):
        def on_recv_rsp(self, rsp_str):
            ret, data = super().on_recv_rsp(rsp_str)
            if ret != RET_OK:
                log.warning('order book push error: %s', data)
                return RET_OK, data
            store.update(data['code'], data)
            return RET_OK, data

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    import os
    PID_PATH.write_text(str(os.getpid()))
    log.info('PID %d written to %s', os.getpid(), PID_PATH)

    ctx = OpenQuoteContext(host=OPEND_HOST, port=OPEND_PORT)
    ctx.set_handler(_Handler())
    ret, err = ctx.subscribe(code_list=codes, subtype_list=[SubType.ORDER_BOOK])
    if ret != RET_OK:
        ctx.close()
        raise RuntimeError(f'subscribe failed: {err}')
    log.info('Subscribed to %d tickers for Level 2 order book', len(codes))

    writer = _ParquetDayWriter(Path(out_dir))
    buffer_rows: list[tuple] = []
    last_flush = time.monotonic()
    stop_event = threading.Event()

    def _flush():
        nonlocal buffer_rows, last_flush
        if buffer_rows:
            df = pd.DataFrame(buffer_rows, columns=[f.name for f in ROW_SCHEMA])
            writer.write(df)
            log.info('Flushed %d rows to Parquet', len(buffer_rows))
            buffer_rows = []
        last_flush = time.monotonic()

    try:
        while not stop_event.is_set():
            time.sleep(snapshot_interval)
            ts = pd.Timestamp.now()
            for code, data in store.snapshot().items():
                for side_key, side_name in (('Bid', 'bid'), ('Ask', 'ask')):
                    for level, (price, size) in enumerate(_bucket_levels(data[side_key], max_levels)):
                        buffer_rows.append((ts, code, side_name, level, price, size))

            if time.monotonic() - last_flush >= flush_interval:
                _flush()
    except KeyboardInterrupt:
        log.info('Stopping feed (KeyboardInterrupt)...')
    finally:
        _flush()
        writer.close()
        ctx.unsubscribe(code_list=codes, subtype_list=[SubType.ORDER_BOOK])
        ctx.close()
        if PID_PATH.exists():
            PID_PATH.unlink()
        log.info('Feed stopped, resources cleaned up.')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        tickers_arg = [t.strip().upper() for t in sys.argv[1].split(',') if t.strip()]
    else:
        tickers_arg = ['AAPL', 'NVDA']
        log.info('No tickers passed on the command line, defaulting to %s', tickers_arg)
    run_feed(tickers_arg)
