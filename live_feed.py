"""
live_feed.py — long-running process meant to run continuously (Task
Scheduler starts it at boot and restarts it daily at 3AM ET to refresh the
symbol universe — see RollingBarCache below for why a restart doesn't lose
data). It doesn't gate on market hours: outside trading sessions Alpaca
simply has no bars to send, so it idles harmlessly rather than exiting.

Subscribes to live 1-minute bars for the screener + conviction-watchlist
universe via Alpaca's IEX websocket feed (free tier, real-time, single
exchange), and republishes each bar immediately over a local ZeroMQ PUB
socket (see live_config.PUB_ADDRESS) — one message per bar, topic-filtered
by ticker, so a strategy process can subscribe to just the symbols it needs
with sub-millisecond local latency. No parquet persistence: this process is
push-only.

Bars are also kept in a rolling in-memory cache (RollingBarCache, retention
set by run_live_feed's history_days param, default 1 day) and served on
request over a second socket (live_config.HISTORY_ADDRESS), so a client that
connects mid-stream can catch up on recent bars before switching to the live
PUB/SUB stream — PUB/SUB itself has no replay. On every (re)start, the cache
is also backfilled from data/ohlcv_1min.parquet for the same history_days
window (see _seed_history), so it's never empty right after a restart —
that file only refreshes once at 5PM ET though, so the backfill is "as of
the last EOD pull," not perfectly current; live bars take over immediately.

Alpaca's free tier docs state websocket streaming is capped at 30 symbols,
but that was empirically tested against this paper account on 2026-07-16
and does NOT apply to bars-only subscriptions: a 200-symbol subscribe_bars
call ran cleanly for 90s with no "insufficient subscription" error and
correctly received bars for 146/200 symbols (the rest simply hadn't printed
a trade on IEX in that short window — not a rejection). Default behavior
below (use_rotation=False) subscribes to the full universe in one call.
RotatingSubscriptionManager is kept as a fallback in case a larger universe
someday does hit a real cap, but it's currently unused.

To stop the task
- PowerShell: Stop-ScheduledTask -TaskName "liveFeed" -TaskPath "/trading/"
- Task Scheduler GUI: find /trading/liveFeed, right-click, End
- Or taskkill /PID <pid> /F using whatever's in data/live_feed.pid (equivalent, just more manual)

Consumers: use live_client.LiveDataClient rather than reading any file.
"""

import json
import logging
import os
import sys
import threading
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas_market_calendars as mcal
import zmq
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from data_ohlcv import _to_alpaca_symbol
from live_config import PUB_ADDRESS, HISTORY_ADDRESS
from utils import get_data, screen_symbols

load_dotenv()

# --- Paths / config ---
WATCHLIST_PATH = Path('data/conviction_watchlist.txt')
PID_PATH = Path('data/live_feed.pid')
SILENCE_CHECK_INTERVAL_SEC = 3600
SUBSCRIPTION_BATCH_SIZE = 25  # only used if use_rotation=True — see module docstring

LOG_DIR = Path('logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / 'live_feed.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger('live_feed')


def is_market_day() -> bool:
    """Duplicated from daily_run.py rather than imported — daily_run.py has
    module-level logging.basicConfig side effects that would clash with this
    module's own log setup if imported. Informational only now — run_live_feed
    no longer exits on a non-trading day, it just logs and keeps running."""
    nyse = mcal.get_calendar('NYSE')
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty


def read_watchlist() -> list[str]:
    """
    Reads the conviction watchlist: one ticker per line, '#' comments and
    blank lines ignored. Missing file is not an error — the watchlist is
    optional and the universe falls back to the screener alone.
    """
    if not WATCHLIST_PATH.exists():
        return []
    tickers = []
    for line in WATCHLIST_PATH.read_text().splitlines():
        line = line.split('#', 1)[0].strip().upper()
        if line:
            tickers.append(line)
    return tickers


def build_subscription_universe(top_k: int = 200) -> list[str]:
    """
    Screener top_k equities+ETFs, unioned with the conviction watchlist
    (which is always included regardless of screener rank).

    Returns internal-format tickers (hyphens for share classes, e.g. 'BF-A')
    — the same convention data/ohlcv_1min.parquet uses — since callers need
    this format both to query historical data and to convert to Alpaca's
    notation (_to_alpaca_symbol) for the live subscription.
    """
    watchlist = read_watchlist()
    screen = screen_symbols()
    screened = (screen['equities'] + screen['etfs'])[:top_k]
    return list(dict.fromkeys(watchlist + screened))  # watchlist first, dedup preserves priority


class RollingBarCache:
    """
    Thread-safe in-memory store of the last `retention_days` of bars per
    ticker, kept so a client that connects mid-stream can request everything
    published so far (see _serve_history) before switching to the live
    PUB/SUB stream. Retention is wall-clock, not a fixed bar count — bars
    older than retention_days age out on every add(), which matters because
    equities only trade ~390 min/day (a count-based window sized for "N
    days" would actually retain far more than N calendar days of history
    for them). No daily reset otherwise, since live_feed.py is meant to run
    continuously (Task Scheduler restarts it daily only to refresh the
    symbol universe, not to clear this cache) — seed() backfills from
    historical data on every (re)start so the cache is never empty.
    """

    def __init__(self, retention_days: float = 1.0):
        self._lock = threading.Lock()
        self._retention = timedelta(days=retention_days)
        self._bars: dict[str, deque] = defaultdict(deque)
        self._last_seen: dict[str, datetime] = {}

    def _prune(self, ticker: str, now: datetime) -> None:
        bars = self._bars[ticker]
        cutoff = now - self._retention
        while bars and datetime.fromisoformat(bars[0]['datetime']) < cutoff:
            bars.popleft()

    def add(self, ticker: str, bar: dict) -> None:
        with self._lock:
            self._bars[ticker].append(bar)
            self._last_seen[ticker] = bar['datetime']
            self._prune(ticker, datetime.now(timezone.utc))

    def seed(self, ticker: str, bars: list[dict]) -> None:
        """Bulk-loads historical bars (oldest first) at startup, ahead of
        any live bars arriving via add()."""
        if not bars:
            return
        with self._lock:
            self._bars[ticker].extend(bars)
            self._last_seen[ticker] = bars[-1]['datetime']
            self._prune(ticker, datetime.now(timezone.utc))

    def history(self, tickers: list[str] | None) -> dict[str, list[dict]]:
        with self._lock:
            if tickers is None:
                return {t: list(bars) for t, bars in self._bars.items()}
            return {t: list(self._bars[t]) for t in tickers if t in self._bars}

    def silent_tickers(self, universe: list[str], since: datetime) -> list[str]:
        with self._lock:
            return [t for t in universe if self._last_seen.get(t, since) <= since]


def _on_bar(pub_socket: zmq.Socket, buffer: RollingBarCache):
    async def handler(bar) -> None:
        ticker = bar.symbol
        message = {
            'ticker': ticker,
            'datetime': bar.timestamp.isoformat(),
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        }
        buffer.add(ticker, message)
        # Multipart: topic frame (ticker) lets subscribers filter server-side
        # via zmq's SUBSCRIBE prefix matching, without receiving every symbol.
        pub_socket.send_multipart([ticker.encode(), json.dumps(message).encode()])
    return handler


def _serve_history(rep_socket: zmq.Socket, buffer: RollingBarCache, stop_event: threading.Event) -> None:
    """
    Blocking request/reply loop, run in its own thread since zmq.REP is
    synchronous and would otherwise block the asyncio stream loop. Polls
    with a timeout so stop_event is checked periodically on shutdown.
    """
    poller = zmq.Poller()
    poller.register(rep_socket, zmq.POLLIN)
    while not stop_event.is_set():
        if not dict(poller.poll(timeout=500)):
            continue
        request = json.loads(rep_socket.recv())
        tickers = request.get('tickers')
        rep_socket.send(json.dumps(buffer.history(tickers)).encode())


def _run_periodic(interval_sec: float, fn, *args, stop_event: threading.Event) -> None:
    while not stop_event.wait(interval_sec):
        try:
            fn(*args)
        except Exception:
            log.error('Periodic task %s failed:\n%s', fn.__name__, traceback.format_exc())


class RotatingSubscriptionManager:
    """
    Cycles subscribe_bars/unsubscribe_bars over batches of the universe so
    each batch gets ~minutes_per_batch of live coverage before rotating to
    the next. Only meant to be used if empirical testing confirms Alpaca's
    free-tier symbol cap applies to bars subscriptions — otherwise subscribe
    to the whole universe once and skip this entirely.
    """

    def __init__(self, stream: StockDataStream, handler, universe: list[str],
                 batch_size: int, minutes_per_batch: float = 5.0):
        self.stream = stream
        self.handler = handler
        self.batches = [universe[i:i + batch_size] for i in range(0, len(universe), batch_size)]
        self.minutes_per_batch = minutes_per_batch
        self._idx = 0
        self._stop_event = threading.Event()

    def start(self) -> None:
        self.stream.subscribe_bars(self.handler, *self.batches[self._idx])
        log.info('Rotation: subscribed to batch %d/%d (%d symbols)',
                  self._idx + 1, len(self.batches), len(self.batches[self._idx]))
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop_event.wait(self.minutes_per_batch * 60):
            old_batch = self.batches[self._idx]
            self._idx = (self._idx + 1) % len(self.batches)
            new_batch = self.batches[self._idx]
            self.stream.unsubscribe_bars(*old_batch)
            self.stream.subscribe_bars(self.handler, *new_batch)
            log.info('Rotation: switched to batch %d/%d (%d symbols)',
                      self._idx + 1, len(self.batches), len(new_batch))

    def stop(self) -> None:
        self._stop_event.set()


def _seed_history(buffer: RollingBarCache, internal_universe: list[str],
                   alpaca_by_internal: dict[str, str], history_days: float) -> None:
    """
    Backfills the cache from data/ohlcv_1min.parquet (via utils.get_data) so
    the cache isn't empty right after a (re)start — that file is internal-
    ticker-keyed and only refreshes once daily at 5PM ET via daily_run.py,
    so this is "as of the last EOD pull," not perfectly current; live bars
    take over the moment the stream connects.
    """
    start = datetime.now(timezone.utc) - timedelta(days=history_days)
    data = get_data(tickers=internal_universe, start=start)
    seeded_tickers = 0
    seeded_rows = 0
    for internal_ticker, df in data.items():
        alpaca_ticker = alpaca_by_internal[internal_ticker]
        bars = [
            {
                'ticker': alpaca_ticker,
                'datetime': ts.isoformat(),
                'open': row.Open, 'high': row.High, 'low': row.Low,
                'close': row.Close, 'volume': row.Volume,
            }
            for ts, row in df.sort_index().iterrows()
        ]
        buffer.seed(alpaca_ticker, bars)
        seeded_tickers += 1
        seeded_rows += len(bars)
    log.info('Backfilled %d rows across %d/%d tickers from data/ohlcv_1min.parquet (last %.1f days)',
              seeded_rows, seeded_tickers, len(internal_universe), history_days)


def run_live_feed(top_k: int = 200, use_rotation: bool = False, history_days: float = 1.0) -> None:
    if not is_market_day():
        log.info('Not a market day — running anyway; expect no bars until the next trading session.')

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))
    log.info('PID %d written to %s', os.getpid(), PID_PATH)

    internal_universe = build_subscription_universe(top_k=top_k)
    alpaca_by_internal = {t: _to_alpaca_symbol(t) for t in internal_universe}
    universe = [alpaca_by_internal[t] for t in internal_universe]
    log.info('Subscription universe: %d symbols', len(universe))

    buffer = RollingBarCache(retention_days=history_days)
    _seed_history(buffer, internal_universe, alpaca_by_internal, history_days)

    zmq_context = zmq.Context()
    pub_socket = zmq_context.socket(zmq.PUB)
    pub_socket.bind(PUB_ADDRESS)
    log.info('Publishing bars on %s', PUB_ADDRESS)

    rep_socket = zmq_context.socket(zmq.REP)
    rep_socket.bind(HISTORY_ADDRESS)
    log.info('Serving history on %s', HISTORY_ADDRESS)

    handler = _on_bar(pub_socket, buffer)
    stream = StockDataStream(
        os.environ['ALPACA_API_KEY'],
        os.environ['ALPACA_SECRET_KEY'],
        feed=DataFeed.IEX,
    )

    stop_event = threading.Event()
    start_ts = datetime.now(timezone.utc)

    threading.Thread(
        target=_serve_history, args=(rep_socket, buffer, stop_event), daemon=True,
    ).start()

    def _check_silence():
        silent = buffer.silent_tickers(universe, since=start_ts)
        if silent:
            log.warning('%d symbols have received no bars yet: %s',
                         len(silent), ', '.join(silent[:20]) + ('...' if len(silent) > 20 else ''))

    threading.Thread(
        target=_run_periodic, args=(SILENCE_CHECK_INTERVAL_SEC, _check_silence),
        kwargs={'stop_event': stop_event}, daemon=True,
    ).start()

    rotation = None
    if use_rotation:
        rotation = RotatingSubscriptionManager(stream, handler, universe, SUBSCRIPTION_BATCH_SIZE)
        rotation.start()
    else:
        stream.subscribe_bars(handler, *universe)

    try:
        stream.run()
    except ValueError as e:
        if 'insufficient subscription' in str(e).lower():
            log.error(
                'Alpaca rejected the subscription (symbol cap likely hit at %d symbols). '
                'Re-run with use_rotation=True or lower SUBSCRIPTION_BATCH_SIZE.',
                len(universe),
            )
        else:
            log.error('Stream stopped: %s', e)
    except Exception:
        log.error('Stream crashed:\n%s', traceback.format_exc())
    finally:
        stop_event.set()
        if rotation is not None:
            rotation.stop()
        pub_socket.close()
        rep_socket.close()
        zmq_context.term()
        if PID_PATH.exists():
            PID_PATH.unlink()


if __name__ == '__main__':
    run_live_feed()
