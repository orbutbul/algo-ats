"""
live/client.py — subscribes to live_feed.py's ZeroMQ bar stream and
reassembles it into per-ticker DataFrames, so strategy/indicator code can
work with pandas the same way it does against historical data, just fed by
push updates instead of a polled file.

    from live.client import LiveDataClient

    client = LiveDataClient(['AAPL', 'NVDA'])   # or LiveDataClient() for all
    ...
    aapl = client.get('AAPL')                   # Open/High/Low/Close/Volume DataFrame
    aapl['Close'].rolling(20).mean()
    ...
    client.stop()

Note the "slow joiner" behavior inherent to ZeroMQ PUB/SUB: a subscriber
that connects after live_feed.py has already started publishing may miss
the first bar or two while the socket handshake completes. This self-heals
within a bar or two and isn't worth working around for a minute-bar feed.
"""

import json
import logging
import threading
from collections import defaultdict, deque
from typing import Iterable

import pandas as pd
import zmq

from live.config import PUB_ADDRESS, HISTORY_ADDRESS

BAR_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
HISTORY_TIMEOUT_MS = 3000

log = logging.getLogger('live_client')


class LiveDataClient:
    def __init__(self, tickers: Iterable[str] | None = None,
                 address: str = PUB_ADDRESS, maxlen: int = 500):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(address)
        if tickers is None:
            self._socket.setsockopt_string(zmq.SUBSCRIBE, '')
        else:
            for ticker in tickers:
                self._socket.setsockopt_string(zmq.SUBSCRIBE, ticker)

        self._lock = threading.Lock()
        self._maxlen = maxlen
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._stop_event = threading.Event()
        # Start listening before requesting catch-up history, so no live bars
        # published during the request round-trip are missed.
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        self._fetch_history(tickers)

    def _fetch_history(self, tickers: Iterable[str] | None) -> None:
        req_socket = self._context.socket(zmq.REQ)
        req_socket.connect(HISTORY_ADDRESS)
        poller = zmq.Poller()
        poller.register(req_socket, zmq.POLLIN)
        req_socket.send(json.dumps({'tickers': list(tickers) if tickers is not None else None}).encode())
        if not poller.poll(timeout=HISTORY_TIMEOUT_MS):
            log.warning('No response from live_feed history endpoint at %s within %dms — '
                        'proceeding with live-only data.', HISTORY_ADDRESS, HISTORY_TIMEOUT_MS)
            req_socket.close(linger=0)
            return
        history = json.loads(req_socket.recv())
        req_socket.close()
        with self._lock:
            for ticker, bars in history.items():
                # Merge catch-up bars with anything the live thread already
                # received during the request round-trip, then dedupe on
                # (ticker, datetime) keeping the latest — resolves the
                # snapshot-plus-tail race without losing or duplicating bars.
                merged = {bar['datetime']: bar for bar in bars}
                for bar in self._bars[ticker]:
                    merged[bar['datetime']] = bar
                ordered = sorted(merged.values(), key=lambda b: b['datetime'])
                # Adapt maxlen up if the server sent more history than our
                # default holds — the client should never silently truncate
                # what live_feed.py's history_days retention actually gave it.
                self._maxlen = max(self._maxlen, len(ordered))
                self._bars[ticker] = deque(ordered, maxlen=self._maxlen)
        log.info('Caught up on history for %d tickers', len(history))

    def _listen(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop_event.is_set():
            if not poller.poll(timeout=500):  # ms; lets stop_event be checked periodically
                continue
            _, payload = self._socket.recv_multipart()
            bar = json.loads(payload)
            with self._lock:
                self._bars[bar['ticker']].append(bar)

    def get(self, ticker: str) -> pd.DataFrame:
        """Returns the rolling buffer of received bars for one ticker."""
        with self._lock:
            bars = list(self._bars.get(ticker, []))
        if not bars:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.DataFrame(bars)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return (
            df.set_index('datetime')
            .rename(columns=str.title)
            [BAR_COLUMNS]
        )

    def snapshot(self) -> dict[str, pd.DataFrame]:
        """Returns {ticker: DataFrame} for every ticker seen so far."""
        with self._lock:
            tickers = list(self._bars.keys())
        return {ticker: self.get(ticker) for ticker in tickers}

    #main return method for prelim strategies
    def snapshot_close(self) -> pd.DataFrame:
        """Returns Close prices for every ticker seen so far, wide-format:
        columns = ticker, rows = datetime. Tickers with bars at different
        timestamps are aligned on the union of datetimes seen (NaN where a
        ticker has no bar at that time)."""
        return pd.DataFrame({ticker: df['Close'] for ticker, df in self.snapshot().items()})

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._socket.close()
        self._context.term()

    def __enter__(self) -> 'LiveDataClient':
        return self

    def __exit__(self, *_exc_info) -> None:
        self.stop()
