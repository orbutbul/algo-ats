"""Profile one 10-year detect call. Read-only. Does not change the detector."""

from __future__ import annotations

import cProfile
import pstats
import time
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from vcp.detector import detect_vcp_breakouts
from vcp.pivots import detect_pivots
from vcp.params import VCPParams
from vcp.vbt_compat import VBT_FLAVOR

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'


def load(ticker: str, start: str) -> pd.DataFrame:
    con = duckdb.connect(DUCK, read_only=True)
    frame = con.execute(
        '''
        SELECT datetime, open, high, low, close, volume
        FROM ohlcv_max_daily
        WHERE ticker = ? AND datetime >= ?
        ORDER BY datetime
        ''',
        [ticker, start],
    ).df()
    con.close()
    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
    frame = frame.set_index('datetime')
    frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return frame


def main() -> None:
    print('vbt', VBT_FLAVOR)
    start = (datetime.now(timezone.utc) - timedelta(days=365 * 10 + 400)).date().isoformat()
    frame = load('NVDA', start)
    print('bars', len(frame), 'start', start)
    params = VCPParams()

    t0 = time.perf_counter()
    pivots = detect_pivots(frame['High'], frame['Low'], params.pivot_up_th, params.pivot_down_th)
    print(f'pivots {len(pivots)} in {time.perf_counter() - t0:.3f}s')

    t0 = time.perf_counter()
    result = detect_vcp_breakouts(frame, ticker='NVDA', params=params)
    print(f'detect {len(result.breakouts)} hits in {time.perf_counter() - t0:.3f}s')

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(3):
        detect_vcp_breakouts(frame, ticker='NVDA', params=params)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)


if __name__ == '__main__':
    main()
