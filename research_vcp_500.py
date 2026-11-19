"""500 random tickers, last 10 years, one default VCP pass.

Read-only. Writes artifacts/vcp_500x10y.json. Prints wall time and payload size.
Warmup bars before the window are loaded so the 200-day MA is defined, then
breakouts earlier than the window are dropped.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import duckdb
import pandas as pd

from vcp.params import VCPParams
from vcp.scan import scan_tickers_parallel

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'
SEED = 17
N = 500


def _window(con: duckdb.DuckDBPyConnection) -> tuple[str, str, str]:
    mx = pd.Timestamp(con.execute('SELECT max(datetime) FROM ohlcv_max_daily').fetchone()[0])
    if mx.tzinfo is None:
        mx = mx.tz_localize('UTC')
    window_start = (mx - pd.DateOffset(years=10)).tz_convert('UTC')
    load_start = window_start - pd.Timedelta(days=400)
    return (
        load_start.date().isoformat(),
        mx.tz_convert('UTC').date().isoformat(),
        window_start.date().isoformat(),
    )


def _eligible(con: duckdb.DuckDBPyConnection, window_start: str, load_end: str) -> list[str]:
    rows = con.execute(
        '''
        SELECT ticker
        FROM ohlcv_max_daily
        WHERE datetime >= ? AND datetime <= ?
        GROUP BY ticker
        HAVING count(*) >= 2400
        ORDER BY ticker
        ''',
        [window_start, load_end],
    ).fetchall()
    return [row[0] for row in rows]


def _same(a: dict, b: dict) -> bool:
    return a['by_ticker'] == b['by_ticker'] and a['n_hits'] == b['n_hits']


def main() -> None:
    con = duckdb.connect(DUCK, read_only=True)
    load_start, load_end, window_start = _window(con)
    eligible = _eligible(con, window_start, load_end)
    con.close()
    print(f'window {window_start} .. {load_end}  load from {load_start}', flush=True)
    print(f'eligible tickers with >= 2400 bars in window: {len(eligible)}', flush=True)
    rng = random.Random(SEED)
    picked = rng.sample(eligible, N) if len(eligible) >= N else list(eligible)
    picked.sort()
    workers = min(8, os.cpu_count() or 1)

    check = picked[:6]
    serial = scan_tickers_parallel(
        duck=DUCK,
        tickers=check,
        load_start=load_start,
        load_end=load_end,
        window_start=window_start,
        params=VCPParams(),
        workers=1,
    )
    parallel = scan_tickers_parallel(
        duck=DUCK,
        tickers=check,
        load_start=load_start,
        load_end=load_end,
        window_start=window_start,
        params=VCPParams(),
        workers=min(2, len(check)),
    )
    if not _same(serial, parallel):
        raise SystemExit('parallel scan diverged from serial on the check set')
    print(f'check ok: {serial["n_hits"]} hits, serial {serial["wall_s"]:.1f}s vs parallel {parallel["wall_s"]:.1f}s', flush=True)

    t = scan_tickers_parallel(
        duck=DUCK,
        tickers=picked,
        load_start=load_start,
        load_end=load_end,
        window_start=window_start,
        params=VCPParams(),
        workers=workers,
    )
    by_ticker = t.pop('by_ticker')
    payload = {
        'summary': {
            **t,
            'seed': SEED,
            'requested': N,
            'window_start': window_start,
            'window_end': load_end,
            'load_start': load_start,
            'eligible': len(eligible),
            'params': VCPParams().to_dict(),
        },
        'tickers': picked,
        'by_ticker': by_ticker,
    }
    dest = Path('artifacts/vcp_500x10y.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload)
    dest.write_text(raw)
    print(json.dumps(payload['summary'], indent=2, default=str), flush=True)
    print(f'wrote {dest.resolve()} ({dest.stat().st_size} bytes, json chars {len(raw)})', flush=True)


if __name__ == '__main__':
    main()
