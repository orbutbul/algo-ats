"""Parallel multi-ticker VCP scan.

Research / validation helper. Each ticker is detected on its own, so a hit
list matches ``detect_vcp_breakouts`` for that frame. Workers only split the
universe; they do not change pivot, shelf, or score math.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from vcp.params import VCPParams


def _json_float(value: float) -> float | None:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hit_record(breakout, index: pd.DatetimeIndex) -> dict:
    start = pd.Timestamp(breakout.contractions[0].peak.index)
    end = pd.Timestamp(breakout.breakout_date)
    tz = index.tz
    if tz is not None:
        start = start.tz_convert(tz) if start.tzinfo else start.tz_localize(tz)
        end = end.tz_convert(tz) if end.tzinfo else end.tz_localize(tz)
    start_loc = int(index.searchsorted(start))
    end_loc = int(index.searchsorted(end))
    if end_loc < len(index) and index[end_loc] == end:
        n_sessions = end_loc - start_loc + 1
        next_loc = end_loc + 1
    else:
        n_sessions = max(0, end_loc - start_loc)
        next_loc = end_loc
    payload = breakout.to_dict()
    payload['depths'] = [_json_float(float(x)) for x in payload['depths']]
    payload['volume_dryup_ratios'] = [_json_float(float(x)) for x in payload['volume_dryup_ratios']]
    payload['quiet_bar_ratios'] = [_json_float(float(x)) for x in payload['quiet_bar_ratios']]
    payload['pattern_start'] = str(start.date())
    payload['pattern_end'] = str(end.date())
    payload['n_pattern_sessions'] = int(n_sessions)
    payload['next_20_dates'] = [str(index[i].date()) for i in range(next_loc, min(len(index), next_loc + 20))]
    return payload


def _scan_worker(task: dict) -> dict:
    """Load and detect one chunk. Top-level so Windows can spawn it."""
    import duckdb

    from vcp.detector import detect_vcp_breakouts

    params = VCPParams(**task['params'])
    load_s = 0.0
    detect_s = 0.0
    hits: list[dict] = []
    n_kept = 0
    con = duckdb.connect(task['duck'], read_only=True)
    try:
        for ticker in task['tickers']:
            t0 = time.perf_counter()
            frame = con.execute(
                '''
                SELECT datetime, open, high, low, close, volume
                FROM ohlcv_max_daily
                WHERE ticker = ? AND datetime >= ? AND datetime <= ?
                ORDER BY datetime
                ''',
                [ticker, task['load_start'], task['load_end']],
            ).df()
            load_s += time.perf_counter() - t0
            if frame.empty:
                continue
            frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
            frame = frame.set_index('datetime')
            frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            t0 = time.perf_counter()
            result = detect_vcp_breakouts(frame, ticker=ticker, params=params)
            detect_s += time.perf_counter() - t0
            window_start = pd.Timestamp(task['window_start'], tz='UTC')
            for breakout in result.breakouts:
                if pd.Timestamp(breakout.breakout_date) < window_start:
                    continue
                n_kept += 1
                hits.append(_hit_record(breakout, frame.index))
    finally:
        con.close()
    return {
        'load_s': load_s,
        'detect_s': detect_s,
        'n_tickers': len(task['tickers']),
        'n_hits': n_kept,
        'hits': hits,
    }


def scan_tickers_parallel(
    *,
    duck: str,
    tickers: list[str],
    load_start: str,
    load_end: str,
    window_start: str,
    params: VCPParams | None = None,
    workers: int = 1,
) -> dict:
    """Detect ``tickers`` over one date window. ``workers`` > 1 splits the list."""
    params = params or VCPParams()
    workers = max(1, min(int(workers), len(tickers) or 1))
    chunks: list[list[str]] = [[] for _ in range(workers)]
    for i, ticker in enumerate(tickers):
        chunks[i % workers].append(ticker)
    tasks = [
        {
            'duck': duck,
            'tickers': chunk,
            'load_start': load_start,
            'load_end': load_end,
            'window_start': window_start,
            'params': params.to_dict(),
        }
        for chunk in chunks
        if chunk
    ]
    t0 = time.perf_counter()
    if len(tasks) == 1:
        parts = [_scan_worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            parts = list(pool.map(_scan_worker, tasks))
    wall_s = time.perf_counter() - t0
    hits: list[dict] = []
    for part in parts:
        hits.extend(part['hits'])
    hits.sort(key=lambda row: (row['ticker'], row['breakout_date'], -row['confidence']))
    by_ticker: dict[str, list[dict]] = {}
    for row in hits:
        by_ticker.setdefault(row['ticker'], []).append(row)
    return {
        'wall_s': wall_s,
        'load_s': sum(part['load_s'] for part in parts),
        'detect_s': sum(part['detect_s'] for part in parts),
        'n_tickers': sum(part['n_tickers'] for part in parts),
        'n_hits': len(hits),
        'workers': len(tasks),
        'by_ticker': by_ticker,
    }


def _bar_dict(ts, row) -> dict:
    return {
        'date': str(pd.Timestamp(ts).date()),
        'open': round(float(row['Open']), 4),
        'high': round(float(row['High']), 4),
        'low': round(float(row['Low']), 4),
        'close': round(float(row['Close']), 4),
        'volume': float(row['Volume']),
    }


def _hit_with_forward_bars(breakout, frame: pd.DataFrame, combo_id: int) -> dict:
    """Pattern date range plus the signal bar and the next 20 sessions."""
    record = _hit_record(breakout, frame.index)
    record['combo_id'] = int(combo_id)
    index = frame.index
    end = pd.Timestamp(breakout.breakout_date)
    if index.tz is not None:
        end = end.tz_convert(index.tz) if end.tzinfo else end.tz_localize(index.tz)
    end_loc = int(index.searchsorted(end))
    if end_loc < len(index) and index[end_loc] == end:
        record['breakout_bar'] = _bar_dict(index[end_loc], frame.iloc[end_loc])
        next_loc = end_loc + 1
    else:
        record['breakout_bar'] = None
        next_loc = end_loc
    stop = min(len(index), next_loc + 20)
    record['next_20'] = [_bar_dict(index[i], frame.iloc[i]) for i in range(next_loc, stop)]
    return record


def _scan_param_sets_worker(task: dict) -> dict:
    """Load each ticker once, run every parameter set, append JSONL hits."""
    import json

    import duckdb

    from vcp.detector import detect_vcp_breakouts

    sets = [(int(item['combo_id']), VCPParams(**item['params'])) for item in task['param_sets']]
    window_start = pd.Timestamp(task['window_start'], tz='UTC')
    load_s = 0.0
    detect_s = 0.0
    n_hits = 0
    con = duckdb.connect(task['duck'], read_only=True)
    out = open(task['out_path'], 'w', encoding='utf-8')
    try:
        for n_done, ticker in enumerate(task['tickers'], start=1):
            t0 = time.perf_counter()
            frame = con.execute(
                '''
                SELECT datetime, open, high, low, close, volume
                FROM ohlcv_max_daily
                WHERE ticker = ? AND datetime >= ? AND datetime <= ?
                ORDER BY datetime
                ''',
                [ticker, task['load_start'], task['load_end']],
            ).df()
            load_s += time.perf_counter() - t0
            if frame.empty:
                continue
            frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
            frame = frame.set_index('datetime')
            frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            try:
                for combo_id, params in sets:
                    t0 = time.perf_counter()
                    result = detect_vcp_breakouts(frame, ticker=ticker, params=params)
                    detect_s += time.perf_counter() - t0
                    for breakout in result.breakouts:
                        if pd.Timestamp(breakout.breakout_date) < window_start:
                            continue
                        out.write(json.dumps(_hit_with_forward_bars(breakout, frame, combo_id), allow_nan=False))
                        out.write('\n')
                        n_hits += 1
            except Exception as exc:
                print(f"worker {task['worker_id']}: skip {ticker}: {exc}", flush=True)
            if n_done % 10 == 0:
                print(f"worker {task['worker_id']}: {n_done}/{len(task['tickers'])} names", flush=True)
    finally:
        out.close()
        con.close()
    return {
        'worker_id': task['worker_id'],
        'load_s': load_s,
        'detect_s': detect_s,
        'n_tickers': len(task['tickers']),
        'n_hits': n_hits,
        'out_path': task['out_path'],
    }


def scan_param_sets_parallel(
    *,
    duck: str,
    tickers: list[str],
    load_start: str,
    load_end: str,
    window_start: str,
    param_sets: list[dict],
    out_dir: str,
    workers: int = 1,
) -> dict:
    """Run many parameter sets. Each worker loads its names once."""
    from pathlib import Path

    workers = max(1, min(int(workers), len(tickers) or 1))
    chunks: list[list[str]] = [[] for _ in range(workers)]
    for i, ticker in enumerate(tickers):
        chunks[i % workers].append(ticker)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    tasks = []
    for worker_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        tasks.append(
            {
                'worker_id': worker_id,
                'duck': duck,
                'tickers': chunk,
                'load_start': load_start,
                'load_end': load_end,
                'window_start': window_start,
                'param_sets': param_sets,
                'out_path': str(dest / f'part_{worker_id}.jsonl'),
            }
        )
    t0 = time.perf_counter()
    if len(tasks) == 1:
        parts = [_scan_param_sets_worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            parts = list(pool.map(_scan_param_sets_worker, tasks))
    return {
        'wall_s': time.perf_counter() - t0,
        'load_s': sum(part['load_s'] for part in parts),
        'detect_s': sum(part['detect_s'] for part in parts),
        'n_tickers': sum(part['n_tickers'] for part in parts),
        'n_hits': sum(part['n_hits'] for part in parts),
        'workers': len(tasks),
        'parts': parts,
    }
