"""Pilot timing and payload size for a VCP parameter sweep log.

Reads data/ohlcv.duckdb from the main checkout (read-only). Runs the default
parameter set plus a few Sobol draws. For each hit, records the pattern date
span and the next 20 sessions, grouped by ticker. No orders, no backtest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import qmc

from vcp.detector import detect_vcp_breakouts
from vcp.params import VCPParams

DUCK = Path(r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb')
ANCHORS = ['SMCI', 'AMD', 'CELH', 'DECK', 'MELI', 'AVGO', 'SNAP', 'NVDA']
N_SOBOL = 4
N_EXTRA = 24
NEXT_N = 20

# Joint box for the simultaneous sweep. Scaled onto Sobol draws in [0, 1].
SPACE = [
    ('min_confidence', 0.40, 0.80),
    ('shelf_tol', 0.012, 0.040),
    ('noise_depth_frac', 0.20, 0.50),
    ('contraction_ratio', 0.60, 0.95),
    ('min_first_depth', 0.04, 0.12),
    ('max_final_depth', 0.07, 0.12),
    ('vdu_max_ratio', 0.60, 0.85),
    ('breakout_vs_dryup_mult', 1.20, 2.20),
]


def _sobol_params(n: int, seed: int = 0) -> list[VCPParams]:
    sampler = qmc.Sobol(d=len(SPACE), scramble=True, seed=seed)
    unit = sampler.random(n)
    base = VCPParams().to_dict()
    out: list[VCPParams] = []
    for row in unit:
        overrides = {}
        for (name, lo, hi), u in zip(SPACE, row):
            overrides[name] = float(lo + (hi - lo) * u)
        out.append(VCPParams(**{**base, **overrides}))
    return out


def _load_many(con: duckdb.DuckDBPyConnection, tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        t0 = time.perf_counter()
        frame = con.execute(
            '''
            SELECT datetime, open, high, low, close, volume
            FROM ohlcv_max_daily
            WHERE ticker = ?
            ORDER BY datetime
            ''',
            [ticker],
        ).df()
        elapsed = time.perf_counter() - t0
        if frame.empty:
            print(f'load {ticker}: missing ({elapsed:.3f}s)', flush=True)
            continue
        frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
        frame = frame.set_index('datetime').sort_index()
        frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        frames[ticker] = frame
        print(f'load {ticker}: {len(frame)} bars in {elapsed:.3f}s', flush=True)
    return frames


def _next_dates(index: pd.DatetimeIndex, breakout: pd.Timestamp, n: int) -> list[str]:
    ts = index
    stamp = pd.Timestamp(breakout)
    if ts.tz is not None and stamp.tzinfo is None:
        stamp = stamp.tz_localize(ts.tz)
    elif ts.tz is not None:
        stamp = stamp.tz_convert(ts.tz)
    loc = int(ts.searchsorted(stamp))
    if loc < len(ts) and ts[loc] == stamp:
        loc += 1
    return [str(ts[i].date()) for i in range(loc, min(len(ts), loc + n))]


def _hit_record(breakout, index: pd.DatetimeIndex) -> dict:
    start = pd.Timestamp(breakout.contractions[0].peak.index)
    end = pd.Timestamp(breakout.breakout_date)
    start_loc = int(index.searchsorted(start.tz_convert(index.tz) if index.tz is not None else start))
    end_loc = int(index.searchsorted(end.tz_convert(index.tz) if index.tz is not None else end))
    if end_loc < len(index) and index[end_loc] == (end.tz_convert(index.tz) if index.tz is not None else end):
        n_sessions = end_loc - start_loc + 1
    else:
        n_sessions = max(0, end_loc - start_loc)
    payload = breakout.to_dict()
    payload.update(
        {
            'pattern_start': str(start.date()),
            'pattern_end': str(end.date()),
            'n_pattern_sessions': int(n_sessions),
            'next_20_dates': _next_dates(index, end, NEXT_N),
        }
    )
    return payload


def _ohlcv_next(frame: pd.DataFrame, dates: list[str]) -> list[dict]:
    if not dates:
        return []
    want = set(dates)
    rows = []
    for ts, row in frame.iterrows():
        key = str(pd.Timestamp(ts).date())
        if key in want:
            rows.append(
                {
                    'date': key,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                }
            )
    return rows


def main() -> None:
    t_all = time.perf_counter()
    con = duckdb.connect(str(DUCK), read_only=True)
    t0 = time.perf_counter()
    counts = con.execute(
        'SELECT ticker, COUNT(*) AS n FROM ohlcv_max_daily GROUP BY ticker ORDER BY ticker'
    ).fetchall()
    t_universe = time.perf_counter() - t0
    universe = [r[0] for r in counts]
    print(f'universe: {len(universe)} tickers in {t_universe:.2f}s', flush=True)

    present_anchors = [t for t in ANCHORS if t in set(universe)]
    extras = [t for t in universe if t not in present_anchors]
    if extras:
        step = max(1, len(extras) // N_EXTRA)
        sampled = extras[::step][:N_EXTRA]
    else:
        sampled = []
    tickers = present_anchors + sampled
    print(f'probe tickers ({len(tickers)}): {", ".join(tickers)}', flush=True)

    t0 = time.perf_counter()
    frames = _load_many(con, tickers)
    t_load = time.perf_counter() - t0
    con.close()

    combos = [VCPParams()] + _sobol_params(N_SOBOL, seed=0)
    by_ticker: dict[str, list[dict]] = {t: [] for t in frames}
    detect_seconds = 0.0
    n_detects = 0
    hits_per_combo = []
    combo_meta = []

    for combo_id, params in enumerate(combos):
        n_hits = 0
        t0 = time.perf_counter()
        for ticker, frame in frames.items():
            result = detect_vcp_breakouts(frame, ticker=ticker, params=params)
            n_detects += 1
            for hit in result.breakouts:
                rec = _hit_record(hit, frame.index)
                rec['combo_id'] = combo_id
                rec['next_20_ohlcv'] = _ohlcv_next(frame, rec['next_20_dates'])
                by_ticker[ticker].append(rec)
                n_hits += 1
        elapsed = time.perf_counter() - t0
        detect_seconds += elapsed
        hits_per_combo.append(n_hits)
        combo_meta.append({'combo_id': combo_id, 'params': params.to_dict(), 'n_hits': n_hits, 'seconds': elapsed})
        print(f'combo {combo_id}: {n_hits} hits in {elapsed:.2f}s', flush=True)

    dates_only = {
        ticker: [{k: v for k, v in hit.items() if k != 'next_20_ohlcv'} for hit in hits]
        for ticker, hits in by_ticker.items()
    }
    dates_bytes = len(json.dumps(dates_only).encode())
    with_bars_bytes = len(json.dumps(by_ticker).encode())
    n_hits_total = sum(hits_per_combo)
    n_loaded = len(frames)
    t_detect_per_call = detect_seconds / n_detects if n_detects else 0.0
    t_load_per = t_load / n_loaded if n_loaded else 0.0

    # Extrapolate an efficient sweep: load each ticker once, then reuse the frame.
    def _project(n_tickers: int, n_combos: int) -> dict:
        seconds = n_tickers * t_load_per + n_tickers * n_combos * t_detect_per_call
        # Hit rate from this probe, which mixes anchors and a spread of the universe.
        hits_per_ticker_combo = (n_hits_total / n_detects) if n_detects else 0.0
        bytes_per_hit = (dates_bytes / n_hits_total) if n_hits_total else 0.0
        bytes_per_hit_bars = (with_bars_bytes / n_hits_total) if n_hits_total else 0.0
        return {
            'n_tickers': n_tickers,
            'n_combos': n_combos,
            'hours': seconds / 3600.0,
            'hits': hits_per_ticker_combo * n_tickers * n_combos,
            'dates_only_mb': bytes_per_hit * hits_per_ticker_combo * n_tickers * n_combos / 1e6,
            'with_next20_ohlcv_mb': bytes_per_hit_bars * hits_per_ticker_combo * n_tickers * n_combos / 1e6,
        }

    summary = {
        'universe_tickers': len(universe),
        'universe_scan_s': t_universe,
        'probe_tickers': list(frames),
        'n_combos': len(combos),
        'load_s': t_load,
        'load_s_per_ticker': t_load_per,
        'detect_s': detect_seconds,
        'detect_s_per_call': t_detect_per_call,
        'hits_per_combo': hits_per_combo,
        'n_hits': n_hits_total,
        'dates_only_bytes': dates_bytes,
        'with_next20_ohlcv_bytes': with_bars_bytes,
        'wall_s': time.perf_counter() - t_all,
        'project_128_full': _project(len(universe), 128),
        'project_256_full': _project(len(universe), 256),
        'project_128_probe_mix': _project(n_loaded, 128),
    }

    out = {
        'summary': summary,
        'combos': combo_meta,
        'by_ticker': dates_only,
    }
    dest = Path('artifacts') / 'vcp_probe.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'wrote {dest.resolve()} ({dest.stat().st_size} bytes)', flush=True)


if __name__ == '__main__':
    main()
