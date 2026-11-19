"""Snapshot breakout identity before the excursion fast path. Read-only."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from vcp.detector import detect_vcp_breakouts

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'
TICKERS = ['NVDA', 'AMD', 'SMCI', 'SNAP', 'AAPL', 'CELH', 'DECK', 'MELI', 'AVGO', 'TSLA', 'MSFT', 'META']


def _load(con, ticker: str) -> pd.DataFrame:
    frame = con.execute(
        '''
        SELECT datetime, open, high, low, close, volume
        FROM ohlcv_max_daily
        WHERE ticker = ? AND datetime >= '2016-01-01'
        ORDER BY datetime
        ''',
        [ticker],
    ).df()
    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
    frame = frame.set_index('datetime')
    frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return frame


def _pack(result) -> list[dict]:
    rows = []
    for b in result.breakouts:
        row = b.to_dict()
        row['contractions'] = [
            {
                'peak_bar': c.peak.bar,
                'peak_px': c.peak.price,
                'peak_ts': str(c.peak.index),
                'trough_bar': c.trough.bar,
                'trough_px': c.trough.price,
                'trough_ts': str(c.trough.index),
                'rec_bar': None if c.recovery_peak is None else c.recovery_peak.bar,
                'rec_px': None if c.recovery_peak is None else c.recovery_peak.price,
                'rec_ts': None if c.recovery_peak is None else str(c.recovery_peak.index),
                'depth_log': c.depth_log,
                'depth_pct': c.depth_pct,
                'mean_volume': c.mean_volume,
                'min_volume': c.min_volume,
                'bars': c.bars,
                'recovery_frac': c.recovery_frac,
                'shelf_price': c.shelf_price,
                'is_micro': c.is_micro,
                'is_noise': c.is_noise,
            }
            for c in b.contractions
        ]
        rows.append(row)
    return rows


def main() -> None:
    con = duckdb.connect(DUCK, read_only=True)
    out = {}
    for ticker in TICKERS:
        frame = _load(con, ticker)
        result = detect_vcp_breakouts(frame, ticker=ticker)
        out[ticker] = _pack(result)
        print(ticker, len(frame), len(out[ticker]), flush=True)
    con.close()
    dest = Path('artifacts/vcp_golden_breakouts.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print('wrote', dest, dest.stat().st_size)


if __name__ == '__main__':
    main()
