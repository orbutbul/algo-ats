"""20-bar path for the viewer parameter cell, including the 1% shelf cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / 'artifacts' / 'vcp_viewer_scan'
PARAMS = ROOT / 'artifacts' / 'vcp_grid_params.json'


def _returns(hit: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    bars = hit.get('next_20') or []
    if len(bars) < 20:
        return None
    base = (hit.get('breakout_bar') or {}).get('close')
    if not base or base <= 0:
        return None
    close = np.array([bar['close'] for bar in bars[:20]], dtype=float)
    high = np.array([bar['high'] for bar in bars[:20]], dtype=float)
    low = np.array([bar['low'] for bar in bars[:20]], dtype=float)
    if np.any(close <= 0) or np.any(high <= 0) or np.any(low <= 0):
        return None
    return close / base - 1.0, high / base - 1.0, low / base - 1.0


def main() -> None:
    prior = json.loads(PARAMS.read_text(encoding='utf-8'))
    n_universe = len(prior['tickers'])
    closes = []
    highs = []
    lows = []
    names = set()
    n_hits = 0
    n_short = 0
    for part in sorted(PARTS.glob('part_*.jsonl')):
        for line in part.read_text(encoding='utf-8').splitlines():
            if not line:
                continue
            hit = json.loads(line)
            n_hits += 1
            names.add(hit['ticker'])
            followed = _returns(hit)
            if followed is None:
                n_short += 1
                continue
            closes.append(followed[0])
            highs.append(followed[1])
            lows.append(followed[2])
    close = np.vstack(closes)
    high = np.vstack(highs)
    low = np.vstack(lows)
    final = close[:, -1]
    mfe = high.max(axis=1)
    mae = low.min(axis=1)
    print(f'universe {n_universe}')
    print(f'hits {n_hits}  names {len(names)}  name_rate {len(names) / n_universe:.2%}')
    print(f'hits_per_name {n_hits / n_universe:.3f}  full_20 {len(final)}  short {n_short}')
    print(
        f'ret20 up {np.mean(final > 0):.1%}  mean {np.mean(final):.1%}  '
        f'median {np.median(final):.1%}  p10 {np.percentile(final, 10):.1%}  '
        f'p90 {np.percentile(final, 90):.1%}'
    )
    print(
        f'mfe median {np.median(mfe):.1%} mean {np.mean(mfe):.1%}  '
        f'mae median {np.median(mae):.1%} mean {np.mean(mae):.1%}'
    )
    print('bar  up%   mean   median')
    for i in range(20):
        col = close[:, i]
        print(f'{i + 1:3d}  {np.mean(col > 0):5.1%}  {np.mean(col):6.1%}  {np.median(col):6.1%}')


if __name__ == '__main__':
    main()
