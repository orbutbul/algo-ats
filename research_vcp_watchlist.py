"""Rebuild the viewer list with the 1% shelf cap.

Same 10-year names as the grid. One parameter set: the viewer cell.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from vcp.scan import scan_param_sets_parallel
from vcp.viewer import VIEW_PARAMS

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'
ROOT = Path(__file__).resolve().parent
PARAMS = ROOT / 'artifacts' / 'vcp_grid_params.json'
OUT_DIR = ROOT / 'artifacts' / 'vcp_viewer_scan'
DEST = ROOT / 'artifacts' / 'vcp_viewer_hits.json'
WORKERS = 8


def main() -> None:
    prior = json.loads(PARAMS.read_text(encoding='utf-8'))
    result = scan_param_sets_parallel(
        duck=DUCK,
        tickers=prior['tickers'],
        load_start=prior['load_start'],
        load_end=prior['window_end'],
        window_start=prior['window_start'],
        param_sets=[{'combo_id': 0, 'params': VIEW_PARAMS.to_dict()}],
        out_dir=str(OUT_DIR),
        workers=WORKERS,
    )
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for part in result['parts']:
        text = Path(part['out_path']).read_text(encoding='utf-8')
        for line in text.splitlines():
            if not line:
                continue
            hit = json.loads(line)
            by_ticker[hit['ticker']].append(
                {
                    'pattern_start': str(hit['pattern_start'])[:10],
                    'pattern_end': str(hit['pattern_end'])[:10],
                    'breakout_date': str(hit['breakout_date'])[:10],
                    'confidence': round(float(hit['confidence']), 3),
                    'footprint': hit['footprint'],
                }
            )
    tickers = []
    for ticker, hits in by_ticker.items():
        hits.sort(key=lambda item: item['breakout_date'])
        tickers.append({'ticker': ticker, 'n': len(hits), 'latest': hits[-1]['breakout_date'], 'hits': hits})
    tickers.sort(key=lambda item: (item['latest'], item['ticker']), reverse=True)
    payload = {
        'contraction_ratio': VIEW_PARAMS.contraction_ratio,
        'shelf_tol': VIEW_PARAMS.shelf_tol,
        'noise_depth_frac': VIEW_PARAMS.noise_depth_frac,
        'shelf_overshoot': VIEW_PARAMS.shelf_overshoot,
        'window_start': prior['window_start'],
        'window_end': prior['window_end'],
        'n_tickers': len(tickers),
        'n_hits': sum(item['n'] for item in tickers),
        'tickers': tickers,
    }
    DEST.write_text(json.dumps(payload), encoding='utf-8')
    print(
        f"wall {result['wall_s'] / 60:.1f} min, {payload['n_tickers']} tickers, "
        f"{payload['n_hits']} hits, wrote {DEST}",
        flush=True,
    )


if __name__ == '__main__':
    main()
