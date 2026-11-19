"""Check whether relaxed knobs recover CELH 2024-02-27, and print Sobol params."""

import json
from pathlib import Path

import duckdb
import pandas as pd

from vcp.detector import detect_vcp_breakouts
from vcp.params import VCPParams

doc = json.loads(Path('artifacts/vcp_probe.json').read_text())
print('SOBOL / DEFAULT PARAMS')
for combo in doc['combos']:
    p = combo['params']
    keys = [
        'min_confidence',
        'shelf_tol',
        'noise_depth_frac',
        'contraction_ratio',
        'min_first_depth',
        'max_final_depth',
        'vdu_max_ratio',
        'breakout_vs_dryup_mult',
    ]
    vals = ' '.join(f'{k}={p[k]:.3f}' for k in keys)
    print(f"combo {combo['combo_id']} hits={combo['n_hits']} {combo['seconds']:.1f}s  {vals}")

con = duckdb.connect(r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb', read_only=True)
frame = con.execute(
    '''
    SELECT datetime, open, high, low, close, volume
    FROM ohlcv_max_daily
    WHERE ticker = 'CELH'
    ORDER BY datetime
    '''
).df()
con.close()
frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
frame = frame.set_index('datetime')
frame.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

variants = {
    'default': VCPParams(),
    'no_trend': VCPParams(require_trend_template=False),
    'loose': VCPParams(
        require_trend_template=False,
        min_confidence=0.30,
        shelf_tol=0.04,
        noise_depth_frac=0.25,
        contraction_ratio=0.90,
        min_first_depth=0.04,
        max_final_depth=0.12,
        vdu_max_ratio=0.90,
        breakout_vs_dryup_mult=1.2,
        breakout_rel_vol=1.1,
    ),
}
for name, params in variants.items():
    result = detect_vcp_breakouts(frame, ticker='CELH', params=params)
    near = [b for b in result.breakouts if str(b.breakout_date.date()).startswith('2024')]
    print(f'CELH {name}: {len(result.breakouts)} hits, 2024={len(near)}')
    for b in near:
        print('  ', b.breakout_date.date(), b.footprint)
    if name == 'loose':
        for b in result.breakouts[-8:]:
            print('  any', b.breakout_date.date(), b.footprint)
