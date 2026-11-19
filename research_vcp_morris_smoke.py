"""One-name check that a Morris hit line is valid JSON with 20 bars."""

import json
from pathlib import Path

from research_vcp_morris import PRIOR, _build_design
from vcp.params import VCPParams
from vcp.scan import scan_param_sets_parallel

runs, _, _ = _build_design()
prior = json.loads(Path(PRIOR).read_text(encoding='utf-8'))
out = Path('artifacts/vcp_morris_smoke')
result = scan_param_sets_parallel(
    duck=r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb',
    tickers=[next(iter(prior['by_ticker']))],
    load_start=prior['summary']['load_start'],
    load_end=prior['summary']['window_end'],
    window_start=prior['summary']['window_start'],
    param_sets=[{'combo_id': 0, 'params': runs[0]['params']}, {'combo_id': -1, 'params': VCPParams().to_dict()}],
    out_dir=str(out),
    workers=1,
)
line = next(Path(result['parts'][0]['out_path']).open(encoding='utf-8'))
hit = json.loads(line)
print(result['n_hits'], hit['ticker'], hit['pattern_start'], hit['pattern_end'], len(hit['next_20']), hit['breakout_bar']['date'])
