"""Print a short readout of the VCP probe artifact."""

import json
from collections import Counter
from pathlib import Path

doc = json.loads(Path('artifacts/vcp_probe.json').read_text())
anchors = {'SMCI', 'AMD', 'CELH', 'DECK', 'MELI', 'AVGO', 'SNAP', 'NVDA'}
by = doc['by_ticker']
print('combo0 hits by ticker')
for ticker, hits in by.items():
    n = sum(1 for h in hits if h['combo_id'] == 0)
    if n:
        print(f'  {ticker:6} {n:3}  anchor={ticker in anchors}')
print('--- known dates ---')
want = {'SMCI': '2024-01-08', 'AMD': '2019-04-03', 'CELH': '2024-02-27'}
for ticker, day in want.items():
    hits = [h for h in by.get(ticker, []) if h['breakout_date'] == day]
    for h in hits:
        print(ticker, h['combo_id'], h['footprint'], h['pattern_start'], h['pattern_end'], len(h['next_20_dates']))
    if not hits:
        print(ticker, day, 'ABSENT')
print('--- combo0 anchors ---')
for ticker in ['SMCI', 'AMD', 'CELH', 'DECK', 'MELI', 'AVGO', 'SNAP', 'NVDA']:
    hits = [h for h in by.get(ticker, []) if h['combo_id'] == 0]
    print(ticker, len(hits))
    for h in hits:
        print('   ', h['breakout_date'], h['pattern_start'], h['pattern_end'], h['footprint'], 'next', len(h['next_20_dates']))

# hit rate split
anchor_calls = 8 * 5
spread_names = [t for t in by if t not in anchors]
anchor_hits = sum(len(by[t]) for t in anchors if t in by)
spread_hits = sum(len(by[t]) for t in spread_names)
print('anchor hits', anchor_hits, 'spread hits', spread_hits, 'spread names', len(spread_names))
print('bytes per dates hit', doc['summary']['dates_only_bytes'] / doc['summary']['n_hits'])
print('bytes per ohlcv hit', doc['summary']['with_next20_ohlcv_bytes'] / doc['summary']['n_hits'])
