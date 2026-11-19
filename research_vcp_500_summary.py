"""Summarize the 500-ticker artifact for the dashboard constants."""

import json
from collections import Counter
from pathlib import Path

doc = json.loads(Path('artifacts/vcp_500x10y.json').read_text())
by = doc['by_ticker']
hits = [h for rows in by.values() for h in rows]
print('tickers_with_hits', len(by))
print('hits', len(hits))
days = Counter(h['breakout_date'] for rows in by.values() for h in rows)
# same ticker same day
dup = 0
for rows in by.values():
    c = Counter(h['breakout_date'] for h in rows)
    dup += sum(n - 1 for n in c.values() if n > 1)
print('same_ticker_same_day_extra', dup)
first = [h['depths'][0] for h in hits if h['depths']]
# depths are log. percent = 1-exp(-d)
import math
pct = sorted((1 - math.exp(-d)) * 100 for d in first if d is not None)
print('first_leg_pct p50', pct[len(pct)//2], 'p90', pct[int(len(pct)*0.9)], 'max', pct[-1])
deep = sum(p >= 50 for p in pct)
print('first_leg_ge_50pct', deep)
conf = sorted(h['confidence'] for h in hits)
print('confidence p50', conf[len(conf)//2], 'min', conf[0], 'max', conf[-1])
print('bytes', Path('artifacts/vcp_500x10y.json').stat().st_size)
print('next20_full', sum(1 for h in hits if len(h['next_20_dates']) == 20), 'of', len(hits))
