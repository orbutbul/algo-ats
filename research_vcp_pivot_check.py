"""Compare a tighter pivot parse against the current one. No detector edits."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from vcp.pivots import Pivot, _alternate, _pivots_from_pivotinfo
from vcp.vbt_compat import vbt


def _fast(high: pd.Series, low: pd.Series, up_th: float, down_th: float):
    info = vbt.PIVOTINFO.run(high, low, up_th=up_th, down_th=down_th)
    conf_value = info.conf_value
    series_v = conf_value if isinstance(conf_value, pd.Series) else conf_value.iloc[:, 0]
    values = series_v.to_numpy(dtype=np.float64, copy=False)
    index = series_v.index
    pivots: list[Pivot] = []
    prev = np.nan
    for i, val in enumerate(values):
        if val != val or (prev == prev and abs(val - prev) <= (1e-08 + 1e-05 * abs(prev))):
            if val == val:
                prev = float(val)
            continue
        kind = 'high' if (prev != prev or val >= prev) else 'low'
        pivots.append(Pivot(index=index[i], price=float(val), kind=kind, bar=i))
        prev = float(val)
    fixed: list[Pivot] = []
    for p in pivots:
        if not fixed:
            fixed.append(p)
            continue
        kind = 'low' if p.price < fixed[-1].price else 'high'
        fixed.append(Pivot(index=p.index, price=p.price, kind=kind, bar=p.bar))
    return _alternate(fixed)


def _load(con, ticker: str, start: str | None) -> pd.DataFrame:
    sql = '''
        SELECT datetime, open, high, low, close, volume
        FROM ohlcv_max_daily
        WHERE ticker = ?
    '''
    params: list = [ticker]
    if start:
        sql += ' AND datetime >= ?'
        params.append(start)
    sql += ' ORDER BY datetime'
    frame = con.execute(sql, params).df()
    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True)
    return frame.set_index('datetime')


def main() -> None:
    import duckdb

    from vcp.detector import detect_vcp_breakouts
    import vcp.pivots as pivots_mod

    con = duckdb.connect(r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb', read_only=True)
    cases = [('NVDA', '2016-01-01'), ('AMD', None), ('SMCI', '2016-01-01'), ('SNAP', None), ('AAPL', '2000-01-01')]
    for ticker, start in cases:
        frame = _load(con, ticker, start)
        high = frame['high']
        low = frame['low']
        old = _pivots_from_pivotinfo(high, low, 0.035, 0.035)
        t0 = time.perf_counter()
        old = _pivots_from_pivotinfo(high, low, 0.035, 0.035)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = _fast(high, low, 0.035, 0.035)
        t_new = time.perf_counter() - t0
        bad = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
        print(
            f'{ticker} bars={len(frame)} pivots {len(old)} vs {len(new)} '
            f'mismatch={len(bad) + abs(len(old) - len(new))} old={t_old:.3f}s new={t_new:.3f}s'
        )
        cols = {'Open': frame['open'], 'High': frame['high'], 'Low': frame['low'], 'Close': frame['close'], 'Volume': frame['volume']}
        ohlcv = pd.DataFrame(cols)
        before = detect_vcp_breakouts(ohlcv, ticker=ticker)
        pivots_mod._pivots_from_pivotinfo = _fast
        after = detect_vcp_breakouts(ohlcv, ticker=ticker)
        pivots_mod._pivots_from_pivotinfo = _pivots_from_pivotinfo
        b = [x.to_dict() for x in before.breakouts]
        a = [x.to_dict() for x in after.breakouts]
        print(f'  breakouts {len(b)} vs {len(a)} equal={b == a}')
    con.close()


if __name__ == '__main__':
    main()
