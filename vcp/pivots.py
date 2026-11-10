"""Swing high/low extraction.

Uses ``vbt.PIVOTINFO`` when vectorbtpro is available; otherwise a directional-
change zigzag. Thresholds can be fixed fractions or ATR multiples (visual
volatility units — better match what the eye treats as a real swing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from vcp.vbt_compat import VBT_FLAVOR, vbt

PivotKind = Literal['high', 'low']


@dataclass(frozen=True)
class Pivot:
    index: pd.Timestamp
    price: float
    kind: PivotKind
    bar: int  # integer location in the source series


def _threshold_at(i: int, price: float, atr: np.ndarray | None, atr_mult: float, pct_th: float) -> float:
    if atr is not None and np.isfinite(atr[i]) and atr[i] > 0:
        return float(atr[i] * atr_mult)
    return float(price * pct_th)


def _zigzag_pivots(
    high: pd.Series,
    low: pd.Series,
    *,
    up_th: float,
    down_th: float,
    atr: pd.Series | None = None,
    atr_mult: float = 1.5,
    threshold_mode: str = 'pct',
) -> list[Pivot]:
    """Confirmed directional-change pivots on High/Low (no look-ahead rewrite)."""
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    index = high.index
    n = len(highs)
    if n == 0:
        return []

    atr_arr = None
    if threshold_mode == 'atr' and atr is not None:
        atr_s = atr.reindex(high.index)
        atr_arr = atr_s.to_numpy(dtype=float)

    pivots: list[Pivot] = []
    # 1 = seeking peak confirmation (after rising), -1 = seeking valley confirmation
    direction = 0
    cand_i = 0
    cand_high = highs[0]
    cand_low = lows[0]

    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        if direction >= 0:
            if h >= cand_high:
                cand_high = h
                cand_i = i
            else:
                th = _threshold_at(i, cand_high, atr_arr, atr_mult, down_th)
                if cand_high > 0 and (cand_high - l) >= th:
                    pivots.append(Pivot(index=index[cand_i], price=float(cand_high), kind='high', bar=cand_i))
                    direction = -1
                    cand_low = l
                    cand_i = i
                    continue
        if direction <= 0:
            if l <= cand_low:
                cand_low = l
                cand_i = i
            else:
                th = _threshold_at(i, cand_low, atr_arr, atr_mult, up_th)
                if cand_low > 0 and (h - cand_low) >= th:
                    pivots.append(Pivot(index=index[cand_i], price=float(cand_low), kind='low', bar=cand_i))
                    direction = 1
                    cand_high = h
                    cand_i = i

    return _alternate(pivots)


def _alternate(pivots: list[Pivot]) -> list[Pivot]:
    """Keep an alternating high/low sequence; prefer more extreme of same kind."""
    if not pivots:
        return []
    out: list[Pivot] = [pivots[0]]
    for p in pivots[1:]:
        last = out[-1]
        if p.kind == last.kind:
            if p.kind == 'high' and p.price >= last.price:
                out[-1] = p
            elif p.kind == 'low' and p.price <= last.price:
                out[-1] = p
        else:
            out.append(p)
    return out


def _pivots_from_pivotinfo(
    high: pd.Series,
    low: pd.Series,
    up_th: float,
    down_th: float,
) -> list[Pivot] | None:
    if not hasattr(vbt, 'PIVOTINFO'):
        return None
    info = vbt.PIVOTINFO.run(high, low, up_th=up_th, down_th=down_th)
    conf_value = None
    conf_type = None
    for name in ('conf_value', 'confirmed_value', 'last_pivot_value'):
        if hasattr(info, name):
            conf_value = getattr(info, name)
            break
    for name in ('conf_type', 'confirmed_type', 'last_pivot_type'):
        if hasattr(info, name):
            conf_type = getattr(info, name)
            break
    if conf_value is None:
        return None

    series_v = conf_value if isinstance(conf_value, pd.Series) else conf_value.iloc[:, 0]
    pivots: list[Pivot] = []
    prev = np.nan
    for i, (ts, val) in enumerate(series_v.items()):
        if pd.isna(val) or (isinstance(prev, float) and not np.isnan(prev) and np.isclose(val, prev)):
            prev = float(val) if not pd.isna(val) else prev
            continue
        kind: PivotKind = 'high'
        if conf_type is not None:
            tseries = conf_type if isinstance(conf_type, pd.Series) else conf_type.iloc[:, 0]
            tval = tseries.iloc[i]
            if isinstance(tval, str):
                kind = 'high' if 'high' in tval.lower() or 'peak' in tval.lower() else 'low'
            else:
                kind = 'high' if (np.isnan(prev) or val >= prev) else 'low'
        else:
            kind = 'high' if (np.isnan(prev) or val >= prev) else 'low'
        pivots.append(Pivot(index=ts, price=float(val), kind=kind, bar=i))
        prev = float(val)

    fixed: list[Pivot] = []
    for p in pivots:
        if not fixed:
            fixed.append(p)
            continue
        kind = 'low' if p.price < fixed[-1].price else 'high'
        fixed.append(Pivot(index=p.index, price=p.price, kind=kind, bar=p.bar))
    return _alternate(fixed)


def detect_pivots(
    high: pd.Series,
    low: pd.Series,
    *,
    up_th: float = 0.035,
    down_th: float = 0.035,
    atr: pd.Series | None = None,
    atr_mult: float = 1.5,
    threshold_mode: str = 'pct',
) -> list[Pivot]:
    """Return confirmed alternating swing highs/lows."""
    # PRO PIVOTINFO is pct-threshold based; use zigzag when ATR mode is requested.
    if threshold_mode == 'pct' and VBT_FLAVOR == 'vectorbtpro':
        pro = _pivots_from_pivotinfo(high, low, up_th, down_th)
        if pro is not None:
            return pro
    return _zigzag_pivots(
        high,
        low,
        up_th=up_th,
        down_th=down_th,
        atr=atr,
        atr_mult=atr_mult,
        threshold_mode=threshold_mode,
    )
