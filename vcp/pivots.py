"""Swing high/low extraction.

Uses ``vbt.PIVOTINFO`` when vectorbtpro is available; otherwise a directional-
change zigzag with the same up_th / down_th semantics (confirm a peak after a
fall of ``down_th``, a valley after a rise of ``up_th``).
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


def _zigzag_pivots(
    high: pd.Series,
    low: pd.Series,
    up_th: float,
    down_th: float,
) -> list[Pivot]:
    """Confirmed directional-change pivots on High/Low (no look-ahead rewrite)."""
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    index = high.index
    n = len(highs)
    if n == 0:
        return []

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
            elif cand_high > 0 and (cand_high - l) / cand_high >= down_th:
                pivots.append(Pivot(index=index[cand_i], price=float(cand_high), kind='high', bar=cand_i))
                direction = -1
                cand_low = l
                cand_i = i
                continue
        if direction <= 0:
            if l <= cand_low:
                cand_low = l
                cand_i = i
            elif cand_low > 0 and (h - cand_low) / cand_low >= up_th:
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
    # Confirmed pivot value + type when available; attribute names vary slightly
    # across PRO builds, so probe a few common ones.
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
            # PRO often encodes 1=peak, -1=valley (or vice versa); also accept strings.
            if isinstance(tval, str):
                kind = 'high' if 'high' in tval.lower() or 'peak' in tval.lower() else 'low'
            else:
                # Infer from price vs neighbors when type encoding is ambiguous:
                # compare to previous confirmed pivot.
                kind = 'high' if (np.isnan(prev) or val >= prev) else 'low'
                if not np.isnan(prev):
                    kind = 'high' if val > prev else 'low'
        else:
            kind = 'high' if (np.isnan(prev) or val >= prev) else 'low'
        # Re-infer kind from alternating extremes for robustness.
        pivots.append(Pivot(index=ts, price=float(val), kind=kind, bar=i))
        prev = float(val)

    # Fix kinds by reconstructing from price sequence (safer than type codes).
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
    up_th: float,
    down_th: float,
) -> list[Pivot]:
    """Return confirmed alternating swing highs/lows."""
    if VBT_FLAVOR == 'vectorbtpro':
        pro = _pivots_from_pivotinfo(high, low, up_th, down_th)
        if pro is not None:
            return pro
    return _zigzag_pivots(high, low, up_th, down_th)
