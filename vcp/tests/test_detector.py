"""Shelf / excursion VCP — High/Low contractions under a shared resistance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vcp.detector import (
    build_coils,
    build_excursions_under_shelf,
    classify_base_excursions,
    detect_vcp_breakouts,
)
from vcp.params import VCPParams
from vcp.pivots import Pivot


def _synthetic_shelf_vcp(n: int = 340) -> pd.DataFrame:
    """Stage-2 advance into a clear shelf with 3 tightening excursions + breakout."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range('2023-01-02', periods=n, tz='UTC')
    close = np.empty(n, dtype=float)
    high = np.empty(n, dtype=float)
    low = np.empty(n, dtype=float)
    volume = np.empty(n, dtype=float)

    price = 100.0
    for i in range(220):
        price *= 1.0 + float(rng.normal(0.0025, 0.006))
        close[i] = price
        high[i] = price * (1.0 + abs(float(rng.normal(0.004, 0.002))))
        low[i] = price * (1.0 - abs(float(rng.normal(0.004, 0.002))))
        volume[i] = float(rng.integers(1_500_000, 2_500_000))

    shelf = close[219] * 1.01
    # Excursion 1: ~20% under shelf, with fractal noise that never reclaims shelf.
    for j in range(18):
        i = 220 + j
        # Dip then partial bounce (noise), then deeper low, then reclaim.
        if j < 6:
            frac = j / 5
            close[i] = shelf * (1.0 - 0.12 * frac)
        elif j < 10:
            close[i] = shelf * (0.88 + 0.04 * ((j - 6) / 3))  # bounce short of shelf
        elif j < 14:
            close[i] = shelf * (0.92 - 0.10 * ((j - 10) / 3))  # deeper low ~20%
        else:
            close[i] = shelf * (0.80 + 0.20 * ((j - 14) / 3))  # reclaim to shelf
        high[i] = max(close[i] * 1.01, close[i])
        low[i] = min(close[i] * 0.99, close[i])
        volume[i] = float(rng.integers(1_100_000, 1_700_000))
    # Pin exact shelf touches and trough.
    high[220] = shelf
    low[233] = shelf * 0.80
    high[237] = shelf * 0.995

    # Excursion 2: ~10% under shelf.
    for j in range(12):
        i = 238 + j
        if j < 6:
            close[i] = shelf * (1.0 - 0.10 * (j / 5))
        else:
            close[i] = shelf * (0.90 + 0.10 * ((j - 6) / 5))
        high[i] = close[i] * 1.008
        low[i] = close[i] * 0.992
        volume[i] = float(rng.integers(700_000, 1_050_000))
    low[244] = shelf * 0.90
    high[249] = shelf * 0.998

    # Excursion 3: tight ~4% dry-up coil.
    for j in range(8):
        i = 250 + j
        if j < 4:
            close[i] = shelf * (1.0 - 0.04 * (j / 3))
        else:
            close[i] = shelf * (0.96 + 0.04 * ((j - 4) / 3))
        high[i] = close[i] * 1.004
        low[i] = close[i] * 0.996
        volume[i] = float(rng.integers(200_000, 340_000))
    low[253] = shelf * 0.96
    high[257] = shelf * 0.997

    for j in range(258, n):
        if j == 258:
            close[j] = shelf * 1.03
            high[j] = shelf * 1.04
            low[j] = shelf * 1.01
            volume[j] = float(rng.integers(2_800_000, 3_500_000))
        else:
            close[j] = shelf * (1.03 + 0.004 * (j - 258))
            high[j] = close[j] * 1.01
            low[j] = close[j] * 0.995
            volume[j] = float(rng.integers(1_500_000, 2_200_000))

    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=idx,
    )


def test_build_coils_are_contiguous_peak_trough_recovery():
    pivots = [
        Pivot(index=pd.Timestamp('2024-01-02'), price=100.0, kind='high', bar=0),
        Pivot(index=pd.Timestamp('2024-01-10'), price=80.0, kind='low', bar=5),
        Pivot(index=pd.Timestamp('2024-01-20'), price=95.0, kind='high', bar=10),
        Pivot(index=pd.Timestamp('2024-01-28'), price=88.0, kind='low', bar=15),
        Pivot(index=pd.Timestamp('2024-02-05'), price=94.0, kind='high', bar=20),
        Pivot(index=pd.Timestamp('2024-02-08'), price=91.0, kind='low', bar=22),
    ]
    idx = pd.bdate_range('2024-01-02', periods=30)
    volume = pd.Series(1_000_000.0, index=idx)
    coils = build_coils(pivots, volume)
    assert len(coils) == 3
    assert coils[1].peak.bar == coils[0].recovery_peak.bar
    assert coils[2].peak.bar == coils[1].recovery_peak.bar


def test_shelf_excursions_ignore_intra_pullback_noise():
    df = _synthetic_shelf_vcp()
    high = df['High'].to_numpy(dtype=float)
    low = df['Low'].to_numpy(dtype=float)
    volume = df['Volume'].to_numpy(dtype=float)
    shelf = float(high[220])
    params = VCPParams(require_trend_template=False, lookback_bars=80, min_base_bars=10)
    raw = build_excursions_under_shelf(high, low, volume, df.index, 220, shelf, params)
    # Fractal bounce inside first pullback must not create an extra shelf-to-shelf T.
    assert len(raw) >= 3
    assert raw[0].depth_pct >= 0.15
    base = classify_base_excursions(raw, params)
    assert len(base) >= 3
    assert base[0].trough.price < base[1].trough.price <= base[2].trough.price * 1.01


def test_detects_synthetic_vcp_breakout():
    df = _synthetic_shelf_vcp()
    params = VCPParams(
        require_trend_template=False,
        lookback_bars=80,
        price_space='pct',
        min_contractions=3,
        pivot_up_th=0.025,
        pivot_down_th=0.025,
        shelf_tol=0.03,
        noise_depth_frac=0.30,
        vdu_max_ratio=0.85,
        vdu_quiet_bar_ratio=0.60,
        breakout_rel_vol=1.2,
        breakout_vs_dryup_mult=1.5,
        min_first_depth=0.05,
        max_final_depth=0.08,
        contraction_ratio=0.85,
        min_base_bars=10,
        min_confidence=0.30,
        higher_low_tol=0.02,
    )
    result = detect_vcp_breakouts(df, ticker='SYN', params=params)
    assert len(result.breakouts) >= 1
    bo = result.breakouts[-1]
    assert bo.n_contractions >= 3
    assert bo.confidence >= params.min_confidence
    assert bo.shelf_price > 0
    assert bo.depths[0] > bo.depths[-1]
    assert bo.breakout_vs_dryup >= params.breakout_vs_dryup_mult
    # Returns reclaim the shared shelf (around the same high).
    for c in bo.contractions:
        assert c.recovery_peak is not None
        assert abs(c.recovery_peak.price - bo.shelf_price) / bo.shelf_price <= params.shelf_tol + 0.01
