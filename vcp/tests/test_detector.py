"""Synthetic VCP path — contiguous peak→trough→recovery coils + breakout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vcp.detector import build_coils, detect_vcp_breakouts
from vcp.params import VCPParams
from vcp.pivots import Pivot


def _synthetic_vcp(n: int = 320) -> pd.DataFrame:
    """Build a Stage-2 uptrend with a textbook 3-contraction coil then breakout."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range('2023-01-02', periods=n, tz='UTC')
    close = np.empty(n, dtype=float)
    volume = np.empty(n, dtype=float)

    price = 100.0
    for i in range(220):
        price *= 1.0 + float(rng.normal(0.0025, 0.008))
        close[i] = price
        volume[i] = float(rng.integers(1_500_000, 2_500_000))

    peak1 = price
    for j, ret in enumerate(np.linspace(0, -0.20, 12)):
        i = 220 + j
        close[i] = peak1 * (1.0 + ret)
        volume[i] = float(rng.integers(1_200_000, 1_800_000))
    low1 = close[231]
    for j, ret in enumerate(np.linspace(0, 0.18, 8)):
        i = 232 + j
        close[i] = low1 * (1.0 + ret)
        volume[i] = float(rng.integers(1_400_000, 2_000_000))

    peak2 = close[239]
    for j, ret in enumerate(np.linspace(0, -0.10, 8)):
        i = 240 + j
        close[i] = peak2 * (1.0 + ret)
        volume[i] = float(rng.integers(700_000, 1_000_000))
    low2 = close[247]
    for j, ret in enumerate(np.linspace(0, 0.08, 6)):
        i = 248 + j
        close[i] = low2 * (1.0 + ret)
        volume[i] = float(rng.integers(800_000, 1_100_000))

    peak3 = close[253]
    for j, ret in enumerate(np.linspace(0, -0.04, 4)):
        i = 254 + j
        close[i] = peak3 * (1.0 + ret)
        volume[i] = float(rng.integers(250_000, 400_000))
    low3 = close[257]
    for j, ret in enumerate(np.linspace(0, 0.03, 3)):
        i = 258 + j
        close[i] = low3 * (1.0 + ret)
        volume[i] = float(rng.integers(300_000, 450_000))

    for j in range(261, n):
        close[j] = peak3 * (1.02 + 0.004 * (j - 261))
        volume[j] = (
            float(rng.integers(2_800_000, 3_500_000))
            if j == 261
            else float(rng.integers(1_500_000, 2_200_000))
        )

    high = close * 1.005
    low = close * 0.995
    open_ = close.copy()
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
    atr = pd.Series(2.0, index=idx)
    coils = build_coils(pivots, volume, atr)
    assert len(coils) == 3
    assert coils[0].recovery_peak is not None
    assert coils[1].peak.bar == coils[0].recovery_peak.bar
    assert coils[1].recovery_peak is not None
    assert coils[2].peak.bar == coils[1].recovery_peak.bar
    assert coils[0].depth_pct == 0.2
    assert coils[0].recovery_frac == (95 - 80) / (100 - 80)


def test_detects_synthetic_vcp_breakout():
    df = _synthetic_vcp()
    params = VCPParams(
        require_trend_template=False,
        lookback_bars=80,
        pivot_threshold_mode='pct',
        pivot_up_th=0.03,
        pivot_down_th=0.03,
        use_atr_depth_checks=False,
        vdu_max_ratio=0.85,
        breakout_rel_vol=1.2,
        min_first_depth=0.05,
        max_final_depth=0.06,
        contraction_ratio=0.65,
        min_base_bars=10,
        min_recovery_frac=0.3,
    )
    result = detect_vcp_breakouts(df, ticker='SYN', params=params)
    assert len(result.breakouts) >= 1
    bo = result.breakouts[-1]
    assert bo.n_contractions >= 2
    assert bo.depths[0] > bo.depths[-1]
    assert bo.breakout_rel_vol >= params.breakout_rel_vol
    # Contiguous chain in the accepted sequence.
    for prev, cur in zip(bo.contractions, bo.contractions[1:]):
        assert prev.recovery_peak is not None
        assert cur.peak.bar == prev.recovery_peak.bar
