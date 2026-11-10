"""Synthetic VCP path — ideal contractions + volume dry-up + breakout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vcp.detector import detect_vcp_breakouts
from vcp.params import VCPParams


def _synthetic_vcp(n: int = 320) -> pd.DataFrame:
    """Build a Stage-2 uptrend with a textbook 3-contraction coil then breakout."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range('2023-01-02', periods=n, tz='UTC')
    close = np.empty(n, dtype=float)
    volume = np.empty(n, dtype=float)

    # Steady grind higher into the base.
    price = 100.0
    for i in range(220):
        price *= 1.0 + float(rng.normal(0.0025, 0.008))
        close[i] = price
        volume[i] = float(rng.integers(1_500_000, 2_500_000))

    # Contraction 1: ~20% off a peak.
    peak1 = price
    for j, ret in enumerate(np.linspace(0, -0.20, 12)):
        i = 220 + j
        close[i] = peak1 * (1.0 + ret)
        volume[i] = float(rng.integers(1_200_000, 1_800_000))
    # Recover toward peak.
    low1 = close[231]
    for j, ret in enumerate(np.linspace(0, 0.18, 8)):
        i = 232 + j
        close[i] = low1 * (1.0 + ret)
        volume[i] = float(rng.integers(1_400_000, 2_000_000))

    # Contraction 2: ~10%.
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

    # Contraction 3: ~4% on very light volume (tight coil).
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

    # Breakout above peak3 on heavy volume.
    for j in range(261, n):
        close[j] = peak3 * (1.02 + 0.004 * (j - 261))
        volume[j] = float(rng.integers(2_800_000, 3_500_000)) if j == 261 else float(
            rng.integers(1_500_000, 2_200_000)
        )

    high = close * 1.005
    low = close * 0.995
    open_ = close.copy()
    return pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=idx,
    )


def test_detects_synthetic_vcp_breakout():
    df = _synthetic_vcp()
    params = VCPParams(
        require_trend_template=False,  # synthetic series is short of full MA geometry
        lookback_bars=80,
        pivot_up_th=0.03,
        pivot_down_th=0.03,
        vdu_max_ratio=0.85,
        breakout_rel_vol=1.2,
        min_first_depth=0.05,
        max_final_depth=0.06,
        contraction_ratio=0.65,
        min_base_bars=10,
    )
    result = detect_vcp_breakouts(df, ticker='SYN', params=params)
    assert len(result.breakouts) >= 1
    bo = result.breakouts[-1]
    assert bo.n_contractions >= 2
    assert bo.depths[0] > bo.depths[-1]
    assert bo.breakout_rel_vol >= params.breakout_rel_vol
