"""Linear-scale history chart with VCP contraction paths and breakout markers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vcp.detector import VCPResult


def plot_vcp_history(
    ohlcv: pd.DataFrame,
    result: VCPResult,
    *,
    title: str | None = None,
    outfile: str | Path | None = None,
    max_breakouts_labeled: int = 12,
) -> Path | None:
    """Price (linear) + volume with zigzag contractions and breakout markers."""
    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    fig, (ax_px, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    fig.patch.set_facecolor('#f7f4ef')
    for ax in (ax_px, ax_vol):
        ax.set_facecolor('#f7f4ef')

    ax_px.plot(df.index, df['Close'], color='#1f2a44', lw=1.1, label='Close')
    ax_px.set_ylabel('Price')

    def _naive(ts) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        return t.tz_convert(None) if t.tzinfo is not None else t

    # Draw each breakout's contraction path.
    cmap = plt.cm.Dark2(np.linspace(0, 1, max(1, len(result.breakouts))))
    for i, bo in enumerate(result.breakouts):
        color = cmap[i % len(cmap)]
        xs: list[pd.Timestamp] = []
        ys: list[float] = []
        for c in bo.contractions:
            xs.extend([_naive(c.peak.index), _naive(c.trough.index)])
            ys.extend([c.peak.price, c.trough.price])
            if c.recovery_peak is not None:
                xs.append(_naive(c.recovery_peak.index))
                ys.append(c.recovery_peak.price)
        ax_px.plot(xs, ys, color=color, lw=1.6, alpha=0.85)
        bdate = _naive(bo.breakout_date)
        ax_px.axhline(bo.pivot_price, color=color, lw=0.7, ls='--', alpha=0.45)
        ax_px.scatter(
            [bdate],
            [bo.pivot_price],
            marker='^',
            s=70,
            color=color,
            zorder=5,
            edgecolors='black',
            linewidths=0.4,
        )
        if i < max_breakouts_labeled:
            ax_px.annotate(
                bo.footprint,
                (bdate, bo.pivot_price),
                textcoords='offset points',
                xytext=(6, 8),
                fontsize=8,
                color=color,
            )

    ax_px.set_title(title or f'{result.ticker} VCP breakouts ({len(result.breakouts)})')
    ax_vol.bar(df.index, df['Volume'], color='#8a8f98', width=1.0, alpha=0.7)
    # Highlight breakout-day volume.
    for bo in result.breakouts:
        bdate = _naive(bo.breakout_date)
        if bdate in df.index:
            ax_vol.bar([bdate], [df.loc[bdate, 'Volume']], color='#c0392b', width=1.2)

    ax_vol.set_ylabel('Volume')
    ax_px.legend(loc='upper left', frameon=False)
    ax_px.grid(True, which='both', alpha=0.25)
    ax_vol.grid(True, alpha=0.25)
    ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_vol.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_vol.xaxis.get_major_locator()))
    fig.tight_layout()

    if outfile is None:
        plt.show()
        return None
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    return outfile
