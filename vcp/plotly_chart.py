"""Interactive Plotly figure: log price + volume with VCP breakout marks."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vcp.detector import VCPResult


def _naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, 'tz', None) is not None:
        return idx.tz_convert(None)
    return idx


def _naive_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if t.tzinfo is not None else t


def build_vcp_figure(
    ohlcv: pd.DataFrame,
    result: VCPResult,
    *,
    title: str | None = None,
) -> go.Figure:
    df = ohlcv.copy()
    df.index = _naive_index(pd.to_datetime(df.index))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close',
            line=dict(color='#1b2a44', width=1.35),
            hovertemplate='%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>',
        ),
        row=1,
        col=1,
    )

    palette = [
        '#0f6a4f',
        '#b45309',
        '#7c3aed',
        '#be123c',
        '#0369a1',
        '#4d7c0f',
        '#c2410c',
        '#334155',
    ]

    # Mark final-coil dry-up bars (quiet volume into the pivot).
    dryup_days: set[pd.Timestamp] = set()
    for bo in result.breakouts:
        final = bo.contractions[-1]
        for ts in df.index[final.peak.bar : final.trough.bar + 1]:
            dryup_days.add(_naive_ts(ts).normalize())

    for i, bo in enumerate(result.breakouts):
        color = palette[i % len(palette)]
        xs: list[pd.Timestamp] = []
        ys: list[float] = []
        for c in bo.contractions:
            xs.extend([_naive_ts(c.peak.index), _naive_ts(c.trough.index)])
            ys.extend([c.peak.price, c.trough.price])
            if c.recovery_peak is not None:
                xs.append(_naive_ts(c.recovery_peak.index))
                ys.append(c.recovery_peak.price)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode='lines+markers',
                name=bo.footprint,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate=f'{bo.footprint}<br>%{{x|%Y-%m-%d}} %{{y:.2f}}<extra></extra>',
                legendgroup=f'bo{i}',
            ),
            row=1,
            col=1,
        )
        bdate = _naive_ts(bo.breakout_date)
        depths_pct = [round((math.exp(d) - 1) * 100, 1) for d in bo.depths]
        fig.add_trace(
            go.Scatter(
                x=[bdate],
                y=[bo.pivot_price],
                mode='markers+text',
                name=f'Breakout {bdate.date()}',
                text=[bo.footprint],
                textposition='top center',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color=color,
                    line=dict(width=0.6, color='#111'),
                ),
                hovertemplate=(
                    f'<b>{bo.ticker} breakout</b><br>'
                    f'%{{x|%Y-%m-%d}}<br>'
                    f'pivot=%{{y:.2f}}<br>'
                    f'stop={bo.stop_price:.2f}<br>'
                    f'depths%={depths_pct}<br>'
                    f'coil vol vs MA={[round(v, 2) for v in bo.volume_dryup_ratios]}<br>'
                    f'breakout vs dry-up={bo.breakout_vs_dryup:.2f}x<br>'
                    f'rel vol={bo.breakout_rel_vol:.2f}'
                    f'<extra></extra>'
                ),
                legendgroup=f'bo{i}',
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=bo.pivot_price,
            line=dict(color=color, width=1, dash='dot'),
            opacity=0.45,
            row=1,
            col=1,
        )

    breakout_days = {_naive_ts(b.breakout_date).normalize() for b in result.breakouts}
    vol_colors = []
    for ts in df.index:
        day = pd.Timestamp(ts).normalize()
        if day in breakout_days:
            vol_colors.append('#c2410c')  # expansion
        elif day in dryup_days:
            vol_colors.append('#0f6a4f')  # dry-up into pivot
        else:
            vol_colors.append('#9aa3af')

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(color=vol_colors),
            hovertemplate='%{x|%Y-%m-%d}<br>Vol=%{y:,.0f}<extra></extra>',
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(type='log', title_text='Price (log)', row=1, col=1, showgrid=True, zeroline=False)
    fig.update_yaxes(title_text='Vol', row=2, col=1, showgrid=False, zeroline=False, nticks=3)
    fig.update_xaxes(title_text=None, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text=None, showticklabels=True, row=2, col=1)
    fig.update_layout(
        title=dict(
            text=title or f'{result.ticker} · {len(result.breakouts)} VCP breakout(s)',
            font=dict(size=20, family='Fraunces, Georgia, serif', color='#1b2a4a'),
        ),
        template='plotly_white',
        height=760,
        margin=dict(l=56, r=24, t=64, b=36),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        paper_bgcolor='#f3efe6',
        plot_bgcolor='#f8f5ee',
        font=dict(family='IBM Plex Sans, Helvetica, sans-serif', color='#243044'),
        hovermode='x unified',
        bargap=0.05,
    )
    return fig
