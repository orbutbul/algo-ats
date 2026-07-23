"""
dashboard.py — regenerates a self-contained, offline-viewable HTML report of
OHLCV data health. Run manually: `python dashboard.py` (writes data/dashboard.html).

Re-runnable: reuses the cached gap report (see gap_detection.load_or_build_gap_report)
unless --force-refresh is passed, since this is a periodic human-facing check,
not a pipeline-critical step.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from gap_detection import load_or_build_gap_report, summarize

DEFAULT_OUT = Path(__file__).parent / 'data' / 'dashboard.html'


def _heatmap_html(gap_report: pd.DataFrame, asset_classes: list[str], title: str, top_n: int | None, first_chart: bool) -> str:
    subset = gap_report[gap_report['asset_class'].isin(asset_classes)]
    if subset.empty:
        return f'<h2>{title}</h2><p>No tracked tickers.</p>'

    pivot = subset['completeness'].unstack('date').sort_index(axis=1)

    if top_n is not None and len(pivot) > top_n:
        worst = pivot.mean(axis=1).sort_values().index[:top_n]
        pivot = pivot.loc[worst]

    pivot = pivot.sort_index()

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=[d.isoformat() for d in pivot.columns],
        y=pivot.index.tolist(),
        colorscale='YlGnBu',
        zmin=0, zmax=100,
        colorbar=dict(title='% complete'),
        hovertemplate='ticker=%{y}<br>date=%{x}<br>completeness=%{z:.1f}%<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        height=max(300, 16 * len(pivot) + 100),
        margin=dict(l=100, r=20, t=40, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs='inline' if first_chart else False)


def _worst_coverage_table(gap_report: pd.DataFrame, top_n: int) -> str:
    by_ticker = gap_report.groupby(level='ticker')
    dates = gap_report.index.get_level_values('date')

    rows = pd.DataFrame({
        'asset_class': by_ticker['asset_class'].first(),
        'mean_completeness': by_ticker['completeness'].mean(),
        'whole_day_gap_count': by_ticker['is_whole_day_gap'].sum(),
        'days_below_threshold': by_ticker['needs_refetch'].sum(),
    })

    present = gap_report[gap_report['actual_minutes'] > 0]
    if not present.empty:
        present_dates = present.reset_index()[['ticker', 'date']]
        first_last = present_dates.groupby('ticker')['date'].agg(['min', 'max'])
        first_last.columns = ['first_date_present', 'last_date_present']
        rows = rows.join(first_last)

    rows = rows.sort_values('mean_completeness').head(top_n)
    rows['mean_completeness'] = (rows['mean_completeness'] * 100).round(1)
    return rows.to_html(classes='data-table', border=0)


def _dead_ticker_table(gap_report: pd.DataFrame) -> str:
    by_ticker = gap_report.groupby(level='ticker')
    mean_completeness = by_ticker['completeness'].mean()
    dead = mean_completeness[mean_completeness == 0.0]
    if dead.empty:
        return '<p>No fully-dead tickers over the lookback window.</p>'
    dead_df = gap_report.loc[gap_report.index.get_level_values('ticker').isin(dead.index)]
    asset_class = dead_df.groupby(level='ticker')['asset_class'].first()
    out = pd.DataFrame({'asset_class': asset_class}).sort_index()
    return out.to_html(classes='data-table', border=0)


def _summary_html(summary: dict) -> str:
    generated_at = summary.pop('generated_at', None)
    rows = []
    for asset_class, stats in summary.items():
        rows.append(
            f"<tr><td>{asset_class}</td><td>{stats['tickers']}</td>"
            f"<td>{stats['tickers_with_whole_day_gap']}</td>"
            f"<td>{stats['tickers_below_threshold']}</td>"
            f"<td>{stats['mean_completeness']:.1%}</td>"
            f"<td>{stats['median_completeness']:.1%}</td></tr>"
        )
    header = (
        '<tr><th>asset class</th><th>tickers</th><th>with whole-day gap</th>'
        '<th>below fill threshold</th><th>mean completeness</th><th>median completeness</th></tr>'
    )
    ts = generated_at.isoformat() if generated_at is not None else 'unknown'
    return (
        f'<p>Gap report generated at: {ts}</p>'
        f'<table class="data-table" border="0"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'
    )


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>OHLCV Data Health Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 40px; }}
  table.data-table {{ border-collapse: collapse; width: 100%; }}
  table.data-table th, table.data-table td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #ddd; }}
  table.data-table th:first-child, table.data-table td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>OHLCV Data Health Dashboard</h1>
{summary}
<h2>Equity / ETF coverage (worst {top_n} tickers)</h2>
{equity_heatmap}
<h2>Crypto coverage</h2>
{crypto_heatmap}
<h2>Worst coverage — top {top_n} tickers</h2>
{worst_table}
<h2>Dead tickers (0% completeness over lookback window)</h2>
{dead_table}
</body>
</html>
"""


def build_dashboard(out_path: Path = DEFAULT_OUT, force: bool = False,
                     max_age_hours: float = 6.0, top_n: int = 50) -> None:
    gap_report = load_or_build_gap_report(max_age_hours=max_age_hours, force=force)
    summary = summarize(gap_report)

    html = _PAGE_TEMPLATE.format(
        summary=_summary_html(summary),
        top_n=top_n,
        equity_heatmap=_heatmap_html(gap_report, ['equities', 'etfs'], 'Equities / ETFs', top_n, first_chart=True),
        crypto_heatmap=_heatmap_html(gap_report, ['cryptos'], 'Crypto', None, first_chart=False),
        worst_table=_worst_coverage_table(gap_report, top_n),
        dead_table=_dead_ticker_table(gap_report),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Regenerate the data-health HTML dashboard.')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--force-refresh', action='store_true',
                    help='Recompute the gap report instead of reusing the cache.')
    p.add_argument('--max-age-hours', type=float, default=6.0,
                    help='Reuse the cached gap report if younger than this.')
    p.add_argument('--top-n', type=int, default=50, help='Rows in the worst-coverage table / heatmap.')
    args = p.parse_args()
    build_dashboard(args.out, force=args.force_refresh, max_age_hours=args.max_age_hours, top_n=args.top_n)
