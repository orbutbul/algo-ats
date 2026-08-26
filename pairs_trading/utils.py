"""
pairs_trading/utils.py — group tickers into pairs-trading candidate pools
using data/metadata.json (yfinance's raw per-ticker dump, equities/ETFs only —
no crypto in that file, and crypto is out of scope here regardless).
"""

import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from plotly.graph_objects import Figure, Scatter
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pairs_trading.kalman import _kalman_filter_hedge_ratio
from pairs_trading.pairs_filter import load_metadata
from pairs_trading.pair_tests import (
    _crossing_dates, _filter_coverage, _full_sample_ols_beta, _load_group_daily_closes,
    _pair_correlation_stats, _pair_cointegration_stats, _rolling_cointegration_stat,
    _rolling_corr_matrices, _rolling_hurst_exponent, _rolling_ols_beta, _rolling_pair_correlation,
    _threshold_crossings,
)

METADATA_PATH = Path(__file__).parent.parent / 'data' / 'metadata.json'

# ETF 'category' (a Morningstar style-box label) -> equity 'sector' name, for
# the handful of categories that map cleanly onto a sector. Most ETF
# categories (broad market/style-box funds, bond funds, regional funds, etc.)
# have no equity-sector counterpart and are intentionally left unmapped.
_SECTOR_ALIASES = {
    'Financial': 'Financial Services',
    'Health': 'Healthcare',
    'Communications': 'Communication Services',
    'Equity Energy': 'Energy',
    'Equity Precious Metals': 'Basic Materials',
}


def find_pair_groups(metadata_path: Path | None = None) -> dict:
    """
    Groups equities/ETFs from data/metadata.json into pairs-trading candidate
    pools by sector, with equities further split by industry.

    Returns:
        {sector: {
            'etfs': [ticker, ...],                    # sector-focused ETFs
            'industries': {industry: [ticker, ...]},  # equities, grouped tight
        }}

    Equities are grouped by (sector, industry) — e.g. all 'Technology' /
    'Semiconductors' names together — the standard tight starting point for
    pairs trading (same narrow business, correlated but not identical).

    ETFs don't have an industry-level field (Morningstar 'category' is a
    style-box label, not a sector), so they're attached at the sector level
    only, via a small alias map (_SECTOR_ALIASES) for the handful of
    categories that map cleanly onto an equity sector (e.g. category
    'Technology' -> sector 'Technology', 'Health' -> 'Healthcare'). Broad
    market/style-box/bond/regional ETFs (e.g. 'Large Blend', 'Foreign Large
    Value') have no equity-sector counterpart and are excluded — this is
    intentional, not a gap.

    Crypto is not in metadata.json and is out of scope entirely.
    """
    metadata_path = metadata_path or METADATA_PATH
    with open(metadata_path) as f:
        data = json.load(f)

    groups: dict = {}

    # Pass 1: equities establish the known sectors (and their industry
    # subgroups) — done first so ETF matching in pass 2 isn't dependent on
    # dict iteration order.
    for ticker, info in data.items():
        if info.get('asset_class') != 'equities':
            continue
        sector = info.get('sector')
        industry = info.get('industry')
        if not sector or not industry:
            continue
        bucket = groups.setdefault(sector, {'etfs': [], 'industries': {}})
        bucket['industries'].setdefault(industry, []).append(ticker)

    # Pass 2: ETFs attach to a sector only if their category maps (directly
    # or via alias) onto a sector that already has equities.
    for ticker, info in data.items():
        if info.get('asset_class') != 'etfs':
            continue
        category = info.get('category')
        if not category:
            continue
        sector = _SECTOR_ALIASES.get(category, category)
        if sector not in groups:
            continue  # category doesn't map onto any equity sector we've seen
        groups[sector]['etfs'].append(ticker)

    return groups


def _cluster_once(tickers: list[str], metadata: dict, n_clusters: int) -> list[list[str]]:
    """One agglomerative-clustering pass on TF-IDF cosine distance between
    longBusinessSummary descriptions, fit fresh on just these tickers."""
    descriptions = [metadata.get(t, {}).get('longBusinessSummary', '') for t in tickers]

    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(descriptions)
    distance = 1 - cosine_similarity(tfidf)

    # 'complete' linkage (minimize max intra-cluster distance) tends to
    # produce more size-balanced clusters than 'average' — verified: on this
    # data 'average' produced one 76-member cluster plus four singletons
    # from an 83-ticker bucket at n_clusters=6, nowhere near size-balanced.
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='complete',
    ).fit_predict(distance)

    sub_groups: dict[int, list[str]] = {}
    for ticker, label in zip(tickers, labels):
        sub_groups.setdefault(int(label), []).append(ticker)
    return list(sub_groups.values())


def _cluster_bucket(tickers: list[str], metadata: dict, max_group_size: int) -> dict[int, list[str]]:
    """
    Splits one industry bucket into sub-clusters no larger than
    max_group_size, via agglomerative clustering on description similarity.

    A single clustering pass isn't enough to guarantee the size cap —
    'complete' linkage clusters are more balanced than 'average' but still
    aren't guaranteed even sizes on real text data, so any resulting
    sub-cluster still over the cap is recursively re-clustered until every
    sub-cluster complies. This is the actual size guarantee; the linkage
    choice above is just about needing fewer recursive passes in practice.
    """
    if len(tickers) <= max_group_size:
        return {1: tickers}

    n_clusters = ceil(len(tickers) / max_group_size)
    clusters = _cluster_once(tickers, metadata, n_clusters)

    result: list[list[str]] = []
    for cluster in clusters:
        if len(cluster) <= max_group_size:
            result.append(cluster)
        else:
            result.extend(_cluster_bucket(cluster, metadata, max_group_size).values())

    return {i + 1: members for i, members in enumerate(result)}


def refine_groups(groups: dict | None = None, max_group_size: int = 15,
                   metadata: dict | None = None) -> dict:
    """
    Splits any industry bucket in find_pair_groups()'s output larger than
    max_group_size into smaller, textually-coherent sub-groups, using
    agglomerative clustering over the same TF-IDF cosine-similarity measure
    pairs_filter.py already computes for its description_score.

    groups   : output of find_pair_groups() (computed if not passed)
    metadata : output of pairs_filter.load_metadata() (loaded if not passed)

    Returns the same {sector: {'etfs': [...], 'industries': {...}}} shape —
    industries above max_group_size are replaced by multiple entries keyed
    '{industry} (1)', '{industry} (2)', etc.; industries at or under the
    threshold are returned unchanged. ETFs are untouched — yfinance ETF
    entries don't carry longBusinessSummary, so there's nothing to cluster
    them by.
    """
    groups = groups if groups is not None else find_pair_groups()
    metadata = metadata if metadata is not None else load_metadata()

    refined: dict = {}
    for sector, bucket in groups.items():
        new_industries: dict[str, list[str]] = {}
        for industry, tickers in bucket['industries'].items():
            if len(tickers) <= max_group_size:
                new_industries[industry] = tickers
                continue
            for i, members in _cluster_bucket(tickers, metadata, max_group_size).items():
                new_industries[f'{industry} ({i})'] = members
        refined[sector] = {'etfs': bucket['etfs'], 'industries': new_industries}

    return refined


def plot_pair_dashboard(
    ticker_a: str,
    ticker_b: str,
    corr_window: int = 75,
    recompute_freq: str = 'W',
    stability_lookback: int = 6,
    corr_threshold: float = 0.65,
    stability_std_cap: float = 0.15,
    coint_pvalue_threshold: float = 0.05,
    min_crossings: int = 6,
    coint_lookback_months: int = 6,
) -> vbt.utils.figure.Figure:
    """
    Diagnostic dashboard for one candidate pair — fetches its own data from
    data/ohlcv.duckdb::massive_1min, no pre-loaded prices required.

    Plots both tickers' cumulative log returns (indexed to 0 at the start of
    their shared history, so the two lines are directly comparable
    regardless of absolute price level) via vbt's plotting accessor, with
    markers where the two lines cross each other — the same
    sign-flip-counting rule pair_tests.compute_cointegration_test() uses for
    its n_crossings, applied here to the normalized price lines themselves
    rather than the hedge-ratio spread, so the crossings are visible
    directly on the chart.

    The title annotation reports the full picture computed fresh for just
    this pair, via the exact same per-pair helpers
    pair_tests.compute_correlation_prefilter()/compute_cointegration_test()
    call internally (_pair_correlation_stats/_pair_cointegration_stats) —
    one canonical implementation shared by the batch pipeline and this
    single-pair dashboard, not a re-derived copy: latest/min/std rolling
    correlation, both-direction cointegration p-values, hedge ratio, and the
    Stage-B spread's own n_crossings (which can differ from the chart's
    normalized-price crossings — they're different series by construction).
    The cointegration stats are windowed to the trailing coint_lookback_months
    (default 6) of price history, same as compute_cointegration_test() —
    the price panel above still plots the full history for visual context.
    """
    daily_close = _load_group_daily_closes([ticker_a, ticker_b])
    daily_close, excluded = _filter_coverage(daily_close)
    if excluded:
        raise ValueError(f'Insufficient coverage for: {excluded} — cannot build a pair dashboard.')

    log_price = np.log(daily_close)
    cum_log = log_price - log_price.iloc[0]
    crossing_dates = _crossing_dates(cum_log[ticker_a] - cum_log[ticker_b])

    fig = cum_log[ticker_a].vbt.plot(trace_kwargs=dict(name=ticker_a))
    cum_log[ticker_b].vbt.plot(trace_kwargs=dict(name=ticker_b), fig=fig)
    fig.add_trace(Scatter(
        x=crossing_dates, y=cum_log[ticker_a].loc[crossing_dates],
        mode='markers', name='crossings',
        marker=dict(symbol='x', size=9, color='black'),
    ))

    returns = np.log(daily_close / daily_close.shift(1)).iloc[1:]
    matrices = _rolling_corr_matrices(returns, corr_window, recompute_freq)
    dates = sorted(matrices.keys())
    corr_stats = _pair_correlation_stats(
        ticker_a, ticker_b, matrices, dates, stability_lookback, corr_threshold, stability_std_cap,
    )
    coint_stats = _pair_cointegration_stats(
        daily_close[ticker_a], daily_close[ticker_b], coint_pvalue_threshold, min_crossings, coint_lookback_months,
    )

    title = (
        f'{ticker_a} vs {ticker_b}  '
        f'&nbsp;|&nbsp;  corr: latest={corr_stats["latest_corr"]:.3f} min={corr_stats["min_corr"]:.3f} '
        f'std={corr_stats["std_corr"]:.3f} ({"PASS" if corr_stats["passed"] else "fail"})'
        f'&nbsp;|&nbsp;  coint: p_ab={coint_stats["pvalue_ab"]:.3f} p_ba={coint_stats["pvalue_ba"]:.3f} '
        f'hedge_ratio={coint_stats["hedge_ratio"]:.3f} spread_crossings={coint_stats["n_crossings"]} '
        f'({"PASS" if coint_stats["passed"] else "fail"})'
        f'&nbsp;|&nbsp;  chart crossings={len(crossing_dates)}'
    )
    fig.update_layout(title=dict(text=title, font=dict(size=12)), yaxis_title='cumulative log return')

    return fig


def plot_kalman_dashboard(
    ticker_a: str,
    ticker_b: str,
    delta: float = 1e-4,
    obs_covariance: float = 1e-3,
    lookback_months: int | None = None,
) -> Figure:
    """
    Three-panel Kalman dynamic-hedge-ratio dashboard for one candidate pair
    — fetches its own data from data/ohlcv.duckdb::massive_1min, no
    pre-loaded prices required (same self-contained pattern as
    plot_pair_dashboard).

    Panels (top to bottom, shared x-axis):
    1. Both tickers' log prices.
    2. The Kalman-filtered hedge ratio beta_t
       (pairs_trading.kalman._kalman_filter_hedge_ratio), run directly on
       the same log prices as panel 1 so hedge ratio and price share a
       scale.
    3. The hedge-ratio-adjusted log price, log(price_a) - beta_t *
       log(price_b) — the mean-reverting series a pairs trade is actually
       built on.

    Panel 3 uses beta_t's updated (post-observation) value at each bar,
    which is fine for visual diagnosis but is NOT a lookahead-safe backtest
    signal — _kalman_filter_hedge_ratio's own 'spread' column (built from
    each bar's pre-observation predicted state) is the lookahead-safe
    series to trade on.

    Opens the figure in the system's default browser via
    fig.show(renderer='browser') (same behavior as calling fig.show() on a
    plain plotly figure outside a notebook) before returning it, so calling
    this function is enough on its own — no separate .show() call needed.
    """
    daily_close = _load_group_daily_closes([ticker_a, ticker_b])
    daily_close, excluded = _filter_coverage(daily_close)
    if excluded:
        raise ValueError(f'Insufficient coverage for: {excluded} — cannot build a Kalman dashboard.')

    if lookback_months is not None:
        cutoff = daily_close.index.max() - pd.DateOffset(months=lookback_months)
        daily_close = daily_close[daily_close.index >= cutoff]

    log_price = np.log(daily_close)
    kalman = _kalman_filter_hedge_ratio(log_price[ticker_a], log_price[ticker_b], delta, obs_covariance)
    adjusted = log_price[ticker_a] - kalman['beta'] * log_price[ticker_b]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=('log price', 'kalman hedge ratio (beta)', 'hedge-ratio-adjusted log price'),
    )
    fig.add_trace(Scatter(x=log_price.index, y=log_price[ticker_a], name=ticker_a), row=1, col=1)
    fig.add_trace(Scatter(x=log_price.index, y=log_price[ticker_b], name=ticker_b), row=1, col=1)
    fig.add_trace(Scatter(x=kalman.index, y=kalman['beta'], name='beta', line=dict(color='black')), row=2, col=1)
    fig.add_trace(Scatter(
        x=adjusted.index, y=adjusted, name=f'{ticker_a} - beta*{ticker_b}', line=dict(color='firebrick'),
    ), row=3, col=1)

    fig.update_layout(
        title=dict(
            text=f'{ticker_a} vs {ticker_b} — Kalman dynamic hedge ratio '
                 f'(delta={delta:g}, obs_covariance={obs_covariance:g})',
            font=dict(size=12),
        ),
        height=800, showlegend=True,
    )
    fig.update_yaxes(title_text='log price', row=1, col=1)
    fig.update_yaxes(title_text='beta', row=2, col=1)
    fig.update_yaxes(title_text='adjusted log price', row=3, col=1)

    fig.show(renderer='browser')
    return fig


def _threshold_crossing_spans(mask: pd.Series) -> list[tuple[int, int]]:
    """
    Contiguous run boundaries (start, end positional indices, inclusive)
    where boolean `mask` is True — NaNs treated as False. Purely a
    plotting helper for plot_pair_diagnostics_dashboard's z-score panel
    (shading |z| > z_entry as candidate trade windows via fig.add_vrect),
    not a statistical computation, so it lives here rather than in
    pair_tests.py alongside the actual test helpers.
    """
    values = mask.fillna(False).to_numpy()
    spans: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(values) - 1))
    return spans


def _axis_key(prefix: str, row: int) -> str:
    """Plotly's subplot axis naming convention: row 1 is bare ('yaxis',
    'xaxis'), rows 2+ are suffixed with the row number ('yaxis2', ...)."""
    return prefix if row == 1 else f'{prefix}{row}'


def _stacked_domains(heights: list[float], spacing: float) -> list[tuple[float, float]]:
    """
    Reimplements make_subplots' own top-to-bottom row -> y-domain
    assignment for an arbitrary subset of relative `heights`, so a set of
    rows can be re-stacked to fill the full [0, 1] vertical span on their
    own (used to give a tab's rows the space a hidden tab's rows would
    otherwise occupy, rather than leaving blank vertical space).

    Returns one (bottom, top) domain fraction per height, in the same
    (top-to-bottom) order as `heights`.
    """
    total = sum(heights)
    available = 1 - spacing * (len(heights) - 1)
    domains = []
    top = 1.0
    for h in heights:
        height_frac = h / total * available
        bottom = max(top - height_frac, 0.0)  # clamp float drift on the last row's bottom edge
        domains.append((bottom, top))
        top = bottom - spacing
    return domains


def _tab_domains(
    tab_rows: list[int], all_row_heights: list[float], spacing: float,
) -> dict[int, tuple[float, float]]:
    """{row: (bottom, top)} for just `tab_rows`, restacked to fill [0, 1]
    -- see _stacked_domains. `all_row_heights` is indexed by (row - 1)."""
    ordered = sorted(tab_rows)
    heights = [all_row_heights[r - 1] for r in ordered]
    return dict(zip(ordered, _stacked_domains(heights, spacing)))


def _apply_tab(
    fig: Figure, tab_name: str, tab_rows: list[int], all_row_heights: list[float],
    n_total_rows: int, trace_tabs: list[str], shape_tabs: list[str], spacing: float,
) -> None:
    """
    Mutates `fig` in place so `tab_name` renders as the initially-active
    tab: hides every trace/shape/subplot-title belonging to another tab,
    and re-stacks `tab_rows`' y-domains to fill the figure (see
    _tab_domains) so the hidden tabs' rows leave no blank space. This is
    the Python-side counterpart to _tab_button_args, which does the same
    thing via a relayout/restyle patch for the in-browser button clicks
    (there's no live Python callback once fig.show() has handed off to
    the browser, so the two can't share one code path -- only the
    domain math via _tab_domains).
    """
    domains = _tab_domains(tab_rows, all_row_heights, spacing)
    last_active = max(tab_rows)
    for i, trace in enumerate(fig.data):
        trace.visible = trace_tabs[i] == tab_name
    for i, shape in enumerate(fig.layout.shapes):
        shape.visible = shape_tabs[i] == tab_name
    for i, ann in enumerate(fig.layout.annotations):
        ann.visible = (i + 1) in tab_rows
    for r in range(1, n_total_rows + 1):
        yaxis = fig.layout[_axis_key('yaxis', r)]
        xaxis = fig.layout[_axis_key('xaxis', r)]
        if r in domains:
            yaxis.domain = domains[r]
            yaxis.visible = True
            xaxis.showticklabels = r == last_active
        else:
            yaxis.domain = [0, 0.001]
            yaxis.visible = False
            xaxis.showticklabels = False


def _tab_button_args(
    tab_name: str, tab_rows: list[int], all_row_heights: list[float], n_total_rows: int,
    trace_tabs: list[str], shape_tabs: list[str], title_prefix: str, spacing: float,
) -> list[dict]:
    """
    [restyle_dict, relayout_dict] for one updatemenus button -- the
    in-browser (plotly.js) equivalent of _apply_tab, expressed as the flat
    dotted/bracket-path attribute dict plotly.js's relayout/restyle expect
    (e.g. 'yaxis3.domain', 'annotations[4].visible', 'shapes[2].visible')
    rather than direct Python object mutation, since a button click has no
    Python-side callback to run once fig.show() has handed off to the
    browser.
    """
    domains = _tab_domains(tab_rows, all_row_heights, spacing)
    last_active = max(tab_rows)
    restyle = {'visible': [t == tab_name for t in trace_tabs]}
    relayout: dict = {'title.text': f'{title_prefix} — {tab_name}'}
    for i, t in enumerate(shape_tabs):
        relayout[f'shapes[{i}].visible'] = t == tab_name
    for r in range(1, n_total_rows + 1):
        ykey, xkey = _axis_key('yaxis', r), _axis_key('xaxis', r)
        relayout[f'annotations[{r - 1}].visible'] = r in tab_rows
        if r in domains:
            relayout[f'{ykey}.domain'] = list(domains[r])
            relayout[f'{ykey}.visible'] = True
            relayout[f'{xkey}.showticklabels'] = r == last_active
        else:
            relayout[f'{ykey}.domain'] = [0, 0.001]
            relayout[f'{ykey}.visible'] = False
            relayout[f'{xkey}.showticklabels'] = False
    return [restyle, relayout]


# Dark-theme palette for plot_pair_diagnostics_dashboard -- plotly_dark's
# default colorway is tuned for scatter/bar traces on a dark background,
# but several of this dashboard's original light-theme picks (black,
# firebrick, steelblue, darkgreen, saddlebrown, gray) read as too dim or
# outright invisible against it, so they're overridden explicitly here
# rather than left at plotly_dark's defaults.
_DASHBOARD_COLORS = dict(
    ticker_a='#4FC3F7', ticker_b='#FFB74D', kalman_beta='#FAFAFA', static_beta='#FFCA28',
    beta_band='rgba(250, 250, 250, 0.18)', scaled_b='#4DB6AC', dynamic_spread='#FF6E6E',
    static_spread='#64B5F6', mean_cross='#FFD54F', zscore='#CE93D8', hline='#B0BEC5',
    vrect='#FF5252', corr='#4DD0E1', coint='#81C784', coint_crit='#FF8A65', hurst='#D7A86E',
    long_entry='#00E676', short_entry='#FF5252',
)

_TAB_ROW_HEIGHTS = [1.2, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.7, 0.7, 0.7]
_TAB_ROWS = {
    'Price & Hedge Ratio': [1, 2, 3, 4],
    'Spread & Signals': [5, 6, 7],
    'Regime Diagnostics': [8, 9, 10],
}
_TAB_SPACING = 0.06


def plot_pair_diagnostics_dashboard(
    ticker_a: str,
    ticker_b: str,
    delta: float = 1e-4,
    obs_covariance: float = 1e-3,
    lookback_months: int | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_window: int = 21,
    corr_window: int = 60,
    coint_window: int = 90,
    coint_stride: int = 5,
    hurst_window: int = 100,
    static_beta_window: int | None = 90,
) -> Figure:
    """
    Ten-panel, tabbed, dark-themed pairs-trading diagnostic dashboard for
    one candidate pair — supersedes plot_kalman_dashboard() for day-to-day
    use (that function is left in place, not removed, until callers
    migrate). Same self-contained fetch pattern: no pre-loaded prices
    required, data comes from data/ohlcv.duckdb::massive_1min via
    _load_group_daily_closes/_filter_coverage, same as
    plot_pair_dashboard/plot_kalman_dashboard.

    All panels/rows are built up front into one Figure (still no Dash, no
    server -- fig.show(renderer='browser') is still the only entry point),
    then grouped into three tabs toggled via a single updatemenus button
    row (see _apply_tab/_tab_button_args): switching tabs hides the other
    tabs' traces/shapes/subplot-titles and re-stacks the active tab's rows
    to fill the full figure height, rather than leaving blank space where
    the hidden rows were.

    Tab "Price & Hedge Ratio" (rows 1-4):
    1. Both tickers' price rebased to 100 at series start (rebased price,
       not log price — this one's a "does this look like a pair" gut
       check, kept in the same units traders read a chart in).
    2. Kalman-filtered beta_t vs. a static OLS hedge ratio: a rolling
       `static_beta_window`-day OLS beta if given, else one flat
       full-sample OLS beta — makes dynamic-vs-static divergence visible
       directly on one axis.
    3. beta_t with a +/-1 std shaded band from the Kalman filter's own
       state variance (kalman._kalman_filter_hedge_ratio's 'beta_var'
       column — the filter's built-in uncertainty, not recomputed here).
    4. Beta-scaled price overlay: ticker_a's log price alongside
       beta_t * ticker_b's log price — the same two quantities the
       "Spread & Signals" tab's Row 1 differences, shown side by side
       rather than subtracted, so scale/drift issues are easier to
       eyeball. Reuses beta_t (Kalman, post-observation) — no refit.

    Tab "Spread & Signals" (rows 5-7):
    5. Hedge-ratio-adjusted log spread, dynamic (beta_t) vs. static beta,
       with mean-crossing ("x") markers at dynamic_adjusted's own zero
       crossings (pair_tests._crossing_dates, same convention
       plot_pair_dashboard uses for its price-line crossings).
    6. Z-score of the dynamic spread (rolling `z_window`-day mean/std),
       with dashed +/-z_entry, +/-z_exit, and zero lines, the |z| >
       z_entry region shaded as candidate trade windows, and long/short
       entry/exit markers at the z-score's own threshold crossings.
    7. NEW: the dynamic spread treated as a single tradeable "price"
       series, with the exact same long/short entry/exit crossing masks
       from Row 6 (pair_tests._threshold_crossings on the z-score,
       computed once and reused here, not recomputed) plotted as markers
       on the spread line itself — reads like a price chart with trade
       markers. Long entries/exits are green up/hollow-down triangles;
       short entries/exits are red down/hollow-up triangles.

    Tab "Regime Diagnostics" (rows 8-10):
    8. Rolling `corr_window`-day correlation of daily log returns.
    9. Rolling Engle-Granger cointegration test statistic over a trailing
       `coint_window` days, recomputed every `coint_stride` days and
       forward-filled (coint() is too expensive to run on every bar — see
       pair_tests._rolling_cointegration_stat), against a horizontal
       5%-critical-value line.
    10. Rolling `hurst_window`-day Hurst exponent of the dynamic spread,
        against the 0.5 random-walk reference line (below = mean-reverting).

    Row 5's dynamic line (and Row 6/7's z-score-derived series, built on
    top of it) uses beta_t's POST-observation value at each bar — fine
    for visual diagnosis, but NOT a lookahead-safe backtest signal, same
    caveat as plot_kalman_dashboard: _kalman_filter_hedge_ratio's own
    'spread' column (built from each bar's PRE-observation predicted
    state) is the lookahead-safe series to actually trade on. Row 7 is a
    visualization of that same non-lookahead-safe series and its
    threshold crossings — not a backtest, no P&L, no VBT.

    Raises ValueError (same as plot_pair_dashboard/plot_kalman_dashboard)
    if coverage filtering excludes either ticker. Opens in the system's
    default browser via fig.show(renderer='browser') before returning fig.
    """
    daily_close = _load_group_daily_closes([ticker_a, ticker_b])
    daily_close, excluded = _filter_coverage(daily_close)
    if excluded:
        raise ValueError(f'Insufficient coverage for: {excluded} — cannot build a diagnostics dashboard.')

    if lookback_months is not None:
        cutoff = daily_close.index.max() - pd.DateOffset(months=lookback_months)
        daily_close = daily_close[daily_close.index >= cutoff]

    rebased = daily_close / daily_close.iloc[0] * 100
    log_price = np.log(daily_close)
    returns = np.log(daily_close / daily_close.shift(1)).iloc[1:]

    kalman = _kalman_filter_hedge_ratio(log_price[ticker_a], log_price[ticker_b], delta, obs_covariance)
    beta_std = np.sqrt(kalman['beta_var'])

    if static_beta_window is None:
        static_beta = pd.Series(
            _full_sample_ols_beta(log_price[ticker_a], log_price[ticker_b]), index=log_price.index,
        )
    else:
        static_beta = _rolling_ols_beta(log_price[ticker_a], log_price[ticker_b], static_beta_window)

    # dynamic_adjusted uses beta_t's post-observation value -- see the
    # lookahead caveat in the docstring above.
    dynamic_adjusted = log_price[ticker_a] - kalman['beta'] * log_price[ticker_b]
    static_adjusted = log_price[ticker_a] - static_beta * log_price[ticker_b]
    scaled_b = kalman['beta'] * log_price[ticker_b]
    mean_cross_dates = _crossing_dates(dynamic_adjusted)

    z_score = (
        (dynamic_adjusted - dynamic_adjusted.rolling(z_window).mean())
        / dynamic_adjusted.rolling(z_window).std()
    )
    # Computed once, reused for both the z-score panel's own markers (Row 6)
    # and the synthetic spread-price panel's markers (Row 7) -- same
    # timestamps, different y-series (z_score vs. dynamic_adjusted).
    long_entries = _threshold_crossings(z_score, -z_entry, 'down')
    long_exits = _threshold_crossings(z_score, -z_exit, 'up')
    short_entries = _threshold_crossings(z_score, z_entry, 'up')
    short_exits = _threshold_crossings(z_score, z_exit, 'down')

    rolling_corr = _rolling_pair_correlation(returns[ticker_a], returns[ticker_b], corr_window)
    coint_stat, coint_crit_5pct = _rolling_cointegration_stat(
        daily_close[ticker_a], daily_close[ticker_b], coint_window, coint_stride,
    )
    hurst = _rolling_hurst_exponent(dynamic_adjusted, hurst_window)

    C = _DASHBOARD_COLORS

    fig = make_subplots(
        rows=10, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=_TAB_ROW_HEIGHTS,
        subplot_titles=(
            'rebased price (=100 at start)',
            'kalman beta vs static OLS beta',
            'kalman beta +/- 1 std',
            'log price vs beta-scaled log price',
            'hedge-ratio-adjusted spread: dynamic vs static',
            'spread z-score',
            'synthetic spread price with entry/exit signals',
            f'rolling {corr_window}d correlation',
            'rolling cointegration test statistic',
            f'rolling {hurst_window}d Hurst exponent',
        ),
    )

    # Row titles as side labels (next to the legend column) rather than
    # centered banners above each row -- yref stays tied to that row's own
    # y-axis 'domain' (not a fixed 'paper' y), so each label automatically
    # re-centers on _apply_tab/_tab_button_args' domain changes when tabs
    # switch, instead of needing a third copy of the domain math.
    for i, ann in enumerate(fig.layout.annotations):
        row = i + 1
        y_suffix = '' if row == 1 else str(row)
        ann.update(
            x=1.02, xref='paper', xanchor='left',
            y=1.0, yref=f'y{y_suffix} domain', yanchor='top',
            textangle=0, align='left', width=170,
            font=dict(size=11),
        )

    trace_tabs: list[str] = []
    shape_tabs: list[str] = []

    def add_trace(trace, row: int, tab: str) -> None:
        trace_tabs.append(tab)
        fig.add_trace(trace, row=row, col=1)

    def add_hline(tab: str, **kwargs) -> None:
        shape_tabs.append(tab)
        fig.add_hline(**kwargs)

    def add_vrect(tab: str, **kwargs) -> None:
        shape_tabs.append(tab)
        fig.add_vrect(**kwargs)

    TAB1, TAB2, TAB3 = 'Price & Hedge Ratio', 'Spread & Signals', 'Regime Diagnostics'

    # --- Tab 1: Price & Hedge Ratio ---
    add_trace(Scatter(x=rebased.index, y=rebased[ticker_a], name=ticker_a, line=dict(color=C['ticker_a'])), 1, TAB1)
    add_trace(Scatter(x=rebased.index, y=rebased[ticker_b], name=ticker_b, line=dict(color=C['ticker_b'])), 1, TAB1)

    add_trace(Scatter(
        x=kalman.index, y=kalman['beta'], name='kalman beta', line=dict(color=C['kalman_beta']),
    ), 2, TAB1)
    add_trace(Scatter(
        x=static_beta.index, y=static_beta, name='static OLS beta', line=dict(color=C['static_beta'], dash='dash'),
    ), 2, TAB1)

    beta_upper = kalman['beta'] + beta_std
    beta_lower = kalman['beta'] - beta_std
    add_trace(Scatter(
        x=beta_upper.index, y=beta_upper, line=dict(width=0), showlegend=False, hoverinfo='skip',
    ), 3, TAB1)
    add_trace(Scatter(
        x=beta_lower.index, y=beta_lower, line=dict(width=0), fill='tonexty',
        fillcolor=C['beta_band'], name='beta +/-1std', hoverinfo='skip',
    ), 3, TAB1)
    add_trace(Scatter(
        x=kalman.index, y=kalman['beta'], name='kalman beta', line=dict(color=C['kalman_beta']), showlegend=False,
    ), 3, TAB1)

    add_trace(Scatter(
        x=log_price.index, y=log_price[ticker_a], name=ticker_a, line=dict(color=C['ticker_a']), showlegend=False,
    ), 4, TAB1)
    add_trace(Scatter(
        x=scaled_b.index, y=scaled_b, name=f'beta * {ticker_b}', line=dict(color=C['scaled_b']),
    ), 4, TAB1)

    # --- Tab 2: Spread & Signals ---
    add_trace(Scatter(
        x=dynamic_adjusted.index, y=dynamic_adjusted, name='dynamic spread', line=dict(color=C['dynamic_spread']),
    ), 5, TAB2)
    add_trace(Scatter(
        x=static_adjusted.index, y=static_adjusted, name='static spread', line=dict(color=C['static_spread'], dash='dash'),
    ), 5, TAB2)
    add_trace(Scatter(
        x=mean_cross_dates, y=dynamic_adjusted.loc[mean_cross_dates], mode='markers', name='mean crossing',
        marker=dict(symbol='x', size=8, color=C['mean_cross']),
    ), 5, TAB2)

    add_trace(Scatter(
        x=z_score.index, y=z_score, name='spread z-score', line=dict(color=C['zscore']),
    ), 6, TAB2)
    for level, dash in ((0.0, 'solid'), (z_entry, 'dash'), (-z_entry, 'dash'), (z_exit, 'dot'), (-z_exit, 'dot')):
        add_hline(TAB2, y=level, line=dict(dash=dash, color=C['hline'], width=1), row=6, col=1)
    for start_i, end_i in _threshold_crossing_spans(z_score.abs() > z_entry):
        add_vrect(
            TAB2, x0=z_score.index[start_i], x1=z_score.index[end_i],
            fillcolor=C['vrect'], opacity=0.12, line_width=0, row=6, col=1,
        )
    _marker_specs = (
        (long_entries, z_score, 'long entry', 'triangle-up', C['long_entry'], False),
        (long_exits, z_score, 'long exit', 'triangle-down-open', C['long_entry'], True),
        (short_entries, z_score, 'short entry', 'triangle-down', C['short_entry'], False),
        (short_exits, z_score, 'short exit', 'triangle-up-open', C['short_entry'], True),
    )
    for mask, series, label, symbol, color, hollow in _marker_specs:
        add_trace(Scatter(
            x=series.index[mask], y=series[mask], mode='markers', name=label,
            marker=dict(symbol=symbol, size=9, color=color, line=dict(width=2, color=color) if hollow else None),
        ), 6, TAB2)

    add_trace(Scatter(
        x=dynamic_adjusted.index, y=dynamic_adjusted, name='synthetic spread price', line=dict(color=C['dynamic_spread']),
        showlegend=False,
    ), 7, TAB2)
    for mask, label, symbol, color, hollow in (
        (long_entries, 'long entry', 'triangle-up', C['long_entry'], False),
        (long_exits, 'long exit', 'triangle-down-open', C['long_entry'], True),
        (short_entries, 'short entry', 'triangle-down', C['short_entry'], False),
        (short_exits, 'short exit', 'triangle-up-open', C['short_entry'], True),
    ):
        add_trace(Scatter(
            x=dynamic_adjusted.index[mask], y=dynamic_adjusted[mask], mode='markers', name=label,
            marker=dict(symbol=symbol, size=11, color=color, line=dict(width=2, color=color) if hollow else None),
            showlegend=False,
        ), 7, TAB2)

    # --- Tab 3: Regime Diagnostics ---
    add_trace(Scatter(x=rolling_corr.index, y=rolling_corr, name='rolling corr', line=dict(color=C['corr'])), 8, TAB3)

    add_trace(Scatter(x=coint_stat.index, y=coint_stat, name='EG test statistic', line=dict(color=C['coint'])), 9, TAB3)
    add_hline(TAB3, y=coint_crit_5pct, line=dict(dash='dash', color=C['coint_crit'], width=1), row=9, col=1)

    add_trace(Scatter(x=hurst.index, y=hurst, name='hurst exponent', line=dict(color=C['hurst'])), 10, TAB3)
    add_hline(TAB3, y=0.5, line=dict(dash='dash', color=C['hline'], width=1), row=10, col=1)

    title_prefix = (
        f'{ticker_a} vs {ticker_b} — pairs-trading diagnostics '
        f'(delta={delta:g}, obs_covariance={obs_covariance:g})'
    )
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=f'{title_prefix} — {TAB1}', font=dict(size=12)),
        height=1000, showlegend=True,
        margin=dict(r=260),
        legend=dict(x=1.22, y=1, xanchor='left', yanchor='top'),
        updatemenus=[dict(
            # bgcolor is light rather than dark-on-dark specifically so the
            # button text can stay a single black font color and still read
            # cleanly whether hovered or not -- plotly.js lightens bgcolor
            # further on hover, and light-hover-on-dark-button is where the
            # previous light-on-dark font color stopped being legible.
            type='buttons', direction='right', x=0.5, xanchor='center', y=1.08, yanchor='top',
            active=0, bgcolor='#CFD8DC', bordercolor='#607D8B', borderwidth=1,
            font=dict(color='black', size=12),
            buttons=[
                dict(label=tab, method='update', args=_tab_button_args(
                    tab, rows, _TAB_ROW_HEIGHTS, 10, trace_tabs, shape_tabs, title_prefix, _TAB_SPACING,
                ))
                for tab, rows in _TAB_ROWS.items()
            ],
        )],
    )
    fig.update_yaxes(title_text='rebased price', row=1, col=1)
    fig.update_yaxes(title_text='beta', row=2, col=1)
    fig.update_yaxes(title_text='beta', row=3, col=1)
    fig.update_yaxes(title_text='log price', row=4, col=1)
    fig.update_yaxes(title_text='adjusted log price', row=5, col=1)
    fig.update_yaxes(title_text='z-score', row=6, col=1)
    fig.update_yaxes(title_text='adjusted log price', row=7, col=1)
    fig.update_yaxes(title_text='correlation', row=8, col=1)
    fig.update_yaxes(title_text='EG stat', row=9, col=1)
    fig.update_yaxes(title_text='H', row=10, col=1)

    _apply_tab(fig, TAB1, _TAB_ROWS[TAB1], _TAB_ROW_HEIGHTS, 10, trace_tabs, shape_tabs, _TAB_SPACING)

    fig.show(renderer='browser')
    return fig
