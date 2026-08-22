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
from plotly.graph_objects import Scatter
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pairs_trading.pairs_filter import load_metadata
from pairs_trading.pair_tests import (
    _crossing_dates, _filter_coverage, _load_group_daily_closes,
    _pair_correlation_stats, _pair_cointegration_stats, _rolling_corr_matrices,
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
