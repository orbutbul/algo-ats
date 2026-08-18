"""
pairs_trading/utils.py — group tickers into pairs-trading candidate pools
using data/metadata.json (yfinance's raw per-ticker dump, equities/ETFs only —
no crypto in that file, and crypto is out of scope here regardless).
"""

import json
from math import ceil
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pairs_trading.pairs_filter import load_metadata

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
