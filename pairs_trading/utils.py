"""
pairs_trading/utils.py — group tickers into pairs-trading candidate pools
using data/metadata.json (yfinance's raw per-ticker dump, equities/ETFs only —
no crypto in that file, and crypto is out of scope here regardless).
"""

import json
from pathlib import Path

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
