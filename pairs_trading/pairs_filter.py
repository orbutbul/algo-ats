"""
pairs_trading/pairs_filter.py — per-symbol pairs-trading candidate scoring.

Complements find_pair_groups() in pairs_trading/utils.py, which buckets the
whole universe into sector/industry groups (a broad "what's out there"
overview). This module instead scores one symbol against the rest of the
universe in data/metadata.json, combining:
  - structural_score : exact-match on industry/sector (equities) or category
                        (ETFs) fields
  - description_score: TF-IDF cosine similarity between tickers'
                        longBusinessSummary free-text descriptions — catches
                        business overlap that sector/industry labels alone
                        miss (e.g. companies in different formal industries
                        that actually compete)
into a combined_score, so you can rank candidates for one specific ticker
rather than just seeing which bucket it falls into.
"""

import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

METADATA_PATH = Path(__file__).parent.parent / 'data' / 'metadata.json'


def load_metadata(path: Path = METADATA_PATH) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _structural_score(a: dict, b: dict) -> tuple[float, dict]:
    """
    Rule-based score from shared industry/sector/category fields.
    Returns (score 0.0–1.0, breakdown dict).
    """
    a_class = a.get('asset_class')
    b_class = b.get('asset_class')
    breakdown = {}

    # Cross-class pairs: no structural fields overlap
    if a_class != b_class:
        breakdown['cross_class'] = True
        return 0.0, breakdown

    breakdown['cross_class'] = False

    if a_class == 'etfs':
        cat_a = a.get('category', '')
        cat_b = b.get('category', '')
        match = cat_a and cat_b and cat_a == cat_b
        breakdown['category_match'] = match
        breakdown['category'] = cat_a if match else f'{cat_a} vs {cat_b}'
        return 1.0 if match else 0.0, breakdown

    # Equities: industry (weight 0.7) + sector (weight 0.3)
    industry_a = a.get('industry', '')
    industry_b = b.get('industry', '')
    sector_a = a.get('sector', '')
    sector_b = b.get('sector', '')

    industry_match = bool(industry_a and industry_b and industry_a == industry_b)
    sector_match = bool(sector_a and sector_b and sector_a == sector_b)

    breakdown['industry_match'] = industry_match
    breakdown['sector_match'] = sector_match
    breakdown['industry'] = industry_a if industry_match else f'{industry_a} vs {industry_b}'
    breakdown['sector'] = sector_a if sector_match else f'{sector_a} vs {sector_b}'

    score = (0.7 if industry_match else 0.0) + (0.3 if sector_match else 0.0)
    return score, breakdown


def compare_pair(sym1: str, sym2: str, metadata: dict | None = None) -> dict:
    """
    Compare two symbols for pairs trading suitability.

    Returns a dict with:
      - structural_score  (0–1): industry/sector/category match
      - description_score (0–1): TF-IDF cosine similarity of business summaries
      - combined_score    (0–1): weighted average
      - breakdown         : field-level details
    """
    if metadata is None:
        metadata = load_metadata()

    sym1, sym2 = sym1.upper(), sym2.upper()

    if sym1 not in metadata:
        raise KeyError(f'{sym1} not found in metadata')
    if sym2 not in metadata:
        raise KeyError(f'{sym2} not found in metadata')

    a, b = metadata[sym1], metadata[sym2]

    struct_score, breakdown = _structural_score(a, b)

    desc_a = a.get('longBusinessSummary', '')
    desc_b = b.get('longBusinessSummary', '')

    if desc_a and desc_b:
        vec = TfidfVectorizer(stop_words='english')
        tfidf = vec.fit_transform([desc_a, desc_b])
        desc_score = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    else:
        desc_score = 0.0

    # Cross-class pairs lean more on description since structural fields don't overlap
    if breakdown.get('cross_class'):
        combined = desc_score
    else:
        combined = 0.6 * struct_score + 0.4 * desc_score

    return {
        'sym1': sym1,
        'sym2': sym2,
        'asset_class_1': a.get('asset_class'),
        'asset_class_2': b.get('asset_class'),
        'structural_score': round(struct_score, 4),
        'description_score': round(desc_score, 4),
        'combined_score': round(combined, 4),
        'breakdown': breakdown,
    }


def find_pairs(
    symbol: str,
    metadata: dict | None = None,
    min_combined: float = 0.3,
    same_class_only: bool = False,
) -> pd.DataFrame:
    """
    Compare one symbol against the entire universe in metadata.json.

    symbol         : the ticker to find partners for
    metadata       : pre-loaded dict (loaded from disk if None)
    min_combined   : minimum combined_score to include in results
    same_class_only: if True, skip cross-class comparisons

    Returns a DataFrame sorted by combined_score descending.
    """
    if metadata is None:
        metadata = load_metadata()

    symbol = symbol.upper()
    if symbol not in metadata:
        raise KeyError(f'{symbol} not found in metadata')

    symbol_class = metadata[symbol].get('asset_class')

    # Pre-fit TF-IDF on all descriptions for efficiency
    symbols = [s for s in metadata if s != symbol]
    if same_class_only:
        symbols = [s for s in symbols if metadata[s].get('asset_class') == symbol_class]

    all_symbols = [symbol] + symbols
    descriptions = [metadata[s].get('longBusinessSummary', '') for s in all_symbols]

    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(descriptions)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

    rows = []
    for i, sym2 in enumerate(symbols):
        a = metadata[symbol]
        b = metadata[sym2]
        struct_score, breakdown = _structural_score(a, b)
        desc_score = float(sims[i])

        if breakdown.get('cross_class'):
            combined = desc_score
        else:
            combined = 0.6 * struct_score + 0.4 * desc_score

        if combined < min_combined:
            continue

        rows.append({
            'symbol': sym2,
            'asset_class': b.get('asset_class'),
            'structural_score': round(struct_score, 4),
            'description_score': round(desc_score, 4),
            'combined_score': round(combined, 4),
            'industry': b.get('industry') or b.get('category', ''),
            'sector': b.get('sector', ''),
            'name': b.get('longName') or b.get('shortName', ''),
        })

    df = pd.DataFrame(rows).sort_values('combined_score', ascending=False).reset_index(drop=True)
    return df


if __name__ == '__main__':
    meta = load_metadata()

    # Single pair comparison
    result = compare_pair('NVDA', 'AMD', meta)
    print(f"\nNVDA vs AMD")
    print(f"  structural : {result['structural_score']}")
    print(f"  description: {result['description_score']}")
    print(f"  combined   : {result['combined_score']}")
    print(f"  breakdown  : {result['breakdown']}")

    # Find all partners for NVDA
    print("\nTop pairs for NVDA:")
    df = find_pairs('NVDA', meta, min_combined=0.3)
    print(df.head(10).to_string(index=False))
