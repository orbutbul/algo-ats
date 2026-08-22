"""
pairs_trading/pipeline.py — the composed pairs-trading pipeline: runs
Stage A (pair_tests.compute_correlation_prefilter) then Stage B
(pair_tests.compute_cointegration_test) over every equity industry
sub-cluster in refined_groups.json (or a caller-supplied groups dict in the
same shape — see utils.find_pair_groups()/refine_groups()), returning one
combined, ranked table across all groups rather than requiring each group
to be wired through both stages by hand.

Sector ETFs (the 'etfs' list alongside each sector's 'industries') are not
included — this pipeline is equities-only.
"""

import json
from pathlib import Path

import pandas as pd

from pairs_trading.pair_tests import compute_correlation_prefilter, compute_cointegration_test

REFINED_GROUPS_PATH = Path(__file__).parent / 'refined_groups.json'
CANDIDATE_PAIRS_PATH = Path(__file__).parent / 'candidate_pairs.csv'
CANDIDATE_PAIRS_RAW_PATH = Path(__file__).parent / '.candidate_pairs_raw.csv'
PROCESSED_GROUPS_PATH = Path(__file__).parent / '.processed_groups.csv'


_RAW_COLUMNS = [
    'sector', 'group', 'ticker_a', 'ticker_b', 'latest_corr', 'min_corr', 'std_corr',
    'passed_stage_a', 'pvalue_ab', 'pvalue_ba', 'cointegrated', 'hedge_ratio',
    'n_crossings', 'passed_stage_b',
]


def find_candidate_pairs(
    groups: dict | None = None,
    corr_window: int = 75,
    recompute_freq: str = 'W',
    stability_lookback: int = 6,
    corr_threshold: float = 0.65,
    stability_std_cap: float = 0.15,
    coint_pvalue_threshold: float = 0.05,
    min_crossings: int = 6,
    coint_lookback_months: int = 6,
    cache_dir: str | Path | None = None,
    output_path: str | Path | None = CANDIDATE_PAIRS_PATH,
    max_groups: int | None = None,
) -> pd.DataFrame:
    """
    Runs Stage A -> Stage B across every equity industry sub-cluster in
    `groups` (loaded from refined_groups.json if not passed — the
    {sector: {'etfs': [...], 'industries': {group: [tickers]}}} shape
    utils.find_pair_groups()/refine_groups() produce).

    Groups too small to pair (<4 tickers, or fewer after Stage A's own
    coverage filtering) are skipped — same as compute_correlation_prefilter
    on its own.

    Resumable: each group's rows are appended to an internal raw store
    (pairs_trading/.candidate_pairs_raw.csv) as soon as that group finishes,
    rather than held in memory until the very end — 274 groups each
    needing their own DB round-trips can run long enough to hit an
    external time limit (verified), and incremental writes mean a
    re-invocation just picks up where the last one left off rather than
    losing all progress. Pass max_groups to process only that many *new*
    groups per call — useful for chunking a full run across several
    shorter invocations. The raw store is append-only and never carries the
    `rank` column (which depends on every group processed so far, so it's
    recomputed fresh, not persisted incrementally) — keeping ranking out of
    the appended file avoids the append-mode writer and the final
    full-rewrite ever fighting over column layout.

    A group that's too small to pair (<4 tickers, or empty after Stage A's
    own coverage filtering) produces zero rows, so it can't be recognized
    as "already done" from the raw store alone — a separate manifest
    (pairs_trading/.processed_groups.csv) records every attempted
    sector+group regardless of whether it produced any pair rows, so those
    groups are correctly skipped on the next call instead of being
    re-attempted (harmlessly slow, but wasting max_groups budget) forever.

    Returns the full accumulated table (every pair Stage A considered, not
    just Stage-B-passed ones — matches compute_correlation_prefilter/
    compute_cointegration_test's own "return everything, not just passing
    rows" convention), re-ranked across all groups processed so far, and
    writes it to output_path (default pairs_trading/candidate_pairs.csv) —
    this file is a full, ranked snapshot regenerated each call, not itself
    used for resumability tracking. Columns: sector, group, ticker_a,
    ticker_b, latest_corr, min_corr, std_corr, passed_stage_a, pvalue_ab,
    pvalue_ba, cointegrated, hedge_ratio, n_crossings, passed_stage_b, rank.

    rank: sorted with Stage-B-passed pairs first, by n_crossings within
    that (more crossings = more realized mean-reversion/trade
    opportunities over the lookback window — the most decision-relevant
    single number Stage B produces) — a position (1 = best), not a score.
    """
    if groups is None:
        with open(REFINED_GROUPS_PATH) as f:
            groups = json.load(f)

    raw_path = CANDIDATE_PAIRS_RAW_PATH
    done_groups = set()
    if PROCESSED_GROUPS_PATH.exists():
        existing = pd.read_csv(PROCESSED_GROUPS_PATH)
        done_groups = set(map(tuple, existing[['sector', 'group']].drop_duplicates().values))

    processed_this_call = 0
    for sector, bucket in groups.items():
        for group_name, tickers in bucket['industries'].items():
            if (sector, group_name) in done_groups:
                continue
            if max_groups is not None and processed_this_call >= max_groups:
                break

            print(f'--- {sector} / {group_name} ({len(tickers)} tickers) ---', flush=True)

            stage_a = compute_correlation_prefilter(
                group_tickers=tickers, corr_window=corr_window, recompute_freq=recompute_freq,
                stability_lookback=stability_lookback, corr_threshold=corr_threshold,
                stability_std_cap=stability_std_cap, cache_dir=cache_dir,
            )
            processed_this_call += 1

            if not stage_a.empty:
                stage_b = compute_cointegration_test(
                    stage_a, coint_pvalue_threshold=coint_pvalue_threshold,
                    min_crossings=min_crossings, lookback_months=coint_lookback_months,
                )
                merged = stage_a.merge(
                    stage_b, on=['ticker_a', 'ticker_b'], how='left', suffixes=('_stage_a', '_stage_b'),
                )
                merged.insert(0, 'group', group_name)
                merged.insert(0, 'sector', sector)
                merged = merged[_RAW_COLUMNS]

                write_header = not raw_path.exists()
                merged.to_csv(raw_path, mode='a', header=write_header, index=False)

            manifest_row = pd.DataFrame([{'sector': sector, 'group': group_name}])
            manifest_row.to_csv(PROCESSED_GROUPS_PATH, mode='a', header=not PROCESSED_GROUPS_PATH.exists(), index=False)

    if not raw_path.exists():
        return pd.DataFrame(columns=_RAW_COLUMNS + ['rank'])

    result = pd.read_csv(raw_path)
    result['passed_stage_b'] = result['passed_stage_b'].map(lambda x: bool(x) if pd.notna(x) else False)
    result['n_crossings'] = result['n_crossings'].fillna(-1)
    result = result.sort_values(
        by=['passed_stage_b', 'n_crossings'], ascending=[False, False],
    ).reset_index(drop=True)
    result.insert(0, 'rank', result.index + 1)

    if output_path is not None:
        result.to_csv(output_path, index=False)

    print(f'{len(result)} pairs across {result["group"].nunique()} groups '
          f'({processed_this_call} new groups processed this call)'
          + (f' -> {output_path}' if output_path is not None else ''))
    return result


if __name__ == '__main__':
    find_candidate_pairs()
