"""
relative_strength/cli.py — opt-in command for the RS proxy.

Not referenced by daily_run, hourly_run, strategy_runner, or Airflow.
The command only reads prices and writes a new CSV/JSON directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relative_strength.calculator import (
    MAX_MISSING_SESSIONS,
    MIN_UNIVERSE_FOR_PERCENTILE,
    PriceSchemaError,
    compute_relative_strength,
)
from relative_strength.io import (
    ADJUSTMENT_DUCKDB,
    ADJUSTMENT_NOTE_DUCKDB,
    PRICE_SOURCE_DUCKDB,
    describe_daily_coverage,
    load_asset_class_map,
    load_ohlcv_max_daily,
    load_price_csv,
    load_repo_asset_classes,
    write_result,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='relative_strength',
        description=(
            'IBD/Minervini-inspired relative-strength proxy (not official IBD RS). '
            'Reads prices and writes a new CSV and JSON file. Does not trade or modify source data.'
        ),
    )
    sub = parser.add_subparsers(dest='command', required=True)

    coverage = sub.add_parser('coverage', help='Report whether ohlcv_max_daily is readable. Read-only.')
    coverage.add_argument('--duckdb-path', type=Path, default=None)

    run = sub.add_parser('run', help='Compute the proxy for one as-of date and write artifacts.')
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='CSV with date,ticker,close and optional asset_class.')
    source.add_argument(
        '--from-duckdb',
        action='store_true',
        help='Read data/ohlcv.duckdb table ohlcv_max_daily read-only. No download.',
    )
    run.add_argument('--duckdb-path', type=Path, default=None)
    run.add_argument('--as-of', required=True, help='YYYY-MM-DD. Later sessions are ignored.')
    run.add_argument('--universe', choices=('stocks', 'etfs', 'unseparated'), default='stocks')
    run.add_argument(
        '--adjustment',
        choices=('split', 'split_and_dividend', 'unknown'),
        default=None,
        help='Label for a CSV input. ohlcv_max_daily is always split_and_dividend.',
    )
    run.add_argument('--benchmark', default='SPY')
    run.add_argument('--benchmark-csv', type=Path, default=None, help='Optional date,ticker,close file for the benchmark.')
    run.add_argument('--asset-class-csv', type=Path, default=None, help='Optional ticker,asset_class map.')
    run.add_argument('--out-dir', type=Path, default=Path('output/relative_strength'))
    run.add_argument('--min-universe', type=int, default=MIN_UNIVERSE_FOR_PERCENTILE)
    run.add_argument('--max-missing-sessions', type=int, default=MAX_MISSING_SESSIONS)
    run.add_argument('--no-rs-line', action='store_true')
    run.add_argument('--tickers', nargs='*', default=None, help='Optional ticker filter for --from-duckdb.')

    args = parser.parse_args(argv)
    try:
        if args.command == 'coverage':
            return _coverage(args.duckdb_path)
        return _run(args)
    except (FileNotFoundError, LookupError, PriceSchemaError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


def _coverage(path: Path | None) -> int:
    report = describe_daily_coverage(path)
    json.dump(report.to_dict(), sys.stdout, indent=2)
    sys.stdout.write('\n')
    if not report.available:
        print(report.gap, file=sys.stderr)
        return 2
    return 0


def _run(args: argparse.Namespace) -> int:
    benchmark_prices = load_price_csv(args.benchmark_csv) if args.benchmark_csv else None
    if args.from_duckdb:
        if args.adjustment not in (None, ADJUSTMENT_DUCKDB):
            print(
                'error: ohlcv_max_daily is split_and_dividend (yfinance auto_adjust=True) and cannot be relabeled.',
                file=sys.stderr,
            )
            return 2
        prices = load_ohlcv_max_daily(args.duckdb_path, tickers=args.tickers)
        asset_class = load_repo_asset_classes()
        if args.asset_class_csv:
            asset_class.update(load_asset_class_map(args.asset_class_csv))
        price_source = PRICE_SOURCE_DUCKDB
        adjustment = ADJUSTMENT_DUCKDB
        adjustment_note = ADJUSTMENT_NOTE_DUCKDB
    else:
        prices = load_price_csv(args.input)
        asset_class = load_asset_class_map(args.asset_class_csv) if args.asset_class_csv else None
        price_source = f'csv:{Path(args.input).name}'
        adjustment = args.adjustment or 'unknown'
        adjustment_note = None

    result = compute_relative_strength(
        prices,
        as_of=args.as_of,
        universe=args.universe,
        asset_class=asset_class,
        benchmark=args.benchmark,
        benchmark_prices=benchmark_prices,
        price_source=price_source,
        adjustment=adjustment,
        adjustment_note=adjustment_note,
        min_universe=args.min_universe,
        max_missing_sessions=args.max_missing_sessions,
        include_rs_line=not args.no_rs_line,
    )
    paths = write_result(result, args.out_dir)
    for warning in result.summary['warnings']:
        print(f'warning: {warning}', file=sys.stderr)
    for kind, path in paths.items():
        print(f'{kind}: {path}')
    print(
        f"as_of_session={result.summary['as_of_session']} "
        f"eligible={result.summary['eligible_count']} "
        f"broad_market_rating={result.summary['broad_market_rating']}"
    )
    return 0
