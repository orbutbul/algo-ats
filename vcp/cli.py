"""CLI deliverable: seed local DuckDB and chart VCP breakouts for a ticker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcp.detector import detect_vcp_breakouts
from vcp.io import OHLCV_DUCKDB_PATH, list_tickers, load_daily_ohlcv, seed_daily_ohlcv
from vcp.params import VCPParams
from vcp.plot import plot_vcp_history
from vcp.vbt_compat import VBT_FLAVOR
from vcp.viewer import run_viewer


def _parse_params(ns: argparse.Namespace) -> VCPParams:
    base = VCPParams()
    overrides = {}
    if ns.lookback is not None:
        overrides['lookback_bars'] = ns.lookback
    if ns.no_trend_template:
        overrides['require_trend_template'] = False
    if ns.min_contractions is not None:
        overrides['min_contractions'] = ns.min_contractions
    return VCPParams(**{**base.to_dict(), **overrides})


def cmd_seed(args: argparse.Namespace) -> int:
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    path = seed_daily_ohlcv(tickers, path=Path(args.duckdb), period=args.period)
    print(f'Seeded {tickers} -> {path}')
    print('Tickers now:', ', '.join(list_tickers(path)))
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    ticker = args.ticker.strip().upper()
    duckdb_path = Path(args.duckdb)
    params = _parse_params(args)
    ohlcv = load_daily_ohlcv(ticker, path=duckdb_path, start=args.start, end=args.end)
    result = detect_vcp_breakouts(ohlcv, ticker=ticker, params=params)

    out = Path(args.out) if args.out else Path('artifacts') / f'{ticker.lower()}_vcp.png'
    plot_vcp_history(ohlcv, result, outfile=out)
    summary = {
        'ticker': ticker,
        'vbt': VBT_FLAVOR,
        'bars': len(ohlcv),
        'breakouts': len(result.breakouts),
        'params': params.to_dict(),
        'events': [b.to_dict() for b in result.breakouts],
        'chart': str(out),
    }
    print(json.dumps(summary, indent=2, default=str))
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f'Wrote {json_path}')
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    run_viewer(host=args.host, port=args.port, duckdb_path=Path(args.duckdb))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    duckdb_path = Path(args.duckdb)
    params = _parse_params(args)
    tickers = (
        [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
        if args.tickers
        else list_tickers(duckdb_path)
    )
    rows = []
    for ticker in tickers:
        try:
            ohlcv = load_daily_ohlcv(ticker, path=duckdb_path, start=args.start, end=args.end)
        except LookupError as exc:
            print(f'skip {ticker}: {exc}')
            continue
        result = detect_vcp_breakouts(ohlcv, ticker=ticker, params=params)
        for b in result.breakouts:
            rows.append(b.to_dict())
        print(f'{ticker}: {len(result.breakouts)} breakout(s)')
    print(json.dumps(rows, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='vcp', description='Volatility Contraction Pattern detector')
    p.add_argument('--duckdb', default=str(OHLCV_DUCKDB_PATH), help='Path to ohlcv.duckdb')
    sub = p.add_subparsers(dest='cmd', required=True)

    seed = sub.add_parser('seed', help='Download daily OHLCV into ohlcv_max_daily')
    seed.add_argument('--tickers', required=True, help='Comma-separated tickers')
    seed.add_argument('--period', default='max')
    seed.set_defaults(func=cmd_seed)

    chart = sub.add_parser('chart', help='Detect VCP breakouts and write a price/volume chart')
    chart.add_argument('ticker')
    chart.add_argument('--start', default=None)
    chart.add_argument('--end', default=None)
    chart.add_argument('--out', default=None, help='PNG path')
    chart.add_argument('--json-out', default=None)
    chart.add_argument('--lookback', type=int, default=None)
    chart.add_argument('--min-contractions', type=int, default=None)
    chart.add_argument('--no-trend-template', action='store_true')
    chart.set_defaults(func=cmd_chart)

    scan = sub.add_parser('scan', help='Scan one or more tickers and print breakout JSON')
    scan.add_argument('--tickers', default=None, help='Comma-separated; default=all in DB')
    scan.add_argument('--start', default=None)
    scan.add_argument('--end', default=None)
    scan.add_argument('--lookback', type=int, default=None)
    scan.add_argument('--min-contractions', type=int, default=None)
    scan.add_argument('--no-trend-template', action='store_true')
    scan.set_defaults(func=cmd_scan)

    view = sub.add_parser('view', help='Open a local web viewer for ticker VCP history')
    view.add_argument('--host', default='127.0.0.1')
    view.add_argument('--port', type=int, default=8765)
    view.set_defaults(func=cmd_view)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
