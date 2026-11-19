"""4x4x4 grid on the three knobs that move the 20-bar path.

contraction_ratio, shelf_tol, and noise_depth_frac each take the four Morris
levels. Every other gate stays at its default. The universe is every name with
at least 2,400 bars in the same 10-year window as the 500-name run. A default
parameter pass is saved as combo_id -1 and is not one of the 64 cells.

Each hit stores the pattern dates, the signal bar, the next 20 sessions, and
the detector confidence. The report ranks cells by the path after the signal.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

from vcp.params import VCPParams
from vcp.scan import scan_param_sets_parallel

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'artifacts' / 'vcp_grid'
HITS_PATH = ROOT / 'artifacts' / 'vcp_grid_hits.jsonl'
PARAMS_PATH = ROOT / 'artifacts' / 'vcp_grid_params.json'
REPORT_PATH = ROOT / 'artifacts' / 'vcp_grid_report.md'
SUMMARY_PATH = ROOT / 'artifacts' / 'vcp_grid_summary.json'
WORKERS = 8
DEFAULT_ID = -1
AXES = (
    ('contraction_ratio', 0.60, 0.95),
    ('shelf_tol', 0.012, 0.040),
    ('noise_depth_frac', 0.20, 0.50),
)
CONF_EDGES = (0.70, 0.75, 0.80, 0.85, 0.90)


def _levels(lo: float, hi: float, n: int = 4) -> list[float]:
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def _window(con: duckdb.DuckDBPyConnection) -> tuple[str, str, str]:
    mx = pd.Timestamp(con.execute('SELECT max(datetime) FROM ohlcv_max_daily').fetchone()[0])
    if mx.tzinfo is None:
        mx = mx.tz_localize('UTC')
    window_start = (mx - pd.DateOffset(years=10)).tz_convert('UTC')
    load_start = window_start - pd.Timedelta(days=400)
    return (
        load_start.date().isoformat(),
        mx.tz_convert('UTC').date().isoformat(),
        window_start.date().isoformat(),
    )


def _eligible(con: duckdb.DuckDBPyConnection, window_start: str, load_end: str) -> list[str]:
    rows = con.execute(
        '''
        SELECT ticker
        FROM ohlcv_max_daily
        WHERE datetime >= ? AND datetime <= ?
        GROUP BY ticker
        HAVING count(*) >= 2400
        ORDER BY ticker
        ''',
        [window_start, load_end],
    ).fetchall()
    return [row[0] for row in rows]


def _design() -> list[dict]:
    grid = {name: _levels(lo, hi) for name, lo, hi in AXES}
    runs = []
    combo_id = 0
    for contraction in grid['contraction_ratio']:
        for shelf in grid['shelf_tol']:
            for noise in grid['noise_depth_frac']:
                params = VCPParams().to_dict()
                params['contraction_ratio'] = float(contraction)
                params['shelf_tol'] = float(shelf)
                params['noise_depth_frac'] = float(noise)
                runs.append(
                    {
                        'combo_id': combo_id,
                        'contraction_ratio': float(contraction),
                        'shelf_tol': float(shelf),
                        'noise_depth_frac': float(noise),
                        'params': params,
                    }
                )
                combo_id += 1
    return runs


def _follow(hit: dict) -> tuple[float, float, float] | None:
    bars = hit.get('next_20') or []
    if len(bars) < 20:
        return None
    base = (hit.get('breakout_bar') or {}).get('close')
    if not base:
        return None
    last = bars[-1]['close']
    if base <= 0 or last <= 0:
        return None
    highs = [bar['high'] for bar in bars]
    lows = [bar['low'] for bar in bars]
    return (last / base - 1.0, max(highs) / base - 1.0, min(lows) / base - 1.0)


def _pct(ordered: list[float], p: float) -> float:
    n = len(ordered)
    i = min(n - 1, max(0, int(round(p * (n - 1)))))
    return float(ordered[i])


def _median(ordered: list[float]) -> float:
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _conf_label(confidence: float) -> str:
    prev = 0.0
    for edge in CONF_EDGES:
        if confidence < edge:
            return f'{prev:.2f}–{edge:.2f}'
        prev = edge
    return f'{CONF_EDGES[-1]:.2f}+'


def _pack(rets: list[float], mfes: list[float], maes: list[float], n_hits: int, n_names: int) -> dict:
    if not rets:
        return {
            'n_hits': n_hits,
            'n_names': n_names,
            'n_full_20': 0,
            'up_rate': None,
            'median_ret20': None,
            'mean_ret20': None,
            'p10_ret20': None,
            'p90_ret20': None,
            'median_mfe': None,
            'median_mae': None,
        }
    ordered = sorted(rets)
    return {
        'n_hits': n_hits,
        'n_names': n_names,
        'n_full_20': len(ordered),
        'up_rate': sum(v > 0 for v in ordered) / len(ordered),
        'median_ret20': _median(ordered),
        'mean_ret20': sum(ordered) / len(ordered),
        'p10_ret20': _pct(ordered, 0.10),
        'p90_ret20': _pct(ordered, 0.90),
        'median_mfe': _median(sorted(mfes)),
        'median_mae': _median(sorted(maes)),
    }


def _score(path: Path) -> dict[int, dict]:
    buckets: dict[int, dict] = defaultdict(
        lambda: {'ret': [], 'mfe': [], 'mae': [], 'names': set(), 'n': 0, 'conf': defaultdict(lambda: {'ret': [], 'mfe': [], 'mae': []})}
    )
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            hit = json.loads(line)
            combo = int(hit['combo_id'])
            bucket = buckets[combo]
            bucket['n'] += 1
            bucket['names'].add(hit['ticker'])
            followed = _follow(hit)
            if followed is None:
                continue
            bucket['ret'].append(followed[0])
            bucket['mfe'].append(followed[1])
            bucket['mae'].append(followed[2])
            band = bucket['conf'][_conf_label(float(hit['confidence']))]
            band['ret'].append(followed[0])
            band['mfe'].append(followed[1])
            band['mae'].append(followed[2])
    scored = {}
    for combo, bucket in buckets.items():
        row = _pack(bucket['ret'], bucket['mfe'], bucket['mae'], bucket['n'], len(bucket['names']))
        row['confidence'] = {
            label: _pack(band['ret'], band['mfe'], band['mae'], len(band['ret']), 0)
            for label, band in sorted(bucket['conf'].items())
        }
        scored[combo] = row
    return scored


def _pct_text(value: float | None) -> str:
    if value is None:
        return '—'
    return f'{value:.1%}'


def _write_report(summary: dict) -> None:
    cells = [row for row in summary['cells']]
    base = summary['baseline']
    lines = [
        '# VCP 20-bar grid',
        '',
        (
            f"{summary['n_names']} names, window {summary['window_start']} to {summary['window_end']}, "
            f"64 cells (4 contraction ratios × 4 shelf tolerances × 4 noise fractions) on {summary['workers']} workers. "
            f"Wall time {summary['wall_s'] / 60:.1f} minutes. Hits file: `artifacts/vcp_grid_hits.jsonl` "
            f"({summary['n_hit_rows']} rows)."
        ),
        '',
        'Every other gate is at its default, including min confidence 0.45 and VDU cap 0.70. '
        'The default parameter set is saved as combo_id -1 and is not one of the 64 cells.',
        '',
        'Return is the close 20 sessions after the breakout divided by the breakout close, minus one. '
        'Up-rate is the share of those returns that are positive. Mean and the 10th percentile describe '
        'the average path and the weak tail. This is the path after the signal, with no costs and no stop.',
        '',
        '## Default parameters on this universe',
        '',
        (
            f"{base['n_hits']} hits on {base['n_names']} names, {base['n_full_20']} with 20 sessions. "
            f"Up-rate {_pct_text(base['up_rate'])}, median {_pct_text(base['median_ret20'])}, "
            f"mean {_pct_text(base['mean_ret20'])}, 10th percentile {_pct_text(base['p10_ret20'])}, "
            f"median best high {_pct_text(base['median_mfe'])}, median worst low {_pct_text(base['median_mae'])}."
        ),
        '',
        '## Where the 20-bar path is stronger',
        '',
        summary['zone_text'],
        '',
        '## Each knob, averaged across the other two',
        '',
    ]
    for axis in summary['margins']:
        lines.append(f"### {axis['name']}")
        lines.append('')
        lines.append('| Level | Cells | Paths | Up-rate | Median | Mean | 10th pct |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|')
        for level in axis['levels']:
            lines.append(
                f"| {level['value']:.3f} | {level['n_cells']} | {level['n_full_20']} | "
                f"{_pct_text(level['up_rate'])} | {_pct_text(level['median_ret20'])} | "
                f"{_pct_text(level['mean_ret20'])} | {_pct_text(level['p10_ret20'])} |"
            )
        lines.append('')
    lines.extend(
        [
            '## Cells in the stronger corner',
            '',
            'A cell is in the corner when its mean and its median both sit in the top quarter of the 64. '
            'Paths are the average of those cells, so a busy cell does not dominate.',
            '',
            '| Contraction | Shelf | Noise | Paths | Up-rate | Median | Mean | 10th pct |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for row in summary['corner']:
        lines.append(
            f"| {row['contraction_ratio']:.3f} | {row['shelf_tol']:.3f} | {row['noise_depth_frac']:.3f} | "
            f"{row['n_full_20']} | {_pct_text(row['up_rate'])} | {_pct_text(row['median_ret20'])} | "
            f"{_pct_text(row['mean_ret20'])} | {_pct_text(row['p10_ret20'])} |"
        )
    lines.extend(
        [
            '',
            '## Detector confidence inside the default set',
            '',
            'Same default parameters, split by the confidence the detector wrote on each hit.',
            '',
            '| Confidence | Paths | Up-rate | Median | Mean | 10th pct |',
            '|---|---:|---:|---:|---:|---:|',
        ]
    )
    for label, band in base['confidence'].items():
        lines.append(
            f"| {label} | {band['n_full_20']} | {_pct_text(band['up_rate'])} | "
            f"{_pct_text(band['median_ret20'])} | {_pct_text(band['mean_ret20'])} | {_pct_text(band['p10_ret20'])} |"
        )
    lines.extend(['', '## All 64 cells', '', '| Contraction | Shelf | Noise | Hits | Paths | Up-rate | Median | Mean | 10th pct | Worst low |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for row in sorted(cells, key=lambda item: -(item['mean_ret20'] or -1)):
        lines.append(
            f"| {row['contraction_ratio']:.3f} | {row['shelf_tol']:.3f} | {row['noise_depth_frac']:.3f} | "
            f"{row['n_hits']} | {row['n_full_20']} | {_pct_text(row['up_rate'])} | "
            f"{_pct_text(row['median_ret20'])} | {_pct_text(row['mean_ret20'])} | "
            f"{_pct_text(row['p10_ret20'])} | {_pct_text(row['median_mae'])} |"
        )
    lines.append('')
    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')


def _margin(cells: list[dict], name: str) -> dict:
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in cells:
        groups[round(row[name], 5)].append(row)
    levels = []
    for value in sorted(groups):
        members = [row for row in groups[value] if row['n_full_20']]
        n = len(members)
        levels.append(
            {
                'value': value,
                'n_cells': n,
                'n_full_20': sum(row['n_full_20'] for row in members),
                'up_rate': sum(row['up_rate'] for row in members) / n,
                'median_ret20': sum(row['median_ret20'] for row in members) / n,
                'mean_ret20': sum(row['mean_ret20'] for row in members) / n,
                'p10_ret20': sum(row['p10_ret20'] for row in members) / n,
            }
        )
    return {'name': name, 'levels': levels}


def _finish(runs: list[dict], scan: dict) -> None:
    scored = _score(HITS_PATH)
    empty = _pack([], [], [], 0, 0)
    empty['confidence'] = {}
    cells = []
    for run in runs:
        row = dict(scored.get(run['combo_id'], empty))
        row['combo_id'] = run['combo_id']
        row['contraction_ratio'] = run['contraction_ratio']
        row['shelf_tol'] = run['shelf_tol']
        row['noise_depth_frac'] = run['noise_depth_frac']
        cells.append(row)
    baseline = scored.get(DEFAULT_ID, empty)
    means = sorted(row['mean_ret20'] for row in cells if row['mean_ret20'] is not None)
    medians = sorted(row['median_ret20'] for row in cells if row['median_ret20'] is not None)
    mean_cut = means[int(0.75 * (len(means) - 1))]
    median_cut = medians[int(0.75 * (len(medians) - 1))]
    corner = [
        row
        for row in cells
        if row['mean_ret20'] is not None
        and row['mean_ret20'] >= mean_cut
        and row['median_ret20'] >= median_cut
    ]
    corner.sort(key=lambda row: -row['mean_ret20'])
    if corner:
        zone_text = (
            f"{len(corner)} cells sit in the top quarter on both mean and median 20-bar return "
            f"(mean at least {mean_cut:.1%}, median at least {median_cut:.1%}). "
            f"Across that corner the average mean is {sum(row['mean_ret20'] for row in corner) / len(corner):.1%}, "
            f"the average median is {sum(row['median_ret20'] for row in corner) / len(corner):.1%}, "
            f"and the average up-rate is {sum(row['up_rate'] for row in corner) / len(corner):.1%}."
        )
    else:
        zone_text = 'No cell landed in the top quarter on both mean and median.'
    summary = {
        'window_start': scan['window_start'],
        'window_end': scan['window_end'],
        'n_names': scan['n_names_planned'],
        'workers': scan['workers'],
        'wall_s': scan['wall_s'],
        'load_s': scan['load_s'],
        'detect_s': scan['detect_s'],
        'n_hit_rows': scan['n_hits'],
        'mean_cut': mean_cut,
        'median_cut': median_cut,
        'baseline': baseline,
        'margins': [_margin(cells, name) for name, _, _ in AXES],
        'corner': corner,
        'zone_text': zone_text,
        'cells': cells,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_report(summary)
    print(zone_text, flush=True)
    print(f'wrote {REPORT_PATH}', flush=True)


def main() -> None:
    runs = _design()
    assert len(runs) == 64
    con = duckdb.connect(DUCK, read_only=True)
    load_start, load_end, window_start = _window(con)
    tickers = _eligible(con, window_start, load_end)
    con.close()
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(
        json.dumps(
            {
                'axes': [{'name': name, 'lo': lo, 'hi': hi, 'levels': _levels(lo, hi)} for name, lo, hi in AXES],
                'frozen': 'VCPParams defaults except the three axes',
                'window_start': window_start,
                'window_end': load_end,
                'load_start': load_start,
                'n_names': len(tickers),
                'tickers': tickers,
                'default_combo_id': DEFAULT_ID,
                'runs': runs,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    param_sets = [{'combo_id': run['combo_id'], 'params': run['params']} for run in runs]
    param_sets.append({'combo_id': DEFAULT_ID, 'params': VCPParams().to_dict()})
    print(
        f'{len(tickers)} names, {window_start} .. {load_end}, {len(param_sets)} sets, {WORKERS} workers',
        flush=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = scan_param_sets_parallel(
        duck=DUCK,
        tickers=tickers,
        load_start=load_start,
        load_end=load_end,
        window_start=window_start,
        param_sets=param_sets,
        out_dir=str(OUT_DIR),
        workers=WORKERS,
    )
    with HITS_PATH.open('w', encoding='utf-8') as dest:
        for part in result['parts']:
            dest.write(Path(part['out_path']).read_text(encoding='utf-8'))
    result['window_start'] = window_start
    result['window_end'] = load_end
    result['n_names_planned'] = len(tickers)
    print(
        f"scan wall {result['wall_s'] / 60:.1f} min, hits {result['n_hits']}, wrote {HITS_PATH}",
        flush=True,
    )
    _finish(runs, result)


if __name__ == '__main__':
    main()
