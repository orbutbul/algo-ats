"""Morris screen of the eight VCP gates.

90 evaluations (10 trajectories x 9 points) on the same 500 names as the
timing run, 8 workers. Each hit stores the pattern date range, the signal
bar, and the next 20 sessions. A default-parameter pass is saved beside the
90 so the report has a baseline. Detection math is unchanged.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vcp.params import VCPParams
from vcp.scan import scan_param_sets_parallel

DUCK = r'C:\Users\Or\Downloads\VSCode\ATS_trading\data\ohlcv.duckdb'
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT / 'artifacts' / 'vcp_500x10y.json'
OUT_DIR = ROOT / 'artifacts' / 'vcp_morris'
HITS_PATH = ROOT / 'artifacts' / 'vcp_morris_hits.jsonl'
PARAMS_PATH = ROOT / 'artifacts' / 'vcp_morris_params.json'
REPORT_PATH = ROOT / 'artifacts' / 'vcp_morris_report.md'
SUMMARY_PATH = ROOT / 'artifacts' / 'vcp_morris_summary.json'

# Ranges already used on the planner. Unit interval maps onto [lo, hi].
SPACE = [
    ('min_confidence', 0.40, 0.80),
    ('shelf_tol', 0.012, 0.040),
    ('noise_depth_frac', 0.20, 0.50),
    ('contraction_ratio', 0.60, 0.95),
    ('min_first_depth', 0.04, 0.12),
    ('max_final_depth', 0.07, 0.12),
    ('vdu_max_ratio', 0.60, 0.85),
    ('breakout_vs_dryup_mult', 1.20, 2.20),
]
N_TRAJ = 10
N_LEVELS = 4
WORKERS = 8
SEED = 17
DEFAULT_ID = -1


def morris_trajectories(k: int, r: int, num_levels: int, seed: int) -> tuple[list[np.ndarray], list[list[int]], float]:
    """Campolongo trajectories. Each step changes one factor by delta."""
    rng = np.random.default_rng(seed)
    grid_jump = num_levels // 2
    delta = grid_jump / (num_levels - 1)
    trajectories: list[np.ndarray] = []
    orders: list[list[int]] = []
    for _ in range(r):
        b_lower = np.tril(np.ones((k + 1, k), dtype=float), -1)
        signs = np.diag(rng.choice(np.array([-1.0, 1.0]), size=k))
        order = rng.permutation(k).tolist()
        permute = np.eye(k)[order]
        x_star = rng.integers(0, num_levels - grid_jump, size=k) / (num_levels - 1)
        ones = np.ones((k + 1, k))
        path = (ones * x_star + (delta / 2.0) * ((2.0 * b_lower - ones) @ signs + ones)) @ permute
        path = np.clip(np.round(path * (num_levels - 1)) / (num_levels - 1), 0.0, 1.0)
        _assert_trajectory(path, delta)
        trajectories.append(path)
        orders.append(order)
    return trajectories, orders, float(delta)


def _assert_trajectory(path: np.ndarray, delta: float) -> None:
    for left, right in zip(path, path[1:]):
        changed = np.where(np.abs(left - right) > 1e-8)[0]
        if changed.size != 1 or abs(abs(right[changed[0]] - left[changed[0]]) - delta) > 1e-6:
            raise RuntimeError(f'Morris step is not a single delta move: {left} -> {right}')
        if np.any(right < -1e-8) or np.any(right > 1 + 1e-8):
            raise RuntimeError('Morris point left the unit box')


def _scale(unit: np.ndarray) -> dict:
    base = VCPParams().to_dict()
    for (name, lo, hi), u in zip(SPACE, unit):
        base[name] = float(lo + float(u) * (hi - lo))
    return base


def _build_design() -> tuple[list[dict], list[dict], float]:
    k = len(SPACE)
    trajectories, orders, delta = morris_trajectories(k, N_TRAJ, N_LEVELS, SEED)
    run_of: dict[tuple, int] = {}
    runs: list[dict] = []
    points: list[dict] = []
    for traj_id, (path, order) in enumerate(zip(trajectories, orders)):
        ids = []
        for step, unit in enumerate(path):
            key = tuple(np.round(unit, 6))
            if key not in run_of:
                run_of[key] = len(runs)
                runs.append({'run_id': run_of[key], 'unit': [float(v) for v in unit], 'params': _scale(unit)})
            ids.append(run_of[key])
        for step, factor in enumerate(order):
            points.append(
                {
                    'trajectory': traj_id,
                    'step': step,
                    'factor': SPACE[factor][0],
                    'factor_index': int(factor),
                    'run_before': ids[step],
                    'run_after': ids[step + 1],
                    'unit_before': runs[ids[step]]['unit'],
                    'unit_after': runs[ids[step + 1]]['unit'],
                }
            )
    return runs, points, delta


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


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _score_hits(path: Path) -> dict[int, dict]:
    buckets: dict[int, dict] = defaultdict(lambda: {'dates': [], 'names': set(), 'ret': [], 'mfe': [], 'mae': []})
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            hit = json.loads(line)
            bucket = buckets[int(hit['combo_id'])]
            bucket['dates'].append((hit['ticker'], hit['breakout_date']))
            bucket['names'].add(hit['ticker'])
            followed = _follow(hit)
            if followed is not None:
                bucket['ret'].append(followed[0])
                bucket['mfe'].append(followed[1])
                bucket['mae'].append(followed[2])
    scored = {}
    for combo_id, bucket in buckets.items():
        counts = Counter(bucket['dates'])
        n_hits = len(bucket['dates'])
        extra = sum(n - 1 for n in counts.values() if n > 1)
        scored[combo_id] = {
            'n_hits': n_hits,
            'n_names': len(bucket['names']),
            'dup_extra': extra,
            'dup_rate': (extra / n_hits) if n_hits else 0.0,
            'n_full_20': len(bucket['ret']),
            'up_rate': (sum(v > 0 for v in bucket['ret']) / len(bucket['ret'])) if bucket['ret'] else None,
            'median_ret20': _median(bucket['ret']),
            'median_mfe': _median(bucket['mfe']),
            'median_mae': _median(bucket['mae']),
        }
    return scored


def _effects(points: list[dict], scored: dict[int, dict], run_to_combo: dict[int, int], delta: float) -> dict:
    metrics = ('n_hits', 'up_rate', 'median_ret20', 'dup_rate', 'median_mae')
    grouped: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in metrics}
    for point in points:
        before = scored.get(run_to_combo[point['run_before']])
        after = scored.get(run_to_combo[point['run_after']])
        if before is None or after is None:
            # A run with zero hits is still a real outcome for counts.
            before = before or _empty_score()
            after = after or _empty_score()
        for metric in metrics:
            y0 = before[metric]
            y1 = after[metric]
            if y0 is None or y1 is None:
                continue
            grouped[metric][point['factor']].append((y1 - y0) / delta)
    summary = {}
    for metric, by_factor in grouped.items():
        rows = []
        for factor, effects in by_factor.items():
            arr = np.asarray(effects, dtype=float)
            rows.append(
                {
                    'factor': factor,
                    'n': int(arr.size),
                    'mu': float(arr.mean()) if arr.size else None,
                    'mu_star': float(np.mean(np.abs(arr))) if arr.size else None,
                    'sigma': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                }
            )
        rows.sort(key=lambda row: -(row['mu_star'] or 0.0))
        summary[metric] = rows
    return summary


def _empty_score() -> dict:
    return {
        'n_hits': 0,
        'n_names': 0,
        'dup_extra': 0,
        'dup_rate': 0.0,
        'n_full_20': 0,
        'up_rate': None,
        'median_ret20': None,
        'median_mfe': None,
        'median_mae': None,
    }


def _pct(value: float | None) -> str:
    if value is None:
        return '—'
    return f'{100 * value:.1f}%'


def _num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return '—'
    return f'{value:.{digits}f}'


def _write_report(summary: dict) -> None:
    effects = summary['effects']
    baseline = summary['baseline']
    pooled = summary['pooled']
    lines = [
        '# VCP Morris screen',
        '',
        f"500 names, window {summary['window_start']} to {summary['window_end']}, "
        f"{summary['n_runs']} parameter sets ({N_TRAJ} trajectories × {len(SPACE) + 1} points) "
        f"on {summary['workers']} workers. Wall time {summary['wall_s'] / 60:.1f} minutes. "
        f"Hits file: `artifacts/vcp_morris_hits.jsonl` ({summary['n_hit_rows']} rows).",
        '',
        'Each row is one detected pattern: ticker, pattern start and end, the breakout bar, '
        'and the next 20 sessions (open, high, low, close, volume), plus footprint, confidence, '
        'depths, and volume dry-up. `combo_id` points at `artifacts/vcp_morris_params.json`.',
        '',
        'Follow-through uses only patterns that still have 20 sessions after the breakout bar. '
        'Return is the 20th close divided by the breakout close, minus one. Up-rate is the share '
        'of those returns that are positive. This is the path after the signal, not a traded backtest.',
        '',
        '## How often, and how the next 20 bars behaved',
        '',
        f"Default parameters: {baseline['n_hits']} hits on {baseline['n_names']} names, "
        f"{baseline['dup_extra']} extra same-day copies, "
        f"up-rate {_pct(baseline['up_rate'])}, median 20-bar return {_pct(baseline['median_ret20'])}, "
        f"median worst low {_pct(baseline['median_mae'])} "
        f"({baseline['n_full_20']} hits had a full 20 sessions).",
        '',
        f"Across the {summary['n_runs']} Morris sets: hit count ran from {summary['hits_min']} to "
        f"{summary['hits_max']} (median {summary['hits_median']:.0f}). "
        f"Pooled over every saved hit with 20 bars: up-rate {_pct(pooled['up_rate'])}, "
        f"median return {_pct(pooled['median_ret20'])}, median best high {_pct(pooled['median_mfe'])}, "
        f"median worst low {_pct(pooled['median_mae'])} "
        f"({pooled['n_full_20']} of {pooled['n_hits']} hits).",
        '',
        '## Which knobs move the result',
        '',
        'Morris μ* is the average absolute change in that outcome for one step in the factor, '
        'scaled as if the step were the full low-to-high range. μ is the signed average: '
        'positive means raising the knob raises the outcome. σ large relative to μ* means the '
        'effect depends on the other knobs.',
        '',
    ]
    titles = {
        'n_hits': 'Hit count (how often)',
        'up_rate': 'Share of breakouts up after 20 sessions (how well)',
        'median_ret20': 'Median 20-session return',
        'dup_rate': 'Share of hits that are extra same-day copies',
        'median_mae': 'Median worst low over the next 20 sessions',
    }
    for metric, title in titles.items():
        lines.append(f'### {title}')
        lines.append('')
        lines.append('| Knob | μ* (full-range scale) | signed μ | σ | steps |')
        lines.append('|---|---:|---:|---:|---:|')
        for row in effects[metric]:
            scale = 1.0 / summary['delta']
            lines.append(
                f"| {row['factor']} | {_num((row['mu_star'] or 0) * scale, 3)} | "
                f"{_num((row['mu'] or 0) * scale, 3)} | {_num((row['sigma'] or 0) * scale, 3)} | {row['n']} |"
            )
        lines.append('')
    freq = effects['n_hits']
    quality = effects['up_rate']
    lines.extend(
        [
            '## What to freeze and what to sweep next',
            '',
            f"Frequency moves most with {freq[0]['factor']}, then {freq[1]['factor']} and {freq[2]['factor']}.",
            f"The 20-bar up-rate moves most with {quality[0]['factor']}, then {quality[1]['factor']} and {quality[2]['factor']}.",
            '',
            'A knob high on hit count and low on up-rate is mostly changing how often a pattern is emitted. '
            'A knob high on up-rate is changing whether the bars after the signal go up. '
            'Freeze a knob at its default when both μ* ranks are near the bottom and σ is not large.',
            '',
        ]
    )
    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')


def _run_scan(runs: list[dict]) -> dict:
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    tickers = prior['tickers']
    window_start = prior['summary']['window_start']
    load_end = prior['summary']['window_end']
    load_start = prior['summary']['load_start']
    param_sets = [{'combo_id': item['run_id'], 'params': item['params']} for item in runs]
    param_sets.append({'combo_id': DEFAULT_ID, 'params': VCPParams().to_dict()})
    print(
        f'{len(tickers)} names, {len(param_sets)} sets, {WORKERS} workers',
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
            text = Path(part['out_path']).read_text(encoding='utf-8')
            dest.write(text)
    result['window_start'] = window_start
    result['window_end'] = load_end
    result['n_names_planned'] = len(tickers)
    print(
        f"scan wall {result['wall_s'] / 60:.1f} min, hits {result['n_hits']}, wrote {HITS_PATH}",
        flush=True,
    )
    return result


def _finish(runs: list[dict], points: list[dict], delta: float, scan: dict) -> None:
    run_to_combo = {item['run_id']: item['run_id'] for item in runs}
    scored = _score_hits(HITS_PATH)
    # Runs that found nothing are absent from the file.
    for item in runs:
        scored.setdefault(item['run_id'], _empty_score())
    scored.setdefault(DEFAULT_ID, _empty_score())
    effects = _effects(points, scored, run_to_combo, delta)
    hit_counts = [scored[item['run_id']]['n_hits'] for item in runs]
    pooled = _score_hits_pooled(HITS_PATH)
    summary = {
        'window_start': scan['window_start'],
        'window_end': scan['window_end'],
        'workers': scan['workers'],
        'wall_s': scan['wall_s'],
        'load_s': scan['load_s'],
        'detect_s': scan['detect_s'],
        'n_runs': len(runs),
        'n_hit_rows': scan['n_hits'],
        'delta': delta,
        'hits_min': int(min(hit_counts)),
        'hits_max': int(max(hit_counts)),
        'hits_median': float(np.median(hit_counts)),
        'baseline': scored[DEFAULT_ID],
        'pooled': pooled,
        'effects': effects,
        'runs': [
            {'run_id': item['run_id'], 'params': {name: item['params'][name] for name, _, _ in SPACE}, **scored[item['run_id']]}
            for item in runs
        ],
    }
    # sets are not JSON serializable inside baseline
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_report(summary)
    print(f'wrote {REPORT_PATH}', flush=True)


def _score_hits_pooled(path: Path) -> dict:
    dates = []
    ret = []
    mfe = []
    mae = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            hit = json.loads(line)
            if int(hit['combo_id']) == DEFAULT_ID:
                continue
            dates.append((hit['ticker'], hit['breakout_date'], hit['combo_id']))
            followed = _follow(hit)
            if followed is not None:
                ret.append(followed[0])
                mfe.append(followed[1])
                mae.append(followed[2])
    return {
        'n_hits': len(dates),
        'n_full_20': len(ret),
        'up_rate': (sum(v > 0 for v in ret) / len(ret)) if ret else None,
        'median_ret20': _median(ret),
        'median_mfe': _median(mfe),
        'median_mae': _median(mae),
    }


def main() -> None:
    runs, points, delta = _build_design()
    assert len(points) == N_TRAJ * len(SPACE)
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(
        json.dumps(
            {
                'space': [{'name': name, 'lo': lo, 'hi': hi} for name, lo, hi in SPACE],
                'delta_unit': delta,
                'n_levels': N_LEVELS,
                'n_trajectories': N_TRAJ,
                'seed': SEED,
                'runs': runs,
                'steps': points,
                'default_combo_id': DEFAULT_ID,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'{len(runs)} unique parameter vectors, {len(points)} Morris steps, delta={delta}', flush=True)
    scan = _run_scan(runs)
    _finish(runs, points, delta, scan)


if __name__ == '__main__':
    main()
