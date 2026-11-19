"""Slice the saved Morris hits by confidence and by the knobs that move the next 20 bars."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'artifacts'


def follow(hit: dict) -> float | None:
    bars = hit.get('next_20') or []
    if len(bars) < 20:
        return None
    base = (hit.get('breakout_bar') or {}).get('close')
    if not base or base <= 0:
        return None
    last = bars[-1]['close']
    if last <= 0:
        return None
    return last / base - 1.0


def pct(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    i = min(n - 1, max(0, int(round(p * (n - 1)))))
    return sorted_vals[i]


def pack(vals: list[float]) -> dict:
    ordered = sorted(vals)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        'n': n,
        'up': sum(v > 0 for v in ordered) / n,
        'median': median,
        'mean': sum(ordered) / n,
        'p10': pct(ordered, 0.10),
        'p90': pct(ordered, 0.90),
    }


def show(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f'\n{title}')
    print(f'{"bin":<28} {"n":>6} {"up":>7} {"median":>8} {"mean":>8} {"p10":>8} {"p90":>8}')
    for label, row in rows:
        print(
            f'{label:<28} {row["n"]:6d} {row["up"]:7.1%} {row["median"]:8.1%} '
            f'{row["mean"]:8.1%} {row["p10"]:8.1%} {row["p90"]:8.1%}'
        )


def main() -> None:
    params = json.loads((ROOT / 'vcp_morris_params.json').read_text(encoding='utf-8'))
    by_run = {int(run['run_id']): run['params'] for run in params['runs']}
    conf_bins: dict[str, list[float]] = defaultdict(list)
    knob_bins: dict[str, list[float]] = defaultdict(list)
    per_combo: dict[int, list[float]] = defaultdict(list)
    edges = (0.70, 0.75, 0.80, 0.85, 0.90)

    with (ROOT / 'vcp_morris_hits.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            hit = json.loads(line)
            ret = follow(hit)
            if ret is None:
                continue
            combo = int(hit['combo_id'])
            per_combo[combo].append(ret)
            if combo != -1:
                continue
            confidence = float(hit['confidence'])
            label = f'{edges[-1]:.2f}+'
            for edge in edges:
                if confidence < edge:
                    prev = 0.0 if edge == edges[0] else edges[edges.index(edge) - 1]
                    label = f'{prev:.2f}–{edge:.2f}'
                    break
            conf_bins[label].append(ret)

    show('Default set, by the pattern’s own confidence', [(k, pack(v)) for k, v in sorted(conf_bins.items())])

    # One row per Morris parameter set. Skip sets too small to trust a rate.
    combo_rows = []
    for combo, vals in per_combo.items():
        if combo == -1 or len(vals) < 80:
            continue
        row = pack(vals)
        p = by_run[combo]
        row['contraction_ratio'] = p['contraction_ratio']
        row['shelf_tol'] = p['shelf_tol']
        row['noise_depth_frac'] = p['noise_depth_frac']
        row['min_confidence'] = p['min_confidence']
        row['vdu_max_ratio'] = p['vdu_max_ratio']
        combo_rows.append(row)

    def level_table(key: str) -> None:
        groups: dict[float, list[dict]] = defaultdict(list)
        for row in combo_rows:
            groups[round(row[key], 5)].append(row)
        out = []
        for level in sorted(groups):
            members = groups[level]
            # Unweighted across parameter sets so a busy set does not dominate the zone.
            out.append(
                (
                    f'{level:.3f}  ({len(members)} sets)',
                    {
                        'n': sum(m['n'] for m in members),
                        'up': sum(m['up'] for m in members) / len(members),
                        'median': sum(m['median'] for m in members) / len(members),
                        'mean': sum(m['mean'] for m in members) / len(members),
                        'p10': sum(m['p10'] for m in members) / len(members),
                        'p90': sum(m['p90'] for m in members) / len(members),
                    },
                )
            )
        show(key, out)

    print(f'\nMorris sets with at least 80 complete 20-bar paths: {len(combo_rows)}')
    medians = sorted(row['median'] for row in combo_rows)
    means = sorted(row['mean'] for row in combo_rows)
    ups = sorted(row['up'] for row in combo_rows)
    print(
        f'median-of-set medians {medians[len(medians)//2]:.1%}, '
        f'set medians span {medians[0]:.1%} to {medians[-1]:.1%}'
    )
    print(
        f'median-of-set means {means[len(means)//2]:.1%}, '
        f'set means span {means[0]:.1%} to {means[-1]:.1%}'
    )
    print(f'set up-rates span {ups[0]:.1%} to {ups[-1]:.1%}')
    for key in ('contraction_ratio', 'shelf_tol', 'noise_depth_frac', 'min_confidence', 'vdu_max_ratio'):
        level_table(key)

    # Best and worst decile of sets by mean 20-bar return.
    ranked = sorted(combo_rows, key=lambda row: row['mean'])
    k = max(1, len(ranked) // 5)
    def zone(name: str, members: list[dict]) -> None:
        print(f'\n{name} ({len(members)} sets)')
        for key in ('contraction_ratio', 'shelf_tol', 'noise_depth_frac', 'min_confidence', 'vdu_max_ratio'):
            vals = sorted(m[key] for m in members)
            print(f'  {key:<20} {vals[0]:.3f} .. {vals[-1]:.3f}  mid {vals[len(vals)//2]:.3f}')
        print(
            f'  mean ret {sum(m["mean"] for m in members)/len(members):.1%}, '
            f'median ret {sum(m["median"] for m in members)/len(members):.1%}, '
            f'up {sum(m["up"] for m in members)/len(members):.1%}'
        )

    zone('lowest-mean sets', ranked[:k])
    zone('highest-mean sets', ranked[-k:])


if __name__ == '__main__':
    main()
