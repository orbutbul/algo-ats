"""Refresh the Fear & Greed files published by whit3rabbit/fear-greed-data.

That repo's weekday job downloads CNN's graph and commits two files: the raw
``json/cnn_output.json`` payload, and ``fear-greed.csv``, which is that tail
spliced onto a frozen archive. This module copies both into ``data/fear_greed/``.
It does not recompute the index.

The GitHub job is scheduled for 6:00 PM ET on weekdays and often commits later
that evening, so a 5:17 PM daily run usually picks up the previous session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

FEAR_GREED_DIR = Path('data/fear_greed')

# Local name, then the raw GitHub URL. fear-greed.csv is stored as
# fear_greed.csv so it matches the file already kept in data/fear_greed/.
GITHUB_FILES: tuple[tuple[str, str], ...] = (
    (
        'cnn_output.json',
        'https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/json/cnn_output.json',
    ),
    (
        'fear_greed.csv',
        'https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv',
    ),
)

_HEADERS = {'User-Agent': 'algo-ats-daily-run', 'Accept': 'application/json, text/csv, */*'}


class FearGreedFetchError(RuntimeError):
    """The download was missing, truncated, or not the file the repo publishes."""


@dataclass(frozen=True)
class FearGreedFile:
    path: Path
    rows: int
    last_date: str | None


def _fetch(url: str) -> bytes:
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.content


def validate_cnn_json(payload: bytes) -> tuple[int, str | None]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FearGreedFetchError(f'cnn_output.json is not JSON: {exc}') from exc
    hist = data.get('fear_and_greed_historical') if isinstance(data, dict) else None
    rows = hist.get('data') if isinstance(hist, dict) else None
    if not isinstance(rows, list) or not rows:
        raise FearGreedFetchError('cnn_output.json has no fear_and_greed_historical.data')
    last = rows[-1].get('x') if isinstance(rows[-1], dict) else None
    last_date = None
    if isinstance(last, (int, float)):
        last_date = datetime.fromtimestamp(last / 1000, timezone.utc).date().isoformat()
    return len(rows), last_date


def validate_fear_greed_csv(payload: bytes) -> tuple[int, str | None]:
    text = payload.decode('utf-8')
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith('Date,Fear Greed'):
        raise FearGreedFetchError('fear_greed.csv is missing the Date,Fear Greed header')
    if len(lines) < 2:
        raise FearGreedFetchError('fear_greed.csv has no data rows')
    return len(lines) - 1, lines[-1].split(',', 1)[0]


def update_fear_greed(dest: Path | None = None, fetch=None) -> list[FearGreedFile]:
    """Download both published files and replace the local copies only if both parse.

    ``fetch`` defaults to an HTTP GET. Tests pass a callable ``(url) -> bytes``.
    """
    dest = FEAR_GREED_DIR if dest is None else Path(dest)
    fetch = _fetch if fetch is None else fetch
    payloads: dict[str, bytes] = {}
    summaries: dict[str, tuple[int, str | None]] = {}
    for name, url in GITHUB_FILES:
        payloads[name] = fetch(url)
        if name.endswith('.json'):
            summaries[name] = validate_cnn_json(payloads[name])
        else:
            summaries[name] = validate_fear_greed_csv(payloads[name])

    dest.mkdir(parents=True, exist_ok=True)
    written: list[FearGreedFile] = []
    for name, _url in GITHUB_FILES:
        target = dest / name
        partial = dest / f'.{name}.partial'
        partial.write_bytes(payloads[name])
        partial.replace(target)
        rows, last_date = summaries[name]
        written.append(FearGreedFile(path=target, rows=rows, last_date=last_date))
    return written
