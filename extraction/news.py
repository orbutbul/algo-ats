"""
extraction/news.py — Benzinga free-tier news collector.

Pulls headlines/teasers from Benzinga's free "Basic Financial News API"
(https://api.benzinga.com/api/v2/news, auth via ?token=, no full article
body or guaranteed ticker tags on the free tier) and appends them to
data/news.duckdb for later point-in-time joins against OHLCV bars in
backtests.

Requires BENZINGA_API_KEY in .env (get one at
https://www.benzinga.com/apis/licensing/register — free tier, no cost).

Usage:
    from extraction.news import download_news
    df = download_news()  # fetches everything since the last successful run
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_DB_PATH = Path('data/news.duckdb')
LAST_RUN_PATH = Path('data/news_last_run.txt')
BENZINGA_NEWS_URL = 'https://api.benzinga.com/api/v2/news'
PAGE_SIZE = 100
MAX_RETRIES = 3

_TABLES = {
    'benzinga_news': {
        'schema': (
            'id BIGINT, created_utc TIMESTAMP, updated_utc TIMESTAMP, '
            'headline VARCHAR, teaser VARCHAR, url VARCHAR, author VARCHAR, '
            'channels VARCHAR, tags VARCHAR'
        ),
        'dedup_cols': ['id'],
    },
    'benzinga_news_tickers': {
        'schema': 'article_id BIGINT, ticker VARCHAR',
        'dedup_cols': ['article_id', 'ticker'],
    },
}

_CASHTAG_RE = re.compile(r'\$([A-Z]{1,5})\b')


def _connect_news_db(read_only: bool = False) -> 'duckdb.DuckDBPyConnection':
    NEWS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(NEWS_DB_PATH), read_only=read_only)
    if not read_only:
        for name, spec in _TABLES.items():
            con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({spec['schema']})")
    return con


def _upsert(con: 'duckdb.DuckDBPyConnection', table: str, df: pd.DataFrame) -> None:
    """Delete any existing rows matching this batch's dedup key, then insert —
    safe to call multiple times for the same articles: later calls replace
    earlier ones rather than duplicating them."""
    if df.empty:
        return
    dedup_cols = _TABLES[table]['dedup_cols']
    key_cols = ', '.join(dedup_cols)
    con.register('_new', df)
    con.execute(f'DELETE FROM {table} WHERE ({key_cols}) IN (SELECT {key_cols} FROM _new)')
    con.execute(f'INSERT INTO {table} SELECT * FROM _new')
    con.unregister('_new')


def _read_last_run() -> datetime | None:
    if not LAST_RUN_PATH.exists():
        return None
    return datetime.fromisoformat(LAST_RUN_PATH.read_text().strip())


def _write_last_run(ts: datetime) -> None:
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(ts.isoformat())


def _extract_cashtags(*texts: str | None) -> list[str]:
    """Regex fallback for when Benzinga's own `stocks` field is empty/thin —
    the free tier doesn't guarantee ticker tagging."""
    tickers = set()
    for text in texts:
        if text:
            tickers.update(_CASHTAG_RE.findall(text))
    return sorted(tickers)


def _fetch_page(token: str, updated_since: int, page: int) -> list[dict]:
    params = {
        'token': token,
        'updatedSince': updated_since,
        'pageSize': PAGE_SIZE,
        'page': page,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                BENZINGA_NEWS_URL, params=params,
                headers={'Accept': 'application/json'}, timeout=15,
            )
            if resp.status_code == 200:
                try:
                    return resp.json() or []
                except ValueError:
                    print(
                        f'  Benzinga returned 200 but an unparseable body '
                        f'(Content-Type: {resp.headers.get("Content-Type")}): '
                        f'{resp.text[:500]!r}'
                    )
                    raise
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                print(f'  Benzinga returned {resp.status_code}, retrying in {wait}s...')
                time.sleep(wait)
                continue
            print(f'  Benzinga returned {resp.status_code}: {resp.text[:500]!r}')
            resp.raise_for_status()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f'  Benzinga request failed ({e}), retrying in {wait}s...')
            time.sleep(wait)
    return []


def download_news() -> pd.DataFrame:
    """
    Fetches all Benzinga articles updated since the last successful run
    (defaulting to 24h ago on first run) and upserts them into
    data/news.duckdb (benzinga_news + benzinga_news_tickers).

    Returns a DataFrame of the articles fetched this run (empty if none).
    """
    token = os.environ['BENZINGA_API_KEY']

    now = datetime.now(timezone.utc)
    last_run = _read_last_run() or (now - timedelta(hours=24))
    updated_since = int(last_run.timestamp())

    articles = []
    page = 0
    while True:
        batch = _fetch_page(token, updated_since, page)
        if not batch:
            break
        articles.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1

    if not articles:
        print('  No new Benzinga articles.')
        _write_last_run(now)
        return pd.DataFrame()

    news_rows = []
    ticker_rows = []
    for a in articles:
        article_id = a.get('id')
        headline = a.get('title') or a.get('headline')
        teaser = a.get('teaser')
        news_rows.append({
            'id': article_id,
            'created_utc': pd.to_datetime(a.get('created'), utc=True, errors='coerce'),
            'updated_utc': pd.to_datetime(a.get('updated'), utc=True, errors='coerce'),
            'headline': headline,
            'teaser': teaser,
            'url': a.get('url'),
            'author': a.get('author'),
            'channels': ','.join(c.get('name', '') for c in a.get('channels', []) or []),
            'tags': ','.join(t.get('name', '') for t in a.get('tags', []) or []),
        })

        api_tickers = {s.get('name', '').upper() for s in a.get('stocks', []) or [] if s.get('name')}
        all_tickers = api_tickers | set(_extract_cashtags(headline, teaser))
        for ticker in sorted(all_tickers):
            ticker_rows.append({'article_id': article_id, 'ticker': ticker})

    news_df = pd.DataFrame(news_rows)
    tickers_df = pd.DataFrame(ticker_rows, columns=['article_id', 'ticker'])

    con = _connect_news_db()
    try:
        _upsert(con, 'benzinga_news', news_df)
        _upsert(con, 'benzinga_news_tickers', tickers_df)
    finally:
        con.close()

    _write_last_run(now)
    print(f'  Saved {len(news_df)} articles -> {NEWS_DB_PATH}')
    return news_df
