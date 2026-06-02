"""
WSB scraper — pulls data from u/wsbapp's custom Reddit posts.

wsbapp uses Reddit's Devvit App Platform, so post content is served as a
gRPC-Web protobuf from devvit-gateway.reddit.com.  We use Playwright to
render the page, intercept that response, decode it, and extract:

  - trending_tickers_daily / _hourly   list[str]
  - karma_trending_daily               list[{username, karma}]
  - commenter_trending_daily           list[{member, score}]
  - session_store                      list[{ticker, name, price, change, ...}]
  - daily_comment_metrics              list[float]  (hourly comment counts × 24)
  - hourly_comment_metrics             list[float]

Usage:
    from data_wsb import fetch_wsbapp_post, get_latest_wsbapp_data
    data = get_latest_wsbapp_data()          # returns dict
    df   = get_latest_wsbapp_data(as_df=True)  # returns DataFrame of session_store
"""

from __future__ import annotations

import struct
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# Protobuf / gRPC helpers
# ---------------------------------------------------------------------------

def _b(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    s = str(v)
    return s.strip("b'").strip("'")


def _f64(int_val) -> float | None:
    try:
        return struct.unpack("d", struct.pack("Q", int(int_val)))[0]
    except Exception:
        return None


def _parse_struct_list(lst: list | dict) -> dict:
    """Convert a protobuf struct field-list into a plain {name: raw_val} dict."""
    result = {}
    for item in (lst if isinstance(lst, list) else [lst]):
        if not isinstance(item, dict):
            continue
        key = _b(item.get("1", ""))
        result[key] = item.get("2", {})
    return result


def _unwrap(val) -> object:
    """Recursively unwrap a single protobuf value into a Python scalar / list / dict."""
    if not isinstance(val, dict):
        return val
    if "3" in val:
        return _b(val["3"])
    if "2" in val:
        return _f64(val["2"])
    if "1" in val:
        return val["1"]
    if "4" in val:
        return bool(val["4"])
    if "5" in val:
        inner = val["5"].get("1", [])
        if isinstance(inner, list):
            return {k: _unwrap(v) for k, v in _parse_struct_list(inner).items()}
        return inner
    if "6" in val:
        inner = val["6"].get("1", [])
        if isinstance(inner, list):
            return [_unwrap(x) for x in inner]
    return val


# ---------------------------------------------------------------------------
# gRPC-Web frame decoder
# ---------------------------------------------------------------------------

def _strip_grpc_frame(body: bytes) -> bytes:
    """Remove the 5-byte gRPC-Web frame header."""
    msg_len = struct.unpack(">I", body[1:5])[0]
    return body[5 : 5 + msg_len]


# ---------------------------------------------------------------------------
# Playwright: render post and capture devvit-gateway response
# ---------------------------------------------------------------------------

def _capture_devvit_proto(post_url: str) -> bytes:
    """Launch a headless browser, load the post, return the raw protobuf bytes."""
    proto: dict[str, bytes] = {}

    def on_response(response):
        if "RenderPostContent" in response.url:
            proto["body"] = response.body()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0"
            )
        )
        page.on("response", on_response)
        page.goto(post_url, wait_until="networkidle", timeout=45_000)
        browser.close()

    if "body" not in proto:
        raise RuntimeError(f"No devvit-gateway response captured for {post_url}")
    return _strip_grpc_frame(proto["body"])


# ---------------------------------------------------------------------------
# Protobuf decode → structured data
# ---------------------------------------------------------------------------

def _decode_proto(proto_bytes: bytes) -> dict:
    import blackboxprotobuf  # lazy import – only needed here

    msg, _ = blackboxprotobuf.decode_message(proto_bytes)
    return msg


def _extract_wsb_data(msg: dict) -> dict:
    """Walk the decoded protobuf and return a clean dict of WSB data."""
    states = msg.get("1", {}).get("1", [])

    # Find the useState-0 entry which holds appState + sessionStore
    for s in states:
        name = _b(s.get("1", ""))
        if "MainApp.useState-0" not in name:
            continue

        fields = _parse_struct_list(s["2"]["5"]["1"])
        value_raw = fields.get("value", {})
        app = _parse_struct_list(value_raw.get("5", {}).get("1", []))
        inner_app = _parse_struct_list(
            app.get("appState", {}).get("5", {}).get("1", [])
        )

        result = {}

        # --- Trending tickers ---
        for key in ("trendingTickersDaily", "trendingTickersHourly"):
            raw = inner_app.get(key, {})
            tickers = []
            for item in raw.get("6", {}).get("1", []):
                t = _b(item.get("3", b""))
                if t:
                    tickers.append(t)
            result[key] = tickers

        # --- Karma trending ---
        for key in ("karmaTrendingDaily", "karmaTrendingHourly"):
            raw = inner_app.get(key, {})
            users = []
            for item in raw.get("6", {}).get("1", []):
                fields_ = _parse_struct_list(item.get("5", {}).get("1", []))
                users.append({
                    "username": _b(fields_.get("username", {}).get("3", b"")),
                    "karma":    _f64(fields_.get("karma", {}).get("2", 0)),
                })
            result[key] = users

        # --- Commenter trending ---
        for key in ("commenterTrendingDaily", "commenterTrendingHourly"):
            raw = inner_app.get(key, {})
            users = []
            for item in raw.get("6", {}).get("1", []):
                fields_ = _parse_struct_list(item.get("5", {}).get("1", []))
                users.append({
                    "member": _b(fields_.get("member", {}).get("3", b"")),
                    "score":  _f64(fields_.get("score", {}).get("2", 0)),
                })
            result[key] = users

        # --- Comment / submission metrics (time-series arrays) ---
        for key in ("dailyCommentMetrics", "hourlyCommentMetrics",
                    "dailySubmissionMetrics", "hourlySubmissionMetrics"):
            raw = inner_app.get(key, {})
            vals = [_f64(x.get("2", 0)) for x in raw.get("6", {}).get("1", [])]
            result[key] = vals

        # --- Session store (stock prices) ---
        ss_raw = app.get("sessionStore", {})
        stocks = []
        for entry in ss_raw.get("6", {}).get("1", []):
            stock = _parse_struct_list(entry.get("5", {}).get("1", []))
            sess = _parse_struct_list(
                stock.get("session", {}).get("5", {}).get("1", [])
            )
            stocks.append({
                "ticker":        _b(stock.get("ticker", {}).get("3", b"")),
                "name":          _b(stock.get("name",   {}).get("3", b"")),
                "type":          _b(stock.get("type",   {}).get("3", b"")),
                "market_status": _b(stock.get("market_status", {}).get("3", b"")),
                "price":         _f64(stock.get("price", {}).get("2", 0)),
                "change":        _f64(sess.get("change", {}).get("2", 0)),
                "change_pct":    _f64(sess.get("change_percent", {}).get("2", 0)),
                "volume":        _f64(sess.get("volume", {}).get("2", 0)),
                "prev_close":    _f64(sess.get("previous_close", {}).get("2", 0)),
                "early_chg":     _f64(sess.get("early_trading_change", {}).get("2", 0)),
                "early_chg_pct": _f64(sess.get("early_trading_change_percent", {}).get("2", 0)),
                "late_chg":      _f64(sess.get("late_trading_change", {}).get("2", 0)),
                "late_chg_pct":  _f64(sess.get("late_trading_change_percent", {}).get("2", 0)),
            })
        result["sessionStore"] = stocks

        # --- Polymarket data ---
        poly_raw = app.get("polymarketData", {})
        markets = []
        for item in poly_raw.get("6", {}).get("1", []):
            fields_ = _parse_struct_list(item.get("5", {}).get("1", []))
            markets.append({
                "title": _b(fields_.get("title", {}).get("3", b"")),
                "url":   _b(fields_.get("url",   {}).get("3", b"")),
            })
        result["polymarketData"] = markets

        return result

    raise RuntimeError("Could not locate MainApp.useState-0 in protobuf state")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_wsbapp_posts(limit: int = 25) -> list[dict]:
    """Return recent posts by u/wsbapp as a list of dicts (id, title, url, created_utc)."""
    url = "https://www.reddit.com/user/wsbapp/submitted.json"
    r = requests.get(url, headers={"User-Agent": "wsb_scraper/1.0"}, params={"limit": limit})
    r.raise_for_status()
    posts = []
    for p in r.json()["data"]["children"]:
        d = p["data"]
        posts.append({
            "id":          d["id"],
            "title":       d["title"],
            "url":         f"https://www.reddit.com{d['permalink']}",
            "created_utc": datetime.fromtimestamp(d["created_utc"], tz=timezone.utc),
        })
    return posts


def fetch_wsbapp_post(post_url: str) -> dict:
    """
    Render a wsbapp post URL and return a dict with all extracted data.

    Keys: trendingTickersDaily, trendingTickersHourly, karmaTrendingDaily,
          karmaTrendingHourly, commenterTrendingDaily, commenterTrendingHourly,
          dailyCommentMetrics, hourlyCommentMetrics, dailySubmissionMetrics,
          hourlySubmissionMetrics, sessionStore, polymarketData
    """
    proto_bytes = _capture_devvit_proto(post_url)
    msg = _decode_proto(proto_bytes)
    return _extract_wsb_data(msg)


def get_latest_wsbapp_data(
    post_type: str = "moves",
    as_df: bool = False,
) -> dict | pd.DataFrame:
    """
    Fetch data from the most recent wsbapp post matching post_type.

    post_type:
        'moves'    → 'What Are Your Moves Tomorrow' (has full stock data)
        'daily'    → 'Daily Discussion Thread'
        'weekend'  → 'Weekend Discussion Thread'

    If as_df=True, returns a DataFrame of session_store stocks instead.
    """
    keywords = {
        "moves":   "What Are Your Moves",
        "daily":   "Daily Discussion",
        "weekend": "Weekend Discussion",
    }
    kw = keywords.get(post_type, post_type)

    posts = list_wsbapp_posts(limit=25)
    matching = [p for p in posts if kw.lower() in p["title"].lower()]
    if not matching:
        raise ValueError(f"No recent wsbapp post matching '{kw}'")

    data = fetch_wsbapp_post(matching[0]["url"])
    data["post_title"]  = matching[0]["title"]
    data["post_date"]   = matching[0]["created_utc"]

    if as_df:
        return pd.DataFrame(data["sessionStore"])

    return data


# ---------------------------------------------------------------------------
# Daily persistence
# ---------------------------------------------------------------------------

WSB_DIR = Path("data/wsb")

_FILES = {
    "trending_tickers":  WSB_DIR / "trending_tickers.parquet",
    "karma_trending":    WSB_DIR / "karma_trending.parquet",
    "commenter_trending": WSB_DIR / "commenter_trending.parquet",
    "session_store":     WSB_DIR / "session_store.parquet",
}


def save_wsb_data(data: dict) -> None:
    """
    Append today's WSB snapshot to the four parquet files in data/wsb/.

    Safe to call multiple times on the same day — existing rows for that
    date are replaced, not duplicated.
    """
    WSB_DIR.mkdir(parents=True, exist_ok=True)

    post_date = data["post_date"]
    if isinstance(post_date, datetime):
        date = post_date.date()
    else:
        date = pd.Timestamp(post_date).date()

    # --- trending_tickers ---
    rows = []
    for tf, key in (("daily", "trendingTickersDaily"), ("hourly", "trendingTickersHourly")):
        for rank, ticker in enumerate(data.get(key, []), 1):
            rows.append({"date": date, "timeframe": tf, "rank": rank, "ticker": ticker})
    _append(pd.DataFrame(rows), _FILES["trending_tickers"], dedup_cols=["date", "timeframe", "ticker"])

    # --- karma_trending ---
    rows = []
    for tf, key in (("daily", "karmaTrendingDaily"), ("hourly", "karmaTrendingHourly")):
        for rank, u in enumerate(data.get(key, []), 1):
            rows.append({"date": date, "timeframe": tf, "rank": rank,
                         "username": u["username"], "karma": u["karma"]})
    _append(pd.DataFrame(rows), _FILES["karma_trending"], dedup_cols=["date", "timeframe", "username"])

    # --- commenter_trending ---
    rows = []
    for tf, key in (("daily", "commenterTrendingDaily"), ("hourly", "commenterTrendingHourly")):
        for rank, u in enumerate(data.get(key, []), 1):
            rows.append({"date": date, "timeframe": tf, "rank": rank,
                         "member": u["member"], "score": u["score"]})
    _append(pd.DataFrame(rows), _FILES["commenter_trending"], dedup_cols=["date", "timeframe", "member"])

    # --- session_store ---
    keep = ["ticker", "name", "type", "market_status",
            "price", "change", "change_pct", "volume", "prev_close"]
    rows = [{**{k: s[k] for k in keep}, "date": date} for s in data.get("sessionStore", [])]
    _append(pd.DataFrame(rows), _FILES["session_store"], dedup_cols=["date", "ticker"])

    print(f"Saved WSB data for {date} → {WSB_DIR}/")


def _append(new_df: pd.DataFrame, path: Path, dedup_cols: list[str]) -> None:
    """Read existing parquet (if any), drop rows matching today's dedup_cols, append new rows."""
    if new_df.empty:
        return
    if path.exists():
        existing = pd.read_parquet(path)
        # Drop any rows already stored for the same date (idempotent re-runs)
        mask = existing[dedup_cols[0]].astype(str).isin(new_df[dedup_cols[0]].astype(str))
        for col in dedup_cols[1:]:
            mask &= existing[col].astype(str).isin(new_df[col].astype(str))
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)


def load_wsb_data(table: str, date_from: str | None = None) -> pd.DataFrame:
    """
    Read a stored WSB table.

    table: 'trending_tickers' | 'karma_trending' | 'commenter_trending' | 'session_store'
    date_from: optional ISO date string, e.g. '2026-05-01' to filter from that date onward.
    """
    path = _FILES[table]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run save_wsb_data() first")
    df = pd.read_parquet(path)
    if date_from:
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(date_from)]
    return df

