"""
WSB scraper — pulls data from u/verified-trader's custom Reddit posts.

verified-trader uses Reddit's Devvit App Platform, embedding a widget as an
iframe. The widget renders clean, readable panels (MENTIONS 24H, SENTIMENT,
LEADERBOARD, TOP HOLDINGS, TOP TRADES, plus a top ticker strip sorted by
absolute % move which we treat as "biggest movers"). No Reddit login is
required to view it. We render the post with Playwright, read the widget's
plain visible text, and parse it by section header.

Usage:
    from extraction.wsb import fetch_wsb_post, get_latest_wsb_data
    data = get_latest_wsb_data()
    df   = get_latest_wsb_data(as_df=True)  # DataFrame of biggest_movers

Debug (inspect raw widget text when the app changes):
    from extraction.wsb import dump_widget_text
    dump_widget_text()
"""

from __future__ import annotations

import base64
import os
import re
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, Frame


REDDIT_USER = "verified-trader"

# ---------------------------------------------------------------------------
# Reddit OAuth helpers
# ---------------------------------------------------------------------------

def _reddit_oauth_token() -> str | None:
    """
    Return a Reddit application-only access token using client credentials
    from env vars REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.
    Returns None if credentials are not set.
    """
    client_id     = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None

    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        headers={
            "Authorization": f"Basic {creds}",
            "User-Agent": "wsb_scraper/2.0",
        },
        data={"grant_type": "client_credentials"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]

# ---------------------------------------------------------------------------
# Playwright: locate and read the Devvit widget
# ---------------------------------------------------------------------------

_IFRAME_SELECTORS = [
    "iframe[src*='devvit']",
    "iframe[src*='reddit-static']",
    "iframe[id*='devvit']",
    "shreddit-post iframe",
    "div[data-testid='post-content'] iframe",
]


def _find_devvit_frame(page: Page) -> Frame | None:
    for selector in _IFRAME_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el:
                frame = el.content_frame()
                if frame:
                    return frame
        except Exception:
            continue
    return None


def _capture_widget_text(post_url: str) -> str:
    """
    Render a verified-trader post and return the Devvit widget's plain
    visible text (default tab state only). No login required.
    """
    return _capture_widget_views(post_url)["base"]


def _click_visible_text(frame: Frame, label: str) -> bool:
    """
    Click a widget tab/button by its exact visible text. The widget renders a
    hidden mobile-layout duplicate of every panel alongside the desktop one,
    so get_by_text often matches 2+ elements — only one has a real bounding
    box. Regular Locator.click() also fails its own visibility check against
    the visible match (element is tiny, 7px font), so we dispatch a raw JS
    click instead of relying on Playwright's actionability checks.
    """
    loc = frame.get_by_text(label, exact=True)
    for i in range(loc.count()):
        box = loc.nth(i).bounding_box()
        if box and box["width"] > 0 and box["height"] > 0:
            loc.nth(i).evaluate("el => el.click()")
            return True
    return False


def _click_sentiment_toggle(frame: Frame) -> bool:
    """Click the 'NN% Bear'/'NN% Bull' button to flip the sentiment panel."""
    for b in frame.query_selector_all("button"):
        t = (b.inner_text() or "").strip()
        if t.endswith("Bear") or t.endswith("Bull"):
            box = b.bounding_box()
            if box and box["width"] > 0:
                b.evaluate("el => el.click()")
                return True
    return False


def _capture_widget_views(post_url: str) -> dict[str, str]:
    """
    Render a verified-trader post and capture the Devvit widget's plain
    visible text under three tab states — no login required:
      - "base":            default state (Comments leaderboard, one sentiment direction)
      - "leaderboard_alt":  Leaderboard switched to its Streaks tab
      - "sentiment_alt":    Sentiment switched to its inverse (Bull<->Bear) direction
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        try:
            # "load" not "networkidle" — Reddit keeps polling and never idles.
            page.goto(post_url, wait_until="load", timeout=60_000)
            page.wait_for_timeout(5_000)

            frame = _find_devvit_frame(page)
            if frame is None:
                raise RuntimeError(f"Devvit widget iframe not found on {post_url}")

            try:
                frame.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            page.wait_for_timeout(3_000)

            views = {"base": frame.inner_text("body")}

            if _click_visible_text(frame, "Streaks"):
                page.wait_for_timeout(1_500)
                views["leaderboard_alt"] = frame.inner_text("body")
                _click_visible_text(frame, "Comments")  # restore, for cleanliness
                page.wait_for_timeout(500)
            else:
                views["leaderboard_alt"] = ""

            if _click_sentiment_toggle(frame):
                page.wait_for_timeout(1_500)
                views["sentiment_alt"] = frame.inner_text("body")
            else:
                views["sentiment_alt"] = ""

            return views
        finally:
            browser.close()


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

_SECTION_HEADERS = ["MENTIONS 24H", "SENTIMENT", "LEADERBOARD", "TOP HOLDINGS", "TOP TRADES"]
_SECTION_END = "CONNECT & VERIFY"
_JUNK_LINES = {"TABLE", "CHART", "COMMENTS", "STREAKS", "·"}

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _num(s: str) -> float | None:
    """Parse '$1,365.50', '-7.95%', '+$44.4k', '×250' etc. into a float."""
    if s is None:
        return None
    s = s.strip().lstrip("×").replace(",", "").replace("$", "").replace("%", "")
    s = s.lstrip("+")
    mult = 1.0
    if s and s[-1] in ("k", "K"):
        mult, s = 1_000.0, s[:-1]
    elif s and s[-1] in ("m", "M"):
        mult, s = 1_000_000.0, s[:-1]
    try:
        return float(s) * mult
    except (TypeError, ValueError):
        return None


def _split_sections(text: str) -> dict[str, list[str]]:
    """
    Split raw widget text into named line-blocks: ticker_strip (everything
    before the first known header) plus one block per known section header,
    each cut off at the next known header (or CONNECT & VERIFY at the end).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    end_idx = next((i for i, ln in enumerate(lines) if ln == _SECTION_END), len(lines))

    # Only search for headers before the end marker — labels like "SENTIMENT"
    # and "MENTIONS" recur as column headers in the trailing per-ticker detail
    # card past CONNECT & VERIFY, and would otherwise corrupt slice boundaries.
    header_positions: list[tuple[str, int]] = []
    for i, ln in enumerate(lines[:end_idx]):
        if ln in _SECTION_HEADERS:
            header_positions.append((ln, i))

    sections: dict[str, list[str]] = {}
    first_header_idx = header_positions[0][1] if header_positions else len(lines)

    # Ticker strip: everything before the first header, starting from the
    # first line that actually looks like a ticker symbol (skips nav chrome
    # like "Closed" / "Opens 9h 54m" / "Daily Thread" / "Me" / "Data" / "Casino").
    strip_start = next(
        (i for i in range(first_header_idx) if _TICKER_RE.match(lines[i])),
        first_header_idx,
    )
    sections["ticker_strip"] = lines[strip_start:first_header_idx]

    for idx, (name, pos) in enumerate(header_positions):
        stop = header_positions[idx + 1][1] if idx + 1 < len(header_positions) else end_idx
        block = [ln for ln in lines[pos + 1: stop] if ln not in _JUNK_LINES]
        sections[name] = block

    return sections


def _parse_ticker_groups(lines: list[str], cols: int) -> list[dict]:
    """Consume `cols` lines at a time: ticker, price, change_pct[, count]."""
    out = []
    for i in range(0, len(lines) - cols + 1, cols):
        group = lines[i:i + cols]
        row = {
            "ticker": group[0],
            "price": _num(group[1]),
            "change_pct": _num(group[2]),
        }
        if cols == 4:
            row["count"] = _num(group[3])
        out.append(row)
    return out


def _parse_biggest_movers(lines: list[str]) -> list[dict]:
    rows = _parse_ticker_groups(lines, cols=3)
    seen = set()
    deduped = []
    for r in rows:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        deduped.append(r)
    return deduped


def _parse_sentiment_state(lines: list[str]) -> dict:
    """
    Parse one Sentiment tab state (either the default direction or the
    inverse, after clicking the 'NN% Bear'/'NN% Bull' toggle). Each ticker
    row shows a bull:bear split (e.g. "24:2") and the dominant side's %.
    """
    if not lines:
        return {"overall_pct": None, "overall_label": None, "tickers": []}

    m = re.match(r"(\d+)%\s*(\w+)", lines[0])
    overall_pct = float(m.group(1)) if m else None
    overall_label = m.group(2) if m else None

    tickers = []
    rest = lines[1:]
    for i in range(0, len(rest) - 2, 3):
        ticker, split, pct = rest[i:i + 3]
        tickers.append({"ticker": ticker, "split": split, "pct": _num(pct)})

    return {"overall_pct": overall_pct, "overall_label": overall_label, "tickers": tickers}


def _combine_sentiment(base_lines: list[str], alt_lines: list[str]) -> dict:
    """
    Combine the default and toggled Sentiment tab states into one dict keyed
    by direction (bearish/bullish), regardless of which one was the default.
    """
    states = [_parse_sentiment_state(base_lines)]
    if alt_lines:
        states.append(_parse_sentiment_state(alt_lines))

    by_label = {s["overall_label"]: s for s in states if s["overall_label"]}
    bear = by_label.get("Bear", {"overall_pct": None, "tickers": []})
    bull = by_label.get("Bull", {"overall_pct": None, "tickers": []})

    return {
        "bearish_pct": bear.get("overall_pct"),
        "bullish_pct": bull.get("overall_pct"),
        "most_bearish": bear.get("tickers", []),
        "most_bullish": bull.get("tickers", []),
    }


def _parse_leaderboard(lines: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(lines) - 1, 2):
        username, count = lines[i:i + 2]
        out.append({"username": username, "count": _num(count)})
    return out


def _parse_leaderboard_streaks(lines: list[str]) -> list[dict]:
    """Parse the Leaderboard panel's Streaks tab: username, 'Nd' streak length."""
    out = []
    for i in range(0, len(lines) - 1, 2):
        username, streak = lines[i:i + 2]
        out.append({"username": username, "streak_days": _num(streak.rstrip("dD"))})
    return out


def _parse_top_trades(lines: list[str]) -> list[dict]:
    out = []
    i = 0
    while i < len(lines):
        ticker = lines[i]
        i += 1
        if i >= len(lines):
            break
        contract = None
        if not lines[i].startswith("×"):
            contract = lines[i]
            i += 1
        if i >= len(lines):
            break
        qty = _num(lines[i])
        i += 1
        if i >= len(lines):
            break
        pnl = _num(lines[i])
        i += 1
        out.append({"ticker": ticker, "contract": contract, "qty": qty, "pnl": pnl})
    return out


def _extract_wsb_data(views: dict[str, str]) -> dict:
    """
    views: {"base": str, "leaderboard_alt": str, "sentiment_alt": str} as
    produced by _capture_widget_views (a bare str is also accepted for
    backward-compat / ad-hoc use, treated as the base-only view).
    """
    if isinstance(views, str):
        views = {"base": views, "leaderboard_alt": "", "sentiment_alt": ""}

    sections = _split_sections(views["base"])
    alt_leaderboard_sections = _split_sections(views.get("leaderboard_alt", "") or "")
    alt_sentiment_sections = _split_sections(views.get("sentiment_alt", "") or "")

    return {
        "biggest_movers": _parse_biggest_movers(sections.get("ticker_strip", [])),
        "mentions":       _parse_ticker_groups(sections.get("MENTIONS 24H", []), cols=4),
        "sentiment":      _combine_sentiment(
                              sections.get("SENTIMENT", []),
                              alt_sentiment_sections.get("SENTIMENT", []),
                          ),
        "leaderboard": {
            "by_comments": _parse_leaderboard(sections.get("LEADERBOARD", [])),
            "by_streak":   _parse_leaderboard_streaks(alt_leaderboard_sections.get("LEADERBOARD", [])),
        },
        "top_holdings":   _parse_ticker_groups(sections.get("TOP HOLDINGS", []), cols=4),
        "top_trades":     _parse_top_trades(sections.get("TOP TRADES", [])),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_wsb_posts(limit: int = 25) -> list[dict]:
    """
    Return recent posts by u/verified-trader as a list of dicts.
    Uses OAuth if REDDIT_CLIENT_ID/SECRET are set, otherwise falls back
    to Playwright (browser-rendered profile page).
    """
    token = _reddit_oauth_token()
    if token:
        return _list_posts_oauth(token, limit)
    return _list_posts_playwright(limit)


def _list_posts_oauth(token: str, limit: int) -> list[dict]:
    url = f"https://oauth.reddit.com/user/{REDDIT_USER}/submitted"
    r = requests.get(
        url,
        headers={"Authorization": f"bearer {token}", "User-Agent": "wsb_scraper/2.0"},
        params={"limit": limit, "sort": "new"},
        timeout=15,
    )
    r.raise_for_status()
    return _parse_listing_json(r.json())


def _list_posts_playwright(limit: int) -> list[dict]:
    """
    Read u/verified-trader's own submitted-posts page on www.reddit.com.

    old.reddit.com now redirects anonymous visitors to a login wall, so it
    can no longer be scraped without auth. www.reddit.com still renders a
    user's public profile for logged-out visitors, listing posts as
    <shreddit-post> custom elements with the fields we need as attributes.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            )
        )
        page.goto(f"https://www.reddit.com/user/{REDDIT_USER}/submitted/", wait_until="load", timeout=45_000)
        page.wait_for_timeout(3_000)

        collected = []
        for el in page.query_selector_all("shreddit-post"):
            title      = (el.get_attribute("post-title") or "").strip()
            permalink  = el.get_attribute("permalink") or ""
            ts         = el.get_attribute("created-timestamp")
            fullname   = el.get_attribute("id") or ""
            post_id    = fullname[3:] if fullname.startswith("t3_") else fullname

            if not title or not permalink:
                continue

            full_url = permalink if permalink.startswith("http") else f"https://www.reddit.com{permalink}"
            created = (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                       if ts else datetime.now(tz=timezone.utc))
            collected.append({"id": post_id, "title": title, "url": full_url, "created_utc": created})
            if len(collected) >= limit:
                break

        browser.close()

    return collected


def _parse_listing_json(data: dict) -> list[dict]:
    posts = []
    for p in data.get("data", {}).get("children", []):
        d = p.get("data", {})
        posts.append({
            "id":          d.get("id", ""),
            "title":       d.get("title", ""),
            "url":         f"https://www.reddit.com{d['permalink']}" if d.get("permalink") else "",
            "created_utc": datetime.fromtimestamp(d["created_utc"], tz=timezone.utc) if d.get("created_utc") else datetime.now(tz=timezone.utc),
        })
    return posts


def fetch_wsb_post(post_url: str) -> dict:
    """
    Render a verified-trader post URL and return a dict with all extracted data.
    Keys: biggest_movers, mentions, sentiment, leaderboard, top_holdings, top_trades
    """
    views = _capture_widget_views(post_url)
    return _extract_wsb_data(views)


def get_latest_wsb_data(
    post_type: str = "any",
    as_df: bool = False,
) -> dict | pd.DataFrame:
    """
    Fetch data from the most recent verified-trader post.

    post_type:
        'any'      → most recent post regardless of title (default)
        'moves'    → title contains 'What Are Your Moves'
        'daily'    → title contains 'Daily Discussion'
        'weekend'  → title contains 'Weekend Discussion'
        any string → used as a raw substring filter on the title

    If as_df=True, returns a DataFrame of biggest_movers.
    """
    keywords = {
        "moves":   "What Are Your Moves",
        "daily":   "Daily Discussion",
        "weekend": "Weekend Discussion",
        "any":     "",
    }
    kw = keywords.get(post_type, post_type)

    posts = list_wsb_posts(limit=25)

    if not posts:
        raise ValueError(f"No posts found for u/{REDDIT_USER} — check the username or credentials")

    matching = [p for p in posts if not kw or kw.lower() in p["title"].lower()]
    if not matching:
        available = "\n  ".join(f"'{p['title']}'" for p in posts[:10])
        raise ValueError(
            f"No post from u/{REDDIT_USER} matching '{kw}'.\n"
            f"Available titles:\n  {available}"
        )

    data = fetch_wsb_post(matching[0]["url"])
    data["post_title"] = matching[0]["title"]
    data["post_date"]  = matching[0]["created_utc"]

    if as_df:
        return pd.DataFrame(data["biggest_movers"])

    return data


# ---------------------------------------------------------------------------
# Debug helper — call this when the widget changes to inspect its raw text
# ---------------------------------------------------------------------------

def check_user(user: str = REDDIT_USER) -> None:
    """
    Quick diagnostic: print the first 10 post titles found for a Reddit user.
    Use this to verify the username and that post discovery is working.

    Example:
        from extraction.wsb import check_user
        check_user()                      # checks REDDIT_USER
        check_user("some-other-account")  # checks a different account
    """
    old = globals()["REDDIT_USER"]
    # Temporarily override so _list_posts_playwright uses the right name
    import extraction.wsb as _self
    _self.REDDIT_USER = user
    try:
        posts = list_wsb_posts(limit=10)
    finally:
        _self.REDDIT_USER = old

    if not posts:
        print(f"No posts found for u/{user}. The username may be wrong or the account has no public posts.")
        return
    print(f"Found {len(posts)} post(s) for u/{user}:")
    for p in posts:
        print(f"  [{p['created_utc'].strftime('%Y-%m-%d')}] {p['title']}")
        print(f"           {p['url']}")


def dump_widget_text(post_url: str | None = None, out_path: str = "widget_text_dump.txt") -> None:
    """
    Render a post and write the widget's raw visible text to out_path.
    Use this to re-derive the section parsers if Reddit changes the widget
    layout again — no protobuf tooling needed, just read the panel labels.
    """
    if post_url is None:
        posts = list_wsb_posts(limit=5)
        if not posts:
            raise RuntimeError("No posts found")
        post_url = posts[0]["url"]

    text = _capture_widget_text(post_url)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"Widget text written to {out_path} ({len(text)} chars)")


# ---------------------------------------------------------------------------
# Daily persistence
# ---------------------------------------------------------------------------

WSB_DIR = Path("data/wsb")

_FILES = {
    "sentiment":           WSB_DIR / "sentiment.parquet",
    "sentiment_most_bearish": WSB_DIR / "sentiment_most_bearish.parquet",
    "sentiment_most_bullish": WSB_DIR / "sentiment_most_bullish.parquet",
    "top_holdings":        WSB_DIR / "top_holdings.parquet",
    "top_trades":          WSB_DIR / "top_trades.parquet",
    "biggest_movers":      WSB_DIR / "biggest_movers.parquet",
    "mentions":            WSB_DIR / "mentions.parquet",
    "leaderboard":         WSB_DIR / "leaderboard.parquet",
    "leaderboard_streaks": WSB_DIR / "leaderboard_streaks.parquet",
}


def save_wsb_data(data: dict) -> None:
    """
    Append this run's WSB snapshot to the parquet files in data/wsb/. The
    `date` column is a local-time timestamp truncated to the hour it was
    captured in, so calls within the same hour replace that hour's row while
    calls in different hours of the same day both accumulate — building an
    intraday history rather than overwriting the day's only snapshot.
    """
    WSB_DIR.mkdir(parents=True, exist_ok=True)

    date = datetime.now().replace(minute=0, second=0, microsecond=0)

    # --- sentiment (one row per hour: both directions' overall score) ---
    sent = data.get("sentiment", {})
    sent_row = {
        "date": date,
        "bearish_pct": sent.get("bearish_pct"),
        "bullish_pct": sent.get("bullish_pct"),
    }
    _append(pd.DataFrame([sent_row]), _FILES["sentiment"], dedup_cols=["date"])

    for direction in ("most_bearish", "most_bullish"):
        rows = [{"date": date, "rank": i + 1, **r}
                for i, r in enumerate(sent.get(direction, []))]
        if rows:
            _append(pd.DataFrame(rows), _FILES[f"sentiment_{direction}"], dedup_cols=["date", "rank"])

    # --- top_holdings ---
    rows = [{"date": date, "rank": i + 1, **r}
            for i, r in enumerate(data.get("top_holdings", []))]
    if rows:
        _append(pd.DataFrame(rows), _FILES["top_holdings"], dedup_cols=["date", "rank"])

    # --- top_trades ---
    rows = [{"date": date, "rank": i + 1, **r}
            for i, r in enumerate(data.get("top_trades", []))]
    if rows:
        _append(pd.DataFrame(rows), _FILES["top_trades"], dedup_cols=["date", "rank"])

    # --- biggest_movers ---
    rows = [{"date": date, "rank": i + 1, **r}
            for i, r in enumerate(data.get("biggest_movers", []))]
    if rows:
        _append(pd.DataFrame(rows), _FILES["biggest_movers"], dedup_cols=["date", "rank"])

    # --- mentions ---
    rows = []
    for i, m in enumerate(data.get("mentions", [])):
        ticker = m.get("ticker", "")
        if ticker:
            rows.append({"date": date, "rank": i + 1, "ticker": ticker,
                         "price": m.get("price"), "change_pct": m.get("change_pct"),
                         "count": m.get("count")})
    if rows:
        _append(pd.DataFrame(rows), _FILES["mentions"], dedup_cols=["date", "ticker"])

    # --- leaderboard (by comments, and by streak length) ---
    lb = data.get("leaderboard", {})
    rows = [{"date": date, "rank": i + 1, **r}
            for i, r in enumerate(lb.get("by_comments", []))]
    if rows:
        _append(pd.DataFrame(rows), _FILES["leaderboard"], dedup_cols=["date", "rank"])

    rows = [{"date": date, "rank": i + 1, **r}
            for i, r in enumerate(lb.get("by_streak", []))]
    if rows:
        _append(pd.DataFrame(rows), _FILES["leaderboard_streaks"], dedup_cols=["date", "rank"])

    print(f"Saved WSB data for {date} -> {WSB_DIR}/")


def _append(new_df: pd.DataFrame, path: Path, dedup_cols: list[str]) -> None:
    if new_df.empty:
        return
    if path.exists():
        existing = pd.read_parquet(path)
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

    table: 'sentiment' | 'sentiment_most_bearish' | 'sentiment_most_bullish'
         | 'top_holdings' | 'top_trades' | 'biggest_movers' | 'mentions'
         | 'leaderboard' | 'leaderboard_streaks'
    date_from: optional ISO date string, e.g. '2026-05-01'
    """
    path = _FILES.get(table)
    if path is None:
        raise ValueError(f"Unknown table '{table}'. Choose from: {list(_FILES)}")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run save_wsb_data() first")
    df = pd.read_parquet(path)
    if date_from:
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(date_from)]
    return df
