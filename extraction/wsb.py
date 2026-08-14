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
import duckdb
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
    visible text under several tab states — no login required:
      - "base":            default landing state (usually the Chat tab)
      - "leaderboard_alt":  Leaderboard switched to its Streaks tab (legacy;
                            no longer reachable post-redesign, kept as a
                            harmless no-op in case Reddit reverts)
      - "sentiment_alt":    legacy Sentiment Bull/Bear toggle (same as above)
      - "portfolio":        Data tab, default Portfolio sub-tab (open positions
                            + trader/exposure summary)
      - "mentions_v2":      Data tab, Sentiment sub-tab (ticker mentions list)

    As of the 2026-08 widget redesign, the old flat MENTIONS 24H / SENTIMENT /
    LEADERBOARD / TOP HOLDINGS / TOP TRADES section layout is gone, replaced
    by tabs (Me / Data / Casino / Chat) with Data itself split into
    Portfolio / Sentiment / BanBets sub-tabs. "base" is kept for the legacy
    parsers (now mostly dead) but "portfolio" and "mentions_v2" are the
    views that actually carry current data.
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

            views["portfolio"] = ""
            views["mentions_v2"] = ""
            if _click_visible_text(frame, "Data"):
                page.wait_for_timeout(1_500)
                views["portfolio"] = frame.inner_text("body")
                if _click_visible_text(frame, "Sentiment"):
                    page.wait_for_timeout(1_500)
                    views["mentions_v2"] = frame.inner_text("body")

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


_PAGINATION_RE = re.compile(r"^[\d,]+[–-][\d,]+ of [\d,]+$")


def _rows_after_marker(lines: list[str], marker: str, cols: int) -> list[list[str]]:
    """
    Find the last occurrence of `marker` (a column header) in `lines`, then
    consume everything after it in groups of `cols`, skipping the trailing
    pagination footer line (e.g. "1–19 of 200") if present.
    """
    try:
        start = len(lines) - 1 - lines[::-1].index(marker)
    except ValueError:
        return []
    data = [ln for ln in lines[start + 1:] if not _PAGINATION_RE.match(ln)]
    return [data[i:i + cols] for i in range(0, len(data) - cols + 1, cols)]


def _parse_positions(lines: list[str]) -> list[dict]:
    """
    Parse the Data > Portfolio sub-tab's open-positions table (post-2026-08
    widget redesign). Row shape: rank, "TICKER $STRIKE", C/P, expiry, DTE,
    holders, qty, value, sentiment %, delta.
    """
    out = []
    for i, group in enumerate(_rows_after_marker(lines, "Δ", cols=10)):
        rank, contract, opt_type, expiry, dte, holders, qty, val, sent_pct, delta = group
        parts = contract.split(maxsplit=1)
        out.append({
            "rank": _num(rank) or (i + 1),
            "ticker": parts[0],
            "strike": _num(parts[1]) if len(parts) > 1 else None,
            "contract_type": opt_type,
            "expiry": expiry,
            "dte": _num(dte.rstrip("dD")),
            "holders": _num(holders),
            "qty": _num(qty),
            "val": _num(val),
            "sent_pct": _num(sent_pct),
            "delta": None if delta.strip() == "—" else _num(delta),
        })
    return out


def _parse_mentions_v2(lines: list[str]) -> list[dict]:
    """
    Parse the Data > Sentiment sub-tab's Ticker List (post-2026-08 widget
    redesign). Row shape: rank, ticker, holders, bull %, mentions, score.
    """
    out = []
    for i, group in enumerate(_rows_after_marker(lines, "Score", cols=6)):
        rank, ticker, holders, bull_pct, mentions, score = group
        out.append({
            "rank": _num(rank) or (i + 1),
            "ticker": ticker,
            "holders": _num(holders),
            "bull_pct": _num(bull_pct),
            "mentions": _num(mentions),
            "score": _num(score),
        })
    return out


def _parse_portfolio_summary(lines: list[str]) -> dict:
    """Overall trader sentiment card at the top of Data > Portfolio: 'NN% BULLISH'."""
    try:
        idx = lines.index("BULLISH")
    except ValueError:
        return {"bullish_pct": None}
    return {"bullish_pct": _num(lines[idx - 1]) if idx > 0 else None}


def _extract_wsb_data(views: dict[str, str]) -> dict:
    """
    views: {"base": str, "leaderboard_alt": str, "sentiment_alt": str,
    "portfolio": str, "mentions_v2": str} as produced by
    _capture_widget_views (a bare str is also accepted for backward-compat /
    ad-hoc use, treated as the base-only view).
    """
    if isinstance(views, str):
        views = {"base": views, "leaderboard_alt": "", "sentiment_alt": "",
                  "portfolio": "", "mentions_v2": ""}

    sections = _split_sections(views["base"])
    alt_leaderboard_sections = _split_sections(views.get("leaderboard_alt", "") or "")
    alt_sentiment_sections = _split_sections(views.get("sentiment_alt", "") or "")

    portfolio_lines = [ln.strip() for ln in views.get("portfolio", "").splitlines() if ln.strip()]
    mentions_v2_lines = [ln.strip() for ln in views.get("mentions_v2", "").splitlines() if ln.strip()]

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
        "positions_v2":   _parse_positions(portfolio_lines),
        "mentions_v2":    _parse_mentions_v2(mentions_v2_lines),
        "portfolio_summary": _parse_portfolio_summary(portfolio_lines),
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
    List u/verified-trader's own recent posts via their public profile page
    on www.reddit.com (new Reddit).

    old.reddit.com now redirects logged-out visitors to a login wall for
    r/wallstreetbets (Reddit policy change), and Reddit closed self-serve
    OAuth app registration in 2026 (the "Responsible Builder Policy" — new
    apps now require prior approval), so this profile page is the only
    unauthenticated path left. www.reddit.com's <shreddit-post> custom
    elements expose title/permalink/timestamp as plain HTML attributes (no
    shadow-DOM piercing needed) — but the list is virtualized, so posts
    scrolled far out of view get unmounted from the DOM. Collected
    incrementally across scroll steps rather than read once at the end.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page.goto(f"https://www.reddit.com/user/{REDDIT_USER}/submitted/", wait_until="load", timeout=45_000)
        page.wait_for_timeout(3_000)

        collected: dict[str, dict] = {}  # permalink -> post, insertion order preserved
        stall_rounds = 0
        while len(collected) < limit and stall_rounds < 3:
            before = len(collected)
            for post in page.query_selector_all("shreddit-post"):
                permalink = post.get_attribute("permalink") or ""
                if not permalink or permalink in collected:
                    continue
                title = post.get_attribute("post-title") or ""
                ts = post.get_attribute("created-timestamp")
                created = datetime.fromisoformat(ts) if ts else datetime.now(tz=timezone.utc)
                post_id = permalink.rstrip("/").split("/")[-2] if "/comments/" in permalink else permalink
                collected[permalink] = {
                    "id": post_id, "title": title,
                    "url": f"https://www.reddit.com{permalink}", "created_utc": created,
                }

            stall_rounds = 0 if len(collected) > before else stall_rounds + 1
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(1_000)

        browser.close()

    return list(collected.values())[:limit]


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

    target = matching[0]
    if posts[0]["id"] != target["id"]:
        # The widget only renders live data on the single most-recent post
        # from the account, regardless of thread type — any older post
        # (even the latest 'moves' thread, if a newer 'daily' thread has
        # since been posted) shows a "this is not the most recent thread"
        # banner instead of data. The widget's content (sentiment/holdings/
        # trades/mentions) is account-wide, not specific to any one thread's
        # topic, so falling back to the true latest post is safe.
        target = posts[0]

    data = fetch_wsb_post(target["url"])
    data["post_title"] = target["title"]
    data["post_date"]  = target["created_utc"]

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

WSB_DB_PATH = Path("data/wsb.duckdb")

_TABLES = {
    "sentiment": {
        "schema": "date DATE, hour INTEGER, bearish_pct DOUBLE, bullish_pct DOUBLE",
        "dedup_cols": ["date", "hour"],
    },
    "sentiment_most_bearish": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, split VARCHAR, pct DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    "sentiment_most_bullish": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, split VARCHAR, pct DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    "top_holdings": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, price DOUBLE, change_pct DOUBLE, count DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    "top_trades": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, contract VARCHAR, qty DOUBLE, pnl DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    "biggest_movers": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, price DOUBLE, change_pct DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    "mentions": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, price DOUBLE, change_pct DOUBLE, count DOUBLE",
        "dedup_cols": ["date", "hour", "ticker"],
    },
    "leaderboard": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, username VARCHAR, count DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    "leaderboard_streaks": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, username VARCHAR, streak_days DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
    # --- post-2026-08 widget redesign (Data > Portfolio / Data > Sentiment) ---
    "mentions_v2": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, holders DOUBLE, "
                  "bull_pct DOUBLE, mentions DOUBLE, score DOUBLE",
        "dedup_cols": ["date", "hour", "ticker"],
    },
    "positions_v2": {
        "schema": "date DATE, hour INTEGER, rank INTEGER, ticker VARCHAR, strike DOUBLE, "
                  "contract_type VARCHAR, expiry VARCHAR, dte DOUBLE, holders DOUBLE, "
                  "qty DOUBLE, val DOUBLE, sent_pct DOUBLE, delta DOUBLE",
        "dedup_cols": ["date", "hour", "rank"],
    },
}


def _connect_wsb_db(read_only: bool = False) -> "duckdb.DuckDBPyConnection":
    WSB_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WSB_DB_PATH), read_only=read_only)
    if not read_only:
        for name, spec in _TABLES.items():
            con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({spec['schema']})")
    return con


def _upsert(con: "duckdb.DuckDBPyConnection", table: str, df: pd.DataFrame) -> None:
    """
    Delete any existing rows matching this batch's dedup key, then insert —
    safe to call multiple times for the same (date, hour): the later call's
    rows replace the earlier ones rather than duplicating them.
    """
    if df.empty:
        return
    dedup_cols = _TABLES[table]["dedup_cols"]
    key_cols = ", ".join(dedup_cols)
    con.register("_new", df)
    con.execute(f"DELETE FROM {table} WHERE ({key_cols}) IN (SELECT {key_cols} FROM _new)")
    con.execute(f"INSERT INTO {table} SELECT * FROM _new")
    con.unregister("_new")


def save_wsb_data(data: dict) -> None:
    """
    Append this run's WSB snapshot to data/wsb.duckdb (one table per section),
    tagged with the UTC hour it was captured in. Safe to call multiple times
    within the same hour — existing rows for that (date, hour) are replaced;
    calls in different hours of the same day both accumulate, building an
    intraday history rather than overwriting the day's only snapshot.
    """
    now = datetime.now(timezone.utc)
    date = now.date()
    hour = now.hour

    con = _connect_wsb_db()
    try:
        # --- sentiment (one row per hour: both directions' overall score) ---
        # The legacy Bull/Bear-toggle section this used to come from no longer
        # exists (2026-08 widget redesign); fall back to the Portfolio tab's
        # single "NN% BULLISH" summary card. Skip the row entirely rather than
        # write an all-null placeholder if neither source has a value.
        sent = data.get("sentiment", {})
        bullish_pct = sent.get("bullish_pct")
        if bullish_pct is None:
            bullish_pct = data.get("portfolio_summary", {}).get("bullish_pct")
        bearish_pct = sent.get("bearish_pct")
        if bullish_pct is not None or bearish_pct is not None:
            sent_row = {
                "date": date,
                "hour": hour,
                "bearish_pct": bearish_pct,
                "bullish_pct": bullish_pct,
            }
            _upsert(con, "sentiment", pd.DataFrame([sent_row]))

        for direction in ("most_bearish", "most_bullish"):
            rows = [{"date": date, "hour": hour, "rank": i + 1, **r}
                    for i, r in enumerate(sent.get(direction, []))]
            _upsert(con, f"sentiment_{direction}", pd.DataFrame(rows))

        # --- top_holdings ---
        rows = [{"date": date, "hour": hour, "rank": i + 1, **r}
                for i, r in enumerate(data.get("top_holdings", []))]
        _upsert(con, "top_holdings", pd.DataFrame(rows))

        # --- top_trades ---
        rows = [{"date": date, "hour": hour, "rank": i + 1, **r}
                for i, r in enumerate(data.get("top_trades", []))]
        _upsert(con, "top_trades", pd.DataFrame(rows))

        # --- biggest_movers ---
        # The ticker-strip this parses no longer reliably exists post-redesign;
        # only keep rows that actually resolved a price, to avoid writing
        # tickers with all-null price/change_pct.
        rows = [{"date": date, "hour": hour, "rank": i + 1, **r}
                for i, r in enumerate(data.get("biggest_movers", []))
                if r.get("price") is not None]
        _upsert(con, "biggest_movers", pd.DataFrame(rows))

        # --- mentions ---
        rows = []
        for i, m in enumerate(data.get("mentions", [])):
            ticker = m.get("ticker", "")
            if ticker:
                rows.append({"date": date, "hour": hour, "rank": i + 1, "ticker": ticker,
                             "price": m.get("price"), "change_pct": m.get("change_pct"),
                             "count": m.get("count")})
        _upsert(con, "mentions", pd.DataFrame(rows))

        # --- leaderboard (by comments, and by streak length) ---
        lb = data.get("leaderboard", {})
        rows = [{"date": date, "hour": hour, "rank": i + 1, **r}
                for i, r in enumerate(lb.get("by_comments", []))]
        _upsert(con, "leaderboard", pd.DataFrame(rows))

        rows = [{"date": date, "hour": hour, "rank": i + 1, **r}
                for i, r in enumerate(lb.get("by_streak", []))]
        _upsert(con, "leaderboard_streaks", pd.DataFrame(rows))

        # --- mentions_v2 (Data > Sentiment ticker list) ---
        rows = [{"date": date, "hour": hour, **r} for r in data.get("mentions_v2", [])]
        _upsert(con, "mentions_v2", pd.DataFrame(rows))

        # --- positions_v2 (Data > Portfolio open positions) ---
        rows = [{"date": date, "hour": hour, **r} for r in data.get("positions_v2", [])]
        _upsert(con, "positions_v2", pd.DataFrame(rows))
    finally:
        con.close()

    print(f"Saved WSB data for {date} {hour:02d}:00 UTC -> {WSB_DB_PATH}")


def load_wsb_data(table: str, date_from: str | None = None) -> pd.DataFrame:
    """
    Read a stored WSB table from data/wsb.duckdb.

    table: 'sentiment' | 'sentiment_most_bearish' | 'sentiment_most_bullish'
         | 'top_holdings' | 'top_trades' | 'biggest_movers' | 'mentions'
         | 'leaderboard' | 'leaderboard_streaks'
    date_from: optional ISO date string, e.g. '2026-05-01'
    """
    if table not in _TABLES:
        raise ValueError(f"Unknown table '{table}'. Choose from: {list(_TABLES)}")
    if not WSB_DB_PATH.exists():
        raise FileNotFoundError(f"{WSB_DB_PATH} not found — run save_wsb_data() first")

    con = _connect_wsb_db(read_only=True)
    try:
        query = f"SELECT * FROM {table}"
        params = []
        if date_from:
            query += " WHERE date >= ?"
            params.append(date_from)
        return con.execute(query, params).df()
    finally:
        con.close()
