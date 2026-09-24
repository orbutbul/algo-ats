"""
prediction_markets/matching.py — pair equivalent bets across two venues.

    from prediction_markets.matching import match_markets
    from prediction_markets.venues.kalshi import KalshiClient, SPORTS_SERIES
    from prediction_markets.venues.polymarket_us import PolymarketUSClient, SPORTS_MARKET_TYPES

    kalshi = KalshiClient().list_markets(series_ticker=SPORTS_SERIES)
    poly_us = PolymarketUSClient().list_markets(market_types=SPORTS_MARKET_TYPES)
    matches = match_markets(kalshi, poly_us)

Deliberately a standalone function over two `list_markets()` DataFrames
rather than a VenueClient method: it's pure (no HTTP, so it works on cached
frames and is unit-testable), symmetric, and never needs to know either
venue's ticker/slug grammar -- each client already translated that into the
normalized proposition columns on models.Market (league, game_id,
event_date, market_type, outcome, line, negated, teams). Rows without a
market_type (futures, props, non-sports) are ignored.

Two steps:
1. Games. Team codes don't line up across venues outside the NFL (.us
   "librty"/"coast" vs Kalshi "LIB"/"CCU", UFC "rauros" vs "ROS"), so within
   each (league, event_date) every left game is scored against every right
   game by team-name similarity under the better of the two team pairings,
   and games are assigned one-to-one, best score first.
2. Propositions. Within a matched game, winner markets pair by team, spreads
   by (team, line), totals by line. `orientation` says whether the two YES/
   long sides pay in the same world ('same') or opposite ones ('inverted');
   the right venue's price and bid/ask are restated in the left market's
   terms (`*_aligned`; an inverted bid is 1 - ask and vice versa), so they
   compare directly. `arb_edge` is the gross per-contract profit of buying
   the left YES at one venue's ask and its complement at the other's
   (= best aligned bid minus the other venue's ask); > 0 means the books
   cross, before fees and before checking depth.

What this does NOT check: settlement rules. Winner markets are paired as
complements across teams ("A wins" vs "B wins" -> inverted), which is only
exact if the game can't end tied/drawn/no-contest -- venues settle those
differently (.us settles NFL ties and UFC draws at $0.50). Compare the two
markets' rules before treating a gap as an arbitrage.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

MATCH_COLUMNS = [
    'league', 'event_date', 'market_type', 'proposition', 'line', 'orientation',
    'left_venue', 'left_market_id', 'left_title', 'left_price', 'left_bid', 'left_ask',
    'right_venue', 'right_market_id', 'right_title', 'right_price', 'right_bid', 'right_ask',
    'right_price_aligned', 'right_bid_aligned', 'right_ask_aligned',
    'price_gap', 'arb_edge', 'game_score',
]

# normalized name -> normalized name, for schools a venue only knows by a
# nickname-style short form (seen live 2026-09-24).
_NAME_ALIASES = {
    'uconn': 'connecticut',
    'umass': 'massachusetts',
    'ut martin': 'tennessee martin',
}


def _norm_name(name: str) -> str:
    s = name.lower().replace('&', ' and ')
    s = re.sub(r'^st\.?\s', 'saint ', s)        # "St. Thomas" -> saint
    s = re.sub(r'\sst\.?(?=\s|$)', ' state', s)  # "Kansas St." -> state
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = ' '.join(s.split())
    return _NAME_ALIASES.get(s, s)


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def team_similarity(code_a: str, names_a, code_b: str, names_b) -> float:
    """1.0 for the same code or same normalized name; 0.9 when one name is a
    whole-word part of the other ("New England" / "New England Patriots");
    0.85 for a plain prefix ("Los Angeles C" / "Los Angeles Chargers");
    otherwise the best difflib ratio across all name pairs."""
    if code_a and code_b and code_a.lower() == code_b.lower():
        return 1.0
    best = 0.0
    for a in (_norm_name(n) for n in names_a or ()):
        for b in (_norm_name(n) for n in names_b or ()):
            if not a or not b:
                continue
            if a == b:
                return 1.0
            if f' {a} ' in f' {b} ' or f' {b} ' in f' {a} ':
                best = max(best, 0.9)
            elif a.startswith(b) or b.startswith(a):
                best = max(best, 0.85)
            else:
                best = max(best, SequenceMatcher(None, a, b).ratio())
    return best


def _games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game_id: league, event_date, and the union of the team
    code -> names maps across that game's markets."""
    rows = []
    for game_id, grp in df.groupby('game_id', sort=False):
        teams: dict[str, tuple[str, ...]] = {}
        for t in grp['teams']:
            for code, names in (t or {}).items():
                teams[code] = tuple(dict.fromkeys((*teams.get(code, ()), *names)))
        rows.append({'game_id': game_id, 'league': grp['league'].iloc[0],
                     'event_date': grp['event_date'].iloc[0], 'teams': teams})
    return pd.DataFrame(rows, columns=['game_id', 'league', 'event_date', 'teams'])


def _game_score(left_teams: dict, right_teams: dict) -> tuple[float, dict[str, str]]:
    """(score, left code -> right code) under the better team pairing; score
    is the weaker of the two team similarities, so both teams must match."""
    if len(left_teams) != 2 or len(right_teams) != 2:
        return 0.0, {}
    (l1, n1), (l2, n2) = left_teams.items()
    (r1, m1), (r2, m2) = right_teams.items()
    straight = min(team_similarity(l1, n1, r1, m1), team_similarity(l2, n2, r2, m2))
    crossed = min(team_similarity(l1, n1, r2, m2), team_similarity(l2, n2, r1, m1))
    if straight >= crossed:
        return straight, {l1: r1, l2: r2}
    return crossed, {l1: r2, l2: r1}


def match_games(left: pd.DataFrame, right: pd.DataFrame, *, min_score: float = 0.8) -> pd.DataFrame:
    """Pair left/right games (same league and event_date) one-to-one.
    Columns: left_game_id, right_game_id, game_score, team_map (left code ->
    right code)."""
    lg, rg = _games(left), _games(right)
    candidates = []
    for (league, event_date), lgrp in lg.groupby(['league', 'event_date']):
        rgrp = rg[(rg['league'] == league) & (rg['event_date'] == event_date)]
        for _, lrow in lgrp.iterrows():
            for _, rrow in rgrp.iterrows():
                score, team_map = _game_score(lrow['teams'], rrow['teams'])
                if score >= min_score:
                    candidates.append((score, lrow['game_id'], rrow['game_id'], team_map))

    candidates.sort(key=lambda c: c[0], reverse=True)
    used_left, used_right, rows = set(), set(), []
    for score, left_id, right_id, team_map in candidates:
        if left_id in used_left or right_id in used_right:
            continue
        used_left.add(left_id)
        used_right.add(right_id)
        rows.append({'left_game_id': left_id, 'right_game_id': right_id,
                     'game_score': score, 'team_map': team_map})
    return pd.DataFrame(rows, columns=['left_game_id', 'right_game_id', 'game_score', 'team_map'])


def _proposition(market_type: str, outcome: str | None, line: float | None) -> str:
    if market_type == 'winner':
        return f'{outcome} wins'
    if market_type == 'spread':
        return f'{outcome} wins by > {line:g}'
    return f'total > {line:g}'


def match_markets(left: pd.DataFrame, right: pd.DataFrame, *, min_game_score: float = 0.8) -> pd.DataFrame:
    """Every pair of markets, one from each frame, that settle on the same
    (or exactly opposite) outcome of the same game. `left`/`right` are
    `list_markets()` outputs from any two venues whose clients fill the
    normalized proposition columns. One row per pair, columns MATCH_COLUMNS;
    `proposition` is phrased in the left venue's team codes."""
    sports_cols = ['game_id', 'market_type']
    left = left.dropna(subset=sports_cols)
    right = right.dropna(subset=sports_cols)
    if left.empty or right.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    games = match_games(left, right, min_score=min_game_score)
    left_by_game = dict(tuple(left.groupby('game_id')))
    right_by_game = dict(tuple(right.groupby('game_id')))

    rows = []
    for g in games.itertuples(index=False):
        team_map = g.team_map
        for lm in left_by_game[g.left_game_id].itertuples(index=False):
            for rm in right_by_game[g.right_game_id].itertuples(index=False):
                if lm.market_type != rm.market_type:
                    continue
                if lm.market_type == 'winner':
                    if team_map.get(lm.outcome) is None or rm.outcome not in team_map.values():
                        continue
                    same_team = team_map[lm.outcome] == rm.outcome
                    same = same_team == (bool(lm.negated) == bool(rm.negated))
                else:
                    if round(lm.line, 2) != round(rm.line, 2):
                        continue
                    if lm.market_type == 'spread' and team_map.get(lm.outcome) != rm.outcome:
                        continue
                    same = bool(lm.negated) == bool(rm.negated)

                rows.append(_match_row(lm, rm, same, g.game_score))
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def _match_row(lm, rm, same: bool, game_score: float) -> dict:
    def flip(p):
        return None if p is None else round(1 - p, 4)

    left_price, left_bid, left_ask = _num(lm.last_price), _num(lm.best_bid), _num(lm.best_ask)
    right_price, right_bid, right_ask = _num(rm.last_price), _num(rm.best_bid), _num(rm.best_ask)
    if same:
        price_al, bid_al, ask_al = right_price, right_bid, right_ask
    else:
        price_al, bid_al, ask_al = flip(right_price), flip(right_ask), flip(right_bid)

    edges = []
    if bid_al is not None and left_ask is not None:
        edges.append(bid_al - left_ask)      # buy left YES, buy its complement on the right
    if left_bid is not None and ask_al is not None:
        edges.append(left_bid - ask_al)      # buy left complement, buy the equivalent on the right
    prop = _proposition(lm.market_type, lm.outcome, lm.line)
    return {
        'league': lm.league, 'event_date': lm.event_date, 'market_type': lm.market_type,
        'proposition': f'NOT {prop}' if lm.negated else prop, 'line': lm.line,
        'orientation': 'same' if same else 'inverted',
        'left_venue': lm.venue, 'left_market_id': lm.market_id, 'left_title': lm.title,
        'left_price': left_price, 'left_bid': left_bid, 'left_ask': left_ask,
        'right_venue': rm.venue, 'right_market_id': rm.market_id, 'right_title': rm.title,
        'right_price': right_price, 'right_bid': right_bid, 'right_ask': right_ask,
        'right_price_aligned': price_al, 'right_bid_aligned': bid_al, 'right_ask_aligned': ask_al,
        'price_gap': (round(price_al - left_price, 4)
                      if price_al is not None and left_price is not None else None),
        'arb_edge': round(max(edges), 4) if edges else None,
        'game_score': game_score,
    }
