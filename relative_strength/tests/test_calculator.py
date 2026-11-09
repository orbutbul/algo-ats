"""
Synthetic tests for the RS proxy. Prices here are hand-built fixtures.
They are not market readings and not a scan of the repo's price history.
"""

from __future__ import annotations

import ast
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from relative_strength.calculator import (
    CALENDAR_LOOKBACK_CALENDAR_DAYS,
    PriceSchemaError,
    average_ranks,
    compute_relative_strength,
    nyse_sessions,
    percentile_from_average_rank,
)

AS_OF = date(2024, 6, 28)  # Friday NYSE session
REPO = Path(__file__).resolve().parents[2]


def mcal_sessions(start: date, end: date) -> list[date]:
    """Independent of relative_strength.nyse_sessions — calls the calendar directly."""
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar('NYSE').schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [ts.date() for ts in schedule.index]


def full_calendar(as_of: date = AS_OF) -> list[date]:
    days = mcal_sessions(as_of - timedelta(days=CALENDAR_LOOKBACK_CALENDAR_DAYS), as_of)
    assert days[-1] == as_of
    assert len(days) >= 253
    return days


def _frame(series_by_ticker: dict, classes: dict[str, str], extra: list[dict] | None = None) -> pd.DataFrame:
    rows = []
    for ticker, series in series_by_ticker.items():
        for day, close in series.items():
            rows.append({
                'date': day,
                'ticker': ticker,
                'close': close,
                'asset_class': classes.get(ticker),
            })
    if extra:
        rows.extend(extra)
    return pd.DataFrame(rows)


def _flat(days: list[date], as_of_price: float, anchors: dict[date, float] | None = None) -> dict[date, float]:
    series = {day: 100.0 for day in days}
    series[days[-1]] = as_of_price
    if anchors:
        series.update(anchors)
    return series


def test_percentile_half_up_not_bankers_rounding():
    # 1 + (3 - 1.5) / 2 * 98 = 74.5. Python round(74.5) is 74; the proxy rounds half up to 75.
    assert round(74.5) == 74
    assert percentile_from_average_rank(1.5, 3) == 75
    assert percentile_from_average_rank(1, 3) == 99
    assert percentile_from_average_rank(3, 3) == 1
    assert average_ranks({'BBB': 5.0, 'AAA': 5.0, 'CCC': 1.0}) == {'AAA': 1.5, 'BBB': 1.5, 'CCC': 3.0}


def test_hand_calculated_weights_and_calendar_anchors():
    days = full_calendar()
    anchors = {
        days[-1 - 63]: 100.0,
        days[-1 - 126]: 100.0,
        days[-1 - 189]: 80.0,
        days[-1 - 252]: 200.0,
    }
    series = {
        'WGT': _flat(days, 140.0, anchors),
        'OTHER': _flat(days, 100.0),
        'SPY': _flat(days, 100.0),
    }
    classes = {'WGT': 'equities', 'OTHER': 'equities', 'SPY': 'etfs'}
    result = compute_relative_strength(
        _frame(series, classes),
        as_of=AS_OF,
        universe='stocks',
        adjustment='split',
        min_universe=2,
        include_rs_line=False,
    )
    # Independent calendar: lookbacks are these NYSE dates, not row offsets.
    expected_days = mcal_sessions(AS_OF - timedelta(days=CALENDAR_LOOKBACK_CALENDAR_DAYS), AS_OF)
    assert result.summary['lookback_anchor_sessions'] == {
        '63': expected_days[-1 - 63].isoformat(),
        '126': expected_days[-1 - 126].isoformat(),
        '189': expected_days[-1 - 189].isoformat(),
        '252': expected_days[-1 - 252].isoformat(),
    }
    # R63 = 140/100 - 1 = 0.40, R126 = 0.40, R189 = 140/80 - 1 = 0.75, R252 = 140/200 - 1 = -0.30
    # S = 0.40*0.40 + 0.20*0.40 + 0.20*0.75 + 0.20*(-0.30) = 0.33
    hand_score = 0.40 * 0.40 + 0.20 * 0.40 + 0.20 * 0.75 + 0.20 * (-0.30)
    assert hand_score == pytest.approx(0.33)
    row = result.ratings.set_index('ticker').loc['WGT']
    assert row['r_63'] == pytest.approx(140 / 100 - 1)
    assert row['r_126'] == pytest.approx(0.40)
    assert row['r_189'] == pytest.approx(140 / 80 - 1)
    assert row['r_252'] == pytest.approx(140 / 200 - 1)
    assert row['raw_score'] == pytest.approx(hand_score)
    assert int(row['rs_proxy']) == 99
    assert int(result.ratings.set_index('ticker').loc['OTHER', 'rs_proxy']) == 1
    assert result.summary['method']['official_ibd_rs'] is False
    assert result.summary['method']['weights'] == {'63': 0.40, '126': 0.20, '189': 0.20, '252': 0.20}
    assert result.summary['adjustment'] == 'split'
    assert 'not dividend-adjusted' in result.summary['adjustment_note']
    assert result.ratings.set_index('ticker').loc['SPY', 'reason'] == 'outside_universe'
    assert int(row['eligible_count']) == 2
    assert bool(row['broad_market_rating']) is False


def test_ties_small_universe_withheld_and_etf_crypto_do_not_change_rank():
    days = full_calendar()
    series = {}
    classes = {}
    for i in range(1, 29):
        ticker = f'T{i:02d}'
        series[ticker] = _flat(days, 100.0 + i)
        classes[ticker] = 'equities'
    series['T29'] = _flat(days, 130.0)
    series['T30'] = _flat(days, 130.0)
    classes['T29'] = 'stocks'
    classes['T30'] = 'equities'
    # Extreme names that must not enter the stock cross-section.
    series['SPY'] = _flat(days, 1000.0)
    classes['SPY'] = 'etfs'
    series['BTCUSDT'] = _flat(days, 1000.0)
    classes['BTCUSDT'] = 'cryptos'
    result = compute_relative_strength(
        _frame(series, classes),
        as_of='2024-06-28',
        universe='stocks',
        include_rs_line=False,
    )

    def spec(rank: float, n: int) -> int:
        return int(math.floor(1 + (n - rank) / (n - 1) * 98 + 0.5))

    assert spec(1.5, 30) == 97
    assert spec(3, 30) == 92
    assert spec(30, 30) == 1

    ratings = result.ratings.set_index('ticker')
    assert int(ratings.loc['T30', 'rs_proxy']) == 97
    assert int(ratings.loc['T29', 'rs_proxy']) == 97
    assert int(ratings.loc['T28', 'rs_proxy']) == 92
    assert int(ratings.loc['T01', 'rs_proxy']) == 1
    assert ratings.loc['T30', 'raw_score'] == pytest.approx(0.30)
    assert ratings.loc['T28', 'raw_score'] == pytest.approx(0.28)
    assert int(ratings.loc['T01', 'eligible_count']) == 30
    assert ratings.loc['SPY', 'reason'] == 'outside_universe'
    assert ratings.loc['BTCUSDT', 'reason'] == 'crypto_excluded'
    assert bool(ratings.loc['T30', 'broad_market_rating']) is False
    assert any(item.startswith('narrow_universe') for item in result.summary['warnings'])
    assert result.summary['broad_market_rating'] is False

    # Same book plus one invalid stock must not move the percentile.
    series['BAD'] = _flat(days, 0.0)
    classes['BAD'] = 'equities'
    again = compute_relative_strength(
        _frame(series, classes), as_of=AS_OF, universe='stocks', include_rs_line=False,
    )
    again_ratings = again.ratings.set_index('ticker')
    assert again_ratings.loc['BAD', 'reason'] == 'invalid_price'
    assert int(again_ratings.loc['T30', 'rs_proxy']) == 97
    assert int(again_ratings.loc['T30', 'eligible_count']) == 30


def test_percentile_withheld_for_singleton_and_small_universe():
    days = full_calendar()
    one = compute_relative_strength(
        _frame({'ONLY': _flat(days, 110.0)}, {'ONLY': 'equities'}),
        as_of=AS_OF,
        universe='stocks',
        min_universe=1,
        include_rs_line=False,
    )
    assert one.summary['eligible_count'] == 1
    assert pd.isna(one.ratings.loc[0, 'rs_proxy'])
    assert 'min_universe_raised_to_2' in one.summary['warnings'][0]

    series = {f'S{i}': _flat(days, 100.0 + i) for i in range(5)}
    classes = {ticker: 'equities' for ticker in series}
    small = compute_relative_strength(
        _frame(series, classes), as_of=AS_OF, universe='stocks', include_rs_line=False,
    )
    assert small.summary['eligible_count'] == 5
    assert small.ratings['rs_proxy'].isna().all()
    assert small.ratings['raw_score'].notna().all()
    assert (small.ratings['eligible'] == True).all()  # noqa: E712 — boolean column
    assert small.ratings['status'].eq('eligible').all()
    assert any(item.startswith('universe_too_small') for item in small.summary['warnings'])
    assert small.summary['broad_market_rating'] is False
    assert 'watchlist' not in small.summary['rating_label']
    assert small.summary['rating_label'] == 'transparent_proxy_not_official_ibd_rs'


def test_short_history_is_not_a_row_count_horizon():
    days = full_calendar()
    short_days = days[-100:]
    result = compute_relative_strength(
        _frame({'NEW': {day: 100.0 for day in short_days}}, {'NEW': 'equities'}),
        as_of=AS_OF,
        universe='stocks',
        include_rs_line=False,
    )
    row = result.ratings.iloc[0]
    assert row['reason'] == 'short_history'
    assert pd.isna(row['raw_score'])
    assert pd.isna(row['rs_proxy'])
    assert row['first_session'] == short_days[0].isoformat()
    anchor_252 = result.summary['lookback_anchor_sessions']['252']
    assert anchor_252 < row['first_session']
    assert '252' in row['reason_detail']
    # 63 sessions are inside the 100 printed sessions, so that return exists.
    # 252 does not. The missing anchor is not replaced by the first printed row.
    assert row['r_63'] == pytest.approx(0.0)
    assert pd.isna(row['r_252'])


def test_gappy_series_keeps_nyse_anchor_and_withholds_score():
    days = full_calendar()
    series = _flat(days, 100.0)
    series[days[-1 - 63]] = 50.0
    series[days[-1 - 73]] = 80.0
    for day in days[-11:-1]:
        del series[day]
    result = compute_relative_strength(
        _frame({'GAP': series}, {'GAP': 'equities'}),
        as_of=AS_OF,
        universe='stocks',
        include_rs_line=False,
    )
    row = result.ratings.iloc[0]
    assert row['reason'] == 'gappy'
    assert int(row['missing_sessions_in_window']) == 10
    assert pd.isna(row['raw_score'])
    # Calendar anchor t-63 is the 50 print. A 10-row shift would use the 80 print (return 0.25).
    assert row['r_63'] == pytest.approx(1.0)
    assert row['r_63'] != pytest.approx(100 / 80 - 1)


def test_missing_anchor_is_not_forward_filled():
    days = full_calendar()
    series = _flat(days, 100.0)
    anchor = days[-1 - 63]
    series[days[-1 - 62]] = 80.0
    series[days[-1 - 64]] = 80.0
    del series[anchor]
    result = compute_relative_strength(
        _frame({'HOLE': series}, {'HOLE': 'equities'}),
        as_of=AS_OF,
        universe='stocks',
        include_rs_line=False,
    )
    row = result.ratings.iloc[0]
    assert row['reason'] == 'missing_price'
    assert pd.isna(row['r_63'])
    assert pd.isna(row['raw_score'])
    assert row['r_126'] == pytest.approx(0.0)
    assert anchor.isoformat() in row['reason_detail']


def test_stale_and_invalid_prices_are_explicit():
    days = full_calendar()
    stale_series = _flat(days, 50.0)
    del stale_series[days[-1]]
    invalid = {
        'ZERO': _flat(days, 0.0),
        'NEG': _flat(days, -5.0),
        'INF': _flat(days, float('inf')),
        'STALE': stale_series,
        'GOOD': _flat(days, 110.0),
    }
    classes = {ticker: 'equities' for ticker in invalid}
    result = compute_relative_strength(
        _frame(invalid, classes), as_of=AS_OF, universe='stocks', min_universe=2, include_rs_line=False,
    )
    reasons = result.ratings.set_index('ticker')['reason']
    assert reasons['ZERO'] == 'invalid_price'
    assert reasons['NEG'] == 'invalid_price'
    assert reasons['INF'] == 'invalid_price'
    assert reasons['STALE'] == 'stale'
    assert 'not forward-filled' in result.ratings.set_index('ticker').loc['STALE', 'reason_detail']
    assert pd.isna(result.ratings.set_index('ticker').loc['STALE', 'r_63'])
    assert reasons['GOOD'] == 'ok'


def test_duplicate_session_prints_are_rejected_and_identical_repeats_are_not():
    days = full_calendar()
    base = _frame({'DUP': _flat(days, 110.0), 'SAME': _flat(days, 120.0)}, {'DUP': 'equities', 'SAME': 'equities'})
    extra = pd.DataFrame([
        {'date': AS_OF, 'ticker': 'DUP', 'close': 999.0, 'asset_class': 'equities'},
        {'date': AS_OF, 'ticker': 'SAME', 'close': 120.0, 'asset_class': 'equities'},
    ])
    result = compute_relative_strength(
        pd.concat([base, extra], ignore_index=True),
        as_of=AS_OF,
        universe='stocks',
        min_universe=2,
        include_rs_line=False,
    )
    reasons = result.ratings.set_index('ticker')['reason']
    assert reasons['DUP'] == 'duplicate_session_prints'
    assert reasons['SAME'] == 'ok'


def test_future_sessions_and_weekend_as_of_do_not_change_the_score():
    days = full_calendar()
    series = {'AAA': _flat(days, 125.0), 'BBB': _flat(days, 110.0)}
    classes = {'AAA': 'equities', 'BBB': 'equities'}
    baseline = compute_relative_strength(
        _frame(series, classes), as_of=AS_OF, universe='stocks', min_universe=2, include_rs_line=False,
    )
    future = date(2024, 7, 1)  # next NYSE session
    dirty = _frame(series, classes, extra=[
        {'date': future, 'ticker': 'AAA', 'close': 10000.0, 'asset_class': 'equities'},
        {'date': date(2024, 6, 30), 'ticker': 'AAA', 'close': 1.0, 'asset_class': 'equities'},  # Sunday
    ])
    later = compute_relative_strength(
        dirty, as_of=AS_OF, universe='stocks', min_universe=2, include_rs_line=False,
    )
    base_row = baseline.ratings.set_index('ticker').loc['AAA']
    later_row = later.ratings.set_index('ticker').loc['AAA']
    assert later_row['raw_score'] == pytest.approx(base_row['raw_score'])
    assert int(later_row['rs_proxy']) == int(base_row['rs_proxy'])
    assert later.summary['ignored_non_session_rows'] >= 1
    assert later.summary['as_of_session'] == '2024-06-28'

    saturday = compute_relative_strength(
        dirty, as_of=date(2024, 6, 29), universe='stocks', min_universe=2, include_rs_line=False,
    )
    assert saturday.summary['stale'] is False
    assert saturday.summary['as_of_session'] == '2024-06-28'
    assert saturday.summary['as_of_requested'] == '2024-06-29'
    assert saturday.ratings.set_index('ticker').loc['AAA', 'raw_score'] == pytest.approx(base_row['raw_score'])


def test_stale_calendar_uses_last_present_session_without_filling():
    days = full_calendar()
    result = compute_relative_strength(
        _frame({'AAA': _flat(days, 125.0)}, {'AAA': 'equities'}),
        as_of=date(2024, 7, 15),
        universe='stocks',
        include_rs_line=False,
    )
    assert result.summary['stale'] is True
    assert result.summary['as_of_session'] == '2024-06-28'
    assert any(item.startswith('stale_as_of') for item in result.summary['warnings'])
    assert result.ratings.iloc[0]['reason'] == 'ok'


def test_rs_line_alignment_and_missing_benchmark():
    days = full_calendar()
    sessions = days[-3:]
    aaa = {sessions[0]: 100.0, sessions[1]: 200.0, sessions[2]: 150.0}
    spy = {sessions[0]: 100.0, sessions[2]: 125.0}  # middle session intentionally absent
    with_bench = compute_relative_strength(
        _frame({'AAA': aaa, 'SPY': spy}, {'AAA': 'equities', 'SPY': 'etfs'}),
        as_of=AS_OF,
        universe='stocks',
        include_rs_line=True,
    )
    line = with_bench.rs_line
    aaa_line = line[line['ticker'] == 'AAA'].set_index('session_date')
    assert list(aaa_line.index) == [day.isoformat() for day in sessions]
    assert aaa_line.loc[sessions[0].isoformat(), 'rs_line'] == pytest.approx(1.0)
    assert pd.isna(aaa_line.loc[sessions[1].isoformat(), 'rs_line'])
    assert pd.isna(aaa_line.loc[sessions[1].isoformat(), 'benchmark_close'])
    assert aaa_line.loc[sessions[2].isoformat(), 'rs_line'] == pytest.approx(150 / 125)
    assert 'not the 1-99' in aaa_line.iloc[0]['note']
    assert (line['ticker'] == 'SPY').sum() == 0
    # Short history still leaves the percentile path intact: the line is separate.
    assert with_bench.ratings.set_index('ticker').loc['AAA', 'reason'] == 'short_history'

    no_bench = compute_relative_strength(
        _frame({'AAA': aaa}, {'AAA': 'equities'}),
        as_of=AS_OF,
        universe='stocks',
        include_rs_line=True,
    )
    assert no_bench.rs_line.empty
    assert no_bench.summary['benchmark_available'] is False
    assert any(item.startswith('benchmark_unavailable') for item in no_bench.summary['warnings'])
    assert no_bench.ratings.set_index('ticker').loc['AAA', 'reason'] == 'short_history'
    assert no_bench.summary['eligible_count'] == with_bench.summary['eligible_count']


def test_unseparated_universe_is_labeled_and_unclassified_stocks_are_not_invented():
    days = full_calendar()
    series = {'STK': _flat(days, 110.0), 'QQQ': _flat(days, 150.0), 'BTCUSDT': _flat(days, 400.0)}
    labeled = compute_relative_strength(
        _frame(series, {'STK': 'equities', 'QQQ': 'etfs', 'BTCUSDT': 'cryptos'}),
        as_of=AS_OF,
        universe='unseparated',
        min_universe=2,
        include_rs_line=False,
    )
    reasons = labeled.ratings.set_index('ticker')['reason']
    assert reasons['STK'] == 'ok'
    assert reasons['QQQ'] == 'ok'
    assert reasons['BTCUSDT'] == 'crypto_excluded'
    assert labeled.summary['broad_market_rating'] is False
    assert any(item.startswith('stock_etf_not_separated') for item in labeled.summary['warnings'])
    assert int(labeled.ratings.set_index('ticker').loc['QQQ', 'rs_proxy']) == 99

    unlabeled = compute_relative_strength(
        pd.DataFrame({
            'date': days,
            'ticker': 'STK',
            'close': 100.0,
        }),
        as_of=AS_OF,
        universe='stocks',
        include_rs_line=False,
    )
    assert unlabeled.ratings.iloc[0]['reason'] == 'unclassified'
    assert unlabeled.summary['eligible_count'] == 0
    assert any(item.startswith('classification_unavailable') for item in unlabeled.summary['warnings'])


def test_input_frame_is_not_modified_and_intraday_bars_are_rejected():
    days = full_calendar()
    frame = _frame({'AAA': _flat(days, 110.0), 'BBB': _flat(days, 120.0)}, {'AAA': 'equities', 'BBB': 'equities'})
    frame['note'] = 'keep'
    frame.index = range(10, 10 + len(frame))
    snapshot = frame.copy(deep=True)
    compute_relative_strength(frame, as_of=AS_OF, universe='stocks', min_universe=2, include_rs_line=False)
    pd.testing.assert_frame_equal(frame, snapshot)

    intraday = pd.DataFrame({
        'datetime': ['2024-06-28 20:00:00+00:00'],
        'symbol': ['AAA'],
        'Close': [10.0],
        'asset_class': ['equities'],
    })
    with pytest.raises(PriceSchemaError, match='intraday'):
        compute_relative_strength(intraday, as_of=AS_OF, universe='stocks')


def test_utc_midnight_daily_bar_keeps_its_session_label():
    # ohlcv_max_daily stores yfinance daily bars at UTC midnight. Converting
    # that instant to New York would move the session to the previous date.
    frame = pd.DataFrame({
        'datetime': [pd.Timestamp('2024-06-28 00:00:00', tz='UTC')],
        'ticker': ['AAA'],
        'close': [10.0],
        'asset_class': ['equities'],
    })
    result = compute_relative_strength(frame, as_of=AS_OF, universe='stocks', include_rs_line=False)
    assert result.summary['as_of_session'] == '2024-06-28'
    assert result.ratings.iloc[0]['last_session'] == '2024-06-28'
    assert result.summary['adjustment'] == 'unknown'
    assert 'not labeled split-adjusted' in result.summary['adjustment_note']


def test_nyse_wrapper_matches_the_calendar_library():
    days = mcal_sessions(date(2024, 1, 1), date(2024, 1, 31))
    assert nyse_sessions(date(2024, 1, 1), date(2024, 1, 31)) == days
    assert date(2024, 1, 1) not in days  # New Year's Day


def test_package_modules_do_not_import_pipeline_or_brokers():
    forbidden = {
        'yfinance', 'requests', 'robinhood_client', 'alpaca', 'vectorbtpro', 'dotenv',
        'extraction', 'strategy', 'utils', 'binance', 'tradingview_screener',
    }
    for path in (REPO / 'relative_strength').glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split('.')[0]]
            else:
                continue
            assert forbidden.isdisjoint(names), f'{path.name} imports {names}'
        if path.name == 'io.py':
            for node in tree.body:
                if isinstance(node, ast.Import):
                    assert all(alias.name.split('.')[0] != 'duckdb' for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert node.module != 'duckdb'


def test_pipeline_modules_do_not_reference_the_calculator():
    for path in REPO.rglob('*.py'):
        if 'relative_strength' in path.parts or '.venv' in path.parts:
            continue
        text = path.read_text(encoding='utf-8')
        assert 'relative_strength' not in text, path
