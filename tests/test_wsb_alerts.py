"""Unit tests for WSB ntfy classification (no Playwright / Reddit)."""

from extraction.wsb import classify_wsb_alerts, WSB_EMPTY_STREAK_ALERT_AFTER


SOFT = [
    'mentions_v2: 0 rows for 2026-11-10 hour=22',
    'positions_v2: 0 rows for 2026-11-10 hour=22',
    'sentiment: no row for 2026-11-10 hour=22',
]


def test_hard_missing_db_alerts():
    alert, log_only = classify_wsb_alerts(['data/wsb.duckdb does not exist'])
    assert alert and not log_only


def test_tab_click_failure_alerts_immediately():
    alert, log_only = classify_wsb_alerts(
        SOFT,
        capture={'data_tab_ok': False, 'sentiment_tab_ok': False},
        empty_streak=1,
    )
    assert not log_only
    assert any('could not click Data tab' in a for a in alert)


def test_widget_empty_markers_are_log_only():
    alert, log_only = classify_wsb_alerts(
        SOFT,
        capture={
            'data_tab_ok': True,
            'sentiment_tab_ok': True,
            'portfolio_empty_widget': True,
            'mentions_empty_widget': True,
        },
        empty_streak=1,
    )
    assert alert == []
    assert log_only == SOFT


def test_single_ambiguous_empty_is_log_only():
    alert, log_only = classify_wsb_alerts(
        SOFT,
        capture={
            'data_tab_ok': True,
            'sentiment_tab_ok': True,
            'portfolio_empty_widget': False,
            'mentions_empty_widget': False,
        },
        empty_streak=1,
    )
    assert alert == []
    assert log_only == SOFT


def test_sustained_empty_streak_alerts():
    alert, log_only = classify_wsb_alerts(
        SOFT,
        capture={
            'data_tab_ok': True,
            'sentiment_tab_ok': True,
            'portfolio_empty_widget': False,
            'mentions_empty_widget': False,
        },
        empty_streak=WSB_EMPTY_STREAK_ALERT_AFTER,
    )
    assert not log_only
    assert any('consecutive UTC hours' in a for a in alert)


def test_no_soft_issues_means_no_alerts():
    assert classify_wsb_alerts([]) == ([], [])
