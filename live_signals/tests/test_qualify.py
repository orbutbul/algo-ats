"""Qualification: freshness, source mix, cooldown, conflicts."""

from datetime import timedelta

from live_signals.models import Confirmation, SignalSource, TakeRecord
from live_signals.qualify import credibility_multiplier, qualify_signal, source_facts
from live_signals.tests.conftest import et, four_news, signal

NOW = et(2026, 11, 9, 11, 0)


def _q(cfg, sig, **kwargs):
    return qualify_signal(sig, NOW, cfg, **kwargs)


def test_neutral_and_low_score_and_stale(cfg):
    assert _q(cfg, signal(sentiment="neutral", score=95)).reason == "neutral"
    assert _q(cfg, signal(score=69)).reason == "score_below_min"
    assert _q(cfg, signal(minutes_ago=46)).reason == "stale"
    assert _q(cfg, signal(minutes_ago=44)).outcome == "enter"
    from dataclasses import replace
    future = replace(signal(), timestamp=NOW + timedelta(minutes=5))
    assert _q(cfg, future).reason == "future_timestamp"


def test_single_reddit_needs_a_higher_bar_or_confirmation(cfg):
    reddit = (SignalSource("https://www.reddit.com/r/stocks/x", "reddit"),)
    assert _q(cfg, signal(score=70, sources=reddit)).reason == "single_social_unconfirmed"
    assert _q(cfg, signal(score=78, sources=reddit)).reason == "single_social_unconfirmed"
    assert _q(cfg, signal(score=88, sources=reddit)).outcome == "enter"
    confirmed = Confirmation(return_since_signal=0.005, rel_volume=1.6)
    weak = _q(cfg, signal(score=78, sources=reddit), confirmation=confirmed)
    assert weak.outcome == "enter"
    assert weak.credibility == 0.55
    # The GOOGL failure mode: one Reddit post at 70, even with a small move, is not enough
    # unless the move clears 0.4% AND volume clears 1.5x. A 70 never clears the confirmed bar (78).
    assert _q(cfg, signal(score=70, sources=reddit), confirmation=confirmed).reason == "single_social_unconfirmed"


def test_social_pair_and_single_news(cfg):
    social = (
        SignalSource("https://www.reddit.com/r/stocks/x", "reddit"),
        SignalSource("https://x.com/some/status/1", "x"),
    )
    assert _q(cfg, signal(score=79, sources=social)).reason == "social_needs_confirmation"
    assert _q(cfg, signal(score=80, sources=social)).outcome == "enter"
    confirmed = Confirmation(return_since_signal=0.004, rel_volume=1.5)
    assert _q(cfg, signal(score=70, sources=social), confirmation=confirmed).outcome == "enter"
    one_news = (SignalSource("https://www.reuters.com/a", "news"),)
    assert _q(cfg, signal(score=73, sources=one_news)).reason == "thin_single_source"
    ok = _q(cfg, signal(score=74, sources=one_news))
    assert ok.outcome == "enter"
    assert ok.credibility == 0.60


def test_two_news_sources_clear_the_floor(cfg):
    sources = four_news()[:2]
    result = _q(cfg, signal(score=70, sources=sources))
    assert result.outcome == "enter"
    assert result.credibility == 0.78


def test_credibility_saturates(cfg):
    assert credibility_multiplier(1, False, False, cfg) == 0.45
    assert credibility_multiplier(4, True, False, cfg) == 1.0
    assert source_facts(signal(), cfg)[0] == 4


def test_duplicate_cooldown_and_conflict(cfg):
    assert _q(cfg, signal(), already_seen=True).reason == "duplicate"
    take = TakeRecord("AAPL", "long", 80, NOW - timedelta(minutes=30), "old")
    cooled = _q(cfg, signal(score=82, signal_id="new"), takes=(take,))
    assert cooled.reason == "cooldown"
    upgraded = _q(cfg, signal(score=88, signal_id="new"), takes=(take,))
    assert upgraded.outcome == "upgrade"
    opposite = TakeRecord("AAPL", "short", 80, NOW - timedelta(minutes=30), "old")
    blocked = _q(cfg, signal(score=82, signal_id="new"), takes=(opposite,))
    assert blocked.reason == "conflict"
    flipped = _q(cfg, signal(score=90, signal_id="new"), takes=(opposite,))
    assert flipped.outcome == "flip"


def test_unknown_source_type_is_treated_as_social(cfg):
    sources = (SignalSource("https://example.com/a", "blog"),)
    assert _q(cfg, signal(score=80, sources=sources)).reason == "single_social_unconfirmed"


def test_no_sources(cfg):
    assert _q(cfg, signal(sources=())).reason == "no_sources"
