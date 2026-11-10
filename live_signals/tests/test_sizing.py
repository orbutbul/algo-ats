"""Sizing curve, caps, and the worked examples from DESIGN.md."""

from live_signals.config import config_with
from live_signals.models import Exposure
from live_signals.sizing import conviction, size_target_with_cluster


def _size(cfg, score, atr, **overrides):
    args = dict(
        score=score, credibility=1.0, session="rth", equity=15_000, price=100.0,
        atr=atr, direction="long", ticker="AAPL", sector=None, exposures=[],
        correlations={}, trades_today=0, new_risk_today=0.0, cfg=cfg,
    )
    args.update(overrides)
    return size_target_with_cluster(**args)


def test_conviction_is_monotonic_and_pinned(cfg):
    scores = [70, 75, 80, 85, 90, 95, 100]
    values = [conviction(score, cfg) for score in scores]
    assert values[0] == 0.15
    assert values[1] > values[0]
    assert values[2] == pytest_approx(0.473252)
    assert values[4] == pytest_approx(0.891028)
    assert values[5] == 1.0
    assert values[6] == 1.0
    assert values == sorted(values)
    assert conviction(69, cfg) == 0.0


def pytest_approx(value):
    import pytest
    return pytest.approx(value, abs=1e-4)


def test_high_vol_shares_rise_with_score(cfg):
    # ATR $4 on a $100 name. Stop = 1.5 * 4 = $6. The 12% cap does not bind.
    qtys = [_size(cfg, score, 4.0).qty for score in (70, 75, 80, 85, 90, 95)]
    assert qtys == sorted(qtys)
    assert len(set(qtys)) == len(qtys)
    assert _size(cfg, 70, 4.0).qty == 2
    assert _size(cfg, 80, 4.0).qty == 7
    assert _size(cfg, 90, 4.0).qty == 13
    assert all(isinstance(qty, int) for qty in qtys)


def test_low_vol_hits_the_position_cap(cfg):
    # ATR $1.20. Stop = $1.80. Score 70 stays inside the cap; 80 and 90 pin at 12%.
    low_70 = _size(cfg, 70, 1.2)
    low_80 = _size(cfg, 80, 1.2)
    low_90 = _size(cfg, 90, 1.2)
    assert low_70.qty == 7
    assert low_70.binding == "risk_budget"
    assert low_80.qty == 18
    assert low_90.qty == 18
    assert low_90.binding == "position_cap"
    assert low_90.notional == 1800
    assert low_70.qty < low_80.qty == low_90.qty


def test_credibility_and_session_shrink_size(cfg):
    full = _size(cfg, 90, 4.0)
    thin = _size(cfg, 90, 4.0, credibility=0.45)
    overnight = _size(cfg, 90, 4.0, credibility=0.45, session="overnight")
    assert full.qty == 13
    assert thin.qty == 6
    assert overnight.qty == 2
    assert overnight.qty < thin.qty < full.qty
    assert _size(cfg, 90, 4.0, session="closed").qty == 0


def test_gross_net_sector_and_cluster_caps(cfg):
    gross = _size(cfg, 90, 4.0, exposures=[
        Exposure("MSFT", "long", 60, 100.0),
        Exposure("NVDA", "short", 60, 100.0),
    ])
    assert gross.qty == 7 and gross.binding == "gross"

    net = _size(cfg, 90, 4.0, exposures=[Exposure("MSFT", "long", 80, 100.0)])
    assert net.qty == 2 and net.binding == "net"

    sector = _size(
        cfg, 90, 4.0, sector="XLK",
        exposures=[Exposure("MSFT", "long", 40, 100.0, sector="XLK")],
    )
    assert sector.qty == 5 and sector.binding == "sector"

    cluster = _size(
        cfg, 90, 4.0,
        exposures=[Exposure("MSFT", "long", 30, 100.0)],
        correlations={("AAPL", "MSFT"): 0.8},
    )
    assert cluster.qty == 7 and cluster.binding == "cluster"

    # A missing correlation does not invent a cluster.
    free = _size(
        cfg, 90, 4.0,
        exposures=[Exposure("MSFT", "long", 30, 100.0)],
        correlations={},
    )
    assert free.qty == 13


def test_daily_budget_and_trade_cap(cfg):
    blocked = _size(cfg, 90, 4.0, trades_today=6)
    assert blocked.qty == 0 and blocked.binding == "max_trades"
    tight = _size(cfg, 90, 4.0, new_risk_today=290.0)
    assert tight.qty == 1 and tight.binding == "daily_risk_budget"


def test_kelly_ceiling_binds_only_when_the_vol_budget_is_raised(cfg):
    # Default base risk is 0.60%, under the 1% Kelly ceiling, so Kelly does not bind.
    assert _size(cfg, 95, 4.0).binding == "risk_budget"
    raised = config_with(base_risk_pct=0.05)
    # Price $50 so the 12% share cap (36 shares) sits above the Kelly qty.
    capped = _size(raised, 95, 4.0, price=50.0)
    assert capped.binding == "kelly"
    assert capped.qty == 25


def test_price_below_one_dollar_is_skipped(cfg):
    assert _size(cfg, 90, 0.1, price=0.5).binding == "price_below_one_dollar"
