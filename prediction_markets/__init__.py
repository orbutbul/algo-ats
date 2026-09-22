"""
prediction_markets/ — venue client library for prediction markets
(Kalshi, Polymarket.com, Polymarket.us, predict.fun).

This is the on-demand, stateless research layer: "what markets does venue X
have right now" and "what has a specific market's price/order book done".
It is deliberately lightweight (sync `requests`, no persistence, no
asyncio/WebSocket) — see .claude/plans/prediction_markets_collector.md for
the separate, much heavier always-on trade-tape daemon design, which is
expected to import these same venue clients (prediction_markets.
venues.*) rather than re-implement their HTTP calls.

Usage:
    from prediction_markets.venues.kalshi import KalshiClient

    client = KalshiClient()
    markets = client.list_markets()               # DataFrame, one row/market
    history = client.get_price_history('SOME-TICKER')  # DataFrame, one row/point
    book = client.get_orderbook('SOME-TICKER')     # OrderBook -> .to_dataframe()
"""
