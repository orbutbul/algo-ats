"""live/moomoo_l2.py — exploratory script for pulling free Level 2 (order book)
data from a moomoo account via the moomoo OpenAPI.

Prerequisites (one-time, manual):
  1. Have a moomoo account with Level 2 market data enabled. For US stocks,
     moomoo gives free real-time Lv2 (10-level order book, via Nasdaq
     TotalView/UTP) to funded accounts — check "Level 2" status under
     Account > Market Data in the app if get_order_book() below returns
     no data or a permission error.
  2. Download and install OpenD (the local gateway daemon), separate from
     the moomoo trading app: https://www.moomoo.com/download/OpenAPI
  3. Launch OpenD and log in with your moomoo account credentials. Leave
     it running — the Python API talks to OpenD over localhost, not to
     moomoo's servers directly.
  4. pip install moomoo-api

Then run this file directly to sanity-check the connection:
    python -m live.moomoo_l2
"""

import datetime as dt
import logging
import time

from moomoo import OpenQuoteContext, OrderBookHandlerBase, RET_OK, SubType

log = logging.getLogger('moomoo_l2')

# Default OpenD listen address.
OPEND_HOST = '127.0.0.1'
OPEND_PORT = 11111


def get_order_book(codes: list[str], num: int = 10):
    """One-shot Level 2 order book snapshot for the given moomoo-format
    codes (e.g. 'US.AAPL'). Returns {code: order_book_dict}."""
    ctx = OpenQuoteContext(host=OPEND_HOST, port=OPEND_PORT)
    try:
        ret, err = ctx.subscribe(code_list=codes, subtype_list=[SubType.ORDER_BOOK])
        if ret != RET_OK:
            raise RuntimeError(f'subscribe failed: {err}')

        books = {}
        for code in codes:
            ret, data = ctx.get_order_book(code, num=num)
            if ret != RET_OK:
                raise RuntimeError(f'get_order_book failed for {code}: {data}')
            books[code] = data
        return books
    finally:
        ctx.close()


def _format_book(data: dict) -> str:
    lines = [f"{data['code']} ({data['name']}) @ {dt.datetime.now().isoformat(timespec='milliseconds')}"]
    lines.append('  Asks (price, size):')
    for price, size, *_ in reversed(data['Ask']):
        lines.append(f'    {price:>10.2f}  {size:>8.0f}')
    lines.append('  Bids (price, size):')
    for price, size, *_ in data['Bid']:
        lines.append(f'    {price:>10.2f}  {size:>8.0f}')
    return '\n'.join(lines)


def stream_order_book(codes: list[str], duration_s: float, out_path: str, num: int = 10) -> None:
    """Subscribes to Level 2 order book pushes for `codes` and writes every
    update received over `duration_s` seconds to a human-readable text file
    at `out_path`."""
    updates = []

    class _Handler(OrderBookHandlerBase):
        def on_recv_rsp(self, rsp_str):
            ret, data = super().on_recv_rsp(rsp_str)
            if ret != RET_OK:
                log.warning('order book push error: %s', data)
                return RET_OK, data
            updates.append(_format_book(data))
            return RET_OK, data

    ctx = OpenQuoteContext(host=OPEND_HOST, port=OPEND_PORT)
    ctx.set_handler(_Handler())
    try:
        ret, err = ctx.subscribe(code_list=codes, subtype_list=[SubType.ORDER_BOOK])
        if ret != RET_OK:
            raise RuntimeError(f'subscribe failed: {err}')

        # Seed the file with an initial snapshot so there's content even if
        # the book doesn't change during the short capture window.
        for code in codes:
            ret, data = ctx.get_order_book(code, num=num)
            if ret == RET_OK:
                updates.append(_format_book(data))

        log.info('Streaming order book for %s for %.0fs...', codes, duration_s)
        time.sleep(duration_s)
    finally:
        ctx.unsubscribe(code_list=codes, subtype_list=[SubType.ORDER_BOOK])
        ctx.close()

    with open(out_path, 'w') as f:
        f.write(f'\n\n{"-" * 60}\n\n'.join(updates))
    log.info('Wrote %d order book updates to %s', len(updates), out_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    stream_order_book(['US.AAPL', 'US.NVDA'], duration_s=10, out_path='data/moomoo_l2_stream.txt')
