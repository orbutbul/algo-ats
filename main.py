import json
from datetime import datetime, timezone
from wsb2 import get_latest_wsbapp_data, save_wsb_data

print("Fetching WSB data (this takes ~20s)...")
data = get_latest_wsbapp_data()

post_date = data['post_date']
if isinstance(post_date, datetime):
    post_date_str = post_date.strftime('%Y-%m-%d %H:%M UTC')
else:
    post_date_str = str(post_date)

print(f"\n{'='*55}")
print(f"  {data['post_title']}")
print(f"  {post_date_str}")
print(f"{'='*55}")

# --- Trending Tickers ---
print("\nTrending Tickers (Daily)")
print("  " + "  ".join(f"${t}" for t in data['trendingTickersDaily']))

print("\nTrending Tickers (Hourly)")
print("  " + "  ".join(f"${t}" for t in data['trendingTickersHourly']))

# --- Session Store ---
print(f"\nStocks  ({len(data['sessionStore'])} tracked)")
print(f"  {'Ticker':<7} {'Price':>10} {'Change':>9} {'Chg%':>7} {'Volume':>14}")
print(f"  {'-'*6} {'-'*10} {'-'*9} {'-'*7} {'-'*14}")
for s in sorted(data['sessionStore'], key=lambda x: x['ticker']):
    print(
        f"  {s['ticker']:<7} "
        f"${s['price']:>9.2f} "
        f"{s['change']:>+9.2f} "
        f"{s['change_pct']:>+6.2f}% "
        f"{int(s['volume'] or 0):>14,}"
    )

# --- Karma Trending ---
print(f"\nTop Karma Users (Daily)")
for i, u in enumerate(data['karmaTrendingDaily'], 1):
    print(f"  {i:>2}. {u['username']:<28} {int(u['karma'] or 0):>5} karma")

# --- Top Commenters ---
print(f"\nTop Commenters (Daily)")
for i, u in enumerate(data['commenterTrendingDaily'], 1):
    print(f"  {i:>2}. {u['member']:<28} score {u['score']:.1f}")

# --- Polymarket ---
if data.get('polymarketData'):
    print(f"\nPolymarket")
    for m in data['polymarketData']:
        print(f"  - {m['title']}")

# --- Export to JSON ---
def make_serializable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        return round(obj, 4)
    return obj

output_path = "wsb_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, default=make_serializable, ensure_ascii=False)

print(f"\nFull data exported to {output_path}")

# Save to daily parquet store
save_wsb_data(data)