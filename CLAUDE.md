# ATS_trading

Automated trading system: research and backtest quantitative equity/crypto strategies,
then run the winners live. Pipeline = data extraction (WSB sentiment, news, OHLCV,
fundamentals) → research/backtest → strategy signal generation → execution (Robinhood/Alpaca)
→ monitoring, orchestrated by Docker + Airflow (`airflow/docker-compose.yaml`), with DuckDB
as the primary data store (`data/*.duckdb`).

## Read before doing anything else

- `oracle_cloud/` — cloud deployment tooling; `DEPLOY.md` has the deploy steps.
- Data pipeline runs on Docker + Airflow, not Windows Task Scheduler. Active DAGs live in
  `airflow/dags/`. Root-level `hourly_run.py` / `daily_run.py` are pre-Airflow leftovers —
  treat as dead code unless told otherwise.
- Live credentials (`.credentials.json`) are bind-mounted into containers and
  auto-refreshed in place by `robinhood_client.py` — never delete or hand-edit that file.

## Backtesting: use VectorBT Pro

VectorBT Pro (`vectorbtpro`, imported as `vbt`) is the standard tool for indicators,
signal generation, portfolio simulation, and performance analysis in this repo. Default to
it for any backtest, parameter sweep, or performance-metric work instead of hand-rolled
pandas loops or a different backtesting library.

- Invoke the `vbt` skill for anything VBT-related (building/debugging indicators,
  portfolios, signal pipelines) — it knows the API surface and this repo's conventions.
- The `mcp__vectorbtpro__*` tools (search, get_source, get_attrs, run_code, find) give
  live access to the installed VBT Pro source/docs — prefer them over guessing API shapes
  or trusting stale training data, since VBT Pro's API changes across versions and this repo
  pins a specific commit (`pyproject.toml` → `vectorbtpro` git rev).
- Vectorize: avoid Python-level loops over bars/symbols where a vectorized VBT/pandas/numpy
  operation exists. Use `vbt.Portfolio.from_signals` (or similar) rather than manual PnL loops.
- New strategies belong in `strategy/` (see `strategy/base.py` for the shared interface) and
  exploratory research in dated notebooks under `research/`.

## Code style

- Every module/script gets a short file-header docstring: what it does, its role in the
  pipeline (extraction / research / strategy / execution / validation), and non-obvious
  inputs or outputs (e.g. which DuckDB table it reads/writes).
- Comments are concise and explain *why*, not what — skip narrating obvious code.
- No speculative abstractions, feature flags, or unused parameters "for future use."
- Match existing patterns in `extraction/` and `strategy/` (e.g. `save_x_data`, `download_x`,
  `validate_x` naming) rather than inventing new conventions.

## Things that improve accuracy here

- Check `validation.py` for the validation function matching any data domain you touch
  (`validate_wsb`, `validate_ohlcv`, `validate_fundamentals`, `validate_news`,
  `validate_screen`) — pipeline changes should keep these passing.
- Tests live in `regime/tests` (pytest, `pyproject.toml` sets `testpaths`). Run
  `uv run pytest` before claiming a regime/indicator change works.
- Before trusting a claim about "what's scheduled" or "what runs the pipeline," check
  `airflow/dags/*.py` directly rather than assuming from file location — orphaned root
  scripts look active but aren't.
- This is real-money-adjacent code (live Robinhood/Alpaca execution). Never loosen risk
  checks, position sizing, or validation gates without being asked, and flag anything that
  looks like it would place a real order during a backtest/research run.
