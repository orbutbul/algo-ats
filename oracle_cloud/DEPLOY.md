# Deploying the pipeline to Oracle Cloud

The pipeline can run on an Oracle Cloud Always Free VM so it keeps running when your local
machine is off. The VM is treated as a **lean, ephemeral buffer**: it collects new news/OHLCV
data between syncs; you pull that data down and merge it into your local `data/*.duckdb` files
whenever you want (see "Sync and wipe" below), then wipe the cloud copy. Local
`data/ohlcv.duckdb` stays the permanent historical store — the VM never needs to hold more than
one sync interval's worth of data. WSB widget data is collected separately, locally (§5) — it
never runs on the cloud.

**Two deploy paths, same pipeline code:**
- **E2.1.Micro + native venv/cron (default, this doc's primary path)** — AMD x86_64, 1 OCPU/1GB,
  always available (no capacity queue). No Docker, no Airflow, no Postgres: `hourly_run.py` and
  `daily_run.py` at the repo root already are complete, self-contained orchestrators (same
  `extraction.*` calls the Airflow DAGs make, same `validation.py` alerting) — cron just calls
  them directly. This is the lighter path and fits 1GB (confirmed: ~560MB peak used memory
  during an hourly run, out of 954MB total).
- **A1.Flex + Docker/Airflow (§4 below)** — ARM, 2 OCPU/12GB, higher spec, but the free Ampere
  pool has been observed out of capacity for 5+ hours straight across every AD in a region. Use
  this path if/when you actually have A1 capacity and want the Airflow UI + DAG semantics.

**WSB widget data stays on your local machine, not the cloud, either path.** Reddit
network-blocks the Playwright-based scraper's requests from cloud datacenter IPs (confirmed: a
403 "You've been blocked by network security" response when tested from the Oracle VM — not a
timing/rendering issue, and there's no quick fix since Reddit closed self-serve OAuth app
registration in 2026). `wsb_run.py` (repo root) runs it locally on a schedule instead — see
"Local: WSB scraping" below. `hourly_run.py` on the cloud VM only covers Benzinga news now.

## 1. Oracle Cloud console setup

1. **Sign up / log in** at oracle.com/cloud/free. Always Free resources are never billed (a
   card is required for identity verification only).
2. **Create a Virtual Cloud Network (VCN)**: Console → *Networking → Virtual Cloud Networks →
   Start VCN Wizard → "Create VCN with Internet Connectivity"*. Accept the defaults (public
   subnet, internet gateway, route table included). Name it e.g. `ats-trading-vcn`.
3. **Lock down the security list**: Console → *Networking → Virtual Cloud Networks → your VCN →
   Security Lists → Default Security List* → Ingress Rules:
   - Keep only **TCP/22 (SSH)** from `0.0.0.0/0` (or narrow it to your home IP if it's static).
   - Do **not** add a rule for port 8080 — if you use the Docker/Airflow path (§4), its webserver
     is reached via SSH tunnel only, never opened on the public IP.
4. **Create the compute instance**: Console → *Compute → Instances → Create Instance*.
   - Name: e.g. `ats-trading`.
   - Image and shape → Edit → **VM.Standard.E2.1.Micro** (AMD) for the default path, or
     **Ampere → VM.Standard.A1.Flex → 2 OCPUs / 12GB memory** for the Docker/Airflow path (§4).
   - Image: **Ubuntu** (latest LTS, e.g. 24.04).
   - Networking: the VCN/subnet from step 2; keep "Assign a public IPv4 address" checked.
   - SSH keys: paste an existing public key, or let Oracle generate a pair and **download the
     private key immediately** (offered once only).
   - Boot volume: default (~50GB) is enough — `data/` stays small on the VM by design.
   - Click **Create**; note the **public IP** once the instance is `RUNNING`.

   **If you hit "Out of host capacity"** (mainly an A1.Flex problem — E2.1.Micro is essentially
   always available): this is a well-known, widely-reported Always Free A1 issue — the free
   Ampere pool is oversubscribed in many regions, with no published ETA for when it frees up.
   Rather than clicking Create repeatedly by hand, use `oracle_cloud/oracle_provision.py`:
   ```
   pip install oci
   oci setup config    # one-time: generates an API key pair, writes ~/.oci/config --
                        # upload the printed public key under Identity -> My Profile -> API Keys

   python oracle_cloud/oracle_provision.py \
     --compartment-id <tenancy or compartment OCID> \
     --subnet-id <public subnet-ats-trading-vcn OCID> \
     --ssh-key-file ~/.ssh/ats_trading.pub
   ```
   Defaults to `--shape VM.Standard.E2.1.Micro`; pass `--shape VM.Standard.A1.Flex` to try for
   A1 instead. It cycles through every availability domain in the region each pass, retries both
   "out of host capacity" and 429 rate-limiting (Oracle throttles repeated launch attempts after
   a few hours of polling — backs off 15 minutes and resumes automatically), and aborts
   immediately on anything else (bad OCID, quota — not worth looping on). On success it sends a
   push notification via the same `validation.send_alert`/ntfy.sh mechanism the pipeline already
   uses. Leave it running in a background terminal for as long as you're willing to wait — if
   your machine sleeps or the terminal closes, just relaunch the same command.
5. **First SSH check**: `ssh -i /path/to/key ubuntu@<public-ip>`.
6. **(Recommended) Reserve the public IP**: Console → *Networking → IP Management → Reserved
   Public IPs* → create one, attach it to the instance's VNIC, so it survives stop/start.

## 2. Deploy the pipeline (native venv + cron)

1. **Install Python, venv, git, and cron** on the VM: `sudo apt update && sudo apt install -y
   python3.12-venv git cron && sudo systemctl enable --now cron` (Ubuntu 24.04 ships Python 3.12
   by default — same version the Docker image already used, `apache/airflow:2.10.4-python3.12`
   — but not `cron`, which this minimal cloud image doesn't include out of the box).
2. **Transfer the repo and secrets** (none of this comes from `git clone` — `.env`,
   `.credentials.json`, and `data/` are all gitignored):
   - `git clone` the repo on the VM (or `tar`/`scp` it from local if the GitHub repo is private
     and you'd rather not set up VM-side git auth just for this).
   - Copy `.env` (fill from `.env.example` at the repo root) to the VM's repo root, with
     `CLAUDE_CREDENTIALS_PATH` changed to a Linux path, e.g.
     `/home/ubuntu/.claude/.credentials.json`.
   - Copy `.credentials.json` itself to that path.
   - Do **not** copy `data/` — the VM starts empty and only ever holds one sync interval's
     worth of data (see "Sync and wipe" below).
3. **Create the venv and install dependencies**:
   ```
   cd ~/ATS_trading
   python3.12 -m venv venv
   venv/bin/pip install -r oracle_cloud/requirements.txt
   ```
   Uses `oracle_cloud/requirements.txt`, not `airflow/requirements.txt` — same dependency list
   minus `vectorbtpro`. The core pipeline doesn't need it: `extraction/ohlcv.py`'s crypto fetch
   was ported from vectorbtpro's `BinanceData` wrapper to plain `python-binance` (already a
   dependency) specifically so this deploy needs no private-repo GitHub PAT at all.
4. **(Optional) Install Playwright's Chromium browser.** `playwright` itself must be installed
   (already covered by step 3 — `validation.py` imports `extraction/wsb.py` at module level,
   which imports `playwright.sync_api`), but the actual Chromium *binary* is never launched on
   the cloud VM — WSB scraping runs locally instead (see "Local: WSB scraping" below), because
   Reddit network-blocks the scraper from cloud datacenter IPs. Skip this step unless you have
   another reason to run a browser here:
   ```
   sudo venv/bin/playwright install-deps chromium
   venv/bin/playwright install chromium
   ```
5. **Add a 2GB swap file** — cheap insurance against OOM-kills on a 1GB box (a memory spike,
   e.g. Chromium under load, would otherwise kill the process outright):
   ```
   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
6. **Verify manually before trusting cron**: `venv/bin/python hourly_run.py`. Confirm it
   completes and writes real rows (`venv/bin/python -c "import duckdb;
   print(duckdb.connect('data/news.duckdb').execute('SELECT count(*) FROM benzinga_news').fetchone())"`).
   Watch `free -h` during this run to confirm memory headroom on this shape.
7. **Add cron entries** (`crontab -e`), each serialized with `flock` so an hourly and daily run
   can never overlap on this 1GB box (neither touches Chromium anymore now that WSB stays
   local, but `daily_run.py`'s OHLCV/fundamentals fetch is still worth not doubling up on):
   ```
   0 * * * *     flock -n /tmp/ats_hourly.lock -c 'cd ~/ATS_trading && venv/bin/python hourly_run.py >> logs/cron.log 2>&1'
   5 17 * * 1-5  flock -n /tmp/ats_daily.lock  -c 'cd ~/ATS_trading && venv/bin/python daily_run.py  >> logs/cron.log 2>&1'
   ```
   Daily is offset to :05 (not :00) so it can never land in the same minute as the hourly job.
   Both scripts already write their own logs and call `validation.send_alert` on failure — same
   alerting behavior Airflow's `on_failure_callback` gave, just driven by the scripts' own
   try/except blocks instead of a scheduler.
8. **Confirm cron actually fires**: check `logs/cron.log` after the next scheduled hour.

## 3. Sync and wipe (run whenever you want, from your local machine)

`oracle_cloud/cloud_sync.py` pulls the cloud's accumulated data into your local `data/*.duckdb`
files, verifies the merge, then wipes the cloud copy — keeping the cloud VM lean indefinitely
regardless of sync cadence, while local history only ever grows.

```
python oracle_cloud/cloud_sync.py --host <public-ip> --key /path/to/private_key
```

(Add `--docker` only if you're syncing against the A1 + Docker/Airflow path from §4 instead —
default assumes the native venv/cron deploy from §2.)

What it does, in order (`data/wsb.duckdb` isn't part of this — WSB never runs on the cloud, see
§5):
1. **Pull**: `rsync`/`scp` `data/news.duckdb`, `data/ohlcv.duckdb` from the VM to a local temp
   dir.
2. **Merge**: `ATTACH`es each pulled file next to the matching local file and upserts every
   table using the same dedup keys the extraction modules already define
   (`extraction/news.py::_TABLES`, `(datetime, ticker)` for the OHLCV tables) — no rows are
   duplicated or lost.
3. **Verify**: compares pulled vs. now-present-locally row counts per table; aborts before
   wiping anything if they don't reconcile.
4. **Wipe**: only after a verified merge, deletes the data *tables* on the VM (not the whole
   files) via SSH (`venv/bin/python oracle_cloud/remote_wipe.py ...`, or the Docker exec equivalent
   with `--docker`). It explicitly leaves `ohlcv_robinhood_last_run.txt`,
   `ohlcv_robinhood_progress.json`, `ohlcv_last_run.txt`, and `news_last_run.txt` untouched, so
   the cloud pipeline's incremental fetchers resume from where they left off instead of
   re-fetching full history next run.

No daemon or listener runs on either end — this is entirely driven by you running the script.

## 4. Alternative: A1.Flex + Docker/Airflow

Use this instead of §2 if you actually have A1.Flex capacity and want the Airflow UI + DAG
retry/monitoring semantics — `airflow/docker-compose.yaml` and `airflow/Dockerfile` are already
built for it (2 OCPU/12GB is enough headroom for Postgres + Airflow scheduler/webserver +
headless Chromium together, with less margin than pre-2026 A1 allowances, so keep an eye on
memory if you add more DAGs later):

1. **Install Docker + the Compose plugin** on the VM (standard Ubuntu Docker install:
   `curl -fsSL https://get.docker.com | sh`, then add your user to the `docker` group).
2. **Transfer the repo and secrets**, as in §2 step 2, plus also copy `airflow/.env`
   (compose-local vars: `CLAUDE_CREDENTIALS_HOST_PATH`, `POSTGRES_PASSWORD`,
   `AIRFLOW_ADMIN_PASSWORD` — see `airflow/docker-compose.yaml` for how they're used) to the VM,
   with `CLAUDE_CREDENTIALS_HOST_PATH` changed to a Linux path.
3. **Build the image**, supplying the GitHub PAT the Dockerfile needs for the private
   `vectorbtpro` install:
   ```
   DOCKER_BUILDKIT=1 docker compose -f airflow/docker-compose.yaml build \
     --build-arg BUILDKIT_INLINE_CACHE=1 \
     --secret id=gh_pat,src=/path/to/pat_file
   ```
   (a plain `docker compose build` won't pass `--secret`; use `docker buildx bake` or build the
   image directly with `docker build --secret ...` and matching build context/tags if your
   Compose version doesn't support `--secret` passthrough.)
4. **Bring the stack up**: `docker compose -f airflow/docker-compose.yaml up -d`. Check
   `docker compose ps` shows all 4 services healthy.
5. **Reach the Airflow UI** (only via tunnel, since 8080 isn't publicly exposed):
   `ssh -L 8080:localhost:8080 ubuntu@<public-ip>`, then open `http://localhost:8080` locally.
6. **Trigger and verify the hourly DAG** (news only — see "Local: WSB scraping" below for why
   WSB isn't part of this DAG):
   ```
   docker compose -f airflow/docker-compose.yaml exec airflow-scheduler airflow dags trigger hourly_run
   ```

## 5. Local: WSB scraping (`wsb_run.py`)

Runs on your own machine, not the cloud, on either deploy path — see the note near the top of
this doc for why (Reddit blocks the scraper from cloud datacenter IPs; confirmed via a direct
403 test from the Oracle VM, not just a timeout/rendering fluke). `wsb_run.py` (repo root) is
the same logic `hourly_run.py` used to run, split out on its own.

**Windows Task Scheduler** (this restores the setup the pipeline used before it moved to
Airflow):
1. Open Task Scheduler → *Create Task*.
2. General: name it e.g. `ATS WSB Hourly`; "Run whether user is logged on or not" if you want it
   to fire even when locked.
3. Triggers → New → *Daily*, recur every 1 day, **Repeat task every: 1 hour**, for a duration of
   *Indefinitely*.
4. Actions → New → Program: path to your venv's `pythonw.exe` (no console window) — e.g.
   `C:\Users\<you>\...\ATS_trading\.venv\Scripts\pythonw.exe`; Arguments: `wsb_run.py`; Start in:
   the repo root (`C:\Users\<you>\...\ATS_trading`).
5. Save. Check `logs\wsb_run.log` after the next hour to confirm it fired.
