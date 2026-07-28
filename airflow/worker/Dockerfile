# ats-worker: runs the actual pipeline code (yfinance, vectorbtpro, playwright, duckdb).
# Independent of the Airflow images — Airflow only launches this as a sibling
# container per task via DockerOperator, so Airflow's own Python/deps never
# have to match this project's.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# BuildKit secret: the GitHub PAT used to pull the private vectorbtpro repo.
# Never baked into an image layer — build with:
#   DOCKER_BUILDKIT=1 docker build --secret id=gh_pat,env=GH_PAT -t ats-worker -f airflow/worker/Dockerfile .
RUN --mount=type=secret,id=gh_pat \
    GH_PAT="$(cat /run/secrets/gh_pat)" \
    && git config --global url."https://${GH_PAT}@github.com/".insteadOf "https://github.com/" \
    && uv sync --frozen --no-dev \
    && git config --global --unset url."https://${GH_PAT}@github.com/".insteadOf

RUN uv run playwright install --with-deps chromium

COPY . .

ENTRYPOINT ["uv", "run", "python"]
