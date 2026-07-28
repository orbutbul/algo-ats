"""
Airflow replacement for daily_run.py — one task per pipeline step, run in the
ats-worker container. Build the image once before enabling this DAG:

    DOCKER_BUILDKIT=1 docker build --secret id=gh_pat,env=GH_PAT \
        -t ats-worker -f airflow/worker/Dockerfile .
"""

import os
from datetime import datetime

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

REPO = os.environ['ATS_REPO_HOST_PATH']  # host path, not the scheduler-container path
MOUNTS = [
    Mount(source=f'{REPO}/data', target='/app/data', type='bind'),
    Mount(source=f'{REPO}/logs', target='/app/logs', type='bind'),
    Mount(source=f'{REPO}/.env', target='/app/.env', type='bind'),
]

default_args = {
    'owner': 'ats-trading',
    'retries': 1,
}

with DAG(
    dag_id='daily_run',
    description='Symbol screening -> 1-min OHLCV -> fundamentals',
    schedule='0 17 * * 1-5',
    start_date=pendulum.datetime(2026, 1, 1, tz='America/New_York'),
    catchup=False,
    default_args=default_args,
    tags=['ats-trading'],
) as dag:

    common = dict(
        image='ats-worker:latest',
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=MOUNTS,
        mount_tmp_dir=False,
        do_xcom_push=True,
        xcom_all=False,  # XCom = last line of stdout
    )

    screen = DockerOperator(
        task_id='screen',
        command=['airflow_tasks.py', 'screen'],
        **common,
    )

    ohlcv = DockerOperator(
        task_id='ohlcv',
        command=[
            'airflow_tasks.py', 'ohlcv',
            "{{ ti.xcom_pull(task_ids='screen') }}",
        ],
        **common,
    )

    fundamentals = DockerOperator(
        task_id='fundamentals',
        command=[
            'airflow_tasks.py', 'fundamentals',
            "{{ ti.xcom_pull(task_ids='screen') }}",
        ],
        **common,
    )

    # Mirrors daily_run.py: both downstream steps depend only on screening,
    # not on each other, and a market-holiday skip is handled inside
    # airflow_tasks.cmd_ohlcv itself (not a DAG-level branch).
    screen >> [ohlcv, fundamentals]
