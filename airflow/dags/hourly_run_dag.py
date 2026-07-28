"""Airflow replacement for hourly_run.py — WSB widget data snapshot."""

import os

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

REPO = os.environ['ATS_REPO_HOST_PATH']
MOUNTS = [
    Mount(source=f'{REPO}/data', target='/app/data', type='bind'),
    Mount(source=f'{REPO}/logs', target='/app/logs', type='bind'),
    Mount(source=f'{REPO}/.env', target='/app/.env', type='bind'),
]

with DAG(
    dag_id='hourly_run',
    description='WSB widget data snapshot',
    schedule='@hourly',
    start_date=pendulum.datetime(2026, 1, 1, tz='America/New_York'),
    catchup=False,
    default_args={'owner': 'ats-trading', 'retries': 1},
    tags=['ats-trading'],
) as dag:

    DockerOperator(
        task_id='wsb',
        image='ats-worker:latest',
        command=['airflow_tasks.py', 'wsb'],
        api_version='auto',
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        mounts=MOUNTS,
        mount_tmp_dir=False,
    )
