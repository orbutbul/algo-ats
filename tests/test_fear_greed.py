"""Fear & Greed GitHub refresh: no network, just the file checks."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from extraction.fear_greed import FearGreedFetchError, update_fear_greed, validate_cnn_json


def _json_payload(day: str = '2026-11-16') -> bytes:
    stamp = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
    return json.dumps({
        'fear_and_greed': {'score': 31.0, 'rating': 'fear'},
        'fear_and_greed_historical': {'data': [{'x': stamp, 'y': 31.0, 'rating': 'fear'}]},
    }).encode()


def _csv_payload() -> bytes:
    return b'Date,Fear Greed,Rating\n2011-01-03,68,greed\n2026-11-16,31.0,fear\n'


def test_update_writes_both_files(tmp_path):
    urls = {}

    def fetch(url: str) -> bytes:
        urls[url] = True
        if url.endswith('.json'):
            return _json_payload()
        return _csv_payload()

    written = update_fear_greed(dest=tmp_path, fetch=fetch)
    by_name = {item.path.name: item for item in written}
    assert set(by_name) == {'cnn_output.json', 'fear_greed.csv'}
    assert by_name['cnn_output.json'].rows == 1
    assert by_name['cnn_output.json'].last_date == '2026-11-16'
    assert by_name['fear_greed.csv'].rows == 2
    assert by_name['fear_greed.csv'].last_date == '2026-11-16'
    assert (tmp_path / 'cnn_output.json').read_bytes().startswith(b'{')
    assert len(urls) == 2


def test_bad_json_does_not_replace_existing_files(tmp_path):
    (tmp_path / 'cnn_output.json').write_text('keep')
    (tmp_path / 'fear_greed.csv').write_text('keep')

    def fetch(url: str) -> bytes:
        if url.endswith('.json'):
            return b'{"fear_and_greed": {}}'
        return _csv_payload()

    with pytest.raises(FearGreedFetchError):
        update_fear_greed(dest=tmp_path, fetch=fetch)
    assert (tmp_path / 'cnn_output.json').read_text() == 'keep'
    assert (tmp_path / 'fear_greed.csv').read_text() == 'keep'


def test_bad_csv_does_not_replace_existing_files(tmp_path):
    (tmp_path / 'fear_greed.csv').write_text('keep')

    def fetch(url: str) -> bytes:
        if url.endswith('.json'):
            return _json_payload()
        return b'not,a,fear,greed,file\n'

    with pytest.raises(FearGreedFetchError):
        update_fear_greed(dest=tmp_path, fetch=fetch)
    assert (tmp_path / 'fear_greed.csv').read_text() == 'keep'


def test_validate_cnn_json_rejects_empty_history():
    with pytest.raises(FearGreedFetchError):
        validate_cnn_json(b'{"fear_and_greed_historical": {"data": []}}')


def test_cloud_sync_skips_missing_fear_greed_files(tmp_path, monkeypatch):
    import oracle_cloud.cloud_sync as sync

    monkeypatch.setattr(sync, 'DATA_DIR', tmp_path / 'data')
    monkeypatch.setattr(sync, 'TMP_DIR', tmp_path / 'data' / '.cloud_sync_tmp')
    commands = []

    def fake_run(cmd, check=False):
        commands.append(cmd)
        if check:
            raise AssertionError(f'scp should not run when the remote file is missing: {cmd}')

        class Result:
            returncode = 1

        return Result()

    monkeypatch.setattr(sync.subprocess, 'run', fake_run)
    sync.pull_fear_greed('203.0.113.5', 'key', 'ubuntu', 22)
    assert commands
    assert all(cmd[0] == 'ssh' for cmd in commands)
    assert not (tmp_path / 'data' / 'fear_greed' / 'cnn_output.json').exists()


def test_cloud_sync_replaces_local_fear_greed_file(tmp_path, monkeypatch):
    import oracle_cloud.cloud_sync as sync

    data_dir = tmp_path / 'data'
    (data_dir / 'fear_greed').mkdir(parents=True)
    (data_dir / 'fear_greed' / 'cnn_output.json').write_text('old')
    monkeypatch.setattr(sync, 'DATA_DIR', data_dir)
    monkeypatch.setattr(sync, 'TMP_DIR', data_dir / '.cloud_sync_tmp')
    monkeypatch.setattr(sync, 'FEAR_GREED_FILES', ('fear_greed/cnn_output.json',))

    def fake_run(cmd, check=False):
        if cmd[0] == 'ssh':
            class Result:
                returncode = 0
            return Result()
        dest = cmd[-1]
        Path(dest).write_text('new json')
        return None

    monkeypatch.setattr(sync.subprocess, 'run', fake_run)
    sync.pull_fear_greed('203.0.113.5', 'key', 'ubuntu', 22)
    assert (data_dir / 'fear_greed' / 'cnn_output.json').read_text() == 'new json'
