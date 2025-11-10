import os
import json
import threading
import types
import pandas as pd
import server_application.proxy as proxy
from server_application.usage_utils import log_usage
import pytest


def setup_workers_csv(tmp_path):
    df = pd.DataFrame([
        {"name": "w1", "url": "http://w1:11434", "enabled": True, "healthy": True, "vram_total_mb": 9000},
        {"name": "w2", "url": "http://w2:11434", "enabled": True, "healthy": True, "vram_total_mb": 24000},
        {"name": "w3", "url": "http://w3:11434", "enabled": True, "healthy": True, "vram_total_mb": 48000},
    ])
    csv_path = tmp_path / proxy.CONFIG_FILE_PATH
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture(autouse=True)
def reset_state():
    # Clear global workers before each test to avoid leakage
    with proxy._STATE_LOCK:
        proxy._WORKERS.clear()


def test_parse_model_size_from_string():
    assert proxy._parse_model_size_from_string("llama3:8b") == 8
    assert proxy._parse_model_size_from_string("mistral:7B") == 7
    assert proxy._parse_model_size_from_string("tiny:500m") == 0.5
    assert proxy._parse_model_size_from_string("500m") == 0.5
    assert proxy._parse_model_size_from_string("0.5B") == 0.5
    assert proxy._parse_model_size_from_string("no-size") is None


def test_estimate_required_vram_mb():
    assert proxy._estimate_required_vram_mb(1) == 8192
    assert proxy._estimate_required_vram_mb(7) == 8192
    assert proxy._estimate_required_vram_mb(10) == 11000  # base//2
    assert proxy._estimate_required_vram_mb(20) == 22000
    assert proxy._estimate_required_vram_mb(50) == 44000
    assert proxy._estimate_required_vram_mb(100) == 81920


def test_vram_tier():
    assert proxy._vram_tier(None) == 'unknown'
    assert proxy._vram_tier(8000) == '8-10GB'
    assert proxy._vram_tier(15000) == '12-16GB'
    assert proxy._vram_tier(24000) == '24GB'
    assert proxy._vram_tier(50000) == '48GB'
    assert proxy._vram_tier(90000) == '80GB+'


def test_worker_registry_refresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_workers_csv(tmp_path)

    def fake_abs(name):
        return str(tmp_path / name)
    monkeypatch.setattr(proxy, '_abs_path', fake_abs)

    proxy._refresh_worker_registry()
    with proxy._STATE_LOCK:
        assert set(proxy._WORKERS.keys()) == {"w1", "w2", "w3"}
        assert proxy._WORKERS['w1']['vram_total_mb'] == 9000


def test_choose_backend_for_model_prefers_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_workers_csv(tmp_path)

    def fake_abs(name):
        return str(tmp_path / name)
    monkeypatch.setattr(proxy, '_abs_path', fake_abs)

    proxy._refresh_worker_registry()
    with proxy._STATE_LOCK:
        proxy._WORKERS['w2']['loaded_models'].add('mymodel')
        proxy._WORKERS['w2']['available_models'].add('mymodel')

    chosen = proxy._choose_backend_for_model('mymodel')
    assert chosen[0] == 'w2'


def test_choose_backend_for_model_size_fit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_workers_csv(tmp_path)

    def fake_abs(name):
        return str(tmp_path / name)
    monkeypatch.setattr(proxy, '_abs_path', fake_abs)

    with open(tmp_path / proxy.MODEL_SIZES_FILE_PATH, 'w') as f:
        f.write('model,size_billion\n')
        f.write('bigmodel,30\n')

    proxy._refresh_worker_registry()
    with proxy._STATE_LOCK:
        for wname in proxy._WORKERS:
            proxy._WORKERS[wname]['available_models'].add('bigmodel')

    chosen = proxy._choose_backend_for_model('bigmodel')
    assert chosen[0] == 'w2'
    # Cleanup: remove bigmodel from availability to avoid leaking into later tests
    with proxy._STATE_LOCK:
        for wname in proxy._WORKERS:
            proxy._WORKERS[wname]['available_models'].discard('bigmodel')


def test_get_available_models_for_enabled_healthy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_workers_csv(tmp_path)

    def fake_abs(name):
        return str(tmp_path / name)
    monkeypatch.setattr(proxy, '_abs_path', fake_abs)

    proxy._refresh_worker_registry()
    with proxy._STATE_LOCK:
        for wname in proxy._WORKERS:
            proxy._WORKERS[wname]['available_models'].clear()
        proxy._WORKERS['w1']['available_models'].update({'a', 'b'})
        proxy._WORKERS['w2']['available_models'].update({'b', 'c'})
    models = proxy._get_available_models_for_enabled_healthy()
    # Ensure expected models are present (ignore incidental extras)
    assert set(['a','b','c']).issubset(set(models))


def test_log_usage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'usage_stats.json'
    monkeypatch.setattr('server_application.usage_utils.USAGE_STATS_PATH', str(path))
    log_usage('alice', 3.14)
    log_usage('bob', 2.0)
    data = json.loads(path.read_text())
    users = [d['user'] for d in data]
    assert users == ['alice', 'bob']
    assert data[0]['duration'] == 3.14
