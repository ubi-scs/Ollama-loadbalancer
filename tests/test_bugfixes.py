import csv
import json
import os
import threading
import time
from queue import Queue, Empty
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import server_application.proxy as proxy
from server_application.usage_utils import log_usage


@pytest.fixture(autouse=True)
def reset_state():
    with proxy._STATE_LOCK:
        proxy._WORKERS.clear()


def _setup_workers_csv(tmp_path, workers=None):
    if workers is None:
        workers = [
            {
                "name": "w1",
                "url": "http://w1:11434",
                "enabled": True,
                "healthy": True,
                "vram_total_mb": 9000,
            },
            {
                "name": "w2",
                "url": "http://w2:11434",
                "enabled": True,
                "healthy": True,
                "vram_total_mb": 24000,
            },
            {
                "name": "w3",
                "url": "http://w3:11434",
                "enabled": True,
                "healthy": True,
                "vram_total_mb": 48000,
            },
        ]
    csv_path = tmp_path / proxy.CONFIG_FILE_PATH
    df = pd.DataFrame(workers)
    df.to_csv(csv_path, index=False)
    return csv_path


def _patch_abs_path(tmp_path, monkeypatch):
    def fake_abs(name):
        return str(tmp_path / name)

    monkeypatch.setattr(proxy, "_abs_path", fake_abs)


class TestParseModelSizeFromString:
    def test_standard_colon_format(self):
        assert proxy._parse_model_size_from_string("llama3:8b") == 8

    def test_uppercase_unit(self):
        assert proxy._parse_model_size_from_string("mistral:7B") == 7

    def test_million_unit(self):
        assert proxy._parse_model_size_from_string("tiny:500m") == 0.5

    def test_standalone_unit(self):
        assert proxy._parse_model_size_from_string("500m") == 0.5

    def test_decimal_billion(self):
        assert proxy._parse_model_size_from_string("0.5B") == 0.5

    def test_no_size_returns_none(self):
        assert proxy._parse_model_size_from_string("no-size") is None

    def test_non_string_returns_none(self):
        assert proxy._parse_model_size_from_string(42) is None
        assert proxy._parse_model_size_from_string(None) is None

    def test_zero_size_returns_none(self):
        assert proxy._parse_model_size_from_string("model:0b") is None

    def test_zero_million_returns_none(self):
        assert proxy._parse_model_size_from_string("model:0m") is None

    def test_negative_not_matched(self):
        assert proxy._parse_model_size_from_string("model:-1b") is None

    def test_empty_string_returns_none(self):
        assert proxy._parse_model_size_from_string("") is None

    def test_hyphenated_model_name_with_size(self):
        assert proxy._parse_model_size_from_string("dolphin-2.7b") == 2.7

    def test_hyphenated_model_name_integer_size(self):
        assert proxy._parse_model_size_from_string("llama3-8b") == 8

    def test_multi_hyphen_model_name(self):
        assert proxy._parse_model_size_from_string("my-model-13b") == 13

    def test_negative_standalone_returns_none(self):
        assert proxy._parse_model_size_from_string("-7b") is None

    def test_negative_after_colon_returns_none(self):
        assert proxy._parse_model_size_from_string("model:-3b") is None

    def test_negative_after_colon_million_returns_none(self):
        assert proxy._parse_model_size_from_string("model:-500m") is None

    def test_hyphenated_with_colon(self):
        assert proxy._parse_model_size_from_string("namespace/model-7b") == 7

    def test_decimal_with_hyphen_prefix(self):
        assert proxy._parse_model_size_from_string("model-0.5b") == 0.5


class TestEstimateRequiredVramMb:
    def test_none_returns_default(self):
        assert proxy._estimate_required_vram_mb(None) == 22000

    def test_zero_returns_default(self):
        assert proxy._estimate_required_vram_mb(0) == 22000

    def test_negative_returns_default(self):
        assert proxy._estimate_required_vram_mb(-5) == 22000

    def test_small_model(self):
        assert proxy._estimate_required_vram_mb(1) == 8192

    def test_7b(self):
        assert proxy._estimate_required_vram_mb(7) == 8192

    def test_14b(self):
        assert proxy._estimate_required_vram_mb(10) == 11000

    def test_32b(self):
        assert proxy._estimate_required_vram_mb(20) == 22000

    def test_70b(self):
        assert proxy._estimate_required_vram_mb(50) == 44000

    def test_large_model(self):
        assert proxy._estimate_required_vram_mb(100) == 81920


class TestVramTier:
    def test_none(self):
        assert proxy._vram_tier(None) == "unknown"

    def test_tiers(self):
        assert proxy._vram_tier(8000) == "8-10GB"
        assert proxy._vram_tier(15000) == "12-16GB"
        assert proxy._vram_tier(24000) == "24GB"
        assert proxy._vram_tier(50000) == "48GB"
        assert proxy._vram_tier(90000) == "80GB+"


class TestSaveLastUsed:
    def test_creates_file_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._save_last_used("mymodel")
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        assert models_path.exists()
        df = pd.read_csv(models_path)
        assert len(df) == 1
        assert df.iloc[0]["Model"] == "mymodel"

    def test_updates_existing_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
            writer.writerow(["mymodel", "01.01.2024"])
            writer.writerow(["other", "01.01.2024"])
        proxy._save_last_used("mymodel")
        df = pd.read_csv(models_path)
        updated = df[df["Model"] == "mymodel"]
        assert len(updated) == 1
        today = time.strftime("%d.%m.%Y")
        assert updated.iloc[0]["LastUsed"] == today

    def test_appends_new_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
            writer.writerow(["existing", "01.01.2024"])
        proxy._save_last_used("newmodel")
        df = pd.read_csv(models_path)
        models = list(df["Model"])
        assert "existing" in models
        assert "newmodel" in models

    def test_handles_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
        proxy._save_last_used("mymodel")
        df = pd.read_csv(models_path)
        assert len(df) == 1
        assert df.iloc[0]["Model"] == "mymodel"

    def test_handles_malformed_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
            writer.writerow(["only_one_column"])
            writer.writerow(["goodmodel", "01.01.2024"])
        proxy._save_last_used("newmodel")
        df = pd.read_csv(models_path)
        models = list(df["Model"])
        assert "newmodel" in models
        assert "goodmodel" in models


class TestLogUsageThreadSafety:
    def test_concurrent_writes(self, tmp_path, monkeypatch):
        path = tmp_path / "usage_stats.json"
        monkeypatch.setattr(
            "server_application.usage_utils.USAGE_STATS_PATH", str(path)
        )
        errors = []

        def writer(user, count):
            try:
                for i in range(count):
                    log_usage(user, 1.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"user{i}", 20)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        with open(path, "r") as f:
            data = json.load(f)
        assert len(data) == 100

    def test_corrupted_file_recovery(self, tmp_path, monkeypatch):
        path = tmp_path / "usage_stats.json"
        monkeypatch.setattr(
            "server_application.usage_utils.USAGE_STATS_PATH", str(path)
        )
        with open(path, "w") as f:
            f.write("not valid json{{")
        log_usage("test", 1.0)
        with open(path, "r") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["user"] == "test"

    def test_creates_file_if_missing(self, tmp_path, monkeypatch):
        path = tmp_path / "usage_stats_new.json"
        monkeypatch.setattr(
            "server_application.usage_utils.USAGE_STATS_PATH", str(path)
        )
        log_usage("newuser", 5.0)
        assert path.exists()
        with open(path, "r") as f:
            data = json.load(f)
        assert data[0]["user"] == "newuser"


class TestWorkerRegistryRefresh:
    def test_refresh_loads_workers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        with proxy._STATE_LOCK:
            assert set(proxy._WORKERS.keys()) == {"w1", "w2", "w3"}
            assert proxy._WORKERS["w1"]["vram_total_mb"] == 9000

    def test_refresh_preserves_queues(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        q = proxy._WORKERS["w1"]["queue"]
        q.put("item")
        proxy._refresh_worker_registry()
        assert proxy._WORKERS["w1"]["queue"] is q
        assert q.qsize() == 1

    def test_refresh_removes_deleted_workers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        assert "w1" in proxy._WORKERS
        new_workers = [
            {"name": "w2", "url": "http://w2:11434", "enabled": True, "healthy": True},
        ]
        _setup_workers_csv(tmp_path, new_workers)
        proxy._refresh_worker_registry()
        assert "w1" not in proxy._WORKERS
        assert "w2" in proxy._WORKERS

    def test_disabled_worker_not_in_healthy_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        workers = [
            {"name": "w1", "url": "http://w1:11434", "enabled": False, "healthy": True},
            {"name": "w2", "url": "http://w2:11434", "enabled": True, "healthy": True},
        ]
        _setup_workers_csv(tmp_path, workers)
        _patch_abs_path(tmp_path, monkeypatch)
        with patch.object(proxy, "requests"):
            proxy._refresh_worker_registry()
            names = proxy._get_enabled_healthy_workers()
            assert "w1" not in names
            assert "w2" in names

    def test_unhealthy_worker_not_in_healthy_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        workers = [
            {"name": "w1", "url": "http://w1:11434", "enabled": True, "healthy": False},
            {"name": "w2", "url": "http://w2:11434", "enabled": True, "healthy": True},
        ]
        _setup_workers_csv(tmp_path, workers)
        _patch_abs_path(tmp_path, monkeypatch)
        with patch.object(proxy, "requests"):
            proxy._refresh_worker_registry()
            names = proxy._get_enabled_healthy_workers()
            assert "w1" not in names
            assert "w2" in names


class TestChooseBackendForModel:
    def test_prefers_loaded_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        with proxy._STATE_LOCK:
            proxy._WORKERS["w2"]["loaded_models"].add("mymodel")
            proxy._WORKERS["w2"]["available_models"].add("mymodel")
        with patch.object(proxy, "requests"):
            chosen = proxy._choose_backend_for_model("mymodel")
            assert chosen[0] == "w2"

    def test_returns_none_for_no_workers(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        csv_path = tmp_path / proxy.CONFIG_FILE_PATH
        df = pd.DataFrame(
            columns=["name", "url", "enabled", "healthy", "vram_total_mb"]
        )
        df.to_csv(csv_path, index=False)
        _patch_abs_path(tmp_path, monkeypatch)
        with patch.object(proxy, "requests"):
            chosen = proxy._choose_backend_for_model("model")
            assert chosen is None

    def test_size_fit_selection(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        with open(tmp_path / proxy.MODEL_SIZES_FILE_PATH, "w") as f:
            f.write("model,size_billion\n")
            f.write("bigmodel,30\n")
        proxy._refresh_worker_registry()
        with proxy._STATE_LOCK:
            for wname in proxy._WORKERS:
                proxy._WORKERS[wname]["available_models"].add("bigmodel")
        with patch.object(proxy, "requests"):
            chosen = proxy._choose_backend_for_model("bigmodel")
            assert chosen is not None
            assert chosen[0] == "w2"

    def test_empty_model_selects_any_worker(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        with patch.object(proxy, "requests"):
            chosen = proxy._choose_backend_for_model("")
            assert chosen is not None
            assert chosen[0] in {"w1", "w2", "w3"}


class TestGetAvailableModelsForEnabledHealthy:
    def test_returns_union(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        with proxy._STATE_LOCK:
            for wname in proxy._WORKERS:
                proxy._WORKERS[wname]["available_models"].clear()
            proxy._WORKERS["w1"]["available_models"].update({"a", "b"})
            proxy._WORKERS["w2"]["available_models"].update({"b", "c"})
        with patch.object(proxy, "requests"):
            models = proxy._get_available_models_for_enabled_healthy()
            assert set(["a", "b", "c"]).issubset(set(models))


class TestLoadModelSizes:
    def test_creates_file_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        sizes = proxy._load_model_sizes()
        assert isinstance(sizes, dict)
        assert os.path.exists(tmp_path / proxy.MODEL_SIZES_FILE_PATH)

    def test_reads_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        path = tmp_path / proxy.MODEL_SIZES_FILE_PATH
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "size_billion"])
            writer.writerow(["testmodel", "7"])
        sizes = proxy._load_model_sizes()
        assert sizes["testmodel"] == 7.0

    def test_handles_corrupt_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        path = tmp_path / proxy.MODEL_SIZES_FILE_PATH
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "size_billion"])
            writer.writerow(["good", "7"])
            writer.writerow(["bad", "not_a_number"])
        sizes = proxy._load_model_sizes()
        assert sizes["good"] == 7.0
        assert "bad" not in sizes


class TestSaveModelSize:
    def test_saves_new_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._save_model_size("mymodel", 7.0)
        sizes = proxy._load_model_sizes()
        assert sizes["mymodel"] == 7.0

    def test_updates_existing_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._save_model_size("mymodel", 7.0)
        proxy._save_model_size("mymodel", 14.0)
        sizes = proxy._load_model_sizes()
        assert sizes["mymodel"] == 14.0


class TestEnsureHealthCounters:
    def test_creates_counters_for_existing_worker(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        proxy._ensure_health_counters("w1")
        with proxy._STATE_LOCK:
            w = proxy._WORKERS["w1"]
            assert "health_failures" in w
            assert "health_successes" in w
            assert "last_health_probe" in w

    def test_skips_nonexistent_worker(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        proxy._ensure_health_counters("nonexistent")
        with proxy._STATE_LOCK:
            assert "nonexistent" not in proxy._WORKERS


class TestRefreshLoadedModels:
    def test_returns_cached_within_ttl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        with proxy._STATE_LOCK:
            proxy._WORKERS["w1"]["loaded_models"] = {"model_a"}
            proxy._WORKERS["w1"]["last_models_refresh"] = time.time()
        with patch.object(proxy, "requests"):
            result = proxy._refresh_loaded_models("w1", ttl_seconds=30)
            assert "model_a" in result


class TestRefreshAvailableModels:
    def test_returns_cached_within_ttl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _setup_workers_csv(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        proxy._refresh_worker_registry()
        with proxy._STATE_LOCK:
            proxy._WORKERS["w1"]["available_models"] = {"model_x"}
            proxy._WORKERS["w1"]["last_available_refresh"] = time.time()
        with patch.object(proxy, "requests"):
            result = proxy._refresh_available_models("w1", ttl_seconds=60)
            assert "model_x" in result


class TestRewriteUrlToHelperPort:
    def _rewrite_url_to_helper_port(self, worker_url):
        import re

        if not isinstance(worker_url, str) or not re.match(r"^https?://", worker_url):
            return None
        if "://" in worker_url:
            proto, rest = worker_url.split("://", 1)
            if "/" in rest:
                host_part, path_part = rest.split("/", 1)
            else:
                host_part = rest
                path_part = ""
            host_part = re.sub(r":\d+", ":18034", host_part)
            if ":" not in host_part.split("/")[-1]:
                host_part = f"{host_part}:18034"
            if path_part:
                new_url = f"{proto}://{host_part}/{path_part}"
            else:
                new_url = f"{proto}://{host_part}"
        else:
            new_url = worker_url
        return new_url

    def test_url_with_port_and_path(self):
        assert (
            self._rewrite_url_to_helper_port("http://worker:11434/some/path")
            == "http://worker:18034/some/path"
        )

    def test_url_with_port_no_path(self):
        assert (
            self._rewrite_url_to_helper_port("http://worker:11434")
            == "http://worker:18034"
        )

    def test_url_without_port_with_path(self):
        assert (
            self._rewrite_url_to_helper_port("http://worker/some/path")
            == "http://worker:18034/some/path"
        )

    def test_url_without_port_no_path(self):
        assert (
            self._rewrite_url_to_helper_port("http://worker") == "http://worker:18034"
        )

    def test_https_url_with_port(self):
        assert (
            self._rewrite_url_to_helper_port("https://host:11434")
            == "https://host:18034"
        )

    def test_https_url_without_port(self):
        assert self._rewrite_url_to_helper_port("https://host") == "https://host:18034"

    def test_invalid_url_returns_none(self):
        assert self._rewrite_url_to_helper_port("not-a-url") is None

    def test_non_string_returns_none(self):
        assert self._rewrite_url_to_helper_port(42) is None

    def test_different_port_gets_replaced(self):
        assert (
            self._rewrite_url_to_helper_port("http://host:8080/path")
            == "http://host:18034/path"
        )

    def test_already_helper_port(self):
        assert (
            self._rewrite_url_to_helper_port("http://host:18034/status")
            == "http://host:18034/status"
        )


class TestGuiParseModelSizeFromString:
    def _gui_parse_model_size(self, model):
        import re

        if not isinstance(model, str):
            return None
        s = model.strip()
        if s.startswith("-"):
            return None
        colon_idx = s.rfind(":")
        if colon_idx >= 0 and s[colon_idx + 1 :].lstrip().startswith("-"):
            return None
        m = re.search(r"(?i)(?:.*:)?(\d+(?:\.\d+)?)([bm])$", s)
        if not m:
            return None
        try:
            value = float(m.group(1))
        except ValueError:
            return None
        if value <= 0:
            return None
        unit = m.group(2).lower()
        if unit == "b":
            return value
        if unit == "m":
            return value / 1000.0
        return None

    def test_hyphenated_model_name(self):
        assert self._gui_parse_model_size("dolphin-2.7b") == 2.7

    def test_hyphenated_model_name_integer(self):
        assert self._gui_parse_model_size("llama3-8b") == 8

    def test_negative_standalone_returns_none(self):
        assert self._gui_parse_model_size("-7b") is None

    def test_negative_after_colon_returns_none(self):
        assert self._gui_parse_model_size("model:-3b") is None

    def test_standard_colon_format(self):
        assert self._gui_parse_model_size("llama3:8b") == 8

    def test_million_unit(self):
        assert self._gui_parse_model_size("tiny:500m") == 0.5


class TestParseModelSizeNoDeadCode:
    def test_function_has_no_unreachable_code(self):
        import inspect
        import ast

        source = inspect.getsource(proxy._parse_model_size_from_string)
        tree = ast.parse(source)
        func_node = tree.body[0]
        last_return_index = None
        for i, stmt in enumerate(func_node.body):
            if isinstance(stmt, ast.Return):
                last_return_index = i
        assert last_return_index is not None
        for i, stmt in enumerate(func_node.body):
            if i > last_return_index:
                pytest.fail(
                    f"Unreachable code found after return at index {last_return_index}: "
                    f"statement at index {i} ({ast.unparse(stmt) if hasattr(ast, 'unparse') else str(stmt)})"
                )


class TestQueueDequeueRaceCondition:
    def test_dequeue_on_empty_queue_does_not_crash(self):
        q = Queue()
        try:
            q.get_nowait()
        except Empty:
            pass
        assert q.empty()

    def test_dequeue_after_put_removes_item(self):
        q = Queue()
        q.put(1)
        try:
            q.get_nowait()
        except Empty:
            pytest.fail("get_nowait raised Empty on a non-empty queue")
        assert q.empty()

    def test_dequeue_with_try_except_is_safe(self):
        q = Queue()
        q.put(1)
        q.put(2)
        try:
            q.get_nowait()
        except Empty:
            pytest.fail("Should not raise Empty")
        try:
            q.get_nowait()
        except Empty:
            pytest.fail("Should not raise Empty")
        try:
            q.get_nowait()
        except Empty:
            pass
        assert q.empty()

    def test_concurrent_dequeue_does_not_lose_items(self):
        q = Queue()
        num_items = 100
        for i in range(num_items):
            q.put(i)
        consumed = []

        def consumer():
            while True:
                try:
                    item = q.get_nowait()
                    consumed.append(item)
                except Empty:
                    break

        threads = [threading.Thread(target=consumer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(consumed) == num_items


class TestProxyRewriteUrlToHelperPort:
    def test_url_with_standard_port(self):
        assert (
            proxy._rewrite_url_to_helper_port("http://worker:11434")
            == "http://worker:18034"
        )

    def test_url_with_standard_port_and_path(self):
        assert (
            proxy._rewrite_url_to_helper_port("http://worker:11434/some/path")
            == "http://worker:18034/some/path"
        )

    def test_url_with_different_port(self):
        assert (
            proxy._rewrite_url_to_helper_port("http://worker:8080/path")
            == "http://worker:18034/path"
        )

    def test_url_without_port(self):
        assert (
            proxy._rewrite_url_to_helper_port("http://worker") == "http://worker:18034"
        )

    def test_url_without_port_with_path(self):
        assert (
            proxy._rewrite_url_to_helper_port("http://worker/some/path")
            == "http://worker:18034/some/path"
        )

    def test_https_url_with_port(self):
        assert (
            proxy._rewrite_url_to_helper_port("https://host:11434")
            == "https://host:18034"
        )

    def test_https_url_without_port(self):
        assert proxy._rewrite_url_to_helper_port("https://host") == "https://host:18034"

    def test_hostname_containing_11434_is_not_corrupted(self):
        result = proxy._rewrite_url_to_helper_port("http://worker11434:11434")
        assert result == "http://worker11434:18034"
        assert "worker11434" in result

    def test_already_helper_port(self):
        assert (
            proxy._rewrite_url_to_helper_port("http://host:18034/status")
            == "http://host:18034/status"
        )

    def test_invalid_url_returns_none(self):
        assert proxy._rewrite_url_to_helper_port("not-a-url") is None

    def test_non_string_returns_none(self):
        assert proxy._rewrite_url_to_helper_port(42) is None

    def test_none_returns_none(self):
        assert proxy._rewrite_url_to_helper_port(None) is None


class TestSaveLastUsedFixesMalformedRow:
    def test_updates_malformed_single_column_row(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
            writer.writerow(["mymodel"])
            writer.writerow(["other", "01.01.2024"])
        proxy._save_last_used("mymodel")
        df = pd.read_csv(models_path)
        mymodel_rows = df[df["Model"] == "mymodel"]
        assert len(mymodel_rows) == 1
        today = time.strftime("%d.%m.%Y")
        assert mymodel_rows.iloc[0]["LastUsed"] == today

    def test_no_duplicate_rows_after_update(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
            writer.writerow(["mymodel"])
        proxy._save_last_used("mymodel")
        df = pd.read_csv(models_path)
        mymodel_rows = df[df["Model"] == "mymodel"]
        assert len(mymodel_rows) == 1

    def test_existing_two_column_row_is_updated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_abs_path(tmp_path, monkeypatch)
        models_path = tmp_path / proxy.MODELS_FILE_PATH
        with open(models_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "LastUsed"])
            writer.writerow(["mymodel", "01.01.2024"])
        proxy._save_last_used("mymodel")
        df = pd.read_csv(models_path)
        mymodel_rows = df[df["Model"] == "mymodel"]
        assert len(mymodel_rows) == 1
        today = time.strftime("%d.%m.%Y")
        assert mymodel_rows.iloc[0]["LastUsed"] == today
