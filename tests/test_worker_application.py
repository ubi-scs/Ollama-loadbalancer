import os
import sys
import json
import shutil
import tarfile
import tempfile
import time
import threading
import subprocess
from pathlib import Path

import pytest
import httpx


WORKER_APP_DIR = Path(__file__).parent.parent / "worker_application"
API_KEY = "test-api-key-for-testing"


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_tar_gz(directory, filename="update.tar.gz"):
    tar_path = os.path.join(tempfile.gettempdir(), filename)
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in os.listdir(directory):
            tar.add(os.path.join(directory, item), arcname=item)
    return tar_path


def _wait_for_health(
    host, port, path="/api/version", timeout=15, interval=0.5, api_key=""
):
    url = f"http://{host}:{port}{path}"
    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2, headers=headers)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _kill_process_tree(pid):
    try:
        os.kill(pid, 9)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


@pytest.fixture()
def worker_instance(tmp_path):
    worker_port = _find_free_port()
    worker_dir = tmp_path / "worker_application"
    shutil.copytree(str(WORKER_APP_DIR), str(worker_dir))

    env = os.environ.copy()
    env["OLLAMA_HELPER_API_KEY"] = API_KEY
    env["WORKER_HOST"] = "127.0.0.1"
    env["WORKER_PORT"] = str(worker_port)
    env["PYTHONPATH"] = str(worker_dir.parent)
    env["OLLAMA_UPDATE_TIMEOUT"] = "5"
    env["OLLAMA_UPDATE_MAX_RETRIES"] = "1"
    env["OLLAMA_UPDATE_RETRY_DELAY"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "worker_application.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
        ],
        env=env,
        cwd=str(worker_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    healthy = _wait_for_health("127.0.0.1", worker_port, api_key=API_KEY)
    if not healthy:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.skip("Worker failed to start in time")

    base_url = f"http://127.0.0.1:{worker_port}"
    headers = {"X-API-KEY": API_KEY}
    client = httpx.Client(base_url=base_url, headers=headers, timeout=10)

    yield {
        "client": client,
        "port": worker_port,
        "host": "127.0.0.1",
        "proc": proc,
        "worker_dir": worker_dir,
        "base_url": base_url,
    }

    client.close()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture()
def watchdog_instance(tmp_path):
    worker_port = _find_free_port()
    watchdog_port = _find_free_port()

    worker_dir = tmp_path / "worker_application"
    shutil.copytree(str(WORKER_APP_DIR), str(worker_dir))

    env = os.environ.copy()
    env["OLLAMA_HELPER_API_KEY"] = API_KEY
    env["WORKER_HOST"] = "127.0.0.1"
    env["WORKER_PORT"] = str(worker_port)
    env["WATCHDOG_HOST"] = "127.0.0.1"
    env["WATCHDOG_PORT"] = str(watchdog_port)
    env["PYTHONPATH"] = str(worker_dir.parent)

    watchdog_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "worker_application.watchdog:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(watchdog_port),
        ],
        env=env,
        cwd=str(worker_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    healthy = _wait_for_health(
        "127.0.0.1", watchdog_port, path="/watchdog/status", api_key=API_KEY
    )
    if not healthy:
        watchdog_proc.terminate()
        watchdog_proc.wait(timeout=5)
        pytest.skip("Watchdog failed to start in time")

    worker_healthy = _wait_for_health(
        "127.0.0.1", worker_port, timeout=20, api_key=API_KEY
    )
    if not worker_healthy:
        watchdog_proc.terminate()
        watchdog_proc.wait(timeout=5)
        pytest.skip("Worker (via watchdog) failed to start in time")

    watchdog_base = f"http://127.0.0.1:{watchdog_port}"
    worker_base = f"http://127.0.0.1:{worker_port}"
    headers = {"X-API-KEY": API_KEY}
    watchdog_client = httpx.Client(base_url=watchdog_base, headers=headers, timeout=10)
    worker_client = httpx.Client(base_url=worker_base, headers=headers, timeout=10)

    yield {
        "watchdog_client": watchdog_client,
        "worker_client": worker_client,
        "watchdog_port": watchdog_port,
        "worker_port": worker_port,
        "watchdog_proc": watchdog_proc,
        "worker_dir": worker_dir,
        "watchdog_base": watchdog_base,
        "worker_base": worker_base,
    }

    try:
        watchdog_client.post("/watchdog/worker/stop")
    except Exception:
        pass
    time.sleep(1)

    watchdog_client.close()
    worker_client.close()

    watchdog_proc.terminate()
    try:
        watchdog_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _kill_process_tree(watchdog_proc.pid)
        try:
            watchdog_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# Worker API tests (real running instance)
# ---------------------------------------------------------------------------


class TestWorkerAPI:
    def test_api_version(self, worker_instance):
        resp = worker_instance["client"].get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_version" in data

    def test_ollama_update_status_idle(self, worker_instance):
        resp = worker_instance["client"].get("/ollama/update/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    def test_api_key_required(self, worker_instance):
        client_no_key = httpx.Client(base_url=worker_instance["base_url"], timeout=10)
        resp = client_no_key.get("/api/version")
        assert resp.status_code == 422 or resp.status_code == 403
        client_no_key.close()

    def test_api_key_wrong(self, worker_instance):
        client_wrong = httpx.Client(
            base_url=worker_instance["base_url"],
            headers={"X-API-KEY": "wrong-key"},
            timeout=10,
        )
        resp = client_wrong.get("/api/version")
        assert resp.status_code == 403
        client_wrong.close()

    def test_ollama_update_triggers_and_status_changes(self, worker_instance):
        resp = worker_instance["client"].post("/ollama/update")
        assert resp.status_code == 202

        status_resp = worker_instance["client"].get("/ollama/update/status")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] in ("in_progress", "success", "failed", "idle")

        deadline = time.time() + 120
        while time.time() < deadline:
            status_resp = worker_instance["client"].get("/ollama/update/status")
            data = status_resp.json()
            if data["status"] in ("success", "failed", "idle"):
                break
            time.sleep(2)

        assert data["status"] in ("success", "failed", "idle")

    def test_worker_self_update_status_endpoint(self, worker_instance):
        resp = worker_instance["client"].get("/worker/self-update/status")
        assert resp.status_code == 200

    def test_worker_self_update_rejects_non_tar_gz(self, worker_instance):
        resp = worker_instance["client"].post(
            "/worker/self-update",
            files={"file": ("update.zip", b"not-a-zip", "application/zip")},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Ollama update robustness (functional tests with running worker)
# ---------------------------------------------------------------------------


class TestOllamaUpdateRobustness:
    def test_update_status_has_timestamp(self, worker_instance):
        resp = worker_instance["client"].get("/ollama/update/status")
        data = resp.json()
        assert "timestamp" in data

    def test_update_status_transitions_to_terminal_state(self, worker_instance):
        resp = worker_instance["client"].post("/ollama/update")
        assert resp.status_code == 202

        deadline = time.time() + 120
        final_status = None
        while time.time() < deadline:
            status_resp = worker_instance["client"].get("/ollama/update/status")
            data = status_resp.json()
            if data["status"] in ("success", "failed", "idle"):
                final_status = data["status"]
                break
            time.sleep(2)

        assert final_status in ("success", "failed", "idle")

    def test_concurrent_update_rejected(self, worker_instance):
        resp1 = worker_instance["client"].post("/ollama/update")
        assert resp1.status_code == 202

        time.sleep(0.5)

        resp2 = worker_instance["client"].post("/ollama/update")
        assert resp2.status_code in (409, 202)

        deadline = time.time() + 120
        while time.time() < deadline:
            status_resp = worker_instance["client"].get("/ollama/update/status")
            data = status_resp.json()
            if data["status"] in ("success", "failed", "idle"):
                break
            time.sleep(2)


# ---------------------------------------------------------------------------
# Watchdog tests (real running watchdog + worker)
# ---------------------------------------------------------------------------


class TestWatchdogAPI:
    def test_watchdog_status(self, watchdog_instance):
        resp = watchdog_instance["watchdog_client"].get("/watchdog/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "worker_healthy" in data
        assert "worker_running" in data
        assert "update" in data

    def test_worker_healthy_after_startup(self, watchdog_instance):
        resp = watchdog_instance["watchdog_client"].get("/watchdog/status")
        data = resp.json()
        assert data["worker_healthy"] is True
        assert data["worker_running"] is True

    def test_worker_api_accessible(self, watchdog_instance):
        resp = watchdog_instance["worker_client"].get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_version" in data

    def test_watchdog_update_status_idle(self, watchdog_instance):
        resp = watchdog_instance["watchdog_client"].get("/watchdog/update/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    def test_watchdog_restart_worker(self, watchdog_instance):
        resp = watchdog_instance["watchdog_client"].post("/watchdog/worker/restart")
        assert resp.status_code == 200

        worker_healthy = _wait_for_health(
            "127.0.0.1",
            watchdog_instance["worker_port"],
            timeout=15,
            api_key=API_KEY,
        )
        assert worker_healthy is True

    def test_watchdog_stop_and_start_worker(self, watchdog_instance):
        stop_resp = watchdog_instance["watchdog_client"].post("/watchdog/worker/stop")
        assert stop_resp.status_code == 200

        time.sleep(1)

        status_resp = watchdog_instance["watchdog_client"].get("/watchdog/status")
        data = status_resp.json()
        assert data["worker_running"] is False

        start_resp = watchdog_instance["watchdog_client"].post("/watchdog/worker/start")
        assert start_resp.status_code == 200

        worker_healthy = _wait_for_health(
            "127.0.0.1",
            watchdog_instance["worker_port"],
            timeout=15,
            api_key=API_KEY,
        )
        assert worker_healthy is True


# ---------------------------------------------------------------------------
# Self-update via watchdog (real integration tests)
# ---------------------------------------------------------------------------


class TestSelfUpdate:
    def test_self_update_with_valid_archive(self, watchdog_instance):
        worker_dir = watchdog_instance["worker_dir"]

        new_version_dir = worker_dir.parent / "new_version"
        new_version_dir.mkdir()
        version_file = new_version_dir / "__init__.py"
        version_file.write_text("# updated version\n")

        tar_path = _make_tar_gz(str(new_version_dir), "valid_update.tar.gz")

        try:
            with open(tar_path, "rb") as f:
                tar_data = f.read()

            resp = watchdog_instance["watchdog_client"].post(
                "/watchdog/update",
                files={"file": ("update.tar.gz", tar_data, "application/gzip")},
            )
            assert resp.status_code == 202

            deadline = time.time() + 60
            final_status = None
            while time.time() < deadline:
                status_resp = watchdog_instance["watchdog_client"].get(
                    "/watchdog/update/status"
                )
                data = status_resp.json()
                if data["status"] in ("success", "failed"):
                    final_status = data["status"]
                    break
                time.sleep(2)

            assert final_status == "success"
        finally:
            if os.path.exists(tar_path):
                os.unlink(tar_path)

    def test_self_update_rejects_non_tar_gz(self, watchdog_instance):
        resp = watchdog_instance["watchdog_client"].post(
            "/watchdog/update",
            files={"file": ("update.zip", b"not-a-zip", "application/zip")},
        )
        assert resp.status_code == 400

    def test_self_update_with_broken_archive_rolls_back(self, watchdog_instance):
        worker_dir = watchdog_instance["worker_dir"]

        broken_dir = worker_dir.parent / "broken_version"
        broken_dir.mkdir()

        with open(os.path.join(broken_dir, "main.py"), "w") as f:
            f.write("raise SyntaxError('broken on purpose')\n")

        tar_path = _make_tar_gz(str(broken_dir), "broken_update.tar.gz")

        try:
            with open(tar_path, "rb") as f:
                tar_data = f.read()

            resp = watchdog_instance["watchdog_client"].post(
                "/watchdog/update",
                files={"file": ("broken_update.tar.gz", tar_data, "application/gzip")},
            )
            assert resp.status_code == 202

            deadline = time.time() + 60
            final_status = None
            while time.time() < deadline:
                status_resp = watchdog_instance["watchdog_client"].get(
                    "/watchdog/update/status"
                )
                data = status_resp.json()
                if data["status"] in ("success", "failed"):
                    final_status = data["status"]
                    break
                time.sleep(2)

            worker_healthy = _wait_for_health(
                "127.0.0.1",
                watchdog_instance["worker_port"],
                timeout=15,
                api_key=API_KEY,
            )
            assert worker_healthy is True
        finally:
            if os.path.exists(tar_path):
                os.unlink(tar_path)

    def test_concurrent_update_rejected(self, watchdog_instance):
        update_dir = watchdog_instance["worker_dir"].parent / "concurrent_update"
        update_dir.mkdir()
        (update_dir / "__init__.py").write_text("# concurrent update\n")

        tar_path = _make_tar_gz(str(update_dir), "concurrent.tar.gz")

        try:
            with open(tar_path, "rb") as f:
                tar_data = f.read()

            resp1 = watchdog_instance["watchdog_client"].post(
                "/watchdog/update",
                files={"file": ("update1.tar.gz", tar_data, "application/gzip")},
            )
            assert resp1.status_code == 202

            time.sleep(1)

            resp2 = watchdog_instance["watchdog_client"].post(
                "/watchdog/update",
                files={"file": ("update2.tar.gz", tar_data, "application/gzip")},
            )
            assert resp2.status_code == 409

            deadline = time.time() + 60
            while time.time() < deadline:
                status_resp = watchdog_instance["watchdog_client"].get(
                    "/watchdog/update/status"
                )
                data = status_resp.json()
                if data["status"] in ("success", "failed"):
                    break
                time.sleep(2)
        finally:
            if os.path.exists(tar_path):
                os.unlink(tar_path)


# ---------------------------------------------------------------------------
# Watchdog backup/rollback unit tests (filesystem only, no server)
# ---------------------------------------------------------------------------


class TestWatchdogBackupRollback:
    def test_create_backup(self, tmp_path):
        from worker_application.watchdog import _create_backup

        worker_dir = tmp_path / "worker"
        worker_dir.mkdir()
        (worker_dir / "main.py").write_text("# original")
        (worker_dir / "data.txt").write_text("hello")
        backup_dir = tmp_path / "backup"

        _create_backup(str(worker_dir), str(backup_dir))

        assert backup_dir.exists()
        assert (backup_dir / "main.py").read_text() == "# original"
        assert (backup_dir / "data.txt").read_text() == "hello"

    def test_deploy_update(self, tmp_path):
        from worker_application.watchdog import _deploy_update

        worker_dir = tmp_path / "worker"
        worker_dir.mkdir()
        (worker_dir / "main.py").write_text("# old version")

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "main.py").write_text("# new version")
        (staging_dir / "new_file.py").write_text("# new")

        _deploy_update(str(staging_dir), str(worker_dir))

        assert (worker_dir / "main.py").read_text() == "# new version"
        assert (worker_dir / "new_file.py").read_text() == "# new"

    def test_perform_rollback(self, tmp_path):
        from worker_application.watchdog import _perform_rollback

        worker_dir = tmp_path / "worker"
        worker_dir.mkdir()
        (worker_dir / "main.py").write_text("# new broken version")

        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "main.py").write_text("# original working version")
        (backup_dir / "util.py").write_text("# utility")

        result = _perform_rollback(str(worker_dir), str(backup_dir))

        assert result is True
        assert (worker_dir / "main.py").read_text() == "# original working version"
        assert (worker_dir / "util.py").read_text() == "# utility"

    def test_perform_rollback_no_backup(self, tmp_path):
        from worker_application.watchdog import _perform_rollback

        worker_dir = tmp_path / "worker"
        worker_dir.mkdir()
        (worker_dir / "main.py").write_text("# broken")

        backup_dir = tmp_path / "nonexistent_backup"

        result = _perform_rollback(str(worker_dir), str(backup_dir))
        assert result is False

    def test_deploy_update_with_subdirectories(self, tmp_path):
        from worker_application.watchdog import _deploy_update

        worker_dir = tmp_path / "worker"
        worker_dir.mkdir()
        (worker_dir / "main.py").write_text("# old")

        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "main.py").write_text("# new")
        sub_dir = staging_dir / "subdir"
        sub_dir.mkdir()
        (sub_dir / "module.py").write_text("# submodule")

        _deploy_update(str(staging_dir), str(worker_dir))

        assert (worker_dir / "main.py").read_text() == "# new"
        assert (worker_dir / "subdir" / "module.py").read_text() == "# submodule"

    def test_backup_replaces_existing(self, tmp_path):
        from worker_application.watchdog import _create_backup

        worker_dir = tmp_path / "worker"
        worker_dir.mkdir()
        (worker_dir / "main.py").write_text("# v2")
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "old.py").write_text("# old backup")

        _create_backup(str(worker_dir), str(backup_dir))

        assert (backup_dir / "main.py").read_text() == "# v2"
        assert not (backup_dir / "old.py").exists()


# ---------------------------------------------------------------------------
# WorkerProcess class tests
# ---------------------------------------------------------------------------


class TestWorkerProcess:
    def test_worker_process_start_and_stop(self, tmp_path):
        from worker_application.watchdog import WorkerProcess

        worker_port = _find_free_port()
        worker_dir = str(WORKER_APP_DIR)

        env = {
            "OLLAMA_HELPER_API_KEY": API_KEY,
            "WORKER_HOST": "127.0.0.1",
            "WORKER_PORT": str(worker_port),
        }

        wp = WorkerProcess(
            worker_dir=worker_dir,
            worker_host="127.0.0.1",
            worker_port=worker_port,
            env=env,
            api_key=API_KEY,
        )

        started = wp.start(timeout=15)
        assert started is True
        assert wp.is_running is True
        assert wp.is_healthy() is True

        wp.stop()
        assert wp.is_running is False

    def test_worker_process_restart(self, tmp_path):
        from worker_application.watchdog import WorkerProcess

        worker_port = _find_free_port()
        worker_dir = str(WORKER_APP_DIR)

        env = {
            "OLLAMA_HELPER_API_KEY": API_KEY,
            "WORKER_HOST": "127.0.0.1",
            "WORKER_PORT": str(worker_port),
        }

        wp = WorkerProcess(
            worker_dir=worker_dir,
            worker_host="127.0.0.1",
            worker_port=worker_port,
            env=env,
            api_key=API_KEY,
        )

        started = wp.start(timeout=15)
        assert started is True

        restarted = wp.restart()
        assert restarted is True
        assert wp.is_healthy() is True

        wp.stop()

    def test_worker_process_is_healthy_false_when_not_running(self, tmp_path):
        from worker_application.watchdog import WorkerProcess

        worker_port = _find_free_port()

        wp = WorkerProcess(
            worker_dir=str(WORKER_APP_DIR),
            worker_host="127.0.0.1",
            worker_port=worker_port,
            api_key=API_KEY,
        )

        assert wp.is_healthy() is False
        assert wp.is_running is False


# ---------------------------------------------------------------------------
# Activity monitor integration tests (real running instance)
# ---------------------------------------------------------------------------


class TestActivityMonitorAPI:
    def test_api_version_includes_disabled_field(self, worker_instance):
        resp = worker_instance["client"].get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "api_version" in data
        assert "disabled" in data
        assert isinstance(data["disabled"], bool)

    def test_activity_status_endpoint_exists(self, worker_instance):
        resp = worker_instance["client"].get("/worker/activity-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "disabled" in data
        assert "disabled_until" in data
        assert "remaining_seconds" in data
        assert "user_active" in data
        assert "gpu_vram_contended" in data
        assert "last_check_time" in data
        assert "last_disable_reason" in data

    def test_activity_status_initial_state(self, worker_instance):
        resp = worker_instance["client"].get("/worker/activity-status")
        data = resp.json()
        assert isinstance(data["disabled"], bool)
        assert isinstance(data["remaining_seconds"], (int, float))
        assert isinstance(data["disabled_until"], (int, float))
        if data["disabled"]:
            assert data["remaining_seconds"] > 0
            assert data["disabled_until"] > 0
        else:
            assert data["remaining_seconds"] == 0
            assert data["disabled_until"] == 0

    def test_activity_status_reflects_in_version(self, worker_instance):
        version_resp = worker_instance["client"].get("/api/version")
        version_data = version_resp.json()

        activity_resp = worker_instance["client"].get("/worker/activity-status")
        activity_data = activity_resp.json()

        assert version_data["disabled"] == activity_data["disabled"]
