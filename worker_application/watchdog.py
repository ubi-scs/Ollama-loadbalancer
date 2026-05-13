import subprocess
import logging
import os
import sys
import shutil
import tarfile
import tempfile
import time
import threading
import json
import enum
import signal
from pathlib import Path
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    status,
    Header,
    Depends,
    UploadFile,
    File,
)
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_NAME = "OLLAMA_HELPER_API_KEY"
API_KEY = os.getenv(API_KEY_NAME)

WATCHDOG_HOST = os.getenv("WATCHDOG_HOST", "0.0.0.0")
WATCHDOG_PORT = int(os.getenv("WATCHDOG_PORT", "8001"))

WORKER_HOST = os.getenv("WORKER_HOST", "0.0.0.0")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8000"))

WORKER_DIR = Path(__file__).parent.resolve()
BACKUP_DIR = WORKER_DIR.parent / "worker_application_backup"

HEALTH_CHECK_INTERVAL = int(os.getenv("WATCHDOG_HEALTH_INTERVAL", "10"))
HEALTH_CHECK_TIMEOUT = int(os.getenv("WATCHDOG_HEALTH_TIMEOUT", "30"))
WORKER_STARTUP_TIMEOUT = int(os.getenv("WATCHDOG_STARTUP_TIMEOUT", "30"))


class UpdateStatus(str, enum.Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


_update_lock = threading.Lock()
_update_status = UpdateStatus.IDLE
_update_message = ""
_update_timestamp = ""


def _set_update_status(status_val, message=""):
    global _update_status, _update_message, _update_timestamp
    with _update_lock:
        _update_status = status_val
        _update_message = message
        _update_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%Z")


def _get_update_status():
    with _update_lock:
        return {
            "status": _update_status.value,
            "message": _update_message,
            "timestamp": _update_timestamp,
        }


async def verify_api_key(
    x_api_key: str = Header(..., description="The secret API key."),
):
    expected = API_KEY
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key not configured",
        )
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )


app = FastAPI(
    title="Ollama Worker Watchdog",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)],
)


class WorkerProcess:
    def __init__(
        self,
        worker_dir: str,
        worker_host: str,
        worker_port: int,
        env: Optional[dict] = None,
        api_key: str = "",
    ):
        self.worker_dir = str(worker_dir)
        self.worker_host = worker_host
        self.worker_port = worker_port
        self.env = env if env is not None else {}
        self.api_key = api_key
        self.process = None
        self._lock = threading.Lock()

    def start(self, timeout=None):
        timeout = timeout or WORKER_STARTUP_TIMEOUT
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                logger.warning("Worker process already running.")
                return True

            env = os.environ.copy()
            env.update(self.env)
            env["WORKER_HOST"] = self.worker_host
            env["WORKER_PORT"] = str(self.worker_port)
            env["PYTHONPATH"] = str(Path(self.worker_dir).parent.resolve())

            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "worker_application.main:app",
                "--host",
                self.worker_host,
                "--port",
                str(self.worker_port),
            ]
            logger.info(f"Starting worker process: {' '.join(cmd)}")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=self.worker_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                logger.error(f"Failed to start worker process: {e}")
                return False

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                logger.error(
                    f"Worker process exited early with code {self.process.returncode}"
                )
                return False
            if self.is_healthy():
                logger.info(f"Worker is healthy on port {self.worker_port}")
                return True
            time.sleep(1)

        logger.error("Worker did not become healthy within startup timeout.")
        return False

    def stop(self, timeout=10):
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                return True

            logger.info("Stopping worker process...")
            self.process.send_signal(signal.SIGTERM)

            try:
                self.process.wait(timeout=timeout)
                logger.info("Worker process stopped gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning("Worker did not stop in time, killing.")
                self.process.kill()
                self.process.wait(timeout=5)

            self.process = None
            return True

    def restart(self, startup_timeout=None):
        self.stop()
        return self.start(timeout=startup_timeout)

    def is_healthy(self):
        if httpx is None:
            return False
        try:
            headers = {}
            if self.api_key:
                headers["X-API-KEY"] = self.api_key
            resp = httpx.get(
                f"http://{self.worker_host}:{self.worker_port}/api/version",
                timeout=3,
                headers=headers,
            )
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None


worker_process = None


def _create_backup(worker_dir, backup_dir):
    backup_path = Path(backup_dir)
    if backup_path.exists():
        shutil.rmtree(str(backup_path))
    shutil.copytree(str(worker_dir), str(backup_path))
    logger.info(f"Backup created at {backup_dir}")
    return True


def _deploy_update(staging_dir, worker_dir):
    staging_path = Path(staging_dir)
    worker_path = Path(worker_dir)

    for item in os.listdir(staging_path):
        src = staging_path / item
        dst = worker_path / item
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(str(dst))
            shutil.copytree(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))

    logger.info(f"Deployed update from {staging_dir} to {worker_dir}")
    return True


def _perform_rollback(worker_dir, backup_dir):
    worker_path = Path(worker_dir)
    backup_path = Path(backup_dir)

    if not backup_path.exists():
        logger.error("No backup found for rollback.")
        return False

    try:
        for item in os.listdir(str(worker_path)):
            item_path = worker_path / item
            if item_path.name == "__pycache__":
                continue
            if item_path.is_dir():
                shutil.rmtree(str(item_path))
            else:
                os.remove(str(item_path))

        for item in os.listdir(str(backup_path)):
            src = backup_path / item
            dst = worker_path / item
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.copytree(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

        logger.info("Rollback completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return False


def _perform_self_update(archive_path, worker_proc: WorkerProcess):
    logger.info("Starting self-update process...")
    _set_update_status(UpdateStatus.IN_PROGRESS, "Self-update: creating backup")

    worker_dir = str(worker_proc.worker_dir)
    backup_dir = str(BACKUP_DIR)

    try:
        _create_backup(worker_dir, backup_dir)
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        _set_update_status(
            UpdateStatus.FAILED, f"Self-update failed: backup error: {e}"
        )
        return

    _set_update_status(UpdateStatus.IN_PROGRESS, "Self-update: extracting archive")
    staging_dir = tempfile.mkdtemp(prefix="worker_update_")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(staging_dir)
    except Exception as e:
        logger.error(f"Failed to extract archive: {e}")
        _set_update_status(
            UpdateStatus.FAILED, f"Self-update failed: extract error: {e}"
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    _set_update_status(UpdateStatus.IN_PROGRESS, "Self-update: stopping worker")
    worker_proc.stop()

    _set_update_status(UpdateStatus.IN_PROGRESS, "Self-update: deploying new version")
    try:
        _deploy_update(staging_dir, worker_dir)
    except Exception as e:
        logger.error(f"Failed to deploy update: {e}")
        _set_update_status(
            UpdateStatus.IN_PROGRESS, "Self-update: rolling back due to deploy failure"
        )
        _perform_rollback(worker_dir, backup_dir)
        _set_update_status(
            UpdateStatus.FAILED, f"Self-update failed: deploy error: {e}"
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
        worker_proc.start()
        return

    shutil.rmtree(staging_dir, ignore_errors=True)

    _set_update_status(UpdateStatus.IN_PROGRESS, "Self-update: starting updated worker")
    started = worker_proc.start(timeout=WORKER_STARTUP_TIMEOUT)

    if not started:
        logger.error("Updated worker failed to start. Rolling back.")
        _set_update_status(
            UpdateStatus.IN_PROGRESS,
            "Self-update: rolling back - worker failed to start",
        )
        _perform_rollback(worker_dir, backup_dir)
        worker_proc.start(timeout=WORKER_STARTUP_TIMEOUT)
        _set_update_status(
            UpdateStatus.FAILED, "Self-update failed: updated worker would not start"
        )
        return

    _set_update_status(UpdateStatus.IN_PROGRESS, "Self-update: verifying worker health")
    healthy = worker_proc.is_healthy()

    if healthy:
        logger.info("Updated worker is healthy. Self-update successful.")
        try:
            if Path(backup_dir).exists():
                shutil.rmtree(str(backup_dir))
                logger.info(f"Removed backup at {backup_dir}")
        except Exception as e:
            logger.warning(f"Could not remove backup: {e}")
        _set_update_status(UpdateStatus.SUCCESS, "Self-update completed successfully")
    else:
        logger.error("Updated worker is not healthy. Rolling back.")
        _set_update_status(
            UpdateStatus.IN_PROGRESS, "Self-update: rolling back - worker unhealthy"
        )
        worker_proc.stop()
        _perform_rollback(worker_dir, backup_dir)
        worker_proc.start(timeout=WORKER_STARTUP_TIMEOUT)
        _set_update_status(
            UpdateStatus.FAILED, "Self-update failed: worker unhealthy after update"
        )

    if os.path.exists(archive_path):
        try:
            os.unlink(archive_path)
        except OSError:
            pass


def _health_monitor(worker_proc: WorkerProcess, stop_event: threading.Event):
    logger.info("Health monitor started.")
    while not stop_event.is_set():
        is_set = stop_event.wait(HEALTH_CHECK_INTERVAL)
        if is_set:
            break
        if worker_proc.is_running and not worker_proc.is_healthy():
            logger.warning("Worker appears unhealthy, checking further...")
            time.sleep(2)
            if worker_proc.is_running and not worker_proc.is_healthy():
                logger.warning("Worker still unhealthy, attempting restart.")
                worker_proc.restart()
        elif not worker_proc.is_running:
            logger.warning("Worker process not running, attempting restart.")
            worker_proc.start(timeout=WORKER_STARTUP_TIMEOUT)

    logger.info("Health monitor stopped.")


_health_stop_event = None
_health_thread = None
worker_process = None


@app.on_event("startup")
def on_startup():
    global worker_process, _health_stop_event, _health_thread
    api_key = os.getenv(API_KEY_NAME)
    if not api_key:
        logger.warning(f"{API_KEY_NAME} not set, watchdog API will be inaccessible.")

    worker_process = WorkerProcess(
        worker_dir=str(WORKER_DIR),
        worker_host=WORKER_HOST,
        worker_port=WORKER_PORT,
        env={API_KEY_NAME: api_key} if api_key else {},
        api_key=api_key or "",
    )

    def _start_worker():
        started = worker_process.start()
        if not started:
            logger.error("Worker failed to start initially. Health monitor will retry.")

    _start_thread = threading.Thread(target=_start_worker, daemon=True)
    _start_thread.start()

    _health_stop_event = threading.Event()
    _health_thread = threading.Thread(
        target=_health_monitor,
        args=(worker_process, _health_stop_event),
        daemon=True,
    )
    _health_thread.start()


@app.on_event("shutdown")
def on_shutdown():
    global _health_stop_event, _health_thread, worker_process
    if _health_stop_event:
        _health_stop_event.set()
    if _health_thread:
        _health_thread.join(timeout=5)
    if worker_process:
        worker_process.stop()


@app.get("/watchdog/status")
def get_status():
    worker_healthy = False
    if worker_process:
        worker_healthy = worker_process.is_healthy()
    return {
        "worker_healthy": worker_healthy,
        "worker_running": worker_process.is_running if worker_process else False,
        "update": _get_update_status(),
    }


@app.post("/watchdog/worker/restart")
def restart_worker():
    if not worker_process:
        raise HTTPException(
            status_code=500, detail="Worker process not managed by watchdog."
        )
    success = worker_process.restart()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to restart worker.")
    return {"message": "Worker restarted successfully."}


@app.post("/watchdog/update", status_code=status.HTTP_202_ACCEPTED)
async def trigger_update(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not worker_process:
        raise HTTPException(
            status_code=500, detail="Worker process not managed by watchdog."
        )

    with _update_lock:
        if _update_status == UpdateStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Update already in progress.",
            )

    logger.info(f"Received self-update request with file: {file.filename}")

    if not file.filename or not file.filename.endswith((".tar.gz", ".tgz")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .tar.gz archive",
        )

    archive_data = await file.read()
    tmp_archive = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    try:
        tmp_archive.write(archive_data)
        tmp_archive.close()
    except Exception as e:
        logger.error(f"Failed to write uploaded archive: {e}")
        try:
            os.unlink(tmp_archive.name)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded archive: {e}",
        )

    background_tasks.add_task(_perform_self_update, tmp_archive.name, worker_process)
    return {"message": "Self-update process has been started in the background."}


@app.get("/watchdog/update/status")
def get_update_status():
    return _get_update_status()


@app.post("/watchdog/worker/stop")
def stop_worker():
    if not worker_process:
        raise HTTPException(
            status_code=500, detail="Worker process not managed by watchdog."
        )
    worker_process.stop()
    return {"message": "Worker stopped."}


@app.post("/watchdog/worker/start")
def start_worker():
    if not worker_process:
        raise HTTPException(
            status_code=500, detail="Worker process not managed by watchdog."
        )
    success = worker_process.start()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start worker.")
    return {"message": "Worker started."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WATCHDOG_HOST, port=WATCHDOG_PORT)
