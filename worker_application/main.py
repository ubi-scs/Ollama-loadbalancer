import subprocess
import logging
import os
import sys
import re
import shutil
import tarfile
import tempfile
import time
import threading
import json
import enum
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

from worker_application.activity_monitor import ActivityMonitor

try:
    import httpx
except ImportError:
    httpx = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_KEY_NAME = "OLLAMA_HELPER_API_KEY"
API_KEY = os.getenv(API_KEY_NAME)

HOST = os.getenv("WORKER_HOST", "0.0.0.0")
PORT = int(os.getenv("WORKER_PORT", "8000"))

WATCHDOG_HOST = os.getenv("WATCHDOG_HOST", "0.0.0.0")
WATCHDOG_PORT = int(os.getenv("WATCHDOG_PORT", "8001"))

OLLAMA_UPDATE_TIMEOUT = int(os.getenv("OLLAMA_UPDATE_TIMEOUT", "300"))
OLLAMA_UPDATE_MAX_RETRIES = int(os.getenv("OLLAMA_UPDATE_MAX_RETRIES", "3"))
OLLAMA_UPDATE_RETRY_DELAY = int(os.getenv("OLLAMA_UPDATE_RETRY_DELAY", "10"))


class OllamaUpdateStatus(str, enum.Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


_ollama_update_lock = threading.Lock()
_ollama_update_status = OllamaUpdateStatus.IDLE
_ollama_update_message = ""
_ollama_update_timestamp = ""

_activity_monitor = ActivityMonitor()


async def verify_api_key(
    x_api_key: str = Header(..., description="The secret API key."),
):
    expected = os.getenv(API_KEY_NAME)
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
    title="Ollama Worker",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)],
)


def run_command(command, timeout=None):
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return process.returncode, process.stdout.strip(), process.stderr.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {' '.join(command)}")
        return -2, "", "Command timed out"
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        return -1, "", f"Command not found: {command[0]}"
    except Exception as e:
        logger.error(
            f"An error occurred while running command '{' '.join(command)}': {e}"
        )
        return -1, "", str(e)


def _set_ollama_update_status(status_val, message=""):
    global _ollama_update_status, _ollama_update_message, _ollama_update_timestamp
    with _ollama_update_lock:
        _ollama_update_status = status_val
        _ollama_update_message = message
        _ollama_update_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%Z")


def _get_ollama_update_status():
    with _ollama_update_lock:
        return {
            "status": _ollama_update_status.value,
            "message": _ollama_update_message,
            "timestamp": _ollama_update_timestamp,
        }


@app.get("/api/version")
def get_api_version():
    return {
        "api_version": "1.0.0",
        "disabled": _activity_monitor.is_disabled(),
    }


@app.get("/gpu/utilization")
def get_gpu_utilization():
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = run_command(cmd)
    if code != 0:
        raise HTTPException(
            status_code=500, detail=f"Failed to query nvidia-smi: {err}"
        )

    try:
        utilization = float(out.split("\n")[0])
        return {"gpu_utilization_percent": utilization}
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=500, detail="Could not parse nvidia-smi output."
        )


@app.get("/gpu/vram")
def get_gpu_vram():
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = run_command(cmd)
    if code != 0:
        raise HTTPException(
            status_code=500, detail=f"Failed to query nvidia-smi: {err}"
        )

    try:
        used, total = map(float, out.split("\n")[0].split(","))
        return {"vram_used_mb": used, "vram_total_mb": total}
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=500, detail="Could not parse nvidia-smi output."
        )


@app.post("/ollama/restart")
def restart_ollama():
    logger.info("Received request to restart Ollama service.")
    cmd = ["sudo", "systemctl", "restart", "ollama.service"]
    code, out, err = run_command(cmd)

    if code != 0:
        logger.error(f"Failed to restart ollama: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to restart ollama: {err}")

    logger.info("Ollama service restarted successfully.")
    return {"message": "Ollama service is restarting."}


def patch_ollama_service():
    service_path = "/etc/systemd/system/ollama.service"
    env_lines = [
        'Environment="OLLAMA_HOST=0.0.0.0"\n',
        'Environment="OLLAMA_FLASH_ATTENTION=1"\n',
        'Environment="OLLAMA_NOHISTORY=1"\n',
    ]
    try:
        with open(service_path, "r") as f:
            lines = f.readlines()
        last_env_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("Environment="):
                last_env_idx = idx

        if last_env_idx is not None:
            insert_idx = last_env_idx + 1

            already_present = any(env_lines[0].strip() in l for l in lines)
            if not already_present:
                lines[insert_idx:insert_idx] = env_lines
                with open(service_path, "w") as f:
                    f.writelines(lines)
                subprocess.run(["sudo", "systemctl", "daemon-reload"], check=False)
    except Exception as e:
        logger.error(f"Failed to patch ollama.service: {e}")


def _run_ollama_update():
    logger.info("Starting Ollama update...")

    with _ollama_update_lock:
        if _ollama_update_status == OllamaUpdateStatus.IN_PROGRESS:
            logger.warning("Ollama update already in progress, skipping.")
            return

    _set_ollama_update_status(OllamaUpdateStatus.IN_PROGRESS, "Update started")

    command_str = "curl -fsSL https://ollama.com/install.sh | sh"
    last_error = ""

    for attempt in range(1, OLLAMA_UPDATE_MAX_RETRIES + 1):
        try:
            logger.info(f"Ollama update attempt {attempt}/{OLLAMA_UPDATE_MAX_RETRIES}")
            _set_ollama_update_status(
                OllamaUpdateStatus.IN_PROGRESS,
                f"Attempt {attempt}/{OLLAMA_UPDATE_MAX_RETRIES}",
            )

            process = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                timeout=OLLAMA_UPDATE_TIMEOUT,
            )
            code = process.returncode
            out = process.stdout.strip()
            err = process.stderr.strip()

            if code == 0:
                logger.info(
                    f"Ollama update script finished successfully. Stdout: {out}"
                )

                logger.info(
                    "Patching ollama.service with required Environment variables."
                )
                patch_ollama_service()

                logger.info("Restarting Ollama service post-update.")
                restart_code, restart_out, restart_err = run_command(
                    ["sudo", "systemctl", "restart", "ollama.service"],
                    timeout=60,
                )
                if restart_code != 0:
                    _set_ollama_update_status(
                        OllamaUpdateStatus.FAILED,
                        f"Update succeeded but restart failed: {restart_err}",
                    )
                    return

                _set_ollama_update_status(
                    OllamaUpdateStatus.SUCCESS, "Update completed successfully"
                )
                return
            else:
                last_error = f"Exit code {code}. Stderr: {err}. Stdout: {out}"
                logger.error(f"Ollama update attempt {attempt} failed: {last_error}")

        except subprocess.TimeoutExpired:
            last_error = f"Attempt {attempt} timed out after {OLLAMA_UPDATE_TIMEOUT}s"
            logger.error(last_error)
        except Exception as e:
            last_error = f"Attempt {attempt} exception: {e}"
            logger.error(last_error)

        if attempt < OLLAMA_UPDATE_MAX_RETRIES:
            logger.info(f"Retrying in {OLLAMA_UPDATE_RETRY_DELAY}s...")
            time.sleep(OLLAMA_UPDATE_RETRY_DELAY)

    _set_ollama_update_status(
        OllamaUpdateStatus.FAILED,
        f"All {OLLAMA_UPDATE_MAX_RETRIES} attempts failed. Last error: {last_error}",
    )
    logger.error(f"Ollama update failed after {OLLAMA_UPDATE_MAX_RETRIES} attempts.")


@app.post("/ollama/update", status_code=status.HTTP_202_ACCEPTED)
def update_ollama(background_tasks: BackgroundTasks):
    with _ollama_update_lock:
        if _ollama_update_status == OllamaUpdateStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ollama update already in progress.",
            )

    logger.info("Received request to update Ollama. Scheduling background task.")
    background_tasks.add_task(_run_ollama_update)
    return {"message": "Ollama update process has been started in the background."}


@app.get("/ollama/update/status")
def get_ollama_update_status():
    return _get_ollama_update_status()


@app.post("/worker/self-update", status_code=status.HTTP_202_ACCEPTED)
async def self_update(file: UploadFile = File(...)):
    if httpx is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="httpx not available, cannot forward to watchdog.",
        )

    if not file.filename or not file.filename.endswith((".tar.gz", ".tgz")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .tar.gz archive",
        )

    archive_data = await file.read()

    watchdog_url = f"http://{WATCHDOG_HOST}:{WATCHDOG_PORT}/watchdog/update"
    api_key = os.getenv(API_KEY_NAME, "")

    try:
        resp = httpx.post(
            watchdog_url,
            headers={"X-API-KEY": api_key},
            files={"file": (file.filename, archive_data, "application/gzip")},
            timeout=30,
        )
        if resp.status_code != status.HTTP_202_ACCEPTED:
            raise HTTPException(
                status_code=resp.status_code,
                detail=resp.json().get("detail", "Watchdog rejected update request."),
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to watchdog service.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error forwarding to watchdog: {e}",
        )

    return {"message": "Self-update request forwarded to watchdog."}


@app.get("/worker/self-update/status")
def get_self_update_status():
    if httpx is None:
        return {"status": "unknown", "message": "httpx not available", "timestamp": ""}

    watchdog_url = f"http://{WATCHDOG_HOST}:{WATCHDOG_PORT}/watchdog/update/status"
    api_key = os.getenv(API_KEY_NAME, "")

    try:
        resp = httpx.get(watchdog_url, headers={"X-API-KEY": api_key}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    return {"status": "unknown", "message": "Cannot reach watchdog", "timestamp": ""}


@app.on_event("startup")
def _start_activity_monitor():
    _activity_monitor.start()


@app.on_event("shutdown")
def _stop_activity_monitor():
    _activity_monitor.stop()


@app.get("/worker/activity-status")
def get_activity_status():
    return _activity_monitor.get_status()


@app.post("/worker/trigger-update", status_code=status.HTTP_202_ACCEPTED)
async def trigger_self_update(background_tasks: BackgroundTasks):
    if httpx is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="httpx not available, cannot forward to watchdog.",
        )

    watchdog_trigger_url = (
        f"http://{WATCHDOG_HOST}:{WATCHDOG_PORT}/watchdog/trigger-update"
    )
    api_key = os.getenv(API_KEY_NAME, "")

    try:
        resp = httpx.post(
            watchdog_trigger_url,
            headers={"X-API-KEY": api_key},
            timeout=30,
        )
        if resp.status_code == status.HTTP_202_ACCEPTED:
            return {"message": "Worker self-update triggered via watchdog."}
        if resp.status_code == status.HTTP_409_CONFLICT:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Update already in progress.",
            )
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.json().get("detail", "Watchdog rejected trigger request."),
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cannot connect to watchdog service.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error forwarding trigger to watchdog: {e}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
