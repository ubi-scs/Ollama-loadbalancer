import subprocess
import logging
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Header, Depends

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_KEY_NAME = "OLLAMA-HELPER-API-Key"
API_KEY = os.getenv(API_KEY_NAME)

if not API_KEY:
    raise RuntimeError(f"{API_KEY_NAME} environment variable not set. Please provide a secret API Key.")


async def verify_api_key(x_api_key: str = Header(..., description="The secret API key.")):
    """
    Dependency to verify the API key in the request header.

    Raises:
        HTTPException: If the API key is missing or invalid.
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key"
        )


app = FastAPI(
    title="Ollama Worker",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)],
)


def run_command(command: list[str]) -> tuple[int, str, str]:
    """Runs a shell command and returns status code, stdout, and stderr."""
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return process.returncode, process.stdout.strip(), process.stderr.strip()
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        return -1, "", f"Command not found: {command[0]}"
    except Exception as e:
        logger.error(f"An error occurred while running command '{' '.join(command)}': {e}")
        return -1, "", str(e)


@app.get("/gpu/utilization")
def get_gpu_utilization():
    """Reports current GPU utilization percentage."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = run_command(cmd)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"Failed to query nvidia-smi: {err}")

    try:
        utilization = float(out.split('\n')[0])
        return {"gpu_utilization_percent": utilization}
    except (ValueError, IndexError):
        raise HTTPException(status_code=500, detail="Could not parse nvidia-smi output.")


@app.get("/gpu/vram")
def get_gpu_vram():
    """Reports current GPU VRAM usage in MiB."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = run_command(cmd)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"Failed to query nvidia-smi: {err}")

    try:
        used, total = map(float, out.split('\n')[0].split(','))
        return {"vram_used_mb": used, "vram_total_mb": total}
    except (ValueError, IndexError):
        raise HTTPException(status_code=500, detail="Could not parse nvidia-smi output.")


@app.post("/ollama/restart")
def restart_ollama():
    """
    Restarts the ollama service.
    """
    logger.info("Received request to restart Ollama service.")
    cmd = ["sudo", "systemctl", "restart", "ollama.service"]
    code, out, err = run_command(cmd)

    if code != 0:
        logger.error(f"Failed to restart ollama: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to restart ollama: {err}")

    logger.info("Ollama service restarted successfully.")
    return {"message": "Ollama service is restarting."}


def _run_ollama_update():
    """The actual update logic to be run in the background."""
    logger.info("Starting Ollama update in the background...")
    command_str = "curl -fsSL https://ollama.com/install.sh | sh"
    logger.info(f"Running command: {command_str}")
    try:
        process = subprocess.run(
            command_str,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        code, out, err = process.returncode, process.stdout.strip(), process.stderr.strip()
    except Exception as e:
        logger.error(f"An exception occurred during Ollama update: {e}")
        return

    if code != 0:
        logger.error(f"Ollama update failed. Stdout: {out}, Stderr: {err}")
    else:
        logger.info(f"Ollama update script finished successfully. Stdout: {out}")

    logger.info("Restarting Ollama service post-update.")
    run_command(["sudo", "systemctl", "restart", "ollama.service"])


@app.post("/ollama/update", status_code=status.HTTP_202_ACCEPTED)
def update_ollama(background_tasks: BackgroundTasks):
    """
    Triggers an update of Ollama by re-running the install script.
    """
    logger.info("Received request to update Ollama. Scheduling background task.")
    background_tasks.add_task(_run_ollama_update)
    return {"message": "Ollama update process has been started in the background."}
