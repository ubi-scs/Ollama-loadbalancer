import subprocess
import logging
import threading
import time
import os

logger = logging.getLogger(__name__)

DISABLE_COOLDOWN_SECONDS = int(os.getenv("WORKER_DISABLE_COOLDOWN", "900"))
CHECK_INTERVAL_SECONDS = int(os.getenv("WORKER_ACTIVITY_CHECK_INTERVAL", "30"))
VRAM_THRESHOLD_PCT = float(os.getenv("WORKER_VRAM_THRESHOLD_PCT", "25"))
ALLOWED_PROCESS_NAMES = os.getenv(
    "WORKER_ALLOWED_GPU_PROCESSES", "ollama,ollama_llm_server,nvidia-smi"
).split(",")


def _run_command(cmd, timeout=10):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"Command {cmd} failed: {e}")
        return -1, "", str(e)


def get_sessions():
    code, out, err = _run_command(["loginctl", "list-sessions", "--no-legend"])
    if code != 0 or not out:
        return []
    sessions = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            session_id = parts[0]
            session_type = _get_session_type(session_id)
            sessions.append(
                {
                    "session_id": session_id,
                    "type": session_type.lower(),
                }
            )
    return sessions


def _get_session_type(session_id):
    code, out, err = _run_command(
        ["loginctl", "show-session", str(session_id), "-p", "Type"]
    )
    if code != 0:
        return ""
    for line in out.splitlines():
        if "=" in line:
            return line.split("=", 1)[1].strip().lower()
    return ""


def is_session_idle(session_id):
    code, out, err = _run_command(
        ["loginctl", "show-session", str(session_id), "-p", "IdleHint"]
    )
    if code == 0 and "yes" in out.lower():
        return True
    if _is_screensaver_active():
        return True
    return False


def _is_screensaver_active():
    code, out, err = _run_command(
        [
            "dbus-send",
            "--session",
            "--print-reply",
            "--dest=org.freedesktop.ScreenSaver",
            "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.GetActive",
        ],
        timeout=5,
    )
    if code != 0:
        return False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("boolean"):
            return "true" in stripped.lower()
    return False


def is_user_active():
    sessions = get_sessions()
    if not sessions:
        return False
    for session in sessions:
        if session["type"] in ("wayland", "x11"):
            if not is_session_idle(session["session_id"]):
                return True
    return False


def get_process_gpu_memory():
    code, out, err = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0 or not out:
        return []
    processes = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                pid = int(parts[0])
                name = parts[1].lower()
                used_mb = float(parts[2])
                processes.append({"pid": pid, "name": name, "used_memory_mb": used_mb})
            except (ValueError, IndexError):
                continue
    return processes


def get_total_vram_mb():
    code, out, err = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0 or not out:
        return 0
    try:
        return float(out.split("\n")[0].strip())
    except (ValueError, IndexError):
        return 0


def is_gpu_vram_contended(total_vram_mb=None, vram_threshold_pct=None):
    if total_vram_mb is None:
        total_vram_mb = get_total_vram_mb()
    if total_vram_mb <= 0:
        return False
    if vram_threshold_pct is None:
        vram_threshold_pct = VRAM_THRESHOLD_PCT

    threshold_mb = total_vram_mb * (vram_threshold_pct / 100.0)
    processes = get_process_gpu_memory()

    own_pid = os.getpid()
    allowed_names_lower = [n.strip().lower() for n in ALLOWED_PROCESS_NAMES]

    for proc in processes:
        if proc["pid"] == own_pid:
            continue
        if proc["name"] in allowed_names_lower:
            continue
        if proc["used_memory_mb"] >= threshold_mb:
            return True

    return False


class ActivityMonitor:
    def __init__(
        self,
        check_interval=None,
        disable_cooldown=None,
        vram_threshold_pct=None,
    ):
        self.check_interval = check_interval or CHECK_INTERVAL_SECONDS
        self.disable_cooldown = disable_cooldown or DISABLE_COOLDOWN_SECONDS
        self.vram_threshold_pct = (
            vram_threshold_pct if vram_threshold_pct is not None else VRAM_THRESHOLD_PCT
        )

        self._lock = threading.Lock()
        self._disabled_until = 0.0
        self._user_active = False
        self._gpu_contended = False
        self._running = False
        self._stop_event = threading.Event()
        self._thread = None
        self._last_check_time = 0.0
        self._last_disable_reason = None

    def check_now(self):
        user_active = is_user_active()
        gpu_contended = is_gpu_vram_contended(
            vram_threshold_pct=self.vram_threshold_pct
        )

        now = time.time()
        reason = None

        with self._lock:
            self._user_active = user_active
            self._gpu_contended = gpu_contended
            self._last_check_time = now

            if user_active:
                self._disabled_until = now + self.disable_cooldown
                reason = "user_active"
            elif gpu_contended:
                self._disabled_until = now + self.disable_cooldown
                reason = "gpu_vram_contended"
            else:
                if now >= self._disabled_until:
                    self._disabled_until = 0.0

            self._last_disable_reason = reason

    def _monitor_loop(self):
        logger.info("Activity monitor started.")
        while not self._stop_event.is_set():
            try:
                self.check_now()
            except Exception as e:
                logger.error(f"Activity monitor check failed: {e}")
            self._stop_event.wait(self.check_interval)

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.check_now()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def is_disabled(self):
        with self._lock:
            return time.time() < self._disabled_until

    def get_status(self):
        with self._lock:
            now = time.time()
            disabled = now < self._disabled_until
            remaining = max(0, self._disabled_until - now) if disabled else 0
            return {
                "disabled": disabled,
                "disabled_until": self._disabled_until if disabled else 0,
                "remaining_seconds": round(remaining, 1),
                "user_active": self._user_active,
                "gpu_vram_contended": self._gpu_contended,
                "last_check_time": self._last_check_time,
                "last_disable_reason": self._last_disable_reason,
            }
