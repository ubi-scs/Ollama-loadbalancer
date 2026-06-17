import subprocess
import logging
import threading
import time
import os
import glob as _glob

try:
    import evdev

    _EVDEV_AVAILABLE = True
except ImportError:
    evdev = None
    _EVDEV_AVAILABLE = False

logger = logging.getLogger(__name__)

DISABLE_COOLDOWN_SECONDS = int(os.getenv("WORKER_DISABLE_COOLDOWN", "60"))
CHECK_INTERVAL_SECONDS = int(os.getenv("WORKER_ACTIVITY_CHECK_INTERVAL", "30"))
IDLE_TIMEOUT_SECONDS = int(os.getenv("WORKER_IDLE_TIMEOUT", "30"))
VRAM_THRESHOLD_PCT = float(os.getenv("WORKER_VRAM_THRESHOLD_PCT", "25"))
ALLOWED_PROCESS_NAMES = os.getenv(
    "WORKER_ALLOWED_GPU_PROCESSES", "ollama,ollama_llm_server,llama-server,llama_server,nvidia-smi"
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


class InputDeviceMonitor:
    def __init__(self, idle_timeout=None):
        self.idle_timeout = (
            idle_timeout if idle_timeout is not None else IDLE_TIMEOUT_SECONDS
        )
        self._last_activity = time.monotonic()
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._threads = []
        self._monitored_paths = set()
        self._available = False

    def is_available(self):
        return _EVDEV_AVAILABLE and self._available

    def start(self):
        if self._running:
            return
        if not _EVDEV_AVAILABLE:
            logger.info("evdev not available, input device monitoring disabled")
            return
        self._running = True
        self._stop_event.clear()
        self._open_devices()
        if self._monitored_paths:
            self._available = True
            t = threading.Thread(target=self._rescan_loop, daemon=True)
            t.start()
            self._threads.append(t)
            logger.info(
                f"Input device monitor started, watching {len(self._monitored_paths)} devices"
            )
        else:
            logger.info(
                "No input devices could be opened, input monitoring unavailable"
            )
        self._last_activity = time.monotonic()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()
        self._monitored_paths.clear()
        self._available = False

    def _open_devices(self):
        try:
            device_paths = evdev.list_devices()
        except Exception:
            device_paths = _glob.glob("/dev/input/event*")
        for path in device_paths:
            if path in self._monitored_paths:
                continue
            try:
                dev = evdev.InputDevice(path)
                if evdev.ecodes.EV_KEY in dev.capabilities():
                    self._monitored_paths.add(path)
                    t = threading.Thread(
                        target=self._monitor_device, args=(path,), daemon=True
                    )
                    t.start()
                    self._threads.append(t)
                    logger.debug(f"Monitoring input device: {dev.name} ({path})")
            except (OSError, PermissionError) as e:
                logger.debug(f"Cannot open {path}: {e}")
            except Exception as e:
                logger.debug(f"Error opening {path}: {e}")

    def _monitor_device(self, path):
        try:
            dev = evdev.InputDevice(path)
            for event in dev.read_loop():
                if self._stop_event.is_set():
                    return
                if event.type == evdev.ecodes.EV_KEY:
                    with self._lock:
                        self._last_activity = time.monotonic()
        except OSError:
            logger.debug(f"Device {path} disconnected or inaccessible")
            self._monitored_paths.discard(path)
        except Exception as e:
            logger.debug(f"Error monitoring {path}: {e}")
            self._monitored_paths.discard(path)

    def _rescan_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(30)
            if self._stop_event.is_set():
                break
            try:
                self._open_devices()
            except Exception as e:
                logger.debug(f"Device rescan failed: {e}")

    def is_user_active(self):
        with self._lock:
            return (time.monotonic() - self._last_activity) < self.idle_timeout

    def get_idle_seconds(self):
        with self._lock:
            return time.monotonic() - self._last_activity


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


def _is_session_idle_loginctl(session_id):
    code, out, err = _run_command(
        ["loginctl", "show-session", str(session_id), "-p", "IdleHint"]
    )
    if code == 0 and "yes" in out.lower():
        return True
    return None


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
        return None
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("boolean"):
            return "true" in stripped.lower()
    return None


def _is_user_idle_loginctl():
    sessions = get_sessions()
    if not sessions:
        return None
    for session in sessions:
        if session["type"] not in ("wayland", "x11"):
            continue
        idle_hint = _is_session_idle_loginctl(session["session_id"])
        if idle_hint is True:
            return True
        if idle_hint is None:
            screensaver = _is_screensaver_active()
            if screensaver is True:
                return True
    return False


class ActivityMonitor:
    def __init__(
        self,
        check_interval=None,
        disable_cooldown=None,
        vram_threshold_pct=None,
        idle_timeout=None,
    ):
        self.check_interval = check_interval or CHECK_INTERVAL_SECONDS
        self.disable_cooldown = disable_cooldown or DISABLE_COOLDOWN_SECONDS
        self.vram_threshold_pct = (
            vram_threshold_pct if vram_threshold_pct is not None else VRAM_THRESHOLD_PCT
        )
        self.idle_timeout = (
            idle_timeout if idle_timeout is not None else IDLE_TIMEOUT_SECONDS
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
        self._input_monitor = InputDeviceMonitor(idle_timeout=self.idle_timeout)

    def _is_user_active(self):
        if self._input_monitor.is_available():
            return self._input_monitor.is_user_active()
        loginctl_idle = _is_user_idle_loginctl()
        if loginctl_idle is False:
            return True
        if loginctl_idle is True:
            return False
        return False

    def check_now(self):
        user_active = self._is_user_active()
        foreign_procs = get_foreign_gpu_processes(
            vram_threshold_pct=self.vram_threshold_pct
        )
        gpu_contended = len(foreign_procs) > 0

        now = time.time()

        with self._lock:
            was_disabled = now < self._disabled_until
            self._user_active = user_active
            self._gpu_contended = gpu_contended
            self._last_check_time = now

            if user_active:
                self._disabled_until = now + self.disable_cooldown
                reason = "user_active"
                idle_secs = (
                    self._input_monitor.get_idle_seconds()
                    if self._input_monitor.is_available()
                    else None
                )
                if not was_disabled:
                    if idle_secs is not None:
                        logger.info(
                            "Worker self-disabling: user activity detected "
                            "(idle %.1fs, below threshold of %ds). "
                            "Worker will be unavailable for %ds.",
                            idle_secs,
                            self.idle_timeout,
                            self.disable_cooldown,
                        )
                    else:
                        logger.info(
                            "Worker self-disabling: user activity detected "
                            "(loginctl/screensaver reports user active). "
                            "Worker will be unavailable for %ds.",
                            self.disable_cooldown,
                        )
                else:
                    logger.debug(
                        "Worker disable extended: user still active "
                        "(idle %.1fs). Cooldown extended by %ds.",
                        idle_secs if idle_secs is not None else -1,
                        self.disable_cooldown,
                    )
            elif gpu_contended:
                self._disabled_until = now + self.disable_cooldown
                reason = "gpu_vram_contended"
                if not was_disabled:
                    proc_details = ", ".join(
                        f"PID={p['pid']} name={p['name']} vram={p['used_memory_mb']:.0f}MB"
                        for p in foreign_procs
                    )
                    logger.info(
                        "Worker self-disabling: GPU VRAM contention detected "
                        "(>=%.0f%% of VRAM threshold). "
                        "Foreign process(es): %s. "
                        "Worker will be unavailable for %ds.",
                        self.vram_threshold_pct,
                        proc_details,
                        self.disable_cooldown,
                    )
                else:
                    logger.debug(
                        "Worker disable extended: GPU VRAM still contended. "
                        "Cooldown extended by %ds.",
                        self.disable_cooldown,
                    )
            else:
                reason = None
                if was_disabled and now >= self._disabled_until:
                    self._disabled_until = 0.0
                    logger.info(
                        "Worker re-enabling: user is idle and GPU VRAM is free. "
                        "Worker is now available for requests."
                    )
                elif was_disabled:
                    logger.debug(
                        "Worker still disabled, cooldown remaining: %.0fs.",
                        self._disabled_until - now,
                    )

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
        self._input_monitor.start()
        if self._input_monitor.is_available():
            logger.info(
                "Activity monitor: using evdev input device monitoring "
                "(%d devices, idle timeout %ds)",
                len(self._input_monitor._monitored_paths),
                self.idle_timeout,
            )
        else:
            logger.info(
                "Activity monitor: evdev not available, falling back to "
                "loginctl/screensaver idle detection"
            )
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.check_now()

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        self._input_monitor.stop()
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
            status = {
                "disabled": disabled,
                "disabled_until": self._disabled_until if disabled else 0,
                "remaining_seconds": round(remaining, 1),
                "user_active": self._user_active,
                "gpu_vram_contended": self._gpu_contended,
                "last_check_time": self._last_check_time,
                "last_disable_reason": self._last_disable_reason,
            }
            if self._input_monitor.is_available():
                status["idle_seconds"] = round(
                    self._input_monitor.get_idle_seconds(), 1
                )
                status["input_devices_monitored"] = len(
                    self._input_monitor._monitored_paths
                )
            return status


def get_foreign_gpu_processes(total_vram_mb=None, vram_threshold_pct=None):
    if total_vram_mb is None:
        total_vram_mb = get_total_vram_mb()
    if total_vram_mb <= 0:
        return []
    if vram_threshold_pct is None:
        vram_threshold_pct = VRAM_THRESHOLD_PCT

    threshold_mb = total_vram_mb * (vram_threshold_pct / 100.0)
    processes = get_process_gpu_memory()

    own_pid = os.getpid()
    allowed_names_lower = [n.strip().lower() for n in ALLOWED_PROCESS_NAMES]

    foreign = []
    for proc in processes:
        if proc["pid"] == own_pid:
            continue
        if proc["name"] in allowed_names_lower:
            continue
        if proc["used_memory_mb"] >= threshold_mb:
            foreign.append(proc)

    return foreign


def is_gpu_vram_contended(total_vram_mb=None, vram_threshold_pct=None):
    return len(get_foreign_gpu_processes(total_vram_mb, vram_threshold_pct)) > 0


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
                name = os.path.basename(parts[1]).lower()
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
