import time
import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from worker_application.activity_monitor import (
    ActivityMonitor,
    InputDeviceMonitor,
    get_sessions,
    _get_session_type,
    _is_session_idle_loginctl,
    _is_user_idle_loginctl,
    _is_screensaver_active,
    get_process_gpu_memory,
    get_total_vram_mb,
    is_gpu_vram_contended,
    _run_command,
    DISABLE_COOLDOWN_SECONDS,
    CHECK_INTERVAL_SECONDS,
    VRAM_THRESHOLD_PCT,
    ALLOWED_PROCESS_NAMES,
    IDLE_TIMEOUT_SECONDS,
    _EVDEV_AVAILABLE,
)


class TestRunCommand:
    def test_successful_command(self):
        code, out, err = _run_command(["echo", "hello"])
        assert code == 0
        assert out == "hello"

    def test_failed_command(self):
        code, out, err = _run_command(["false"])
        assert code != 0

    def test_command_not_found(self):
        code, out, err = _run_command(["nonexistent_command_xyz"])
        assert code == -1

    def test_command_timeout(self):
        code, out, err = _run_command(["sleep", "10"], timeout=1)
        assert code in (-1, -2)


class TestGetSessionType:
    @patch("worker_application.activity_monitor._run_command")
    def test_parses_type(self, mock_run):
        mock_run.return_value = (0, "Type=wayland", "")
        assert _get_session_type("1") == "wayland"

    @patch("worker_application.activity_monitor._run_command")
    def test_parses_x11_type(self, mock_run):
        mock_run.return_value = (0, "Type=x11", "")
        assert _get_session_type("2") == "x11"

    @patch("worker_application.activity_monitor._run_command")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        assert _get_session_type("1") == ""

    @patch("worker_application.activity_monitor._run_command")
    def test_returns_empty_on_no_equals(self, mock_run):
        mock_run.return_value = (0, "no_equals_sign", "")
        assert _get_session_type("1") == ""


class TestGetSessions:
    @patch("worker_application.activity_monitor._get_session_type")
    @patch("worker_application.activity_monitor._run_command")
    def test_parses_sessions(self, mock_run, mock_type):
        mock_run.return_value = (
            0,
            "1  1000  user1  seat0  4394  user  -  no  -\n2  1000  user2  seat0  9433  user  tty1  no  -",
            "",
        )
        mock_type.side_effect = lambda sid: "wayland" if sid == "1" else "x11"
        sessions = get_sessions()
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "1"
        assert sessions[0]["type"] == "wayland"
        assert sessions[1]["session_id"] == "2"
        assert sessions[1]["type"] == "x11"

    @patch("worker_application.activity_monitor._get_session_type")
    @patch("worker_application.activity_monitor._run_command")
    def test_queries_type_per_session(self, mock_run, mock_type):
        mock_run.return_value = (
            0,
            "1  1000  user1  seat0  4394  user  -  no  -\n2  1000  user2  -  9433  manager  -  no  -",
            "",
        )
        mock_type.side_effect = lambda sid: {"1": "wayland", "2": "unspecified"}[sid]
        sessions = get_sessions()
        assert sessions[0]["type"] == "wayland"
        assert sessions[1]["type"] == "unspecified"

    @patch("worker_application.activity_monitor._run_command")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        sessions = get_sessions()
        assert sessions == []

    @patch("worker_application.activity_monitor._run_command")
    def test_returns_empty_on_empty_output(self, mock_run):
        mock_run.return_value = (0, "", "")
        sessions = get_sessions()
        assert sessions == []


class TestIsSessionIdleLoginctl:
    @patch("worker_application.activity_monitor._run_command")
    def test_idle_yes(self, mock_run):
        mock_run.return_value = (0, "IdleHint=yes", "")
        assert _is_session_idle_loginctl("1") is True

    @patch("worker_application.activity_monitor._run_command")
    def test_idle_no(self, mock_run):
        mock_run.return_value = (0, "IdleHint=no", "")
        assert _is_session_idle_loginctl("1") is None

    @patch("worker_application.activity_monitor._run_command")
    def test_idle_on_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        assert _is_session_idle_loginctl("1") is None


class TestIsScreensaverActive:
    @patch("worker_application.activity_monitor._run_command")
    def test_screensaver_active(self, mock_run):
        mock_run.return_value = (
            0,
            "method return time=1234 sender=:1.23 -> destination=:1.378 serial=1 reply_serial=2\n   boolean true",
            "",
        )
        assert _is_screensaver_active() is True

    @patch("worker_application.activity_monitor._run_command")
    def test_screensaver_inactive(self, mock_run):
        mock_run.return_value = (
            0,
            "method return time=1234 sender=:1.23 -> destination=:1.378 serial=1 reply_serial=2\n   boolean false",
            "",
        )
        assert _is_screensaver_active() is False

    @patch("worker_application.activity_monitor._run_command")
    def test_screensaver_dbus_failure(self, mock_run):
        mock_run.return_value = (
            1,
            "",
            "Error org.freedesktop.DBus.Error.ServiceUnknown",
        )
        assert _is_screensaver_active() is None

    @patch("worker_application.activity_monitor._run_command")
    def test_screensaver_no_boolean_in_output(self, mock_run):
        mock_run.return_value = (0, "method return time=1234\n   string unexpected", "")
        assert _is_screensaver_active() is None


class TestIsUserIdleLoginctl:
    @patch("worker_application.activity_monitor._is_screensaver_active")
    @patch("worker_application.activity_monitor._is_session_idle_loginctl")
    @patch("worker_application.activity_monitor.get_sessions")
    def test_idle_when_loginctl_says_idle(self, mock_sessions, mock_idle, mock_ss):
        mock_sessions.return_value = [{"session_id": "1", "type": "wayland"}]
        mock_idle.return_value = True
        assert _is_user_idle_loginctl() is True

    @patch("worker_application.activity_monitor._is_screensaver_active")
    @patch("worker_application.activity_monitor._is_session_idle_loginctl")
    @patch("worker_application.activity_monitor.get_sessions")
    def test_not_idle_when_loginctl_says_active(
        self, mock_sessions, mock_idle, mock_ss
    ):
        mock_sessions.return_value = [{"session_id": "1", "type": "wayland"}]
        mock_idle.return_value = None
        mock_ss.return_value = False
        assert _is_user_idle_loginctl() is False

    @patch("worker_application.activity_monitor._is_screensaver_active")
    @patch("worker_application.activity_monitor._is_session_idle_loginctl")
    @patch("worker_application.activity_monitor.get_sessions")
    def test_idle_when_screensaver_active(self, mock_sessions, mock_idle, mock_ss):
        mock_sessions.return_value = [{"session_id": "1", "type": "wayland"}]
        mock_idle.return_value = None
        mock_ss.return_value = True
        assert _is_user_idle_loginctl() is True

    @patch("worker_application.activity_monitor.get_sessions")
    def test_no_sessions_returns_none(self, mock_sessions):
        mock_sessions.return_value = []
        assert _is_user_idle_loginctl() is None

    @patch("worker_application.activity_monitor._is_session_idle_loginctl")
    @patch("worker_application.activity_monitor.get_sessions")
    def test_unspecified_session_skipped(self, mock_sessions, mock_idle):
        mock_sessions.return_value = [{"session_id": "1", "type": "unspecified"}]
        mock_idle.return_value = None
        assert _is_user_idle_loginctl() is False


class TestInputDeviceMonitor:
    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", False)
    def test_not_available_when_evdev_missing(self):
        monitor = InputDeviceMonitor(idle_timeout=300)
        assert monitor.is_available() is False

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_start_no_devices(self, mock_evdev):
        mock_evdev.list_devices.return_value = []
        monitor = InputDeviceMonitor(idle_timeout=300)
        monitor.start()
        assert monitor.is_available() is False
        monitor.stop()

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_is_user_active_initially(self, mock_evdev):
        mock_evdev.list_devices.return_value = []
        monitor = InputDeviceMonitor(idle_timeout=300)
        assert monitor.is_user_active() is True
        assert monitor.get_idle_seconds() < 1

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_is_user_active_after_timeout(self, mock_evdev):
        monitor = InputDeviceMonitor(idle_timeout=300)
        monitor._last_activity = time.monotonic() - 600
        assert monitor.is_user_active() is False
        assert monitor.get_idle_seconds() >= 300

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_start_stop_lifecycle(self, mock_evdev):
        mock_evdev.list_devices.return_value = []
        monitor = InputDeviceMonitor(idle_timeout=300)
        monitor.start()
        monitor.stop()
        assert monitor._running is False

    def test_stop_without_start(self):
        monitor = InputDeviceMonitor(idle_timeout=300)
        monitor.stop()
        assert monitor._running is False

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_device_opens_with_ev_key(self, mock_evdev):
        mock_device = MagicMock()
        mock_device.name = "Test Keyboard"
        mock_device.capabilities.return_value = {1: []}
        mock_evdev.InputDevice.return_value = mock_device
        mock_evdev.ecodes.EV_KEY = 1
        mock_evdev.list_devices.return_value = ["/dev/input/event2"]

        monitor = InputDeviceMonitor(idle_timeout=300)
        with patch.object(monitor, "_monitor_device"):
            monitor._open_devices()
        assert "/dev/input/event2" in monitor._monitored_paths

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_device_without_ev_key_skipped(self, mock_evdev):
        mock_device = MagicMock()
        mock_device.name = "Power Button"
        mock_device.capabilities.return_value = {}
        mock_evdev.InputDevice.return_value = mock_device
        mock_evdev.ecodes.EV_KEY = 1
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]

        monitor = InputDeviceMonitor(idle_timeout=300)
        with patch.object(monitor, "_monitor_device"):
            monitor._open_devices()
        assert "/dev/input/event0" not in monitor._monitored_paths

    @patch("worker_application.activity_monitor._EVDEV_AVAILABLE", True)
    @patch("worker_application.activity_monitor.evdev", create=True)
    def test_permission_error_handled(self, mock_evdev):
        mock_evdev.InputDevice.side_effect = PermissionError("Permission denied")
        mock_evdev.list_devices.return_value = ["/dev/input/event0"]

        monitor = InputDeviceMonitor(idle_timeout=300)
        monitor._open_devices()
        assert len(monitor._monitored_paths) == 0


class TestActivityMonitorIsUserActive:
    @patch("worker_application.activity_monitor._is_user_idle_loginctl")
    def test_uses_input_monitor_when_available(self, mock_loginctl):
        mock_loginctl.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = True
        mock_input.is_user_active.return_value = True
        monitor._input_monitor = mock_input

        assert monitor._is_user_active() is True
        mock_input.is_user_active.assert_called_once()

    @patch("worker_application.activity_monitor._is_user_idle_loginctl")
    def test_falls_back_to_loginctl_when_input_unavailable(self, mock_loginctl):
        mock_loginctl.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = False
        monitor._input_monitor = mock_input

        result = monitor._is_user_active()
        assert result is True
        mock_loginctl.assert_called()

    @patch("worker_application.activity_monitor._is_user_idle_loginctl")
    def test_loginctl_idle_means_not_active(self, mock_loginctl):
        mock_loginctl.return_value = True
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = False
        monitor._input_monitor = mock_input

        assert monitor._is_user_active() is False

    @patch("worker_application.activity_monitor._is_user_idle_loginctl")
    def test_loginctl_none_means_not_active(self, mock_loginctl):
        mock_loginctl.return_value = None
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = False
        monitor._input_monitor = mock_input

        assert monitor._is_user_active() is False

    @patch("worker_application.activity_monitor._is_user_idle_loginctl")
    def test_input_monitor_idle_takes_priority_over_loginctl(self, mock_loginctl):
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = True
        mock_input.is_user_active.return_value = False
        monitor._input_monitor = mock_input

        assert monitor._is_user_active() is False
        mock_loginctl.assert_not_called()


class TestGetProcessGpuMemory:
    @patch("worker_application.activity_monitor._run_command")
    def test_parses_processes(self, mock_run):
        mock_run.return_value = (
            0,
            "1234,python,512\n5678,ollama,4096\n",
            "",
        )
        procs = get_process_gpu_memory()
        assert len(procs) == 2
        assert procs[0]["pid"] == 1234
        assert procs[0]["name"] == "python"
        assert procs[0]["used_memory_mb"] == 512

    @patch("worker_application.activity_monitor._run_command")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        procs = get_process_gpu_memory()
        assert procs == []

    @patch("worker_application.activity_monitor._run_command")
    def test_skips_invalid_lines(self, mock_run):
        mock_run.return_value = (0, "invalid_line\n1234,python,512\n", "")
        procs = get_process_gpu_memory()
        assert len(procs) == 1
        assert procs[0]["pid"] == 1234


class TestGetTotalVramMb:
    @patch("worker_application.activity_monitor._run_command")
    def test_parses_vram(self, mock_run):
        mock_run.return_value = (0, "24576\n", "")
        assert get_total_vram_mb() == 24576

    @patch("worker_application.activity_monitor._run_command")
    def test_returns_zero_on_failure(self, mock_run):
        mock_run.return_value = (1, "", "error")
        assert get_total_vram_mb() == 0


class TestIsGpuVramContended:
    @patch("worker_application.activity_monitor.get_process_gpu_memory")
    @patch("worker_application.activity_monitor.get_total_vram_mb")
    def test_no_contention(self, mock_vram, mock_procs):
        mock_vram.return_value = 24000
        mock_procs.return_value = [
            {"pid": 100, "name": "ollama", "used_memory_mb": 10000},
        ]
        assert is_gpu_vram_contended() is False

    @patch("worker_application.activity_monitor.get_process_gpu_memory")
    @patch("worker_application.activity_monitor.get_total_vram_mb")
    def test_contention_with_foreign_process(self, mock_vram, mock_procs):
        mock_vram.return_value = 24000
        mock_procs.return_value = [
            {"pid": 100, "name": "ollama", "used_memory_mb": 10000},
            {"pid": 200, "name": "python", "used_memory_mb": 13000},
        ]
        assert is_gpu_vram_contended() is True

    @patch("worker_application.activity_monitor.get_process_gpu_memory")
    @patch("worker_application.activity_monitor.get_total_vram_mb")
    def test_own_pid_excluded(self, mock_vram, mock_procs):
        mock_vram.return_value = 24000
        own_pid = os.getpid()
        mock_procs.return_value = [
            {"pid": own_pid, "name": "worker", "used_memory_mb": 20000},
        ]
        assert is_gpu_vram_contended() is False

    @patch("worker_application.activity_monitor.get_process_gpu_memory")
    @patch("worker_application.activity_monitor.get_total_vram_mb")
    def test_nvidia_smi_excluded(self, mock_vram, mock_procs):
        mock_vram.return_value = 24000
        mock_procs.return_value = [
            {"pid": 999, "name": "nvidia-smi", "used_memory_mb": 20000},
        ]
        assert is_gpu_vram_contended() is False

    @patch("worker_application.activity_monitor.get_process_gpu_memory")
    @patch("worker_application.activity_monitor.get_total_vram_mb")
    def test_zero_vram_returns_false(self, mock_vram, mock_procs):
        mock_vram.return_value = 0
        mock_procs.return_value = []
        assert is_gpu_vram_contended() is False

    @patch("worker_application.activity_monitor.get_process_gpu_memory")
    @patch("worker_application.activity_monitor.get_total_vram_mb")
    def test_custom_threshold(self, mock_vram, mock_procs):
        mock_vram.return_value = 24000
        mock_procs.return_value = [
            {"pid": 200, "name": "python", "used_memory_mb": 6000},
        ]
        assert is_gpu_vram_contended(vram_threshold_pct=30) is False
        assert is_gpu_vram_contended(vram_threshold_pct=20) is True


class TestActivityMonitor:
    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_initial_state_is_enabled(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = False
        monitor._input_monitor = mock_input
        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            assert monitor.is_disabled() is False

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_user_active_disables_monitor(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.check_now()
            assert monitor.is_disabled() is True

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_gpu_contended_disables_monitor(self, mock_gpu):
        mock_gpu.return_value = True
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            assert monitor.is_disabled() is True

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_cooldown_extends_on_repeated_activity(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=10)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.check_now()
            first_disabled_until = monitor._disabled_until

            time.sleep(0.1)
            monitor.check_now()
            assert monitor._disabled_until >= first_disabled_until

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_becomes_enabled_after_cooldown(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=1)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.check_now()
            assert monitor.is_disabled() is True

        time.sleep(1.5)
        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            assert monitor.is_disabled() is False

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_get_status_returns_correct_fields(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            status = monitor.get_status()
            assert "disabled" in status
            assert "disabled_until" in status
            assert "remaining_seconds" in status
            assert "user_active" in status
            assert "gpu_vram_contended" in status
            assert "last_check_time" in status
            assert "last_disable_reason" in status

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_status_shows_disable_reason_user_active(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.check_now()
            status = monitor.get_status()
            assert status["disabled"] is True
            assert status["last_disable_reason"] == "user_active"
            assert status["user_active"] is True

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_status_shows_disable_reason_gpu(self, mock_gpu):
        mock_gpu.return_value = True
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            status = monitor.get_status()
            assert status["disabled"] is True
            assert status["last_disable_reason"] == "gpu_vram_contended"
            assert status["gpu_vram_contended"] is True

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_remaining_seconds_decreases(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.check_now()
            status1 = monitor.get_status()
            time.sleep(1)
            status2 = monitor.get_status()
            assert status2["remaining_seconds"] < status1["remaining_seconds"]

    def test_start_and_stop(self):
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with (
            patch.object(monitor, "_is_user_active", return_value=False),
            patch(
                "worker_application.activity_monitor.is_gpu_vram_contended",
                return_value=False,
            ),
        ):
            monitor.start()
            assert monitor._running is True
            monitor.stop()
            assert monitor._running is False

    def test_double_start_is_noop(self):
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with (
            patch.object(monitor, "_is_user_active", return_value=False),
            patch(
                "worker_application.activity_monitor.is_gpu_vram_contended",
                return_value=False,
            ),
        ):
            monitor.start()
            monitor.start()
            assert monitor._running is True
            monitor.stop()

    def test_stop_when_not_started_is_noop(self):
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        monitor.stop()
        assert monitor._running is False

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_disabled_until_is_zero_when_enabled(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            status = monitor.get_status()
            assert status["disabled"] is False
            assert status["disabled_until"] == 0

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_cooldown_period_keeps_disabled(self, mock_gpu):
        mock_gpu.return_value = False
        cooldown = 5
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=cooldown)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.check_now()

        with patch.object(monitor, "_is_user_active", return_value=False):
            time.sleep(1)
            monitor.check_now()
            assert monitor.is_disabled() is True

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_env_vars_override_defaults(self, mock_gpu):
        mock_gpu.return_value = False
        with patch.dict(
            os.environ,
            {
                "WORKER_DISABLE_COOLDOWN": "1800",
                "WORKER_ACTIVITY_CHECK_INTERVAL": "60",
                "WORKER_VRAM_THRESHOLD_PCT": "75",
                "WORKER_ALLOWED_GPU_PROCESSES": "ollama,python",
                "WORKER_IDLE_TIMEOUT": "600",
            },
        ):
            import importlib
            import worker_application.activity_monitor as am

            importlib.reload(am)
            assert am.DISABLE_COOLDOWN_SECONDS == 1800
            assert am.CHECK_INTERVAL_SECONDS == 60
            assert am.VRAM_THRESHOLD_PCT == 75.0
            assert am.ALLOWED_PROCESS_NAMES == ["ollama", "python"]
            assert am.IDLE_TIMEOUT_SECONDS == 600
            os.environ.pop("WORKER_DISABLE_COOLDOWN", None)
            os.environ.pop("WORKER_ACTIVITY_CHECK_INTERVAL", None)
            os.environ.pop("WORKER_VRAM_THRESHOLD_PCT", None)
            os.environ.pop("WORKER_ALLOWED_GPU_PROCESSES", None)
            os.environ.pop("WORKER_IDLE_TIMEOUT", None)
            importlib.reload(am)

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_status_includes_idle_info_with_input_monitor(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=60)
        mock_input = MagicMock()
        mock_input.is_available.return_value = True
        mock_input.get_idle_seconds.return_value = 42.5
        mock_input._monitored_paths = {"/dev/input/event2"}
        monitor._input_monitor = mock_input

        with patch.object(monitor, "_is_user_active", return_value=False):
            monitor.check_now()
            status = monitor.get_status()
            assert "idle_seconds" in status
            assert status["idle_seconds"] == 42.5
            assert "input_devices_monitored" in status
            assert status["input_devices_monitored"] == 1


class TestActivityMonitorBackgroundThread:
    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_monitor_detects_changes_in_background(self, mock_gpu):
        mock_gpu.return_value = False
        monitor = ActivityMonitor(check_interval=1, disable_cooldown=10)
        with patch.object(monitor, "_is_user_active", return_value=True):
            monitor.start()
            time.sleep(2.5)
            assert monitor.is_disabled() is True

        with patch.object(monitor, "_is_user_active", return_value=False):
            time.sleep(12)
            assert monitor.is_disabled() is False

        monitor.stop()

    @patch("worker_application.activity_monitor.is_gpu_vram_contended")
    def test_monitor_handles_exceptions_gracefully(self, mock_gpu):
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("test error")
            return False

        mock_gpu.return_value = False

        monitor = ActivityMonitor(check_interval=1, disable_cooldown=10)
        with patch.object(monitor, "_is_user_active", side_effect=side_effect):
            monitor.start()
            time.sleep(3)
            assert monitor._running is True
            monitor.stop()
