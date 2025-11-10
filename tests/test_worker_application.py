import subprocess
import sys
import importlib


def test_run_command_success(monkeypatch):
    monkeypatch.setenv('OLLAMA_HELPER_API_KEY', 'dummy')
    # Lazy import after setting env
    if 'worker_application.main' in sys.modules:
        importlib.reload(sys.modules['worker_application.main'])
    else:
        import worker_application.main  # noqa: F401
    import worker_application.main as main

    def fake_run(cmd, capture_output, text, check):
        class P:
            returncode = 0
            stdout = 'ok'
            stderr = ''
        return P()
    monkeypatch.setattr(subprocess, 'run', fake_run)
    code, out, err = main.run_command(['echo', 'hi'])
    assert code == 0 and out == 'ok' and err == ''


def test_run_command_not_found(monkeypatch):
    monkeypatch.setenv('OLLAMA_HELPER_API_KEY', 'dummy')
    if 'worker_application.main' in sys.modules:
        importlib.reload(sys.modules['worker_application.main'])
    else:
        import worker_application.main  # noqa: F401
    import worker_application.main as main

    def fake_run(cmd, capture_output, text, check):
        raise FileNotFoundError
    monkeypatch.setattr(subprocess, 'run', fake_run)
    code, out, err = main.run_command(['missing'])
    assert code == -1 and 'Command not found' in err


def test_run_command_exception(monkeypatch):
    monkeypatch.setenv('OLLAMA_HELPER_API_KEY', 'dummy')
    if 'worker_application.main' in sys.modules:
        importlib.reload(sys.modules['worker_application.main'])
    else:
        import worker_application.main  # noqa: F401
    import worker_application.main as main

    def fake_run(cmd, capture_output, text, check):
        raise RuntimeError('boom')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    code, out, err = main.run_command(['bad'])
    assert code == -1 and 'boom' in err
