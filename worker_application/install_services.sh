#!/usr/bin/env bash
#
# install_services.sh — one-click install / update for the
# ollama-worker and ollama-watchdog systemd services.
#
# Usage:
#   sudo ./install_services.sh          # install + start
#   sudo ./install_services.sh --stop   # stop services
#   sudo ./install_services.sh --status # check status
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR"
VENV_DIR="$APP_DIR/venv"

ENV_FILE="$APP_DIR/ollama_worker.env"

WORKER_SERVICE_NAME="ollama-worker"
WATCHDOG_SERVICE_NAME="ollama-watchdog"

WORKER_SERVICE_FILE="/etc/systemd/system/${WORKER_SERVICE_NAME}.service"
WATCHDOG_SERVICE_FILE="/etc/systemd/system/${WATCHDOG_SERVICE_NAME}.service"

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }
err()  { echo "[ERROR] $*" >&2; }

require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        err "This script must be run as root (use sudo)."
        exit 1
    fi
}

# ── venv + deps ──────────────────────────────────────────────────────────────

setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi

    log "Installing / updating Python dependencies ..."
    "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel -q
    "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt" -q
}

# ── systemd unit files ───────────────────────────────────────────────────────

write_worker_service() {
    log "Writing ${WORKER_SERVICE_NAME}.service ..."
    cat > "$WORKER_SERVICE_FILE" <<EOF
[Unit]
Description=Ollama Worker API
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${APP_DIR}
ExecStart=${VENV_DIR}/bin/python -m uvicorn worker_application.main:app --host \${WORKER_HOST} --port \${WORKER_PORT}
EnvironmentFile=${ENV_FILE}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

write_watchdog_service() {
    log "Writing ${WATCHDOG_SERVICE_NAME}.service ..."
    cat > "$WATCHDOG_SERVICE_FILE" <<EOF
[Unit]
Description=Ollama Worker Watchdog
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=${APP_DIR}
ExecStart=${VENV_DIR}/bin/python -m uvicorn worker_application.watchdog:app --host \${WATCHDOG_HOST} --port \${WATCHDOG_PORT}
EnvironmentFile=${ENV_FILE}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
}

# ── env file check ───────────────────────────────────────────────────────────

check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        err "Environment file not found: $ENV_FILE"
        exit 1
    fi

    if grep -q '^OLLAMA_HELPER_API_KEY=your-secret-api-key$' "$ENV_FILE"; then
        warn "OLLAMA_HELPER_API_KEY is still set to the default value."
        warn "Edit $ENV_FILE before deploying in production."
    fi

    if ! grep -q '^WATCHDOG_HOST=' "$ENV_FILE"; then
        warn "WATCHDOG_HOST not set in $ENV_FILE — will use default 0.0.0.0"
    fi
    if ! grep -q '^WATCHDOG_PORT=' "$ENV_FILE"; then
        warn "WATCHDOG_PORT not set in $ENV_FILE — will use default 8001"
    fi
}

# ── systemctl wrappers ───────────────────────────────────────────────────────

reload_and_enable() {
    log "Reloading systemd daemon ..."
    systemctl daemon-reload

    log "Enabling ${WATCHDOG_SERVICE_NAME} ..."
    systemctl enable "$WATCHDOG_SERVICE_NAME"

    log "Enabling ${WORKER_SERVICE_NAME} ..."
    systemctl enable "$WORKER_SERVICE_NAME"
}

start_services() {
    log "Starting ${WATCHDOG_SERVICE_NAME} (watchdog starts the worker) ..."
    systemctl start "$WATCHDOG_SERVICE_NAME"

    log "Starting ${WORKER_SERVICE_NAME} ..."
    systemctl start "$WORKER_SERVICE_NAME"
}

stop_services() {
    log "Stopping ${WORKER_SERVICE_NAME} ..."
    systemctl stop "$WORKER_SERVICE_NAME" || true

    log "Stopping ${WATCHDOG_SERVICE_NAME} ..."
    systemctl stop "$WATCHDOG_SERVICE_NAME" || true
}

show_status() {
    echo ""
    echo "=== ${WATCHDOG_SERVICE_NAME} ==="
    systemctl status "$WATCHDOG_SERVICE_NAME" --no-pager || true
    echo ""
    echo "=== ${WORKER_SERVICE_NAME} ==="
    systemctl status "$WORKER_SERVICE_NAME" --no-pager || true
    echo ""
}

# ── main ─────────────────────────────────────────────────────────────────────

install() {
    require_root
    check_env
    setup_venv
    write_watchdog_service
    write_worker_service
    reload_and_enable

    stop_services
    start_services

    log ""
    log "Installation complete."
    log ""
    log "  Worker API:   http://<host>:<WORKER_PORT>/api/version"
    log "  Watchdog API: http://<host>:<WATCHDOG_PORT>/watchdog/status"
    log ""
    log "Use '$0 --status' to check service health."
    log "Use '$0 --stop'   to stop both services."
    log "Use '$0 --start'  to start both services again."
    log "Re-run this script any time to update the unit files, deps, or restart."
    log ""

    show_status
}

case "${1:-}" in
    --stop)
        require_root
        stop_services
        ;;
    --start)
        require_root
        start_services
        ;;
    --status)
        require_root
        show_status
        ;;
    --uninstall)
        require_root
        stop_services
        systemctl disable "$WATCHDOG_SERVICE_NAME" || true
        systemctl disable "$WORKER_SERVICE_NAME" || true
        rm -f "$WATCHDOG_SERVICE_FILE" "$WORKER_SERVICE_FILE"
        systemctl daemon-reload
        log "Services uninstalled."
        ;;
    --help|-h)
        echo "Usage: sudo $0 [--stop|--start|--status|--uninstall|--help]"
        echo ""
        echo "  (no args)    Install / update services and start them"
        echo "  --stop       Stop both services"
        echo "  --start      Start both services"
        echo "  --status     Show status of both services"
        echo "  --uninstall  Remove systemd units and disable services"
        echo "  --help       Show this help"
        ;;
    *)
        install
        ;;
esac