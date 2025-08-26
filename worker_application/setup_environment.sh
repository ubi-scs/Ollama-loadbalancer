#!/usr/bin/env bash

set -euo pipefail

# Get the directory of the script
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# Ensure pip is up to date
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# Install dependencies if requirements.txt exists
if [ -f "$APP_DIR/requirements.txt" ]; then
  echo "[INFO] Installing/updating dependencies..."
  "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
fi