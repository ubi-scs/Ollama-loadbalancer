#!/usr/bin/env bash
set -euo pipefail

PORT=11434
ALLOWED_HOST="scs-ai-proxy"

echo "[INFO] Resolving $ALLOWED_HOST..."
ALLOWED_IP=$(getent ahosts "$ALLOWED_HOST" | awk '{print $1; exit}')

if [[ -z "$ALLOWED_IP" ]]; then
  echo "[ERROR] Could not resolve $ALLOWED_HOST" >&2
  exit 1
fi

echo "[INFO] Allowing $ALLOWED_IP to access port $PORT..."
sudo ufw allow from "$ALLOWED_IP" to any port "$PORT"

echo "[INFO] Denying all other access to port $PORT..."
sudo ufw deny "$PORT"

echo "[INFO] Reloading UFW..."
sudo ufw reload

echo "[INFO] Firewall rule applied: Only $ALLOWED_HOST ($ALLOWED_IP) can access port $PORT."