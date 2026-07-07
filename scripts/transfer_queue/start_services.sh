#!/usr/bin/env bash
set -euo pipefail

VENV="${VIME_VENV:-/mnt/data1/yibo/vime-workspace/.venv}"
SERVICE_DIR="${TQ_SERVICE_DIR:-/mnt/data1/yibo/vime-workspace/services/transfer_queue}"
RAY_NODE_IP="${RAY_NODE_IP:-172.16.1.248}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6380}"
RAY_CLIENT_PORT="${RAY_CLIENT_PORT:-20001}"
RAY_ADDRESS="${RAY_NODE_IP}:${RAY_GCS_PORT}"
PID_FILE="${SERVICE_DIR}/transfer_queue.pid"
LOG_FILE="${SERVICE_DIR}/transfer_queue.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SCRIPT="${SCRIPT_DIR}/tq_service.py"
EXPORT_SCRIPT="${SCRIPT_DIR}/export_config.py"
CLIENT_CONFIG="${SERVICE_DIR}/client_config.pkl"

mkdir -p "${SERVICE_DIR}"

if ! "${VENV}/bin/ray" status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
  ray_args=(
    start
    --head
    "--node-ip-address=${RAY_NODE_IP}"
    "--port=${RAY_GCS_PORT}"
    "--ray-client-server-port=${RAY_CLIENT_PORT}"
    --dashboard-port=8266
    --include-dashboard=false
    --num-cpus=8
    --num-gpus=0
    --min-worker-port=21000
    --max-worker-port=22000
    --disable-usage-stats
  )
  "${VENV}/bin/ray" "${ray_args[@]}"
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "TransferQueue service already running with PID $(cat "${PID_FILE}")"
else
  service_args=(
    -u
    "${SERVICE_SCRIPT}"
    "--ray-address=${RAY_ADDRESS}"
    --polling-mode
  )
  nohup "${VENV}/bin/python" "${service_args[@]}" >"${LOG_FILE}" 2>&1 &
  echo "$!" >"${PID_FILE}"
fi

for _ in $(seq 1 30); do
  if grep -q "TRANSFER_QUEUE_SERVICE_READY" "${LOG_FILE}" 2>/dev/null; then
    "${VENV}/bin/python" "${EXPORT_SCRIPT}"       --ray-address="${RAY_ADDRESS}"       --output="${CLIENT_CONFIG}"
    echo "Ray GCS: ${RAY_ADDRESS}"
    echo "Ray Client: ray://${RAY_NODE_IP}:${RAY_CLIENT_PORT}"
    echo "TransferQueue PID: $(cat "${PID_FILE}")"
    exit 0
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    cat "${LOG_FILE}"
    exit 1
  fi
  sleep 1
done

echo "TransferQueue service did not become ready within 30 seconds"
cat "${LOG_FILE}"
exit 1
