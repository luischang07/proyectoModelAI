#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/mnt/c/Users/luisc/OneDrive - Instituto Tecnológico de Culiacán/Universidad/Semestre_9/tesis/proyectoModelAI"
cd "$PROJECT_DIR"
# Kill existing workers and child training processes
pkill -f 'celery -A backend.tasks.celery_app' || true
pkill -f 'run_unsupervised_training.py' || true
sleep 1
mkdir -p logs
LOG=logs/celery.log
# Start celery using the venv python to avoid sourcing
VENV_PY="$PROJECT_DIR/.venv-wsl/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Venv python not found: $VENV_PY" >&2
  exit 1
fi
echo "Starting celery with $VENV_PY -m celery, logging to $LOG"
nohup "$VENV_PY" -m celery -A backend.tasks.celery_app worker --loglevel=info --pool=solo --concurrency=1 > "$LOG" 2>&1 &
echo $! > logs/celery.pid
sleep 2
echo "=== initial logs (last 200 lines) ==="
tail -n 200 "$LOG" || true
