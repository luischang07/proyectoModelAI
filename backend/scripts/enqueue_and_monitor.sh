#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/mnt/c/Users/luisc/OneDrive - Instituto Tecnológico de Culiacán/Universidad/Semestre_9/tesis/proyectoModelAI"
cd "$PROJECT_DIR"
VENV_PY="$PROJECT_DIR/.venv-wsl/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Venv python not found: $VENV_PY" >&2
  exit 1
fi
LOG=logs/celery.log
# Enqueue task via venv python
TASK_ID=$("$VENV_PY" -c "from backend.tasks.unsupervised_tasks import train_unsupervised_task; res = train_unsupervised_task.apply_async((54321, 'celery_test_monitor', '/mnt/c/Users/luisc/Downloads/V3_10_11_2021/V3_10_11_2021/FLIGHT', 1, 8, 64, 0.2)); print(res.id)")
echo "Enqueued task: $TASK_ID"

# Monitor logs and nvidia-smi for 30s
DURATION=30
END=$((SECONDS + DURATION))
while [ $SECONDS -lt $END ]; do
  echo "--- TIMESTAMP: $(date +%T) ---"
  echo '--- last 40 lines of celery.log ---'
  tail -n 40 "$LOG" || true
  echo '--- nvidia-smi ---'
  nvidia-smi || true
  sleep 3
done

echo '--- FINAL LOG SNIPPET (200 lines) ---'
tail -n 200 "$LOG" || true

echo '--- redis meta for task ---'
redis-cli get "celery-task-meta-$TASK_ID" || true
