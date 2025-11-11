#!/bin/bash
# Script para iniciar Celery en WSL2 con soporte GPU

echo "🔥 Iniciando Celery worker en WSL2..."

# Configurar variables de entorno para CUDA
export CUDA_VISIBLE_DEVICES=0

# IMPORTANTE: Usar 'solo' pool en lugar de 'prefork' para evitar problemas con CUDA
# El pool 'solo' ejecuta tareas en el proceso principal, evitando fork()
celery -A backend.tasks.celery_app worker \
    --loglevel=info \
    --pool=solo \
    --concurrency=1

