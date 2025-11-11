"""Celery app configuration"""
from celery import Celery
from backend.config import settings

celery_app = Celery(
    "anomaly_detection",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Registrar las tareas automáticamente
celery_app.autodiscover_tasks(['backend.tasks'])

# Importar explícitamente los módulos de tareas para asegurar registro
from backend.tasks import training_tasks, inference_tasks, unsupervised_tasks
