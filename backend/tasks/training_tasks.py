"""Celery tasks para entrenamiento"""
from backend.tasks.celery_app import celery_app
from backend.database import SessionLocal
from backend.controllers.training_controller import TrainingController

@celery_app.task(bind=True)
def train_model_task(self, job_id: str):
    """Task asíncrona de entrenamiento"""
    db = SessionLocal()
    try:
        result = TrainingController.execute_training(job_id, db)
        return result
    finally:
        db.close()
