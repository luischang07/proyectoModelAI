"""Celery tasks para inferencia"""
from backend.tasks.celery_app import celery_app
from backend.database import SessionLocal
from backend.controllers.inference_controller import InferenceController

@celery_app.task(bind=True)
def predict_task(self, job_id: str):
    """Task asíncrona de inferencia"""
    db = SessionLocal()
    try:
        result = InferenceController.execute_inference(job_id, db)
        return result
    finally:
        db.close()
