"""Celery tasks para entrenamiento"""
from backend.tasks.celery_app import celery_app
from backend.database import SessionLocal
from backend.controllers.training_controller import TrainingController
from backend.models.db_models import TrainingJob

@celery_app.task(bind=True, soft_time_limit=7200, time_limit=7300)  # 2 horas max
def train_model_task(self, job_id: str):
    """Task asíncrona de entrenamiento con timeout y manejo de errores"""
    db = SessionLocal()
    try:
        result = TrainingController.execute_training(job_id, db)
        return result
    except Exception as e:
        # Marcar el job como fallido en la base de datos
        try:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = f"Error en tarea Celery: {str(e)}"
                db.commit()
        except Exception as db_error:
            print(f"❌ Error actualizando estado del job: {db_error}")
        
        # Re-lanzar la excepción para que Celery la registre
        raise
    finally:
        db.close()
