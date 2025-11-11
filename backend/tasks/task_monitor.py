"""
Monitor para detectar tareas de Celery que fallan sin reportar error
(ej: OOM kill, contenedor reiniciado, etc.)
"""
import time
from celery import states
from backend.database import SessionLocal
from backend.models.db_models import TrainingJob
from backend.tasks.celery_app import celery_app


def check_stuck_tasks():
    """
    Revisa tareas que llevan mucho tiempo en estado 'running' pero Celery
    no las tiene activas. Esto pasa cuando Celery se reinicia por OOM.
    """
    db = SessionLocal()
    try:
        # Buscar jobs en 'running' que empezaron hace más de 10 minutos
        stuck_threshold = time.time() - (10 * 60)  # 10 minutos
        
        jobs = db.query(TrainingJob).filter(
            TrainingJob.status == "running"
        ).all()
        
        for job in jobs:
            # Verificar si la tarea existe en Celery
            task_id = job.celery_task_id
            
            if not task_id:
                print(f"⚠️  Job {job.job_id} sin task_id de Celery")
                continue
            
            # Inspeccionar estado en Celery
            result = celery_app.AsyncResult(task_id)
            
            # Si Celery no conoce la tarea o está en estado terminal
            if result.state in [states.FAILURE, states.REVOKED, None]:
                print(f"❌ Job {job.job_id} perdido - actualizando a failed")
                job.status = "failed"
                job.error_message = f"Tarea perdida. Estado Celery: {result.state}. Posible OOM o reinicio del worker."
                db.commit()
            elif result.state == states.PENDING:
                # PENDING puede significar que la tarea no existe
                job_start_time = job.started_at.timestamp() if job.started_at else time.time()
                if job_start_time < stuck_threshold:
                    print(f"❌ Job {job.job_id} atascado en PENDING - actualizando a failed")
                    job.status = "failed"
                    job.error_message = "Tarea atascada en estado PENDING. Celery worker pudo haber fallado."
                    db.commit()
    except Exception as e:
        print(f"❌ Error monitoreando tareas: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("🔍 Iniciando monitor de tareas...")
    while True:
        check_stuck_tasks()
        time.sleep(30)  # Revisar cada 30 segundos
