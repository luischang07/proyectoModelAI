"""API endpoints para entrenamiento"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers.training_controller import TrainingController
from backend.schemas.training import TrainingStart, TrainingJobCreate, TrainingStatus
from backend.tasks.training_tasks import train_model_task

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/start", response_model=TrainingJobCreate)
def start_training(request: TrainingStart, db: Session = Depends(get_db)):
    """Inicia un nuevo entrenamiento"""
    job = TrainingController.create_training_job(
        db=db,
        images_folder=request.images_folder,
        masks_folder=request.masks_folder,
        patch_size=request.patch_size,
        stride=request.stride,
        batch_size=request.batch_size,
        epochs=request.epochs,
        backbone=request.backbone,
        encoder_weights=request.encoder_weights,
    )
    
    # Lanzar tarea asíncrona
    train_model_task.delay(job.job_id)
    
    return TrainingJobCreate(
        job_id=job.job_id,
        status=job.status,
        message="Entrenamiento iniciado correctamente"
    )

@router.get("/status/{job_id}", response_model=TrainingStatus)
def get_training_status(job_id: str, db: Session = Depends(get_db)):
    """Obtiene el estado de un entrenamiento"""
    from celery import states
    from backend.tasks.celery_app import celery_app
    
    job = TrainingController.get_training_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    # Si el job está "running", verificar el estado real en Celery
    if job.status == "running":
        from datetime import datetime
        
        # Si no hay celery_task_id después de 1 minuto, el worker crasheó antes de registrarlo
        if not job.celery_task_id:
            if job.started_at:
                # Usar datetime sin timezone para ser compatible con SQLite
                elapsed = (datetime.now() - job.started_at).total_seconds()
                if elapsed > 60:  # Más de 1 minuto sin task_id
                    job.status = "failed"
                    job.error_message = "Tarea perdida. Celery worker crasheó antes de comenzar (posible OOM)."
                    db.commit()
        else:
            # Hay celery_task_id, verificar su estado en Celery
            result = celery_app.AsyncResult(job.celery_task_id)
            
            # Si Celery dice que falló o no existe la tarea
            if result.state == states.FAILURE:
                job.status = "failed"
                job.error_message = f"Tarea falló en Celery: {str(result.info)}"
                db.commit()
            elif result.state == states.REVOKED:
                job.status = "failed"  
                job.error_message = "Tarea cancelada o revocada en Celery."
                db.commit()
            elif result.state == states.SUCCESS:
                # La tarea terminó pero no se actualizó el estado
                job.status = "completed"
                db.commit()
            elif result.state == states.PENDING:
                # PENDING significa que Celery no conoce esta tarea
                # Si el job lleva más de 1 minuto en "running" pero Celery dice PENDING,
                # entonces el worker se reinició y perdió la tarea
                if job.started_at:
                    # Usar datetime sin timezone para ser compatible con SQLite
                    elapsed = (datetime.now() - job.started_at).total_seconds()
                    if elapsed > 60:  # Más de 1 minuto
                        job.status = "failed"
                        job.error_message = "Tarea perdida. Celery worker se reinició (posible OOM)."
                        db.commit()
    
    metrics = None
    if job.current_loss is not None:
        metrics = {
            "loss": job.current_loss,
            "iou_score": job.current_iou,
            "val_loss": job.val_loss,
            "val_iou_score": job.val_iou
        }
    
    return TrainingStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        metrics=metrics,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )

@router.delete("/cancel/{job_id}")
def cancel_training(job_id: str, db: Session = Depends(get_db)):
    """Cancela un entrenamiento"""
    success = TrainingController.cancel_training(db, job_id)
    if not success:
        raise HTTPException(status_code=400, detail="No se pudo cancelar el job")
    return {"message": "Job cancelado", "job_id": job_id}
