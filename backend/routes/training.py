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
    job = TrainingController.get_training_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
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
