"""
Rutas API para entrenamiento NO SUPERVISADO
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from backend.database import get_db
from backend.controllers import unsupervised_controller
from backend.tasks.unsupervised_tasks import train_unsupervised_task

router = APIRouter(prefix="/unsupervised", tags=["Unsupervised Training"])


class UnsupervisedTrainingRequest(BaseModel):
    """Request para iniciar entrenamiento NO SUPERVISADO."""
    model_name: str = Field(..., description="Nombre del modelo a entrenar")
    images_folder: str = Field(..., description="Ruta a carpeta con imágenes (sin máscaras)")
    epochs: int = Field(default=50, ge=1, le=200, description="Número de épocas")
    batch_size: int = Field(default=16, ge=1, le=64, description="Tamaño del batch")
    latent_dim: int = Field(default=128, ge=32, le=512, description="Dimensión del espacio latente")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Fracción para validación")


class TrainingResponse(BaseModel):
    """Respuesta al iniciar entrenamiento."""
    job_id: str
    status: str
    message: str


@router.post("/train", response_model=TrainingResponse)
def start_unsupervised_training(
    request: UnsupervisedTrainingRequest,
    db: Session = Depends(get_db)
):
    """
    Inicia entrenamiento NO SUPERVISADO (Autoencoder).
    
    **Solo requiere imágenes**, NO necesita máscaras etiquetadas.
    Ideal para detectar anomalías cuando no hay datos etiquetados.
    """
    try:
        # Crear job en la base de datos
        job = unsupervised_controller.create_job(
            db=db,
            model_name=request.model_name,
            images_folder=request.images_folder,
            epochs=request.epochs,
            batch_size=request.batch_size
        )
        
        # Lanzar tarea asíncrona en Celery
        task = train_unsupervised_task.delay(
            job_id=job.job_id,
            model_name=request.model_name,
            images_folder=request.images_folder,
            epochs=request.epochs,
            batch_size=request.batch_size,
            latent_dim=request.latent_dim,
            validation_split=request.validation_split
        )
        
        # Guardar task_id de Celery para poder cancelar
        job.celery_task_id = task.id
        db.commit()
        
        return TrainingResponse(
            job_id=job.job_id,
            status=job.status,
            message=f"Entrenamiento NO SUPERVISADO iniciado. Job ID: {job.job_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=dict)
def get_unsupervised_training_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Obtiene el estado de un trabajo de entrenamiento NO SUPERVISADO.
    
    Returns:
        - status: pending/running/completed/failed
        - progress: Porcentaje de progreso (0-100)
        - message: Detalles adicionales
        - metrics: Métricas de entrenamiento (si está disponible)
    """
    try:
        job = unsupervised_controller.get_job_status(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} no encontrado"
            )
        
        # Convertir progress a porcentaje (0-100)
        progress_pct = int((job.progress or 0) * 100)
        
        response = {
            "job_id": job.job_id,
            "status": job.status,
            "progress": progress_pct,
            "current_epoch": job.current_epoch or 0,
            "total_epochs": job.total_epochs or 50,
            "error_message": job.error_message or "",
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
        
        # Añadir métricas si existen
        metrics = {}
        if job.current_loss is not None:
            metrics["loss"] = job.current_loss
        if job.val_loss is not None:
            metrics["val_loss"] = job.val_loss
        if job.final_loss is not None:
            metrics["final_loss"] = job.final_loss
        
        if metrics:
            response["metrics"] = metrics
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cancel/{job_id}", response_model=dict)
def cancel_unsupervised_training(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Cancela un trabajo de entrenamiento NO SUPERVISADO.
    """
    from backend.tasks.celery_app import celery_app
    
    try:
        job = unsupervised_controller.get_job_status(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} no encontrado"
            )
        
        if job.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400,
                detail=f"No se puede cancelar un job con status: {job.status}"
            )
        
        # Revocar tarea en Celery si existe task_id
        if job.celery_task_id:
            try:
                celery_app.control.revoke(job.celery_task_id, terminate=True, signal='SIGKILL')
            except Exception as e:
                print(f"⚠️ Error al revocar tarea Celery: {e}")
        
        # Actualizar estado a cancelado
        unsupervised_controller.update_job_status(
            db, job_id, "cancelled", progress=0
        )
        
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Entrenamiento cancelado exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
