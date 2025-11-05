"""
Rutas API para entrenamiento NO SUPERVISADO
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from backend.database import get_db
from backend.controllers import unsupervised_controller
from backend.tasks.unsupervised_tasks import train_unsupervised_task

router = APIRouter(prefix="/api/unsupervised", tags=["Unsupervised Training"])


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
    job_id: int
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
        train_unsupervised_task.delay(
            job_id=job.id,
            model_name=request.model_name,
            images_folder=request.images_folder,
            epochs=request.epochs,
            batch_size=request.batch_size,
            latent_dim=request.latent_dim,
            validation_split=request.validation_split
        )
        
        return TrainingResponse(
            job_id=job.id,
            status=job.status,
            message=f"Entrenamiento NO SUPERVISADO iniciado. Job ID: {job.id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
