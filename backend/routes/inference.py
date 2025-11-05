"""API endpoints para inferencia"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers.inference_controller import InferenceController
from backend.schemas.inference import InferencePredict, InferenceJobCreate, InferenceStatus
from backend.tasks.inference_tasks import predict_task

router = APIRouter(prefix="/inference", tags=["inference"])

@router.post("/predict", response_model=InferenceJobCreate)
def start_prediction(request: InferencePredict, db: Session = Depends(get_db)):
    """Inicia una predicción"""
    job = InferenceController.create_inference_job(
        db=db,
        image_path=request.image_path,
        model_id=request.model_id,
        threshold=request.threshold,
        stride=request.stride,
        batch_size=request.batch_size,
    )
    
    # Lanzar tarea asíncrona
    predict_task.delay(job.job_id)
    
    return InferenceJobCreate(
        job_id=job.job_id,
        status=job.status,
        message="Predicción iniciada correctamente"
    )

@router.get("/status/{job_id}", response_model=InferenceStatus)
def get_inference_status(job_id: str, db: Session = Depends(get_db)):
    """Obtiene el estado de una inferencia"""
    job = InferenceController.get_inference_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    
    stats = None
    if job.output_path:
        stats = {
            "anomaly_pixels": job.anomaly_pixels,
            "total_pixels": job.total_pixels,
            "anomaly_percentage": job.anomaly_percentage
        }
    
    return InferenceStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        output_path=job.output_path,
        stats=stats,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at
    )
