"""Controller para operaciones de inferencia"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from sqlalchemy.orm import Session
from backend.models.db_models import InferenceJob, MLModel
from backend.models.ml_models import MLModelWrapper
from backend.models.procesamiento import (
    load_image,
    normalize_image,
    create_patches,
    reconstruct_from_patches,
    save_mask_geotiff,
)
from backend.config import settings


class InferenceController:
    """Controlador para lógica de inferencia"""
    
    @staticmethod
    def create_inference_job(
        db: Session,
        image_path: str,
        model_id: str,
        threshold: float = 0.5,
        stride: int = 256,
        batch_size: int = 16,
    ) -> InferenceJob:
        """Crea un nuevo job de inferencia"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"pred_{timestamp}"
        
        job = InferenceJob(
            job_id=job_id,
            status="queued",
            image_path=image_path,
            model_id=model_id,
            threshold=threshold,
            stride=stride,
            batch_size=batch_size
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    
    @staticmethod
    def get_inference_job(db: Session, job_id: str) -> Optional[InferenceJob]:
        """Obtiene un job por ID"""
        return db.query(InferenceJob).filter(InferenceJob.job_id == job_id).first()
    
    @staticmethod
    def update_job_status(db: Session, job_id: str, status: str, **kwargs) -> Optional[InferenceJob]:
        """Actualiza estado del job"""
        job = InferenceController.get_inference_job(db, job_id)
        if not job:
            return None
        
        job.status = status
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        
        if status == "running" and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in ["completed", "failed", "cancelled"]:
            job.completed_at = datetime.utcnow()
        
        db.commit()
        db.refresh(job)
        return job
    
    @staticmethod
    def execute_inference(
        job_id: str,
        db: Session,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Ejecuta la inferencia (llamado por Celery)"""
        job = InferenceController.get_inference_job(db, job_id)
        if not job:
            return {"success": False, "error": "Job no encontrado"}
        
        try:
            InferenceController.update_job_status(db, job_id, "running", progress=0.1)
            
            # 1. Obtener modelo de DB
            ml_model_entry = db.query(MLModel).filter(MLModel.model_id == job.model_id).first()
            if not ml_model_entry:
                raise ValueError(f"Modelo {job.model_id} no encontrado")
            
            # 2. Cargar modelo ML
            ml_model = MLModelWrapper(ml_model_entry.model_path)
            InferenceController.update_job_status(db, job_id, "running", progress=0.2)
            
            # 3. Cargar y normalizar imagen
            image, profile = load_image(job.image_path)
            image = normalize_image(image)
            InferenceController.update_job_status(db, job_id, "running", progress=0.3)
            
            # 4. Crear parches
            patch_size = ml_model_entry.input_shape[0]
            x_patches, _, coords = create_patches(
                image,
                mask=None,
                patch_size=patch_size,
                stride=job.stride
            )
            InferenceController.update_job_status(db, job_id, "running", progress=0.4)
            
            # 5. Predecir en batches
            probs = []
            total_batches = (len(x_patches) + job.batch_size - 1) // job.batch_size
            for i in range(0, len(x_patches), job.batch_size):
                batch = x_patches[i:i + job.batch_size]
                batch_probs = ml_model.model.predict(batch, verbose=0)
                probs.append(batch_probs)
                
                # Actualizar progreso
                batch_num = i // job.batch_size + 1
                progress = 0.4 + 0.4 * (batch_num / total_batches)
                InferenceController.update_job_status(db, job_id, "running", progress=progress)
            
            probs = np.concatenate(probs, axis=0)
            probs2d = probs[..., 0]  # (N, patch, patch)
            
            # 6. Reconstruir mosaico
            mosaic = reconstruct_from_patches(
                probs2d,
                coords,
                out_shape=image.shape[:2],
                patch_size=patch_size,
                reduce="mean"
            )
            InferenceController.update_job_status(db, job_id, "running", progress=0.9)
            
            # 7. Umbral y guardar
            mask_bin = (mosaic >= job.threshold).astype(np.uint8)
            
            output_filename = f"pred_{job_id}.tif"
            output_path = settings.OUTPUT_DIR / output_filename
            save_mask_geotiff(str(output_path), mask_bin, profile, dtype="uint8")
            
            # 8. Calcular estadísticas
            anomaly_pixels = int(np.sum(mask_bin))
            total_pixels = int(mask_bin.size)
            anomaly_percentage = float((anomaly_pixels / total_pixels) * 100)
            
            # 9. Actualizar job
            InferenceController.update_job_status(
                db,
                job_id,
                "completed",
                progress=1.0,
                output_path=str(output_path),
                anomaly_pixels=anomaly_pixels,
                total_pixels=total_pixels,
                anomaly_percentage=anomaly_percentage
            )
            
            return {
                "success": True,
                "output_path": str(output_path),
                "anomaly_percentage": anomaly_percentage
            }
            
        except Exception as e:
            InferenceController.update_job_status(db, job_id, "failed", error_message=str(e))
            return {"success": False, "error": str(e)}
