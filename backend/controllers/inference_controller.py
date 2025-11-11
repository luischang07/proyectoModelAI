"""Controller para operaciones de inferencia"""
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch

from sqlalchemy.orm import Session
from backend.models.db_models import InferenceJob, MLModel
from backend.models.ml_models import MLModelWrapper
from backend.models.procesamiento import (
    load_image,
    normalize_image,
    create_patches,
    reconstruct_from_patches,
    save_mask_geotiff,
    convert_windows_path_to_wsl,
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
            # Convertir ruta de Windows a WSL si es necesario
            image_path = convert_windows_path_to_wsl(job.image_path)
            image, profile = load_image(image_path)
            image = normalize_image(image)
            InferenceController.update_job_status(db, job_id, "running", progress=0.3)
            
            # 4. Crear parches
            # Para modelos no supervisados, obtener patch_size del modelo cargado
            if hasattr(ml_model, 'model_type') and ml_model.model_type == 'unsupervised_autoencoder':
                # Autoencoder: usar el tamaño de la primera imagen del dataset de entrenamiento
                from backend.models.procesamiento import PATCH_SIZE
                patch_size = PATCH_SIZE
            else:
                # UNet supervisado: usar input_shape de la DB
                patch_size = ml_model_entry.input_shape[0]
            
            x_patches, _, coords = create_patches(
                image,
                mask=None,
                patch_size=patch_size,
                stride=job.stride
            )
            InferenceController.update_job_status(db, job_id, "running", progress=0.4)
            
            # 5. Predecir en batches con PyTorch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ml_model.model.eval()
            
            probs = []
            total_batches = (len(x_patches) + job.batch_size - 1) // job.batch_size
            is_unsupervised = hasattr(ml_model, 'model_type') and ml_model.model_type == 'unsupervised_autoencoder'
            
            with torch.no_grad():
                for i in range(0, len(x_patches), job.batch_size):
                    batch = x_patches[i:i + job.batch_size]
                    
                    # Transponer de (N, H, W, C) a (N, C, H, W) para PyTorch
                    batch = np.transpose(batch, (0, 3, 1, 2))
                    
                    # Convertir a tensor y mover a device
                    batch_tensor = torch.from_numpy(batch).float().to(device)
                    
                    # Predecir
                    output = ml_model.model(batch_tensor)
                    
                    if is_unsupervised:
                        # Para autoencoder: calcular error de reconstrucción MSE
                        # output es la reconstrucción (N, C, H, W)
                        mse = torch.mean((output - batch_tensor) ** 2, dim=1, keepdim=True)  # (N, 1, H, W)
                        batch_probs = mse.cpu().numpy()
                        # Transponer (N, 1, H, W) -> (N, H, W, 1)
                        batch_probs = np.transpose(batch_probs, (0, 2, 3, 1))
                    else:
                        # Para UNet supervisado: usar la salida directa
                        batch_probs = output.cpu().numpy()
                        # Convertir de (N, C, H, W) -> (N, H, W, C)
                        batch_probs = np.transpose(batch_probs, (0, 2, 3, 1))
                    
                    probs.append(batch_probs)
                    
                    # Actualizar progreso
                    batch_num = i // job.batch_size + 1
                    progress = 0.4 + 0.4 * (batch_num / total_batches)
                    InferenceController.update_job_status(db, job_id, "running", progress=progress)
            
            probs = np.concatenate(probs, axis=0)
            probs2d = probs[..., 0]  # Extraer canal único (N, H, W)
            
            # 6. Reconstruir mosaico
            mosaic = reconstruct_from_patches(
                probs2d,
                coords,
                out_shape=image.shape[:2],
                patch_size=patch_size,
                reduce="mean"
            )
            InferenceController.update_job_status(db, job_id, "running", progress=0.9)
            
            # 7. Normalizar y aplicar umbral
            if is_unsupervised:
                # Para autoencoder: normalizar el mapa de error MSE a [0, 1]
                mosaic_min = mosaic.min()
                mosaic_max = mosaic.max()
                if mosaic_max > mosaic_min:
                    mosaic = (mosaic - mosaic_min) / (mosaic_max - mosaic_min)
                else:
                    mosaic = np.zeros_like(mosaic)
            
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
