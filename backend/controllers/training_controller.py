"""Controller para operaciones de entrenamiento"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from sqlalchemy.orm import Session
from backend.models.db_models import TrainingJob, MLModel
from backend.models.ml_models import MLModelWrapper, prepare_data_for_training
from backend.config import settings
from backend.utils.gpu_config import print_device_info, configure_gpu, auto_configure
from sklearn.model_selection import train_test_split


class TrainingController:
    """Controlador para lógica de entrenamiento"""
    
    @staticmethod
    def create_training_job(
        db: Session,
        images_folder: str,
        masks_folder: str,
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 8,
        epochs: int = 50,
        backbone: str = "resnet34",
        encoder_weights: Optional[str] = None,
    ) -> TrainingJob:
        """Crea un nuevo job de entrenamiento en la DB"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"train_{timestamp}"
        
        job = TrainingJob(
            job_id=job_id,
            status="queued",
            images_folder=images_folder,
            masks_folder=masks_folder,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            total_epochs=epochs,
            backbone=backbone,
            encoder_weights=encoder_weights,
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        return job
    
    @staticmethod
    def get_training_job(db: Session, job_id: str) -> Optional[TrainingJob]:
        """Obtiene un job por ID"""
        return db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    
    @staticmethod
    def update_job_status(
        db: Session,
        job_id: str,
        status: str,
        **kwargs
    ) -> Optional[TrainingJob]:
        """Actualiza el estado de un job"""
        job = TrainingController.get_training_job(db, job_id)
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
    def execute_training(
        job_id: str,
        db: Session,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el entrenamiento (función principal que llama Celery)
        
        Args:
            job_id: ID del job
            db: Sesión de DB
            progress_callback: Callback para actualizar progreso (epoch, loss, iou)
        
        Returns:
            Dict con resultados del entrenamiento
        """
        job = TrainingController.get_training_job(db, job_id)
        if not job:
            return {"success": False, "error": "Job no encontrado"}
        
        try:
            # Configurar GPU antes de comenzar entrenamiento
            print("\n" + "="*60)
            print("🚀 INICIANDO ENTRENAMIENTO")
            print("="*60)
            auto_configure(verbose=True)
            print("="*60 + "\n")
            
            # Actualizar a running
            TrainingController.update_job_status(db, job_id, "running")
            
            # 1. Preparar datos
            x_data, y_data, num_channels = prepare_data_for_training(
                job.images_folder,
                job.masks_folder,
                patch_size=job.patch_size,
                stride=job.stride
            )
            
            # 2. Split train/val
            x_train, x_val, y_train, y_val = train_test_split(
                x_data, y_data, test_size=0.2, random_state=42
            )
            
            # Actualizar stats
            TrainingController.update_job_status(
                db,
                job_id,
                "running",
                training_patches=len(x_train),
                validation_patches=len(x_val),
                num_channels=num_channels
            )
            
            # 3. Construir modelo
            ml_model = MLModelWrapper()
            input_shape = (job.patch_size, job.patch_size, num_channels)
            ml_model.build_and_compile(
                input_shape=input_shape,
                n_classes=1,
                backbone=job.backbone,
                encoder_weights=job.encoder_weights
            )
            
            # 4. Callback personalizado para progreso (PyTorch style)
            class ProgressCallback:
                def __init__(self, job_id, db_session, total_epochs):
                    self.job_id = job_id
                    self.db = db_session
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    progress = (epoch + 1) / self.total_epochs
                    
                    TrainingController.update_job_status(
                        self.db,
                        self.job_id,
                        "running",
                        current_epoch=epoch + 1,
                        progress=progress,
                        current_loss=float(logs.get("loss", 0.0)),
                        current_iou=float(logs.get("iou_score", 0.0)),
                        val_loss=float(logs.get("val_loss", 0.0)),
                        val_iou=float(logs.get("val_iou_score", 0.0))
                    )
                    
                    if progress_callback:
                        progress_callback(epoch + 1, logs)
            
            # 5. Entrenar
            history = ml_model.train(
                x_train,
                y_train,
                x_val,
                y_val,
                epochs=job.epochs,
                batch_size=job.batch_size,
                callbacks=[ProgressCallback(job_id, db, job.epochs)]
            )
            
            # 6. Guardar modelo
            model_path = ml_model.save_model(settings.MODELS_DIR)
            
            # 7. Guardar history
            history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            history_path = settings.MODELS_DIR / f"history_{job_id}.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history_dict, f, indent=2)
            
            # 8. Obtener métricas finales
            final_loss = float(history.history["loss"][-1])
            final_iou = float(history.history["iou_score"][-1])
            val_loss_final = float(history.history["val_loss"][-1])
            val_iou_final = float(history.history["val_iou_score"][-1])
            
            # 9. Actualizar job a completed
            TrainingController.update_job_status(
                db,
                job_id,
                "completed",
                progress=1.0,
                final_loss=final_loss,
                final_iou=final_iou,
                val_loss=val_loss_final,
                val_iou=val_iou_final,
                model_path=model_path,
                history_path=str(history_path),
                history=history_dict
            )
            
            # 10. Crear entrada en MLModel
            model_id = Path(model_path).stem
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            # Obtener número de parámetros (PyTorch style)
            num_params = sum(p.numel() for p in ml_model.model.parameters())
            
            ml_model_entry = MLModel(
                model_id=model_id,
                architecture="U-Net",
                backbone=job.backbone,
                input_shape=[job.patch_size, job.patch_size, num_channels],
                num_parameters=num_params,
                training_job_id=job_id,
                epochs_trained=job.epochs,
                final_iou=final_iou,
                final_loss=final_loss,
                model_path=model_path,
                history_path=str(history_path),
                file_size_mb=file_size,
                training_images=len(x_data)
            )
            
            db.add(ml_model_entry)
            db.commit()
            
            return {
                "success": True,
                "model_path": model_path,
                "model_id": model_id,
                "final_iou": final_iou,
                "final_loss": final_loss
            }
            
        except Exception as e:
            # Actualizar job a failed
            TrainingController.update_job_status(
                db,
                job_id,
                "failed",
                error_message=str(e)
            )
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def cancel_training(db: Session, job_id: str) -> bool:
        """Cancela un entrenamiento en curso"""
        job = TrainingController.get_training_job(db, job_id)
        if not job:
            return False
        
        if job.status in ["queued", "running"]:
            TrainingController.update_job_status(db, job_id, "cancelled")
            return True
        
        return False
