"""
Controlador para entrenamiento NO SUPERVISADO con Autoencoder

Este controlador maneja el entrenamiento de modelos que NO requieren máscaras.
Solo usa imágenes para aprender patrones normales.
"""

import os
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.db_models import TrainingJob, MLModel
from backend.models import procesamiento
from backend.utils.gpu_config import print_device_info, configure_gpu, auto_configure

def create_job(
    db: Session,
    model_name: str,
    images_folder: str,
    epochs: int = 50,
    batch_size: int = 16,
    job_type: str = "unsupervised_training"
) -> TrainingJob:
    """Crea un nuevo trabajo de entrenamiento no supervisado."""
    job_id = f"unsupervised_{uuid.uuid4().hex[:8]}"
    
    job = TrainingJob(
        job_id=job_id,
        status="queued",
        images_folder=images_folder,
        masks_folder="",  # No se usan máscaras en no supervisado
        batch_size=batch_size,
        epochs=epochs,
        total_epochs=epochs,
        backbone="autoencoder",  # Indicar que es autoencoder
        created_at=datetime.now()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job_status(db: Session, job_id: str) -> Optional[TrainingJob]:
    """Obtiene el estado actual de un trabajo."""
    return db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()


def update_job_status(
    db: Session, 
    job_id: str, 
    status: str, 
    progress: Optional[int] = None,
    current_epoch: Optional[int] = None,
    logs: Optional[str] = None,
    result: Optional[str] = None,
    metrics: Optional[dict] = None
):
    """Actualiza el estado del trabajo."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if job:
        job.status = status
        if progress is not None:
            job.progress = progress / 100.0  # Convertir a 0.0-1.0
        if current_epoch is not None:
            job.current_epoch = current_epoch
        if metrics:
            job.current_loss = metrics.get("loss")
            job.val_loss = metrics.get("val_loss")
            if status == "completed":
                job.final_loss = metrics.get("loss")
        if status == "running" and job.started_at is None:
            job.started_at = datetime.now()
        if status in ["completed", "failed", "cancelled"]:
            job.completed_at = datetime.now()
        if result:
            job.error_message = result if status == "failed" else None
        db.commit()


def save_model_metadata(
    db: Session,
    model_name: str,
    model_path: str,
    input_shape: tuple,
    latent_dim: int,
    threshold: float,
    metrics: dict,
    training_job_id: str,
    epochs_trained: int
):
    """Guarda metadatos del modelo entrenado."""
    model_id = f"autoencoder_{uuid.uuid4().hex[:8]}"
    
    metadata = MLModel(
        model_id=model_id,
        name=model_name,
        architecture="autoencoder",
        backbone="autoencoder",
        input_shape={
            "height": input_shape[0],
            "width": input_shape[1],
            "channels": input_shape[2],
            "latent_dim": latent_dim,
            "anomaly_threshold": float(threshold)
        },
        training_job_id=training_job_id,
        epochs_trained=epochs_trained,
        final_loss=float(metrics.get("loss", 0)),
        model_path=model_path,
        created_at=datetime.now(),
        is_active=True
    )
    db.add(metadata)
    db.commit()
    return metadata


def execute_unsupervised_training(
    job_id: str,
    model_name: str,
    images_folder: str,
    epochs: int = 50,
    batch_size: int = 16,
    latent_dim: int = 128,
    validation_split: float = 0.2
):
    """
    Ejecuta el entrenamiento NO SUPERVISADO del autoencoder.
    
    IMPORTANTE: Solo usa IMÁGENES, NO requiere máscaras.
    """
    db = next(get_db())
    
    try:
        # Try to configure GPUs (PyTorch). This will print device info.
        try:
            auto_configure(verbose=True)
        except Exception:
            pass

        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO NO SUPERVISADO (PyTorch)")
        print("="*60)

        # Obtener job para verificar cancelación
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} no encontrado")

        # Actualizar estado inicial
        update_job_status(db, job_id, "running", 0)

        # Función para verificar si el job fue cancelado
        def check_cancelled():
            db.refresh(job)
            return job.status == "cancelled"

        # ========== 1. OBTENER RUTAS DE IMÁGENES (lazy loading) ==========
        update_job_status(db, job_id, "running", 10)
        
        # Importar PyTorch y Dataset personalizado
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, random_split
        except Exception as e:
            raise RuntimeError(f"PyTorch no está disponible en el entorno: {e}")

        # Obtener rutas de imágenes sin cargarlas en memoria
        image_paths = procesamiento.get_image_paths(images_folder)
        if len(image_paths) == 0:
            raise ValueError(f"No se encontraron imágenes en {images_folder}")
        
        print(f"🖼️ Total de imágenes encontradas: {len(image_paths)}")

        update_job_status(db, job_id, "running", 20)

        # Crear dataset lazy (no carga imágenes en memoria)
        full_dataset = procesamiento.LazyImageDataset(
            image_paths=image_paths,
            patch_size=procesamiento.PATCH_SIZE,
            cancel_check_fn=check_cancelled
        )
        
        # Split train/validation
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        
        print(f"📊 Train: {train_size} imágenes, Val: {val_size} imágenes")

        update_job_status(db, job_id, "running", 30)

        # Forzar uso de GPU - no permitir CPU
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU no disponible. El entrenamiento requiere GPU CUDA.")
        
        device = torch.device('cuda')
        print(f"🔧 Usando dispositivo: {device} (GPU forzada)")

        # DataLoaders con num_workers=0 para evitar problemas de memoria
        # pin_memory=True acelera transferencias a GPU
        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Evitar problemas de multiprocessing
            pin_memory=(device.type == 'cuda')
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda')
        )

        channels = full_dataset.num_channels
        patch_size = full_dataset.patch_size

        # Simple convolutional autoencoder
        class ConvAutoencoder(nn.Module):
            def __init__(self, in_channels=channels, latent_dim=128, img_size=patch_size):
                super().__init__()
                # Calcular tamaño después de 3 convoluciones con stride=2
                encoded_size = img_size // 8
                flattened_size = 128 * encoded_size * encoded_size
                
                self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.Flatten(),
                    nn.Linear(flattened_size, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, flattened_size),
                    nn.Unflatten(1, (128, encoded_size, encoded_size)),
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                z = self.encoder(x)
                xrec = self.decoder(z)
                return xrec

        model = ConvAutoencoder(in_channels=channels, latent_dim=latent_dim, img_size=patch_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        update_job_status(db, job_id, "running", 40)

        # Training loop
        epochs_completed = 0
        for epoch in range(epochs):
            # Verificar si el job fue cancelado
            db.refresh(job)
            if job.status == "cancelled":
                print("🛑 Entrenamiento cancelado por el usuario")
                return "Training cancelled by user"
            
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                xb = batch.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, xb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                
                # Liberar memoria después de cada batch
                del xb, out, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            epoch_loss = running_loss / len(train_loader.dataset)

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xb = batch.to(device)
                    out = model(xb)
                    l = criterion(out, xb)
                    val_loss += l.item() * xb.size(0)
                    
                    # Liberar memoria después de cada batch
                    del xb, out, l
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            val_loss = val_loss / len(val_loader.dataset)

            epochs_completed = epoch + 1
            progress = 45 + int((epoch + 1) / epochs * 40)
            
            # Print progress para que aparezca en logs de Celery
            print(f"📊 Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            update_job_status(db, job_id, "running", progress, current_epoch=epoch+1, metrics={"loss": epoch_loss, "val_loss": val_loss})

        update_job_status(db, job_id, "running", 85)

        # Compute reconstruction errors on validation set
        print("📊 Calculando threshold de anomalías...")
        model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                xb = batch.to(device)  # Original image
                out = model(xb)  # Reconstructed image
                
                # Calcular error por imagen
                error = torch.mean((xb - out) ** 2, dim=(1, 2, 3))  # MSE por imagen
                reconstruction_errors.extend(error.cpu().numpy())
                
                # Liberar memoria
                del xb, out, error
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        reconstruction_errors = np.array(reconstruction_errors)
        threshold = float(np.percentile(reconstruction_errors, 95))
        print(f"🎯 Threshold calculado: {threshold:.6f}")

        update_job_status(db, job_id, "running", 90)

        # Save model and threshold
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"{model_name}.pt"
        try:
            # Guardar con metadata para poder reconstruir el modelo
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'model_type': 'unsupervised_autoencoder',
                    'in_channels': channels,
                    'latent_dim': latent_dim,
                    'img_size': patch_size,
                },
                'threshold': threshold,
            }
            torch.save(checkpoint, str(model_path))
        except Exception as e:
            print(f"⚠️ Error guardando el modelo: {e}")

        threshold_path = models_dir / f"{model_name}_threshold.txt"
        threshold_path.write_text(str(threshold))

        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.model_path = str(model_path)

        update_job_status(db, job_id, "running", 95)

        final_metrics = {"loss": float(epoch_loss), "val_loss": float(val_loss)}

        save_model_metadata(
            db=db,
            model_name=model_name,
            model_path=str(model_path),
            input_shape=(procesamiento.PATCH_SIZE, procesamiento.PATCH_SIZE, channels),
            latent_dim=latent_dim,
            threshold=threshold,
            metrics=final_metrics,
            training_job_id=job.job_id if job else "",
            epochs_trained=epochs_completed
        )

        update_job_status(db, job_id, "completed", 100, result=f"Modelo guardado: {model_path}", metrics=final_metrics)

        return str(model_path)
        
    except Exception as e:
        error_msg = f"Error en entrenamiento: {str(e)}"
        update_job_status(db, job_id, "failed", result=error_msg)
        raise
    finally:
        db.close()


__all__ = [
    "create_job",
    "get_job_status",
    "update_job_status",
    "execute_unsupervised_training"
]
