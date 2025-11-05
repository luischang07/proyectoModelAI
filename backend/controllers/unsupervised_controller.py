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
from backend.models.architecture_autoencoder import (
    build_autoencoder,
    compile_autoencoder
)
from tensorflow import keras


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


def update_job_status(
    db: Session, 
    job_id: int, 
    status: str, 
    progress: Optional[int] = None,
    current_epoch: Optional[int] = None,
    logs: Optional[str] = None,
    result: Optional[str] = None,
    metrics: Optional[dict] = None
):
    """Actualiza el estado del trabajo."""
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
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
    job_id: int,
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
        # Actualizar estado inicial
        update_job_status(db, job_id, "running", 0)
        
        # ========== 1. CARGAR SOLO IMÁGENES (sin máscaras) ==========
        update_job_status(db, job_id, "running", 10)
        
        # Cargar solo imágenes (x_data)
        x_data = procesamiento.load_images_only(images_folder)
        
        if len(x_data) == 0:
            raise ValueError(f"No se encontraron imágenes en {images_folder}")
        
        update_job_status(db, job_id, "running", 20)
        
        # ========== 2. DIVIDIR EN TRAIN/VALIDATION ==========
        # Solo dividimos las imágenes (no hay y_data)
        from sklearn.model_selection import train_test_split
        
        x_train, x_val = train_test_split(
            x_data,
            test_size=validation_split,
            random_state=42
        )
        
        update_job_status(db, job_id, "running", 30)
        
        # ========== 3. CONSTRUIR AUTOENCODER ==========
        update_job_status(db, job_id, "running", 35)
        
        input_shape = (procesamiento.PATCH_SIZE, procesamiento.PATCH_SIZE, x_train.shape[-1])
        
        autoencoder, encoder, decoder = build_autoencoder(
            input_shape=input_shape,
            latent_dim=latent_dim
        )
        
        compile_autoencoder(autoencoder, learning_rate=0.001)
        
        update_job_status(db, job_id, "running", 40)
        
        # ========== 4. ENTRENAR EL MODELO ==========
        update_job_status(db, job_id, "running", 45)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: update_job_status(
                    db, job_id, "running",
                    progress=45 + int((epoch + 1) / epochs * 40),
                    current_epoch=epoch + 1,
                    metrics={"loss": logs['loss'], "val_loss": logs['val_loss']}
                )
            )
        ]
        
        # ¡IMPORTANTE! En autoencoder, X e Y son iguales
        # El modelo aprende a reconstruir la entrada
        history = autoencoder.fit(
            x_train, x_train,  # x=entrada, y=misma entrada (reconstrucción)
            validation_data=(x_val, x_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # ========== 5. CALCULAR UMBRAL DE ANOMALÍA ==========
        update_job_status(db, job_id, "running", 85)
        
        # Predecir en conjunto de validación
        reconstructed = autoencoder.predict(x_val, verbose=0)
        reconstruction_errors = np.mean(np.square(x_val - reconstructed), axis=(1, 2, 3))
        
        # Usar percentil 95 como umbral
        threshold = float(np.percentile(reconstruction_errors, 95))
        
        update_job_status(db, job_id, "running", 90)
        
        # ========== 6. GUARDAR MODELO ==========
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"{model_name}.keras"
        autoencoder.save(str(model_path))
        
        # Guardar también el encoder y decoder por separado
        encoder.save(str(models_dir / f"{model_name}_encoder.keras"))
        decoder.save(str(models_dir / f"{model_name}_decoder.keras"))
        
        # Guardar umbral en archivo de texto
        threshold_path = models_dir / f"{model_name}_threshold.txt"
        threshold_path.write_text(str(threshold))
        
        # Actualizar job con model_path
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.model_path = str(model_path)
        
        update_job_status(
            db, job_id, "running", 95
        )
        
        # ========== 7. GUARDAR METADATOS ==========
        final_metrics = {
            "loss": float(history.history["loss"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "mae": float(history.history["mae"][-1]),
            "val_mae": float(history.history["val_mae"][-1])
        }
        
        epochs_completed = len(history.history['loss'])
        
        save_model_metadata(
            db=db,
            model_name=model_name,
            model_path=str(model_path),
            input_shape=input_shape,
            latent_dim=latent_dim,
            threshold=threshold,
            metrics=final_metrics,
            training_job_id=job.job_id if job else "",
            epochs_trained=epochs_completed
        )
        
        # ========== 8. FINALIZAR ==========
        result_message = (
            f"✓ Entrenamiento NO SUPERVISADO completado\n"
            f"- Tipo: Autoencoder para detección de anomalías\n"
            f"- Imágenes entrenamiento: {len(x_train)}\n"
            f"- Imágenes validación: {len(x_val)}\n"
            f"- Épocas completadas: {epochs_completed}\n"
            f"- Loss final: {final_metrics['loss']:.6f}\n"
            f"- Val Loss final: {final_metrics['val_loss']:.6f}\n"
            f"- Umbral de anomalía: {threshold:.6f}\n"
            f"- Modelo guardado: {model_path}"
        )
        
        update_job_status(
            db, job_id, "completed", 100,
            result=result_message,
            metrics=final_metrics
        )
        
        return str(model_path)
        
    except Exception as e:
        error_msg = f"Error en entrenamiento: {str(e)}"
        update_job_status(db, job_id, "failed", result=error_msg)
        raise
    finally:
        db.close()


__all__ = [
    "create_job",
    "update_job_status",
    "execute_unsupervised_training"
]
