"""Modelos de base de datos (SQLAlchemy)"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from backend.database import Base


class Job(Base):
    """Modelo genérico para trabajos (entrenamiento, inferencia, etc.)"""
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    job_type = Column(String, nullable=False)  # supervised_training, unsupervised_training, inference
    progress = Column(Integer, default=0)
    logs = Column(Text, nullable=True)
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelMetadata(Base):
    """Metadatos de modelos entrenados"""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    architecture = Column(String, nullable=False)  # unet, autoencoder
    input_shape = Column(String, nullable=False)
    training_type = Column(String, nullable=False)  # supervised, unsupervised
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingJob(Base):
    """Modelo para trabajos de entrenamiento"""
    __tablename__ = "training_jobs"
    
    job_id = Column(String, primary_key=True, index=True, nullable=False)
    status = Column(String, default="queued")  # queued, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    
    # Configuración
    images_folder = Column(String, nullable=False)
    masks_folder = Column(String, nullable=False)
    patch_size = Column(Integer, default=256)
    stride = Column(Integer, default=128)
    batch_size = Column(Integer, default=8)
    epochs = Column(Integer, default=50)
    backbone = Column(String, default="resnet34")
    encoder_weights = Column(String, nullable=True)
    
    # Progreso
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=50)
    
    # Métricas
    current_loss = Column(Float, nullable=True)
    current_iou = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    final_iou = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    val_iou = Column(Float, nullable=True)
    
    # Historia completa (JSON)
    history = Column(JSON, nullable=True)
    
    # Paths
    model_path = Column(String, nullable=True)
    history_path = Column(String, nullable=True)
    
    # Celery task ID para cancelación
    celery_task_id = Column(String, nullable=True)
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Stats
    training_patches = Column(Integer, nullable=True)
    validation_patches = Column(Integer, nullable=True)
    num_channels = Column(Integer, nullable=True)


class InferenceJob(Base):
    """Modelo para trabajos de inferencia"""
    __tablename__ = "inference_jobs"
    
    job_id = Column(String, primary_key=True, index=True, nullable=False)
    status = Column(String, default="queued")  # queued, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    
    # Configuración
    image_path = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    threshold = Column(Float, default=0.5)
    stride = Column(Integer, default=256)
    batch_size = Column(Integer, default=16)
    
    # Resultados
    output_path = Column(String, nullable=True)
    anomaly_pixels = Column(Integer, nullable=True)
    total_pixels = Column(Integer, nullable=True)
    anomaly_percentage = Column(Float, nullable=True)
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class MLModel(Base):
    """Modelo para almacenar info de modelos entrenados"""
    __tablename__ = "ml_models"
    
    model_id = Column(String, primary_key=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    
    # Info del modelo
    architecture = Column(String, default="U-Net")
    backbone = Column(String, default="resnet34")
    input_shape = Column(JSON, nullable=False)  # [H, W, C]
    num_parameters = Column(Integer, nullable=True)
    
    # Training info
    training_job_id = Column(String, nullable=True)
    epochs_trained = Column(Integer, nullable=False)
    final_iou = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    
    # Paths
    model_path = Column(String, nullable=False)
    history_path = Column(String, nullable=True)
    
    # Stats
    file_size_mb = Column(Float, nullable=True)
    training_images = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    description = Column(Text, nullable=True)
