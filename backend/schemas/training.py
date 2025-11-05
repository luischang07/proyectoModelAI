"""Schemas Pydantic para Training API"""
from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class TrainingStart(BaseModel):
    """Request para iniciar entrenamiento"""
    images_folder: str = Field(..., description="Carpeta con imágenes .tif")
    masks_folder: str = Field(..., description="Carpeta con máscaras .tif")
    patch_size: int = Field(256, ge=64, le=1024)
    stride: int = Field(128, ge=32, le=512)
    batch_size: int = Field(8, ge=1, le=64)
    epochs: int = Field(50, ge=1, le=500)
    val_split: float = Field(0.2, ge=0.1, le=0.5)
    backbone: str = Field("resnet34", description="Backbone del U-Net")
    encoder_weights: Optional[str] = Field(None, description="'imagenet' o None")


class TrainingStatus(BaseModel):
    """Response con estado del entrenamiento"""
    job_id: str
    status: str  # queued, running, completed, failed, cancelled
    progress: float = Field(..., ge=0.0, le=1.0)
    current_epoch: int
    total_epochs: int
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TrainingHistory(BaseModel):
    """Response con historial completo de entrenamiento"""
    job_id: str
    epochs: List[int]
    loss: List[float]
    val_loss: List[float]
    iou_score: List[float]
    val_iou_score: List[float]


class TrainingJobCreate(BaseModel):
    """Response al crear un job de entrenamiento"""
    job_id: str
    status: str
    message: str
