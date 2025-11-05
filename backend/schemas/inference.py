"""Schemas Pydantic para Inference API"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class InferencePredict(BaseModel):
    """Request para predicción"""
    image_path: str = Field(..., description="Ruta a imagen .tif para inferencia")
    model_id: str = Field(..., description="ID del modelo a usar")
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    stride: int = Field(256, ge=32, le=512)
    batch_size: int = Field(16, ge=1, le=64)


class InferenceStatus(BaseModel):
    """Response con estado de inferencia"""
    job_id: str
    status: str  # queued, running, completed, failed, cancelled
    progress: float = Field(..., ge=0.0, le=1.0)
    output_path: Optional[str] = None
    stats: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class InferenceJobCreate(BaseModel):
    """Response al crear un job de inferencia"""
    job_id: str
    status: str
    message: str
