"""Schemas Pydantic para Models API"""
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Información de un modelo guardado"""
    model_id: str
    name: Optional[str] = None
    architecture: str
    backbone: str
    input_shape: List[int]
    num_parameters: Optional[int] = None
    epochs_trained: int
    final_iou: Optional[float] = None
    final_loss: Optional[float] = None
    file_size_mb: Optional[float] = None
    training_images: Optional[int] = None
    created_at: datetime
    is_active: bool
    description: Optional[str] = None


class ModelsList(BaseModel):
    """Lista de modelos"""
    models: List[ModelInfo]
    total: int


class ModelDelete(BaseModel):
    """Response al eliminar modelo"""
    model_id: str
    message: str
    success: bool
