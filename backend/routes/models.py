"""API endpoints para modelos"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.db_models import MLModel
from backend.schemas.models import ModelInfo, ModelsList, ModelDelete
import os

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/", response_model=ModelsList)
def list_models(db: Session = Depends(get_db)):
    """Lista todos los modelos"""
    models = db.query(MLModel).filter(MLModel.is_active == True).all()
    
    models_info = [
        ModelInfo(
            model_id=m.model_id,
            name=m.name,
            architecture=m.architecture,
            backbone=m.backbone,
            input_shape=m.input_shape,
            num_parameters=m.num_parameters,
            epochs_trained=m.epochs_trained,
            final_iou=m.final_iou,
            final_loss=m.final_loss,
            file_size_mb=m.file_size_mb,
            training_images=m.training_images,
            created_at=m.created_at,
            is_active=m.is_active,
            description=m.description
        )
        for m in models
    ]
    
    return ModelsList(models=models_info, total=len(models_info))

@router.get("/{model_id}", response_model=ModelInfo)
def get_model(model_id: str, db: Session = Depends(get_db)):
    """Obtiene info de un modelo específico"""
    model = db.query(MLModel).filter(MLModel.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    return ModelInfo(
        model_id=model.model_id,
        name=model.name,
        architecture=model.architecture,
        backbone=model.backbone,
        input_shape=model.input_shape,
        num_parameters=model.num_parameters,
        epochs_trained=model.epochs_trained,
        final_iou=model.final_iou,
        final_loss=model.final_loss,
        file_size_mb=model.file_size_mb,
        training_images=model.training_images,
        created_at=model.created_at,
        is_active=model.is_active,
        description=model.description
    )

@router.delete("/{model_id}", response_model=ModelDelete)
def delete_model(model_id: str, db: Session = Depends(get_db)):
    """Elimina un modelo (soft delete)"""
    model = db.query(MLModel).filter(MLModel.model_id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    model.is_active = False
    db.commit()
    
    return ModelDelete(
        model_id=model_id,
        message="Modelo desactivado correctamente",
        success=True
    )
