"""__init__.py para backend.models"""
from backend.models.db_models import TrainingJob, InferenceJob, MLModel
# Los módulos que necesiten MLModelWrapper deben importarlo directamente

__all__ = [
    "TrainingJob",
    "InferenceJob",
    "MLModel",
]
