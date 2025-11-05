"""
Tareas de Celery para entrenamiento NO SUPERVISADO
"""

from backend.tasks.celery_app import celery_app
from backend.controllers import unsupervised_controller


@celery_app.task(name="backend.tasks.unsupervised_tasks.train_unsupervised_task")
def train_unsupervised_task(
    job_id: int,
    model_name: str,
    images_folder: str,
    epochs: int = 50,
    batch_size: int = 16,
    latent_dim: int = 128,
    validation_split: float = 0.2
):
    """
    Tarea asíncrona para entrenar un modelo NO SUPERVISADO (Autoencoder).
    
    Solo requiere imágenes, NO necesita máscaras.
    """
    return unsupervised_controller.execute_unsupervised_training(
        job_id=job_id,
        model_name=model_name,
        images_folder=images_folder,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        validation_split=validation_split
    )
