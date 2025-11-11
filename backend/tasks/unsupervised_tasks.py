"""
Tareas de Celery para entrenamiento NO SUPERVISADO
"""

import subprocess
import sys
from pathlib import Path
from celery.utils.log import get_task_logger
from backend.tasks.celery_app import celery_app

# Logger de Celery para que aparezca en consola del worker
logger = get_task_logger(__name__)


@celery_app.task(name="backend.tasks.unsupervised_tasks.train_unsupervised_task")
def train_unsupervised_task(
    job_id: int,
    model_name: str,
    images_folder: str,
    epochs: int = 50,
    batch_size: int = 16,
    latent_dim: int = 128,
    validation_split: float = 0.2,
    debug: bool = False,
):
    """
    Tarea asíncrona para entrenar un modelo NO SUPERVISADO (Autoencoder).
    
    Solo requiere imágenes, NO necesita máscaras.
    
    IMPORTANTE: Ejecuta en un proceso separado para evitar problemas de CUDA con fork.
    """
    # Ruta al script independiente de entrenamiento
    script_path = Path(__file__).parent.parent / "scripts" / "run_unsupervised_training.py"
    
    # Argumentos para el script
    args = [
        str(job_id),
        model_name,
        images_folder,
        str(epochs),
        str(batch_size),
        str(latent_dim),
        str(validation_split)
    ]
    
    # Crear un entorno limpio para el subproceso
    # Importante: NO heredar variables de CUDA del proceso padre
    import os
    clean_env = os.environ.copy()
    
    # Limpiar cualquier estado de CUDA previo
    cuda_vars_to_remove = [
        'CUDA_MODULE_LOADING',
        'TF_CUDNN_DETERMINISTIC',
        'TF_DETERMINISTIC_OPS'
    ]
    for var in cuda_vars_to_remove:
        clean_env.pop(var, None)
    
    # Configurar variables necesarias
    clean_env['PYTHONUNBUFFERED'] = '1'
    # No forzamos siempre CUDA_VISIBLE_DEVICES a '0' — si el worker ya lo tiene
    # configurado, no lo sobrescribimos. Forzar el valor erróneo puede producir
    # CUDA_ERROR_NO_DEVICE en entornos con mapeos distintos.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        clean_env['CUDA_VISIBLE_DEVICES'] = '0'  # Usar GPU 0 si no existe
    # PyTorch doesn't require TF-specific env vars. Keep environment minimal.
    # Si solicitamos debug, habilitar diagnóstico dentro del proceso hijo
    if debug:
        clean_env['MM_GPU_DEBUG'] = '1'
        clean_env['MM_GPU_DEBUG_ONLY'] = '1'
    
    # Ejecutar en proceso separado (spawn, no fork)
    # Esto evita heredar el estado de CUDA del proceso padre
    # Usamos Popen y stream de stdout para que el worker de Celery muestre
    # el progreso en tiempo real (imprime líneas según el script las emite).
    process = subprocess.Popen(
        [sys.executable, str(script_path)] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=clean_env
    )

    output_lines = []
    try:
        # Iterar sobre las líneas de salida mientras el proceso corre
        if process.stdout is not None:
            for raw in iter(process.stdout.readline, ''):
                line = raw.rstrip('\n')
                if line:
                    # Usar logger.warning para que aparezca en consola con loglevel=WARNING
                    logger.warning(line)
                    output_lines.append(line)
        process.wait()
    except Exception:
        # Asegurar terminación si hay excepción
        try:
            process.kill()
        except Exception:
            pass
        raise

    combined = "\n".join(output_lines)
    if process.returncode != 0:
        logger.error(f"❌ Error en entrenamiento (exit {process.returncode}):")
        logger.error(combined)
        raise RuntimeError(f"Training failed (exit {process.returncode}). See logs above.")

    # Devolver la salida combinada como resultado
    return combined
