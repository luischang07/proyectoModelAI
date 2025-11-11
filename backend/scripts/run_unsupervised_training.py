"""
Script independiente para ejecutar entrenamiento NO SUPERVISADO.
Se ejecuta en un proceso separado para evitar conflictos de CUDA con Celery.

USO:
    python run_unsupervised_training.py <job_id> <model_name> <images_folder> <epochs> <batch_size> <latent_dim> <validation_split>
"""

import sys
import os
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en el path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Ejecuta el entrenamiento con los parámetros de línea de comandos."""
    if len(sys.argv) != 8:
        print("❌ Error: Se requieren 7 argumentos")
        print("USO: python run_unsupervised_training.py <job_id> <model_name> <images_folder> <epochs> <batch_size> <latent_dim> <validation_split>")
        sys.exit(1)
    
    # Parse argumentos
    job_id = sys.argv[1]  # String ID (e.g., 'unsupervised_7f1ad897')
    model_name = sys.argv[2]
    images_folder = sys.argv[3]
    epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    latent_dim = int(sys.argv[6])
    validation_split = float(sys.argv[7])
    
    print(f"🚀 Iniciando entrenamiento NO SUPERVISADO...")
    print(f"   Job ID: {job_id}")
    print(f"   Modelo: {model_name}")
    print(f"   Imágenes: {images_folder}")
    print(f"   Epochs: {epochs}, Batch: {batch_size}, Latent: {latent_dim}")
    
    # Opcional: diagnóstico GPU — sólo si se solicita desde el entorno del proceso
    # Para activar, exportar MM_GPU_DEBUG=1 en el entorno del proceso que ejecuta este script
    if os.environ.get("MM_GPU_DEBUG") == "1":
        print("=== GPU DEBUG START ===")
        for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH"):
            print(f"ENV {k} = {os.environ.get(k)!r}")
        try:
            import subprocess
            subprocess.run(['bash', '-lc', 'ls -l /dev | egrep "nvidia|dxg" || true'], check=False)
        except Exception:
            pass
        try:
            import torch
            print("torch.__version__=", torch.__version__)
            print("torch.cuda.is_available()=", torch.cuda.is_available())
            print("torch.cuda.device_count()=", torch.cuda.device_count())
            if torch.cuda.is_available():
                print("torch.cuda.get_device_name(0)=", torch.cuda.get_device_name(0))
        except Exception as e:
            print('PyTorch import error:', repr(e))
        print("=== GPU DEBUG END ===")
        # Si sólo queremos ejecutar el diagnóstico, salir antes de iniciar el entrenamiento
        if os.environ.get("MM_GPU_DEBUG_ONLY") == "1":
            print("Exiting after GPU debug because MM_GPU_DEBUG_ONLY=1")
            sys.exit(0)

    # IMPORTANTE: Importar el controlador AQUÍ, no al inicio del archivo
    # Esto asegura que PyTorch se importe en un proceso limpio
    from backend.controllers import unsupervised_controller
    
    # Ejecutar entrenamiento
    unsupervised_controller.execute_unsupervised_training(
        job_id=job_id,
        model_name=model_name,
        images_folder=images_folder,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        validation_split=validation_split
    )
    
    print(f"✅ Entrenamiento completado exitosamente")

if __name__ == "__main__":
    main()
