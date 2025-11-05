"""Inicia Celery worker"""
import os
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    print("🔥 Iniciando Celery worker...")
    
    # Usar el python del venv actual
    python_exe = sys.executable
    venv_path = Path(python_exe).parent
    
    # Path al celery del venv
    celery_exe = venv_path / "celery.exe" if os.name == "nt" else venv_path / "celery"
    
    if not celery_exe.exists():
        print(f"❌ No se encontró celery en {celery_exe}")
        print(f"Usando celery desde PATH...")
        celery_cmd = "celery"
    else:
        celery_cmd = str(celery_exe)
    
    subprocess.run([
        celery_cmd,
        "-A", "backend.tasks.celery_app",
        "worker",
        "--loglevel=info",
        "--pool=solo"  # Para Windows
    ])
