"""Inicia Celery worker"""
import os
import subprocess

if __name__ == "__main__":
    print("🔥 Iniciando Celery worker...")
    subprocess.run([
        "celery",
        "-A", "backend.tasks.celery_app",
        "worker",
        "--loglevel=info",
        "--pool=solo"  # Para Windows
    ])
