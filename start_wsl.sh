#!/bin/bash
# Script para iniciar el backend en WSL2 con GPU

cd "$(dirname "$0")"
source .venv-wsl/bin/activate
python start_backend.py
