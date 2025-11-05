# 🌾 U-Net Anomaly Detection System - Arquitectura MVC

Sistema completo de detección de anomalías en imágenes multiespectrales usando U-Net con arquitectura MVC + APIs REST.

## 🏗️ Arquitectura

```
├── backend/                    # API REST (FastAPI)
│   ├── models/                 # Capa MODEL (Datos & ML)
│   │   ├── db_models.py        # SQLAlchemy models
│   │   ├── ml_models.py        # U-Net wrapper
│   │   ├── procesamiento.py    # Procesamiento de imágenes
│   │   └── architecture_unet.py
│   ├── controllers/            # Capa CONTROLLER (Lógica)
│   │   ├── training_controller.py
│   │   └── inference_controller.py
│   ├── routes/                 # Endpoints API
│   │   ├── training.py
│   │   ├── inference.py
│   │   └── models.py
│   ├── tasks/                  # Celery tasks (async)
│   │   ├── training_tasks.py
│   │   └── inference_tasks.py
│   └── main.py                 # FastAPI app
│
├── frontend_desktop/           # Capa VIEW (PyQt5)
│   ├── views/
│   │   ├── main_window.py
│   │   ├── training_view.py
│   │   ├── inference_view.py
│   │   └── results_view.py
│   ├── utils/
│   │   └── api_client.py       # Cliente HTTP
│   └── main.py
│
├── data/                       # Datasets
├── models/                     # Modelos entrenados
├── output/                     # Resultados
└── logs/                       # Logs
```

## 📋 Requisitos Previos

### 1. Python 3.11+

### 2. Redis (para Celery)

**Windows - Opción Recomendada: Docker Compose**

```powershell
# Asegúrate de tener Docker Desktop instalado
docker-compose up -d

# Verificar
docker ps
```

**Alternativas:**

- **WSL2 + Ubuntu**: `wsl --install` → `sudo apt install redis-server`
- **Memurai**: https://www.memurai.com/ (Redis nativo para Windows)

## 🚀 Instalación

### 1. Clonar y crear entorno

```powershell
cd proyectoModelAI
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

### 2. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

## ▶️ Ejecución

### Terminal 1: Redis

```powershell
# Si usas Docker Compose (Recomendado)
docker-compose up -d
# Redis corre en background, no necesitas mantener la terminal abierta

# Si usas WSL
wsl -d Ubuntu
redis-server

# Si usas Memurai, se ejecuta como servicio automático
```

### Terminal 2: Backend API

```powershell
python start_backend.py

# O manualmente:
# python -m uvicorn backend.main:app --reload
```

Accede a la documentación interactiva:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Terminal 3: Celery Worker

```powershell
python start_celery.py

# O manualmente:
# celery -A backend.tasks.celery_app worker --loglevel=info --pool=solo
```

### Terminal 4: Frontend Desktop

```powershell
python start_app.py

# O manualmente:
# python frontend_desktop/main.py
```

## 📱 Uso de la Aplicación

### 1. Entrenar Modelo

1. Abre la pestaña **"📚 Entrenar Modelo"**
2. Selecciona carpeta de **imágenes** (.tif)
3. Selecciona carpeta de **máscaras** (.tif)
4. Configura parámetros:
   - Patch Size: 256 (recomendado)
   - Stride: 128 (50% overlap)
   - Batch Size: 8 (ajustar según GPU)
   - Epochs: 50
5. Click en **"🚀 INICIAR ENTRENAMIENTO"**
6. Monitorea el progreso en tiempo real

### 2. Inferencia

1. Abre la pestaña **"🔍 Inferencia"**
2. Selecciona imagen de prueba
3. Elige modelo entrenado
4. Ajusta umbral (0.5 por defecto)
5. Click en **"🎯 PREDECIR ANOMALÍAS"**
6. Revisa resultados en `output/`

### 3. Ver Resultados

1. Abre la pestaña **"📊 Resultados"**
2. Ve lista de modelos entrenados
3. Click en **"🔄 Actualizar"** para refrescar

## 🔌 API Endpoints

### Training

```http
POST   /api/v1/training/start
GET    /api/v1/training/status/{job_id}
DELETE /api/v1/training/cancel/{job_id}
```

### Inference

```http
POST   /api/v1/inference/predict
GET    /api/v1/inference/status/{job_id}
```

### Models

```http
GET    /api/v1/models/
GET    /api/v1/models/{model_id}
DELETE /api/v1/models/{model_id}
```

## 🐛 Troubleshooting

### Backend no se conecta

```powershell
# Verificar que Redis esté corriendo
redis-cli ping
# Debe responder: PONG

# Verificar que FastAPI esté corriendo
curl http://localhost:8000/health
```

### Celery no procesa tareas

```powershell
# Verificar logs de Celery
# Debe decir: "ready" o "celery@... ready"

# Reiniciar Celery
# Ctrl+C en la terminal de Celery
python start_celery.py
```

### PyQt5 no se muestra correctamente

```powershell
# Reinstalar PyQt5
pip uninstall PyQt5
pip install PyQt5==5.15.9
```

## 📊 Estructura de Datos

### Carpetas Requeridas

```
data/
├── train/           # Imágenes originales multiespectrales
│   ├── vuelo1.tif
│   ├── vuelo2.tif
│   └── ...
└── masks/           # Máscaras binarias de segmentación
    ├── vuelo1.tif   # Mismo nombre que la imagen correspondiente
    ├── vuelo2.tif
    └── ...
```

### 📸 Formato de Imágenes Originales

- **Formato**: `.tif` o `.tiff` (GeoTIFF)
- **Tipo**: Imágenes multiespectrales capturadas con dron UAV
- **Canales**: RGB, NIR (Infrarrojo Cercano), RedEdge, etc.
  - Depende de tu cámara multiespectral (e.g., Parrot Sequoia, MicaSense)
- **Ubicación**: `data/train/`

### 🎯 Formato de Máscaras de Segmentación

- **Formato**: `.tif` o `.tiff` (GeoTIFF)
- **Tipo**: Máscaras binarias de anotación
- **Valores**:
  - `0` (negro) = Área sana/normal del cultivo
  - `1` (blanco) = Área con anomalía/problema detectado
- **Requisitos**:
  - ⚠️ **Mismo nombre** que la imagen correspondiente (ej: `vuelo1.tif` → `vuelo1.tif`)
  - ⚠️ **Mismas dimensiones** (ancho × alto) que la imagen
  - ⚠️ Se recomienda conservar la georeferenciación (opcional)
- **Ubicación**: `data/masks/`

### 🛠️ Generar Máscaras en Cero (Cultivo Sano)

Si solo tienes imágenes de cultivo sano sin anomalías, usa el script:

```powershell
python generate_zero_masks.py
```

Esto creará máscaras completamente negras (valor 0 = todo sano) automáticamente.

### 📝 Herramientas para Crear Máscaras

- **QGIS** (gratuito) - Para imágenes georreferenciadas
- **LabelMe** - Para anotación manual
- **GIMP/Photoshop** - Edición de imágenes
- **Python + OpenCV** - Automatización programática

## 🎯 Próximos Pasos

- [ ] Añadir autenticación (JWT)
- [ ] Implementar frontend web (React)
- [ ] Agregar data augmentation
- [ ] Soporte para modelos pre-entrenados
- [ ] Dashboard de métricas (Grafana)
- [ ] Docker deployment

## 📄 Licencia

MIT License

## 👥 Autores

Sistema desarrollado para detección de anomalías en cultivos con imágenes UAV multiespectrales.
