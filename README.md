# 🌾 U-Net Anomaly Detection System - Arquitectura MVC

Sistema completo de detección de anomalías en imágenes multiespectrales con **entrenamiento supervisado Y no supervisado** usando arquitectura MVC + APIs REST.

## ✨ Características Principales

- 🎯 **Entrenamiento Supervisado**: U-Net con máscaras etiquetadas (alta precisión)
- 🧠 **Entrenamiento No Supervisado**: Autoencoder sin máscaras (no requiere etiquetado)
- 🚀 **Backend REST API**: FastAPI con procesamiento asíncrono (Celery)
- 🖥️ **Frontend Desktop**: Interfaz gráfica PyQt5 intuitiva
- 📊 **Monitoreo en tiempo real**: Progreso y métricas en vivo
- 🐳 **Docker Ready**: Redis containerizado para producción

## 🏗️ Arquitectura

```
├── backend/                    # API REST (FastAPI)
│   ├── models/                 # Capa MODEL (Datos & ML)
│   │   ├── db_models.py        # SQLAlchemy models
│   │   ├── ml_models.py        # U-Net wrapper
│   │   ├── procesamiento.py    # Procesamiento de imágenes
│   │   ├── architecture_unet.py        # U-Net para supervisado
│   │   └── architecture_autoencoder.py # Autoencoder para no supervisado
│   ├── controllers/            # Capa CONTROLLER (Lógica)
│   │   ├── training_controller.py      # Entrenamiento supervisado
│   │   ├── unsupervised_controller.py  # Entrenamiento no supervisado
│   │   └── inference_controller.py
│   ├── routes/                 # Endpoints API
│   │   ├── training.py         # POST /training/start (supervisado)
│   │   ├── unsupervised.py     # POST /unsupervised/train (no supervisado)
│   │   ├── inference.py
│   │   └── models.py
│   ├── tasks/                  # Celery tasks (async)
│   │   ├── training_tasks.py
│   │   ├── unsupervised_tasks.py
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

### 🐳 Opción A: Docker (Recomendado para Producción)

```bash
# Iniciar todos los servicios con un solo comando
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

📚 **[Guía completa de Docker: DOCKER_GUIDE.md](DOCKER_GUIDE.md)**

### 💻 Opción B: Ejecución Local (Desarrollo)

#### Terminal 1: Redis

```powershell
# Si usas Docker solo para Redis (Recomendado)
docker-compose up -d redis

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

### 🎯 Seleccionar Tipo de Entrenamiento

La aplicación ahora soporta **DOS modos de entrenamiento**:

#### 1️⃣ **Supervisado (con máscaras)**

- ✅ Requiere: Imágenes + Máscaras etiquetadas
- ✅ Modelo: U-Net
- ✅ Ideal para: Alta precisión en anomalías conocidas
- ✅ Usa cuando: Tienes datos etiquetados manualmente

#### 2️⃣ **No Supervisado (sin máscaras)**

- ✅ Requiere: Solo imágenes (sin máscaras)
- ✅ Modelo: Autoencoder
- ✅ Ideal para: Detectar anomalías desconocidas
- ✅ Usa cuando: NO tienes máscaras etiquetadas

📚 **[Lee la guía completa: TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

---

### 1. Entrenar Modelo (SUPERVISADO)

1. Abre la pestaña **"📚 Entrenar Modelo"**
2. Selecciona **"Supervisado (con máscaras)"** en el dropdown
3. Selecciona carpeta de **imágenes** (.tif)
4. Selecciona carpeta de **máscaras** (.tif) ← **Requerido**
5. Configura parámetros:
   - Patch Size: 128 (recomendado, ajustado para memoria)
   - Stride: 64 (50% overlap)
   - Batch Size: 4-8 (ajustar según GPU)
   - Epochs: 25-50
   - Backbone: resnet34, efficientnetb0, etc.
6. Click en **"🚀 INICIAR ENTRENAMIENTO"**
7. Monitorea progreso: Loss, IoU Score en tiempo real

### 2. Entrenar Modelo (NO SUPERVISADO)

1. Abre la pestaña **"📚 Entrenar Modelo"**
2. Selecciona **"No Supervisado (sin máscaras - Autoencoder)"** en el dropdown
3. Selecciona carpeta de **imágenes** (.tif) - Solo imágenes normales/sanas
4. Campo de máscaras se deshabilita automáticamente ← **No requerido**
5. Configura parámetros:
   - Batch Size: 16 (más alto para autoencoder)
   - Epochs: 50-100 (necesita más épocas)
   - Latent Dim: 128 (tamaño del espacio latente)
6. Click en **"🚀 INICIAR ENTRENAMIENTO"**
7. Monitorea progreso: Loss, MAE en tiempo real

⚠️ **Importante para No Supervisado**:

- Entrena **SOLO con imágenes NORMALES/SANAS**
- El modelo aprende qué es "normal"
- En inferencia detectará anomalías por alto error de reconstrucción

### 3. Inferencia

1. Abre la pestaña **"🔍 Inferencia"**
2. Selecciona imagen de prueba
3. Elige modelo entrenado (supervisado o no supervisado)
4. Ajusta umbral (0.5 por defecto)
5. Click en **"🎯 PREDECIR ANOMALÍAS"**
6. Revisa resultados en `output/`

### 4. Ver Resultados

1. Abre la pestaña **"📊 Resultados"**
2. Ve lista de modelos entrenados
3. Click en **"🔄 Actualizar"** para refrescar

## 🔌 API Endpoints

### Training (Supervisado)

```http
POST   /api/v1/training/start
GET    /api/v1/training/status/{job_id}
DELETE /api/v1/training/cancel/{job_id}
```

### Training (No Supervisado) ✨ **NUEVO**

```http
POST   /api/v1/unsupervised/train
```

**Request Body Example:**

```json
{
  "model_name": "autoencoder_cultivo1",
  "images_folder": "data/images",
  "epochs": 50,
  "batch_size": 16,
  "latent_dim": 128,
  "validation_split": 0.2
}
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

### 📁 Carpetas Requeridas

#### Para Entrenamiento **SUPERVISADO** (con máscaras)

```
data/
├── images/          # Imágenes originales multiespectrales
│   ├── vuelo1.tif
│   ├── vuelo2.tif
│   └── ...
└── masks/           # Máscaras binarias de segmentación
    ├── vuelo1.tif   # Mismo nombre que la imagen correspondiente
    ├── vuelo2.tif
    └── ...
```

#### Para Entrenamiento **NO SUPERVISADO** (sin máscaras) ✨ **NUEVO**

```
data/
└── images/          # Solo imágenes NORMALES/SANAS
    ├── sano_001.tif
    ├── sano_002.tif
    ├── sano_003.tif
    └── ...
```

⚠️ **Importante**: Para no supervisado, usa **SOLO imágenes sin anomalías** (cultivo sano).

---

### 📸 Formato de Imágenes Originales

- **Formato**: `.tif` o `.tiff` (GeoTIFF)
- **Tipo**: Imágenes multiespectrales capturadas con dron UAV
- **Canales**: RGB, NIR (Infrarrojo Cercano), RedEdge, etc.
  - Depende de tu cámara multiespectral (e.g., Parrot Sequoia, MicaSense)
- **Ubicación**: `data/images/`

### 🎯 Formato de Máscaras de Segmentación (Solo para Supervisado)

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
- **🚫 NO requerido** para entrenamiento no supervisado

### 🛠️ Generar Máscaras en Cero (Cultivo Sano)

Si solo tienes imágenes de cultivo sano sin anomalías, usa el script:

```powershell
python generate_zero_masks.py
```

Esto creará máscaras completamente negras (valor 0 = todo sano) automáticamente.

**Alternativa**: Usa el modo **No Supervisado** que no requiere máscaras en absoluto! 🎉

### 📝 Herramientas para Crear Máscaras (Solo Supervisado)

- **QGIS** (gratuito) - Para imágenes georreferenciadas
- **LabelMe** - Para anotación manual
- **GIMP/Photoshop** - Edición de imágenes
- **Python + OpenCV** - Automatización programática

## � Comparación: Supervisado vs No Supervisado

| Característica               | Supervisado         | No Supervisado      |
| ---------------------------- | ------------------- | ------------------- |
| **Requiere máscaras**        | ✅ Sí               | ❌ No               |
| **Modelo**                   | U-Net               | Autoencoder         |
| **Precisión**                | ⭐⭐⭐⭐⭐ Alta     | ⭐⭐⭐ Media        |
| **Tiempo preparación**       | 🕐 Alto (etiquetar) | ⚡ Rápido           |
| **Detecta anomalías nuevas** | ❌ Solo conocidas   | ✅ Cualquiera       |
| **Cantidad de datos**        | Media (100-1000)    | Alta (1000+)        |
| **Uso típico**               | Alta precisión      | Exploración inicial |

📚 **[Guía completa: TRAINING_GUIDE.md](TRAINING_GUIDE.md)**

## Próximos Pasos

- [x] Entrenamiento no supervisado (Autoencoder)
- [x] Interfaz para elegir tipo de entrenamiento
- [ ] Inferencia con Autoencoder (detectar anomalías)
- [ ] Añadir autenticación (JWT)
- [ ] Implementar frontend web (React)
- [ ] Agregar data augmentation
- [ ] Soporte para modelos pre-entrenados
- [ ] Dashboard de métricas (Grafana)
- [ ] Docker deployment completo

## 📚 Documentación Adicional

- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - 🐳 Guía completa de despliegue con Docker
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Guía completa de entrenamiento supervisado vs no supervisado
- **[OPTIMIZACIONES_MEMORIA.md](OPTIMIZACIONES_MEMORIA.md)** - Optimizaciones de memoria para entrenamiento
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Estado actual del proyecto y arquitectura
- **[README_MVC.md](README_MVC.md)** - Documentación técnica de arquitectura MVC

## 📄 Licencia

MIT License

## 👥 Autores

Sistema desarrollado para detección de anomalías en cultivos con imágenes UAV multiespectrales.
