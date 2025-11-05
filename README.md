# U-Net para detección de anomalías en imágenes GeoTIFF

Este proyecto implementa un flujo de segmentación semántica para detectar anomalías (p.ej., estrés hídrico) en imágenes multiespectrales `.tif` usando un modelo U-Net.

## Estructura

- `procesamiento.py`: utilidades para cargar/normalizar imágenes, crear parches, reconstruir mosaicos y guardar máscaras GeoTIFF.
- `architecture_unet.py`: función `build_unet` para construir y compilar un U-Net con `segmentation-models` (backbones tipo `resnet34`).
- `training_model.py`: script de entrenamiento. Carga datos, divide train/val, entrena y guarda el modelo y el historial.
- `results_visualization.py`: reconstruye el conjunto de validación, carga el último modelo y muestra predicciones vs. etiquetas.
- `predict_full_image.py`: infiere sobre una imagen completa `.tif`, mosaica la predicción y guarda una máscara georreferenciada.

## Requisitos

- Python 3.11 recomendado (para compatibilidad con binarios de Windows).
- GPU NVIDIA opcional (para acelerar entrenamiento con TensorFlow + CUDA/cuDNN compatibles).

Instala dependencias dentro de tu entorno virtual:

```powershell
# Activar entorno si aplica
& ".\.venv\Scripts\Activate.ps1"

# Instalar dependencias
python -m pip install -r requirements.txt
```

Notas:

- `segmentation-models==1.0.1` requiere establecer el framework `tf.keras`. El código ya lo configura.
- Si tienes problemas con TensorFlow en Windows, considera instalar la versión recomendada para tu GPU/CPU desde la guía oficial de TensorFlow.

## Entrenamiento

1. Coloca tus archivos `mi_campo.tif` y `mi_mascara_estres.tif` en la raíz del proyecto, o ajusta las rutas en `training_model.py`.
2. Ejecuta:

```powershell
python training_model.py
```

El modelo se guardará en `models/unet_estres_hidrico_YYYYMMDD_HHMMSS.keras` y el historial en `models/history_YYYYMMDD_HHMMSS.json`.

## Visualización de resultados

```powershell
python results_visualization.py
```

Se mostrarán algunos parches de validación con su predicción.

## Inferencia en imagen completa

Ajusta `IMAGE_PATH` en `predict_full_image.py` y ejecuta:

```powershell
python predict_full_image.py
```

Se generará `prediccion_anomalias.tif` con la máscara binaria georreferenciada.

## Consejos

- Para imágenes grandes, reduce el `BATCH_SIZE` o usa `STRIDE < PATCH` para traslape y predicciones más suaves.
- Si tus imágenes tienen más de 3 canales, usa `encoder_weights=None` en `training_model.py`.
