# 📊 Esquema de Base de Datos

Base de datos SQLite: `app.db`

## Tablas

### 1. `jobs` - Trabajos Genéricos

Modelo genérico para rastrear cualquier tipo de trabajo (entrenamiento, inferencia, etc.).

| Campo        | Tipo     | Descripción           | Notas                                                       |
| ------------ | -------- | --------------------- | ----------------------------------------------------------- |
| `id`         | Integer  | ID único (PK)         | Auto-incrementado                                           |
| `model_name` | String   | Nombre del modelo     | Required                                                    |
| `status`     | String   | Estado del trabajo    | `pending`, `running`, `completed`, `failed`                 |
| `job_type`   | String   | Tipo de trabajo       | `supervised_training`, `unsupervised_training`, `inference` |
| `progress`   | Integer  | Progreso (0-100)      | Porcentaje                                                  |
| `logs`       | Text     | Logs del proceso      | Opcional                                                    |
| `result`     | Text     | Resultado del trabajo | Opcional                                                    |
| `created_at` | DateTime | Fecha de creación     | UTC                                                         |
| `updated_at` | DateTime | Última actualización  | UTC                                                         |

---

### 2. `model_metadata` - Metadatos de Modelos

Información general de modelos entrenados.

| Campo           | Tipo     | Descripción               | Notas                        |
| --------------- | -------- | ------------------------- | ---------------------------- |
| `id`            | Integer  | ID único (PK)             | Auto-incrementado            |
| `name`          | String   | Nombre único del modelo   | Unique, Required             |
| `path`          | String   | Ruta al archivo .pt       | Required                     |
| `architecture`  | String   | Arquitectura usada        | `unet`, `autoencoder`        |
| `input_shape`   | String   | Forma de entrada          | Ejemplo: `512x512x3`         |
| `training_type` | String   | Tipo de entrenamiento     | `supervised`, `unsupervised` |
| `metrics`       | JSON     | Métricas de entrenamiento | IoU, Loss, etc.              |
| `created_at`    | DateTime | Fecha de creación         | UTC                          |

---

### 3. `training_jobs` - Trabajos de Entrenamiento

Registro detallado de entrenamientos (supervisados y no supervisados).

#### Campos Principales

| Campo      | Tipo   | Descripción              | Notas                                                   |
| ---------- | ------ | ------------------------ | ------------------------------------------------------- |
| `job_id`   | String | ID único del job (PK)    | Primary Key, Indexed. Ejemplo: `unsupervised_a3f7b2c1`  |
| `status`   | String | Estado del entrenamiento | `queued`, `running`, `completed`, `failed`, `cancelled` |
| `progress` | Float  | Progreso (0.0-1.0)       | 0.0 = 0%, 1.0 = 100%                                    |

#### Configuración del Entrenamiento

| Campo             | Tipo    | Descripción          | Default                |
| ----------------- | ------- | -------------------- | ---------------------- |
| `images_folder`   | String  | Carpeta de imágenes  | Required               |
| `masks_folder`    | String  | Carpeta de máscaras  | Required (supervisado) |
| `patch_size`      | Integer | Tamaño de parches    | 256                    |
| `stride`          | Integer | Stride para parches  | 128                    |
| `batch_size`      | Integer | Tamaño del batch     | 8                      |
| `epochs`          | Integer | Número de épocas     | 50                     |
| `backbone`        | String  | Backbone de la red   | `resnet34`             |
| `encoder_weights` | String  | Pesos pre-entrenados | Opcional               |

#### Progreso y Métricas

| Campo           | Tipo    | Descripción        | Notas                        |
| --------------- | ------- | ------------------ | ---------------------------- |
| `current_epoch` | Integer | Época actual       | Durante entrenamiento        |
| `total_epochs`  | Integer | Total de épocas    | 50 por defecto               |
| `current_loss`  | Float   | Loss actual        | Actualizado cada época       |
| `current_iou`   | Float   | IoU actual         | Solo supervisado             |
| `final_loss`    | Float   | Loss final         | Al terminar                  |
| `final_iou`     | Float   | IoU final          | Solo supervisado             |
| `val_loss`      | Float   | Validation loss    | Si hay validación            |
| `val_iou`       | Float   | Validation IoU     | Solo supervisado             |
| `history`       | JSON    | Historial completo | Todas las métricas por época |

#### Archivos y Celery

| Campo            | Tipo   | Descripción             | Notas            |
| ---------------- | ------ | ----------------------- | ---------------- |
| `model_path`     | String | Ruta al modelo guardado | `.pt` file       |
| `history_path`   | String | Ruta al historial       | `.json` file     |
| `celery_task_id` | String | ID de tarea Celery      | Para cancelación |

#### Información de Error

| Campo           | Tipo | Descripción      | Notas    |
| --------------- | ---- | ---------------- | -------- |
| `error_message` | Text | Mensaje de error | Si falla |

#### Timestamps

| Campo          | Tipo     | Descripción              | Notas |
| -------------- | -------- | ------------------------ | ----- |
| `created_at`   | DateTime | Creación del job         | UTC   |
| `started_at`   | DateTime | Inicio del entrenamiento | UTC   |
| `completed_at` | DateTime | Finalización             | UTC   |

#### Estadísticas

| Campo                | Tipo    | Descripción              | Notas      |
| -------------------- | ------- | ------------------------ | ---------- |
| `training_patches`   | Integer | Parches de entrenamiento | Generados  |
| `validation_patches` | Integer | Parches de validación    | Generados  |
| `num_channels`       | Integer | Canales de entrada       | 3 para RGB |

---

### 4. `inference_jobs` - Trabajos de Inferencia

Registro de predicciones/inferencias realizadas.

#### Campos Principales

| Campo      | Tipo   | Descripción           | Notas                                                   |
| ---------- | ------ | --------------------- | ------------------------------------------------------- |
| `job_id`   | String | ID único del job (PK) | Primary Key, Indexed. Ejemplo: `inference_f5d8b123`     |
| `status`   | String | Estado de inferencia  | `queued`, `running`, `completed`, `failed`, `cancelled` |
| `progress` | Float  | Progreso (0.0-1.0)    | 0.0 = 0%, 1.0 = 100%                                    |

#### Configuración

| Campo        | Tipo    | Descripción            | Default  |
| ------------ | ------- | ---------------------- | -------- |
| `image_path` | String  | Ruta a imagen          | Required |
| `model_id`   | String  | ID del modelo usado    | Required |
| `threshold`  | Float   | Umbral de decisión     | 0.5      |
| `stride`     | Integer | Stride para inferencia | 256      |
| `batch_size` | Integer | Tamaño del batch       | 16       |

#### Resultados

| Campo                | Tipo    | Descripción      | Notas              |
| -------------------- | ------- | ---------------- | ------------------ |
| `output_path`        | String  | Ruta a resultado | Imagen con máscara |
| `anomaly_pixels`     | Integer | Píxeles anómalos | Conteo             |
| `total_pixels`       | Integer | Total de píxeles | De la imagen       |
| `anomaly_percentage` | Float   | % de anomalías   | 0.0-100.0          |

#### Error y Timestamps

| Campo           | Tipo     | Descripción          | Notas    |
| --------------- | -------- | -------------------- | -------- |
| `error_message` | Text     | Mensaje de error     | Si falla |
| `created_at`    | DateTime | Creación del job     | UTC      |
| `started_at`    | DateTime | Inicio de inferencia | UTC      |
| `completed_at`  | DateTime | Finalización         | UTC      |

---

### 5. `ml_models` - Modelos Entrenados

Catálogo de modelos ML disponibles.

#### Identificación

| Campo      | Tipo   | Descripción              | Notas                                                 |
| ---------- | ------ | ------------------------ | ----------------------------------------------------- |
| `model_id` | String | ID único del modelo (PK) | Primary Key, Indexed. Ejemplo: `autoencoder_a3f7b2c1` |
| `name`     | String | Nombre descriptivo       | Opcional                                              |

#### Información del Modelo

| Campo            | Tipo    | Descripción          | Default                |
| ---------------- | ------- | -------------------- | ---------------------- |
| `architecture`   | String  | Arquitectura         | `U-Net`, `Autoencoder` |
| `backbone`       | String  | Backbone usado       | `resnet34`             |
| `input_shape`    | JSON    | Forma de entrada     | `[H, W, C]`            |
| `num_parameters` | Integer | Número de parámetros | Calculado              |

#### Información de Entrenamiento

| Campo             | Tipo    | Descripción               | Notas            |
| ----------------- | ------- | ------------------------- | ---------------- |
| `training_job_id` | String  | ID del job que lo entrenó | FK lógica        |
| `epochs_trained`  | Integer | Épocas entrenadas         | Required         |
| `final_iou`       | Float   | IoU final                 | Solo supervisado |
| `final_loss`      | Float   | Loss final                | Required         |

#### Archivos

| Campo          | Tipo   | Descripción            | Notas    |
| -------------- | ------ | ---------------------- | -------- |
| `model_path`   | String | Ruta al archivo .pt    | Required |
| `history_path` | String | Ruta al historial JSON | Opcional |

#### Estadísticas

| Campo             | Tipo    | Descripción        | Notas         |
| ----------------- | ------- | ------------------ | ------------- |
| `file_size_mb`    | Float   | Tamaño del archivo | En MB         |
| `training_images` | Integer | Imágenes usadas    | Para entrenar |

#### Metadata

| Campo         | Tipo     | Descripción       | Notas            |
| ------------- | -------- | ----------------- | ---------------- |
| `created_at`  | DateTime | Fecha de creación | UTC              |
| `is_active`   | Boolean  | Modelo activo     | True por defecto |
| `description` | Text     | Descripción       | Opcional         |

---

## Relaciones

### Lógicas (No FK físicas)

- `ml_models.training_job_id` → `training_jobs.job_id`
- `inference_jobs.model_id` → `ml_models.model_id`

### Diagrama de Flujo

```
training_jobs → ml_models → inference_jobs
     ↓              ↓            ↓
  [.pt file]   [metadata]   [predictions]
```

---

## Estados Válidos

### Training Jobs

- `queued`: En cola, esperando procesamiento
- `running`: Entrenamiento en progreso
- `completed`: Finalizado exitosamente
- `failed`: Error durante entrenamiento
- `cancelled`: Cancelado por usuario

### Inference Jobs

- `queued`: En cola
- `running`: Inferencia en progreso
- `completed`: Finalizado exitosamente
- `failed`: Error durante inferencia
- `cancelled`: Cancelado por usuario

---

## Inicialización

Para reinicializar la base de datos:

```bash
python reset_database.py
```

Para verificar estado:

```bash
python check_db.py
```

---

## Notas Técnicas

- **Motor**: SQLite
- **ORM**: SQLAlchemy
- **Ubicación**: `app.db` (raíz del proyecto)
- **Migraciones**: Manual (via scripts)
- **Backups**: Se recomienda hacer backup antes de migraciones

---

## Changelog

### 2025-11-09

- ✅ Agregado campo `celery_task_id` a `training_jobs` para cancelación de tareas
- ✅ **BREAKING CHANGE**: Eliminados campos `id` auto-incrementales de `training_jobs`, `inference_jobs` y `ml_models`
- ✅ Migración a IDs basados en strings:
  - `training_jobs.job_id` ahora es Primary Key (ejemplo: `unsupervised_a3f7b2c1`)
  - `inference_jobs.job_id` ahora es Primary Key (ejemplo: `inference_f5d8b123`)
  - `ml_models.model_id` ahora es Primary Key (ejemplo: `autoencoder_a3f7b2c1`)
- ✅ Actualizados controladores y rutas para usar `job_id: str` en lugar de `job_id: int`
- ✅ Base de datos reinicializada con nuevo esquema
