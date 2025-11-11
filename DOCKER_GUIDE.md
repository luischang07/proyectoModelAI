# 🐳 Docker Deployment Guide

Guía completa para desplegar ModelAI usando Docker y Docker Compose.

## 📋 Requisitos Previos

### Instalación de Docker

#### Windows

1. Descarga [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Instala y reinicia el sistema
3. Verifica: `docker --version`

#### Linux (Ubuntu/Debian)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### GPU Support (Opcional pero Recomendado)

Si tienes GPU NVIDIA, instala **nvidia-docker** para acelerar el entrenamiento:

#### Linux

```bash
# Agregar repositorio nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Instalar nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Windows

Docker Desktop con WSL2 ya incluye soporte para GPU automáticamente.

Verifica:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Inicio Rápido

### Opción 1: Con GPU (Recomendado)

```bash
# 1. Crear archivo .env (copiar desde .env.example)
cp .env.example .env

# 2. Iniciar servicios con GPU
docker-compose up -d

# 3. Ver logs
docker-compose logs -f
```

### Opción 2: Sin GPU (CPU Only - Desarrollo)

```bash
# 1. Usar docker-compose para desarrollo
docker-compose -f docker-compose.dev.yml up -d

# 2. Ver logs
docker-compose -f docker-compose.dev.yml logs -f
```

### Opción 3: Usando el Script Helper (Linux/Mac/WSL)

```bash
# Dar permisos de ejecución
chmod +x docker-manager.sh

# Iniciar servicios
./docker-manager.sh start

# Ver ayuda
./docker-manager.sh help
```

---

## 📦 Arquitectura de Contenedores

```
┌─────────────────────────────────────────────────┐
│                   Docker Host                    │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐│
│  │   Redis     │  │   Backend   │  │  Celery  ││
│  │   :6379     │◄─┤   :8000     │◄─┤  Worker  ││
│  │             │  │             │  │          ││
│  └─────────────┘  └─────────────┘  └──────────┘│
│         ▲                ▲               ▲      │
│         │                │               │      │
│         └────────────────┴───────────────┘      │
│              modelai-network (bridge)           │
│                                                  │
│  Volúmenes:                                     │
│  • redis_data/ (persistencia Redis)             │
│  • ./data/ (datasets)                           │
│  • ./models/ (modelos entrenados)               │
│  • ./output/ (predicciones)                     │
│  • ./logs/ (logs)                               │
│  • ./app.db (base de datos SQLite)             │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Comandos Útiles

### Gestión de Servicios

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f backend
docker-compose logs -f celery

# Detener servicios
docker-compose down

# Reiniciar servicios
docker-compose restart

# Ver estado de servicios
docker-compose ps
```

### Reconstruir Imágenes

```bash
# Reconstruir todas las imágenes
docker-compose build --no-cache

# Reconstruir e iniciar
docker-compose up -d --build
```

### Acceso a Contenedores

```bash
# Shell en backend
docker exec -it modelai-backend bash

# Shell en celery
docker exec -it modelai-celery bash

# Redis CLI
docker exec -it modelai-redis redis-cli
```

### Limpieza

```bash
# Detener y eliminar contenedores
docker-compose down

# Detener y eliminar contenedores + volúmenes
docker-compose down -v

# Eliminar TODO (contenedores, volúmenes, imágenes)
docker-compose down -v --rmi all
```

---

## 🔍 Verificación de Servicios

### Health Checks

```bash
# Redis
docker exec modelai-redis redis-cli ping
# Debe responder: PONG

# Backend API
curl http://localhost:8000/health
# Debe responder: {"status":"healthy"}

# Celery Worker
docker exec modelai-celery celery -A backend.tasks.celery_app inspect ping
# Debe responder con workers activos
```

### Ver Métricas GPU

```bash
# Dentro del contenedor
docker exec -it modelai-backend nvidia-smi

# Desde el host
nvidia-smi
```

---

## 🌐 Endpoints Disponibles

Una vez iniciado, puedes acceder a:

- **Backend API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📂 Volúmenes Persistentes

Los siguientes directorios se montan como volúmenes para persistencia:

| Host Path   | Container Path | Propósito                 |
| ----------- | -------------- | ------------------------- |
| `./data/`   | `/app/data/`   | Datasets de entrenamiento |
| `./models/` | `/app/models/` | Modelos entrenados (.pt)  |
| `./output/` | `/app/output/` | Predicciones/resultados   |
| `./logs/`   | `/app/logs/`   | Logs de aplicación        |
| `./app.db`  | `/app/app.db`  | Base de datos SQLite      |

---

## ⚙️ Variables de Entorno

Edita `.env` para configurar:

```env
# Redis
REDIS_URL=redis://redis:6379/0

# Base de datos
DATABASE_URL=sqlite:///./app.db

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Celery
CELERY_WORKER_CONCURRENCY=4
CELERY_WORKER_POOL=solo

# PyTorch
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Modelos
PATCH_SIZE=128
BATCH_SIZE=16
```

---

## 🐛 Troubleshooting

### Backend no inicia

```bash
# Ver logs detallados
docker-compose logs backend

# Verificar que Redis esté corriendo
docker ps | grep redis

# Reiniciar backend
docker-compose restart backend
```

### Celery no procesa tareas

```bash
# Ver logs de celery
docker-compose logs celery

# Verificar conexión a Redis
docker exec modelai-celery redis-cli -h redis ping

# Reiniciar celery
docker-compose restart celery
```

### Celery se cierra inesperadamente (exited with code 0)

Si ves `modelai-celery exited with code 0` durante el entrenamiento:

```bash
# 1. Ver logs completos para encontrar el error
docker-compose logs celery --tail 100

# 2. Verificar uso de memoria
docker stats modelai-celery

# 3. Posibles causas y soluciones:

# a) Error de memoria (OOM) - Reducir batch size
# Editar .env:
BATCH_SIZE=8  # o menor

# b) Error en el código - Ver logs de Python
docker-compose logs celery | grep -i "error\|exception\|traceback"

# c) Configurar restart automático en docker-compose.yml:
# restart: unless-stopped  (ya está configurado)

# 4. Reiniciar y monitorear
docker-compose up -d celery
docker-compose logs -f celery
```

### Error de GPU no disponible

```bash
# Verificar nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Si falla, usar docker-compose.dev.yml (CPU only)
docker-compose -f docker-compose.dev.yml up -d
```

### Puerto 8000 ya en uso

```bash
# Encontrar proceso usando el puerto
# Windows
netstat -ano | findstr :8000

# Linux
lsof -i :8000

# Cambiar puerto en docker-compose.yml:
ports:
  - "8001:8000"  # Host:Container
```

---

## 🔄 Actualización del Código

Cuando hagas cambios en el código:

```bash
# Opción 1: Reiniciar sin rebuild (usa volúmenes montados)
docker-compose restart

# Opción 2: Rebuild completo (si cambian dependencias)
docker-compose up -d --build

# Opción 3: Rebuild desde cero
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 🚀 Producción

### Recomendaciones

1. **No uses SQLite en producción**

   - Cambia a PostgreSQL:

   ```yaml
   postgres:
     image: postgres:15
     environment:
       POSTGRES_DB: modelai
       POSTGRES_USER: admin
       POSTGRES_PASSWORD: secure_password
   ```

2. **Usa NGINX como reverse proxy**

   ```yaml
   nginx:
     image: nginx:alpine
     ports:
       - '80:80'
       - '443:443'
     volumes:
       - ./nginx.conf:/etc/nginx/nginx.conf
   ```

3. **Habilita autenticación** (JWT)

4. **Configura límites de recursos**

   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4'
         memory: 8G
   ```

5. **Usa secrets para credenciales**
   ```yaml
   secrets:
     - db_password
     - redis_password
   ```

---

## 📚 Más Información

- [Documentación Docker](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

---

## ✅ Checklist de Despliegue

- [ ] Docker y Docker Compose instalados
- [ ] GPU drivers instalados (si aplica)
- [ ] nvidia-docker instalado (si aplica)
- [ ] Archivo `.env` configurado
- [ ] Directorios `data/`, `models/`, `output/` creados
- [ ] Servicios iniciados: `docker-compose up -d`
- [ ] Health checks pasando: Redis, Backend, Celery
- [ ] Swagger UI accesible: http://localhost:8000/docs
- [ ] Frontend conectando correctamente

---

**Última actualización**: 10 de Noviembre de 2025
