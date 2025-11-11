#!/bin/bash

# Script para gestionar la infraestructura Docker del proyecto ModelAI

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="modelai"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    cat << EOF
🚀 ModelAI Docker Manager

Uso: ./docker-manager.sh [comando]

Comandos disponibles:
    start       Iniciar todos los servicios (Redis, Backend, Celery)
    stop        Detener todos los servicios
    restart     Reiniciar todos los servicios
    logs        Ver logs de todos los servicios
    logs-back   Ver logs solo del backend
    logs-celery Ver logs solo de Celery
    build       Re-construir las imágenes Docker
    clean       Detener y limpiar contenedores, volúmenes e imágenes
    status      Ver estado de los servicios
    shell-back  Abrir shell en el contenedor del backend
    shell-celery Abrir shell en el contenedor de Celery
    redis-cli   Conectar a Redis CLI
    health      Verificar health de los servicios

Ejemplos:
    ./docker-manager.sh start
    ./docker-manager.sh logs-celery
    ./docker-manager.sh clean

EOF
}

# Función para verificar Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker no está instalado${NC}"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker daemon no está corriendo${NC}"
        exit 1
    fi
}

# Función para verificar nvidia-docker (GPU)
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ GPU detectada${NC}"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}⚠️  No se detectó GPU NVIDIA (continuando sin GPU)${NC}"
    fi
}

# Comandos
case "$1" in
    start)
        check_docker
        echo -e "${GREEN}🚀 Iniciando servicios...${NC}"
        check_gpu
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
        echo -e "${GREEN}✅ Servicios iniciados${NC}"
        echo -e "${YELLOW}📊 Ver logs: ./docker-manager.sh logs${NC}"
        ;;
    
    stop)
        check_docker
        echo -e "${YELLOW}🛑 Deteniendo servicios...${NC}"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down
        echo -e "${GREEN}✅ Servicios detenidos${NC}"
        ;;
    
    restart)
        check_docker
        echo -e "${YELLOW}🔄 Reiniciando servicios...${NC}"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME restart
        echo -e "${GREEN}✅ Servicios reiniciados${NC}"
        ;;
    
    logs)
        check_docker
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f
        ;;
    
    logs-back|logs-backend)
        check_docker
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f backend
        ;;
    
    logs-celery)
        check_docker
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f celery
        ;;
    
    build)
        check_docker
        echo -e "${YELLOW}🔨 Construyendo imágenes...${NC}"
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build --no-cache
        echo -e "${GREEN}✅ Imágenes construidas${NC}"
        ;;
    
    clean)
        check_docker
        echo -e "${RED}🧹 Limpiando contenedores, volúmenes e imágenes...${NC}"
        read -p "¿Estás seguro? Esto eliminará TODOS los datos (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v --rmi all
            echo -e "${GREEN}✅ Limpieza completada${NC}"
        else
            echo -e "${YELLOW}❌ Limpieza cancelada${NC}"
        fi
        ;;
    
    status)
        check_docker
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
        ;;
    
    shell-back|shell-backend)
        check_docker
        docker exec -it ${PROJECT_NAME}-backend bash
        ;;
    
    shell-celery)
        check_docker
        docker exec -it ${PROJECT_NAME}-celery bash
        ;;
    
    redis-cli)
        check_docker
        docker exec -it ${PROJECT_NAME}-redis redis-cli
        ;;
    
    health)
        check_docker
        echo -e "${GREEN}🏥 Verificando health de servicios...${NC}"
        echo ""
        echo "Redis:"
        docker exec ${PROJECT_NAME}-redis redis-cli ping || echo -e "${RED}❌ Redis no responde${NC}"
        echo ""
        echo "Backend:"
        curl -s http://localhost:8000/health || echo -e "${RED}❌ Backend no responde${NC}"
        echo ""
        echo "Celery:"
        docker exec ${PROJECT_NAME}-celery celery -A backend.tasks.celery_app inspect ping || echo -e "${RED}❌ Celery no responde${NC}"
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    *)
        echo -e "${RED}❌ Comando desconocido: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
