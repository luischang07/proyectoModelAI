"""
Script para reinicializar la base de datos
"""
from pathlib import Path
from backend.database import Base, engine
from backend.models import db_models  # Import para cargar todos los modelos

def reset_database():
    """Elimina y recrea todas las tablas"""
    db_path = Path("app.db")
    
    print("🗑️  Eliminando base de datos antigua...")
    if db_path.exists():
        db_path.unlink()
        print("✅ Base de datos eliminada")
    
    print("🔨 Creando nuevas tablas...")
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos reinicializada con éxito")
    
    print("\n📊 Tablas creadas:")
    for table in Base.metadata.tables.keys():
        print(f"  • {table}")

if __name__ == "__main__":
    reset_database()
