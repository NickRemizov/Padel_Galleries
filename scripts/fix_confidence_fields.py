#!/usr/bin/env python3
"""
Безопасное исправление старых записей photo_faces без правильных значений recognition_confidence
С dry-run режимом, backup и возможностью отката
"""
import asyncio
import asyncpg
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

try:
    from dotenv import load_dotenv
    # Try to load from python/.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        # Try loading from current directory .env
        load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using existing environment variables.")

async def create_backup(conn):
    """Создает backup таблицу photo_faces"""
    backup_name = f"photo_faces_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[BACKUP] Creating backup table: {backup_name}...")
    
    await conn.execute(f"""
        CREATE TABLE {backup_name} AS 
        SELECT * FROM photo_faces
    """)
    
    count = await conn.fetchval(f"SELECT COUNT(*) FROM {backup_name}")
    print(f"[BACKUP] ✓ Backed up {count} records to {backup_name}\n")
    return backup_name

async def show_changes(conn):
    """Показывает что будет изменено (DRY RUN)"""
    print("=" * 60)
    print("DRY RUN - ЧТО БУДЕТ ИЗМЕНЕНО")
    print("=" * 60)
    sys.stdout.flush()  # Принудительная очистка буфера
    
    # 1. Verified faces без recognition_confidence = 1.0
    print("\n[1] Verified faces without recognition_confidence = 1.0:")
    sys.stdout.flush()
    
    verified_faces = await conn.fetch("""
        SELECT id, photo_id, person_id, verified, 
               insightface_confidence, recognition_confidence
        FROM photo_faces
        WHERE verified = true 
          AND (recognition_confidence IS NULL OR recognition_confidence != 1.0)
        LIMIT 10
    """)
    
    count1 = await conn.fetchval("""
        SELECT COUNT(*) FROM photo_faces
        WHERE verified = true 
          AND (recognition_confidence IS NULL OR recognition_confidence != 1.0)
    """)
    
    print(f"   Найдено записей: {count1}")
    sys.stdout.flush()
    
    if verified_faces:
        print("   Примеры (первые 10):")
        sys.stdout.flush()
        for face in verified_faces:
            face_id = str(face['id'])[:8]
            current_conf = face['recognition_confidence']
            print(f"     - ID: {face_id}... | verified: {face['verified']} | current conf: {current_conf} -> будет: 1.0")
            sys.stdout.flush()
    else:
        print("   Нет записей для изменения")
        sys.stdout.flush()
    
    # 2. Faces с person_id но без recognition_confidence
    print("\n[2] Faces with person_id but no recognition_confidence:")
    sys.stdout.flush()
    
    auto_faces = await conn.fetch("""
        SELECT id, photo_id, person_id, verified,
               insightface_confidence, recognition_confidence
        FROM photo_faces
        WHERE person_id IS NOT NULL 
          AND verified = false
          AND recognition_confidence IS NULL
        LIMIT 10
    """)
    
    count2 = await conn.fetchval("""
        SELECT COUNT(*) 
        FROM photo_faces
        WHERE person_id IS NOT NULL 
          AND verified = false
          AND recognition_confidence IS NULL
    """)
    
    print(f"   Найдено записей: {count2}")
    sys.stdout.flush()
    
    if auto_faces:
        print("   Примеры (первые 10):")
        sys.stdout.flush()
        for face in auto_faces:
            face_id = str(face['id'])[:8]
            print(f"     - ID: {face_id}... | person_id: {str(face['person_id'])[:8]}... | будет conf: 0.85")
            sys.stdout.flush()
    else:
        print("   Нет записей для изменения")
        sys.stdout.flush()
    
    print("\n" + "=" * 60)
    sys.stdout.flush()
    
    return count1, count2

async def apply_fixes(conn):
    """Применяет исправления"""
    print("\n[APPLYING FIXES]")
    print("=" * 60)
    
    # 1. Исправить verified faces
    print("[1/2] Fixing verified faces...")
    result1 = await conn.execute("""
        UPDATE photo_faces
        SET recognition_confidence = 1.0
        WHERE verified = true 
          AND (recognition_confidence IS NULL OR recognition_confidence != 1.0)
    """)
    count1 = int(result1.split()[-1])
    print(f"   ✓ Fixed {count1} verified faces\n")
    
    # 2. Исправить auto-recognized faces
    print("[2/2] Fixing auto-recognized faces...")
    result2 = await conn.execute("""
        UPDATE photo_faces
        SET recognition_confidence = 0.85
        WHERE person_id IS NOT NULL 
          AND verified = false
          AND recognition_confidence IS NULL
    """)
    count2 = int(result2.split()[-1])
    print(f"   ✓ Fixed {count2} auto-recognized faces\n")
    
    return count1, count2

async def verify_fixes(conn):
    """Проверяет что все исправлено"""
    print("[VERIFICATION]")
    print("=" * 60)
    
    wrong_verified = await conn.fetchval("""
        SELECT COUNT(*) FROM photo_faces
        WHERE verified = true 
          AND (recognition_confidence IS NULL OR recognition_confidence != 1.0)
    """)
    
    missing_conf = await conn.fetchval("""
        SELECT COUNT(*) FROM photo_faces
        WHERE person_id IS NOT NULL 
          AND recognition_confidence IS NULL
    """)
    
    print(f"Verified faces without conf=1.0: {wrong_verified}")
    print(f"Faces with person_id without conf: {missing_conf}\n")
    
    if wrong_verified == 0 and missing_conf == 0:
        print("✅ Все проблемы исправлены!")
        return True
    else:
        print(f"⚠️  Осталось проблем: {wrong_verified + missing_conf}")
        return False

async def main():
    print("=" * 60)
    print("SAFE FIX CONFIDENCE FIELDS")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    # Подключение к БД
    db_url = os.getenv('PV_POSTGRES_URL') or os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        print("ERROR: PV_POSTGRES_URL or POSTGRES_URL not set")
        return

    conn = await asyncpg.connect(db_url, statement_cache_size=0)
    
    try:
        # ШАГ 1: DRY RUN - показать что будет изменено
        count1, count2 = await show_changes(conn)
        
        if count1 == 0 and count2 == 0:
            print("\n✅ Нет записей для исправления!")
            return
        
        # ШАГ 2: Спросить подтверждение
        print("\nПродолжить исправление? (yes/no): ", end='')
        sys.stdout.flush()
        
        # Читаем из stdin
        response = sys.stdin.readline().strip().lower()
        
        if response not in ['yes', 'y']:
            print("❌ Отменено пользователем")
            return
        
        # ШАГ 3: Создать backup
        backup_name = await create_backup(conn)
        
        # ШАГ 4: Применить исправления
        fixed1, fixed2 = await apply_fixes(conn)
        
        # ШАГ 5: Проверить результат
        success = await verify_fixes(conn)
        
        # ШАГ 6: Показать итог
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Backup table: {backup_name}")
        print(f"Total fixed: {fixed1 + fixed2}")
        print(f"  - Verified faces: {fixed1}")
        print(f"  - Auto-recognized: {fixed2}")
        
        if success:
            print("\n✅ Исправление завершено успешно!")
        else:
            print("\n⚠️  Есть оставшиеся проблемы, проверьте данные")
        
        print("\nДля отката изменений выполните:")
        print(f"  DROP TABLE photo_faces;")
        print(f"  ALTER TABLE {backup_name} RENAME TO photo_faces;")
        print("=" * 60)
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
