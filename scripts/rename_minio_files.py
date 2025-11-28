"""
Скрипт для переименования файлов в MinIO
Убирает URL-encoding из имён файлов

Пример:
  Queen%20of%20the%20Court%2020-09-25-201.jpg
  ->
  Queen of the Court 20-09-25-201.jpg
"""

import os
import sys
from urllib.parse import unquote
from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource

# Конфигурация MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9200")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "2o5CBBoM/ynAEcrcxViXmvcqDs4UAFXM")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET_NAME = "galleries"

# Режим работы
DRY_RUN = "--dry-run" in sys.argv or "-n" in sys.argv


def main():
    print(f"=== MinIO File Rename Script ===")
    print(f"Endpoint: {MINIO_ENDPOINT}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Dry run: {DRY_RUN}")
    print()
    
    # Подключение к MinIO
    try:
        endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
        secure = MINIO_ENDPOINT.startswith("https://")
        
        client = Minio(
            endpoint=endpoint,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=secure
        )
        print("[OK] Connected to MinIO")
    except Exception as e:
        print(f"[ERROR] Failed to connect to MinIO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Проверка bucket
    if not client.bucket_exists(BUCKET_NAME):
        print(f"[ERROR] Bucket '{BUCKET_NAME}' does not exist")
        sys.exit(1)
    
    print(f"[OK] Bucket '{BUCKET_NAME}' exists")
    print()
    
    # Получение списка файлов
    print("Scanning files...")
    objects = list(client.list_objects(BUCKET_NAME, recursive=True))
    print(f"Found {len(objects)} files")
    print()
    
    # Поиск файлов с URL-encoded именами
    files_to_rename = []
    for obj in objects:
        old_name = obj.object_name
        new_name = unquote(old_name)  # Декодируем URL-encoding
        
        if old_name != new_name:
            files_to_rename.append((old_name, new_name, obj.size))
    
    print(f"Files to rename: {len(files_to_rename)}")
    print()
    
    if not files_to_rename:
        print("No files need renaming. Done!")
        return
    
    # Показываем первые 10 примеров
    print("Examples (first 10):")
    for old_name, new_name, size in files_to_rename[:10]:
        print(f"  {old_name[:60]}...")
        print(f"    -> {new_name[:60]}...")
        print()
    
    if DRY_RUN:
        print("=== DRY RUN MODE - No changes made ===")
        print(f"Would rename {len(files_to_rename)} files")
        print("Run without --dry-run to apply changes")
        return
    
    # Подтверждение
    print(f"About to rename {len(files_to_rename)} files.")
    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return
    
    # Переименование файлов
    print()
    print("Renaming files...")
    success_count = 0
    error_count = 0
    
    for i, (old_name, new_name, size) in enumerate(files_to_rename):
        try:
            # MinIO не поддерживает rename напрямую
            # Нужно: copy -> delete
            
            # 1. Копируем файл с новым именем
            client.copy_object(
                BUCKET_NAME,
                new_name,
                CopySource(BUCKET_NAME, old_name)
            )
            
            # 2. Удаляем старый файл
            client.remove_object(BUCKET_NAME, old_name)
            
            success_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(files_to_rename)}")
                
        except S3Error as e:
            print(f"  [ERROR] {old_name}: {e}")
            error_count += 1
        except Exception as e:
            print(f"  [ERROR] {old_name}: {e}")
            error_count += 1
    
    print()
    print(f"=== Done ===")
    print(f"Renamed: {success_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
