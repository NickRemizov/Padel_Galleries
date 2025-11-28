#!/usr/bin/env python3
"""
System Status Verification Script

Проверяет текущее состояние системы после миграции:
- Наличие PlayerDatabase (должна отсутствовать)
- Конфигурация verified_threshold
- Статус Python API
- Статус PostgreSQL

Usage:
    python scripts/check_system_status.py
"""

import os
import sys
import requests
import json
from pathlib import Path

# Colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Configuration
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://23.88.61.20:8001")

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}")
    print(f"{text}")
    print(f"{'='*80}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")


def check_player_database():
    """Check if PlayerDatabase (SQLite) still exists"""
    print_header("1. Проверка PlayerDatabase (SQLite)")
    
    # Check if database.py exists
    db_file = Path("python/services/database.py")
    if db_file.exists():
        print_error("Файл python/services/database.py ВСЁ ЕЩЁ СУЩЕСТВУЕТ")
        print_info("Нужно: Удалить этот файл")
        return False
    else:
        print_success("Файл python/services/database.py удалён")
    
    # Check if players.db exists
    players_db = Path("players.db")
    if players_db.exists():
        print_warning(f"SQLite база players.db найдена (размер: {players_db.stat().st_size} bytes)")
        print_info("Можно удалить: rm players.db")
    else:
        print_success("SQLite база players.db отсутствует")
    
    return True


def check_config_api():
    """Check config API and verified_threshold"""
    print_header("2. Проверка конфигурации (verified_threshold)")
    
    try:
        response = requests.get(f"{FASTAPI_URL}/api/v2/config", timeout=10)
        
        if response.status_code != 200:
            print_error(f"API недоступен: {response.status_code}")
            print_info(f"Response: {response.text}")
            return False
        
        config = response.json()
        print_info(f"Config response: {json.dumps(config, indent=2)}")
        
        quality_filters = config.get("quality_filters", {})
        
        # Check verified_threshold
        if "verified_threshold" in quality_filters:
            threshold = quality_filters["verified_threshold"]
            print_success(f"verified_threshold настроен: {threshold * 100:.0f}%")
            
            if threshold == 0.99:
                print_info("Значение по умолчанию: 99% (строгий режим)")
            elif threshold == 0.6:
                print_info("Значение по умолчанию: 60% (мягкий режим)")
            else:
                print_info(f"Пользовательское значение: {threshold * 100:.0f}%")
        else:
            print_warning("verified_threshold отсутствует в конфигурации (может быть в defaults)")
            print_info(f"Доступные поля quality_filters: {list(quality_filters.keys())}")
        
        # Show other settings
        print_info(f"min_detection_score: {quality_filters.get('min_detection_score', 'N/A')}")
        print_info(f"min_blur_score: {quality_filters.get('min_blur_score', 'N/A')}")
        print_info(f"min_face_size: {quality_filters.get('min_face_size', 'N/A')}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print_error(f"Не могу подключиться к FastAPI: {FASTAPI_URL}")
        print_info("Проверьте что Python backend запущен")
        return False
    except Exception as e:
        print_error(f"Ошибка: {str(e)}")
        import traceback
        print_info(f"Traceback: {traceback.format_exc()}")
        return False


def check_postgres_connection():
    """Check PostgreSQL connection via API"""
    print_header("3. Проверка подключения к PostgreSQL")
    
    try:
        response = requests.get(f"{FASTAPI_URL}/api/v2/statistics", timeout=10)
        
        if response.status_code != 200:
            print_error(f"API недоступен: {response.status_code}")
            return False
        
        stats = response.json()
        
        print_success("PostgreSQL подключен и работает")
        print_info(f"Людей в базе: {stats.get('people_count', 0)}")
        print_info(f"Всего лиц: {stats.get('total_faces', 0)}")
        print_info(f"Галерей: {stats.get('gallery_count', 0)}")
        print_info(f"Фотографий: {stats.get('photo_count', 0)}")
        
        return True
        
    except Exception as e:
        print_error(f"Ошибка: {str(e)}")
        return False


def check_people_api():
    """Check People API endpoints"""
    print_header("4. Проверка People API")
    
    try:
        # Test GET /api/crud/people
        response = requests.get(f"{FASTAPI_URL}/api/crud/people", timeout=10)
        
        if response.status_code == 200:
            people = response.json()
            print_success(f"People API работает (найдено {len(people)} людей)")
            
            # Show first 3 people
            for person in people[:3]:
                print_info(f"  - {person.get('name', 'Unknown')} (ID: {person.get('id', 'N/A')[:8]}...)")
            
            return True
        else:
            print_error(f"People API недоступен: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Ошибка: {str(e)}")
        return False


def check_faces_api():
    """Check Faces API endpoints"""
    print_header("5. Проверка Faces API")
    
    try:
        response = requests.post(
            f"{FASTAPI_URL}/api/faces/get-photo-faces",
            json={"photo_id": "test"},  # Will fail but shows endpoint exists
            timeout=10
        )
        
        # Any response except 404 means endpoint exists
        if response.status_code == 404:
            print_error("Faces API endpoint не найден (404)")
            return False
        else:
            print_success(f"Faces API доступен (status: {response.status_code})")
            return True
            
    except Exception as e:
        print_error(f"Ошибка: {str(e)}")
        return False


def main():
    """Run all checks"""
    print(f"\n{Colors.BOLD}Проверка состояния системы Galeries{Colors.ENDC}")
    print(f"FastAPI URL: {FASTAPI_URL}\n")
    
    results = []
    
    # Run checks
    results.append(("PlayerDatabase удалён", check_player_database()))
    results.append(("Конфигурация работает", check_config_api()))
    results.append(("PostgreSQL подключен", check_postgres_connection()))
    results.append(("People API работает", check_people_api()))
    results.append(("Faces API работает", check_faces_api()))
    
    # Summary
    print_header("ИТОГО")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print(f"\n{Colors.BOLD}Пройдено: {passed}/{total}{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}Система работает корректно!{Colors.ENDC}")
        return 0
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}Обнаружены проблемы, требуется внимание{Colors.ENDC}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
