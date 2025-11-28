"""
Тестовый скрипт для проверки CRUD API
Запускать после перезапуска Python сервера
"""

import os

# Получаем URL сервера из переменных окружения или используем значение по умолчанию
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")

print(f"=" * 60)
print(f"ТЕСТИРОВАНИЕ CRUD API")
print(f"Сервер: {FASTAPI_URL}")
print(f"=" * 60)

import urllib.request
import urllib.error
import json

def test_endpoint(method: str, endpoint: str, data: dict = None):
    """Тестирует один эндпоинт"""
    url = f"{FASTAPI_URL}{endpoint}"
    print(f"\n{method} {endpoint}")
    print("-" * 40)
    
    try:
        if data:
            json_data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=json_data, method=method)
            req.add_header('Content-Type', 'application/json')
        else:
            req = urllib.request.Request(url, method=method)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"✅ УСПЕХ (статус: {response.status})")
            # Показываем первые 3 элемента если это список
            if isinstance(result, list):
                print(f"   Получено записей: {len(result)}")
                for item in result[:3]:
                    if isinstance(item, dict):
                        name = item.get('name') or item.get('title') or item.get('id')
                        print(f"   - {name}")
                if len(result) > 3:
                    print(f"   ... и ещё {len(result) - 3}")
            elif isinstance(result, dict):
                for key, value in list(result.items())[:5]:
                    print(f"   {key}: {value}")
            return True, result
            
    except urllib.error.HTTPError as e:
        print(f"❌ ОШИБКА HTTP {e.code}: {e.reason}")
        try:
            error_body = e.read().decode('utf-8')
            print(f"   {error_body[:200]}")
        except:
            pass
        return False, None
        
    except urllib.error.URLError as e:
        print(f"❌ ОШИБКА СОЕДИНЕНИЯ: {e.reason}")
        print(f"   Проверьте, что сервер запущен на {FASTAPI_URL}")
        return False, None
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return False, None


def run_tests():
    """Запускает все тесты"""
    results = []
    
    # Тест 1: Проверка здоровья сервера
    print("\n" + "=" * 60)
    print("1. ПРОВЕРКА СЕРВЕРА")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/health")
    results.append(("Здоровье сервера", success))
    
    if not success:
        print("\n⚠️  СЕРВЕР НЕ ОТВЕЧАЕТ!")
        print("Инструкции:")
        print("1. Подключитесь к серверу по SSH")
        print("2. Перезапустите Python сервер")
        print("3. Запустите этот скрипт снова")
        return
    
    # Тест 2: Galleries
    print("\n" + "=" * 60)
    print("2. ТЕСТИРОВАНИЕ GALLERIES (Галереи)")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/api/crud/galleries")
    results.append(("GET /api/crud/galleries", success))
    
    # Тест 3: Photographers
    print("\n" + "=" * 60)
    print("3. ТЕСТИРОВАНИЕ PHOTOGRAPHERS (Фотографы)")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/api/crud/photographers")
    results.append(("GET /api/crud/photographers", success))
    
    # Тест 4: Locations
    print("\n" + "=" * 60)
    print("4. ТЕСТИРОВАНИЕ LOCATIONS (Локации)")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/api/crud/locations")
    results.append(("GET /api/crud/locations", success))
    
    # Тест 5: Organizers
    print("\n" + "=" * 60)
    print("5. ТЕСТИРОВАНИЕ ORGANIZERS (Организаторы)")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/api/crud/organizers")
    results.append(("GET /api/crud/organizers", success))
    
    # Тест 6: People
    print("\n" + "=" * 60)
    print("6. ТЕСТИРОВАНИЕ PEOPLE (Персоны)")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/api/crud/people")
    results.append(("GET /api/crud/people", success))
    
    # Тест 7: Stats
    print("\n" + "=" * 60)
    print("7. ТЕСТИРОВАНИЕ STATS (Статистика)")
    print("=" * 60)
    success, _ = test_endpoint("GET", "/api/crud/stats/recognition")
    results.append(("GET /api/crud/stats/recognition", success))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print("-" * 40)
    print(f"Успешно: {passed}/{len(results)}")
    print(f"Провалено: {failed}/{len(results)}")
    
    if failed == 0:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("Можно переходить к Этапу 2")
    else:
        print("\n⚠️  ЕСТЬ ОШИБКИ!")
        print("Скопируй этот вывод и отправь мне")


if __name__ == "__main__":
    run_tests()
