#!/usr/bin/env python3
"""
Скрипт для проверки данных в таблице people.
Запуск на сервере: cd /home/nickr/python && source venv/bin/activate && python check_people_data.py
"""
import os
import sys
sys.path.insert(0, '/home/nickr/python')

import asyncio
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def check_people():
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("ERROR: DATABASE_URL not found in .env")
        return
    
    print(f"Connecting...")
    
    # Подключение к БД
    conn = await asyncpg.connect(database_url)
    
    rows = await conn.fetch("""
        SELECT 
            id, 
            real_name, 
            tournament_results,
            pg_typeof(tournament_results) as db_type
        FROM people 
        LIMIT 3
    """)
    
    print("=== Данные из таблицы people ===\n")
    
    for row in rows:
        print(f"ID: {row['id']}")
        print(f"Name: {row['real_name']}")
        print(f"tournament_results: {repr(row['tournament_results'])}")
        print(f"Python type: {type(row['tournament_results'])}")
        print(f"DB type: {row['db_type']}")
        print("-" * 40)
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_people())
