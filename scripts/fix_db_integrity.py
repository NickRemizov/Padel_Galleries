#!/usr/bin/env python3
"""
Database Integrity Fixer - Standalone version
Fixes data consistency issues found by check_db_integrity.py
"""
import asyncio
import asyncpg
import json
import os
from datetime import datetime

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

async def fix_integrity():
    # Load report
    try:
        with open('/tmp/db_integrity_report.json', 'r') as f:
            report = json.load(f)
    except FileNotFoundError:
        print("Error: No integrity report found. Run check_db_integrity.py first.")
        return
    
    print("=" * 60)
    print("DATABASE INTEGRITY FIXER")
    print("=" * 60)
    print(f"Report from: {report['timestamp']}")
    print(f"Total issues: {report['total_issues']}")
    print()
    
    # Show issues
    issues = report['issues']
    print("Issues found:")
    print(f"[1] Verified faces without 100% confidence: {issues['verified_without_100_confidence']['count']}")
    print(f"[2] Faces with person_id but no embedding: {issues['person_without_embedding']['count']}")
    print(f"[3] Orphaned records (broken person_id): {issues['orphaned_person_id']['count']}")
    print(f"[4] Duplicate embeddings: {issues['duplicate_embeddings']['count']}")
    print(f"[5] Verified faces without person_id: {issues['verified_without_person']['count']}")
    print(f"[6] Malformed embeddings: {issues['malformed_embeddings']['count']}")
    print()
    
    # Ask which to fix
    print("Which issues do you want to fix?")
    print("Enter numbers separated by commas (e.g., 1,3,5) or 'all' to fix everything:")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        to_fix = [1, 2, 3, 4, 5, 6]
    else:
        try:
            to_fix = [int(x.strip()) for x in choice.split(',')]
        except ValueError:
            print("Invalid input. Exiting.")
            return
    
    # Get database URL from environment
    db_url = os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        print("Error: POSTGRES_URL or DATABASE_URL environment variable not set")
        return
    
    # Connect to database
    conn = await asyncpg.connect(db_url)
    
    try:
        fixed = {}
        
        if 1 in to_fix:
            print("\n[1] Fixing verified faces without 100% confidence...")
            result = await conn.execute("""
                UPDATE photo_faces 
                SET recognition_confidence = 1.0 
                WHERE verified = true 
                  AND (recognition_confidence IS NULL OR recognition_confidence < 1.0)
            """)
            count = int(result.split()[-1])
            fixed['verified_without_100_confidence'] = count
            print(f"   Fixed: {count} records")
        
        if 2 in to_fix:
            print("\n[2] Fixing faces with person_id but no embedding...")
            result = await conn.execute("""
                UPDATE photo_faces 
                SET person_id = NULL, verified = false 
                WHERE person_id IS NOT NULL
                  AND insightface_descriptor IS NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM face_descriptors fd 
                    WHERE fd.person_id = photo_faces.person_id
                  )
            """)
            count = int(result.split()[-1])
            fixed['person_without_embedding'] = count
            print(f"   Fixed: {count} records")
        
        if 3 in to_fix:
            print("\n[3] Fixing orphaned records...")
            result = await conn.execute("""
                UPDATE photo_faces 
                SET person_id = NULL, verified = false 
                WHERE person_id IS NOT NULL 
                  AND NOT EXISTS (SELECT 1 FROM people p WHERE p.id = person_id)
            """)
            count = int(result.split()[-1])
            fixed['orphaned_person_id'] = count
            print(f"   Fixed: {count} records")
        
        if 4 in to_fix:
            print("\n[4] Fixing duplicate embeddings...")
            result = await conn.execute("""
                DELETE FROM face_descriptors
                WHERE id NOT IN (
                  SELECT DISTINCT ON (person_id, source_image_id) id
                  FROM face_descriptors
                  ORDER BY person_id, source_image_id, created_at DESC
                )
            """)
            count = int(result.split()[-1])
            fixed['duplicate_embeddings'] = count
            print(f"   Fixed: {count} records")
        
        if 5 in to_fix:
            print("\n[5] Fixing verified faces without person_id...")
            result = await conn.execute("""
                UPDATE photo_faces 
                SET verified = false 
                WHERE verified = true 
                  AND person_id IS NULL
            """)
            count = int(result.split()[-1])
            fixed['verified_without_person'] = count
            print(f"   Fixed: {count} records")
        
        if 6 in to_fix:
            print("\n[6] Fixing malformed embeddings...")
            result = await conn.execute("""
                UPDATE photo_faces 
                SET insightface_descriptor = NULL 
                WHERE insightface_descriptor IS NOT NULL 
                  AND (
                    insightface_descriptor::text NOT LIKE '[%]'
                    OR insightface_descriptor::text LIKE '%null%'
                    OR insightface_descriptor::text LIKE '%nan%'
                  )
            """)
            count = int(result.split()[-1])
            fixed['malformed_embeddings'] = count
            print(f"   Fixed: {count} records")
        
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        total_fixed = sum(fixed.values())
        for key, value in fixed.items():
            print(f"{key}: {value} records fixed")
        print(f"\nTotal records fixed: {total_fixed}")
        print("=" * 60)
        
    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(fix_integrity())
