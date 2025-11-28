#!/usr/bin/env python3
"""
Database Integrity Checker - Standalone version
Checks PostgreSQL database for data consistency issues without requiring project dependencies
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

async def check_integrity():
    # Get database URL from environment
    db_url = os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        print("Error: POSTGRES_URL or DATABASE_URL environment variable not set")
        return
    
    print("=" * 60)
    print("DATABASE INTEGRITY CHECK")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Connect to database
    conn = await asyncpg.connect(db_url)
    
    try:
        issues = {}
        total_issues = 0
        
        # Check 1: Verified faces without 100% confidence
        print("[1/6] Checking verified faces without 100% confidence...")
        rows = await conn.fetch("""
            SELECT id, person_id, recognition_confidence, verified 
            FROM photo_faces 
            WHERE verified = true 
              AND (recognition_confidence IS NULL OR recognition_confidence < 1.0)
        """)
        count = len(rows)
        issues['verified_without_100_confidence'] = {
            'count': count,
            'records': [dict(r) for r in rows[:10]]  # First 10 examples
        }
        total_issues += count
        print(f"   Found: {count}")
        
        # Check 2: Faces with person_id but no embedding
        print("[2/6] Checking faces with person_id but no embedding...")
        rows = await conn.fetch("""
            SELECT pf.id, pf.person_id, pf.photo_id
            FROM photo_faces pf
            WHERE pf.person_id IS NOT NULL
              AND pf.insightface_descriptor IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM face_descriptors fd 
                WHERE fd.person_id = pf.person_id
              )
        """)
        count = len(rows)
        issues['person_without_embedding'] = {
            'count': count,
            'records': [dict(r) for r in rows[:10]]
        }
        total_issues += count
        print(f"   Found: {count}")
        
        # Check 3: Orphaned records (broken person_id)
        print("[3/6] Checking orphaned records...")
        rows = await conn.fetch("""
            SELECT pf.id, pf.person_id, pf.photo_id
            FROM photo_faces pf
            WHERE pf.person_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM people p WHERE p.id = pf.person_id
              )
        """)
        count = len(rows)
        issues['orphaned_person_id'] = {
            'count': count,
            'records': [dict(r) for r in rows[:10]]
        }
        total_issues += count
        print(f"   Found: {count}")
        
        # Check 4: Duplicate embeddings (same person, same photo)
        print("[4/6] Checking duplicate embeddings...")
        rows = await conn.fetch("""
            SELECT person_id, source_image_id, COUNT(*) as count
            FROM face_descriptors
            GROUP BY person_id, source_image_id
            HAVING COUNT(*) > 1
        """)
        count = len(rows)
        issues['duplicate_embeddings'] = {
            'count': count,
            'records': [dict(r) for r in rows[:10]]
        }
        total_issues += count
        print(f"   Found: {count}")
        
        # Check 5: Verified faces without person_id
        print("[5/6] Checking verified faces without person_id...")
        rows = await conn.fetch("""
            SELECT id, person_id, verified 
            FROM photo_faces 
            WHERE verified = true 
              AND person_id IS NULL
        """)
        count = len(rows)
        issues['verified_without_person'] = {
            'count': count,
            'records': [dict(r) for r in rows[:10]]
        }
        total_issues += count
        print(f"   Found: {count}")
        
        # Check 6: Malformed embeddings
        print("[6/6] Checking malformed embeddings...")
        rows = await conn.fetch("""
            SELECT id, person_id 
            FROM photo_faces 
            WHERE insightface_descriptor IS NOT NULL 
              AND (
                insightface_descriptor::text NOT LIKE '[%]'
                OR insightface_descriptor::text LIKE '%null%'
                OR insightface_descriptor::text LIKE '%nan%'
              )
        """)
        count = len(rows)
        issues['malformed_embeddings'] = {
            'count': count,
            'records': [dict(r) for r in rows[:10]]
        }
        total_issues += count
        print(f"   Found: {count}")
        
        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for key, value in issues.items():
            print(f"{key}: {value['count']} issues")
        print(f"\nTotal issues found: {total_issues}")
        print("=" * 60)
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_issues': total_issues,
            'issues': issues
        }
        
        with open('/tmp/db_integrity_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: /tmp/db_integrity_report.json")
        
    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(check_integrity())
