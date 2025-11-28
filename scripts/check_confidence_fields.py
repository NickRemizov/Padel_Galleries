#!/usr/bin/env python3
"""
Confidence Fields Checker
Checks if insightface_confidence and recognition_confidence fields are properly populated
"""
import asyncio
import asyncpg
import os
from datetime import datetime

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using existing environment variables.")

async def check_confidence_fields():
    db_url = os.getenv('PV_POSTGRES_URL') or os.getenv('POSTGRES_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        print("Error: PV_POSTGRES_URL or POSTGRES_URL environment variable not set")
        return
    
    print("=" * 60)
    print("CONFIDENCE FIELDS CHECK")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    conn = await asyncpg.connect(db_url)
    
    try:
        # Check 1: Schema check
        print("[1/6] Checking if confidence fields exist in schema...")
        schema = await conn.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'photo_faces' 
              AND column_name IN ('insightface_confidence', 'recognition_confidence')
            ORDER BY column_name
        """)
        
        if len(schema) != 2:
            print("   ERROR: Missing confidence fields!")
            for row in schema:
                print(f"   Found: {row['column_name']} ({row['data_type']})")
            return
        else:
            print("   OK: Both fields exist")
            for row in schema:
                print(f"   - {row['column_name']}: {row['data_type']}")
        
        # Check 2: Verified faces with recognition_confidence = 1.0
        print("\n[2/6] Checking verified faces have recognition_confidence = 1.0...")
        verified_correct = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE verified = true 
              AND recognition_confidence = 1.0
        """)
        
        verified_wrong = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE verified = true 
              AND (recognition_confidence IS NULL OR recognition_confidence != 1.0)
        """)
        
        print(f"   Correct: {verified_correct}")
        print(f"   Wrong: {verified_wrong}")
        
        if verified_wrong > 0:
            examples = await conn.fetch("""
                SELECT id, person_id, recognition_confidence, verified, created_at
                FROM photo_faces 
                WHERE verified = true 
                  AND (recognition_confidence IS NULL OR recognition_confidence != 1.0)
                LIMIT 5
            """)
            print("   Examples of wrong records:")
            for row in examples:
                print(f"     - ID {row['id']}: recognition_confidence = {row['recognition_confidence']}")
        
        # Check 3: Faces with person_id but no recognition_confidence
        print("\n[3/6] Checking faces with person_id but no recognition_confidence...")
        count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE person_id IS NOT NULL 
              AND recognition_confidence IS NULL
        """)
        print(f"   Found: {count}")
        
        if count > 0:
            examples = await conn.fetch("""
                SELECT id, person_id, verified, created_at
                FROM photo_faces 
                WHERE person_id IS NOT NULL 
                  AND recognition_confidence IS NULL
                LIMIT 5
            """)
            print("   Examples:")
            for row in examples:
                print(f"     - ID {row['id']}: verified = {row['verified']}, created = {row['created_at']}")
        
        # Check 4: Auto-recognized faces (verified=false) with confidence
        print("\n[4/6] Checking auto-recognized faces...")
        auto_with_conf = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE verified = false 
              AND person_id IS NOT NULL
              AND recognition_confidence IS NOT NULL
        """)
        
        auto_without_conf = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE verified = false 
              AND person_id IS NOT NULL
              AND recognition_confidence IS NULL
        """)
        
        print(f"   With recognition_confidence: {auto_with_conf}")
        print(f"   Without recognition_confidence: {auto_without_conf}")
        
        # Check 5: Statistics
        print("\n[5/6] Confidence statistics...")
        stats = await conn.fetchrow("""
            SELECT 
              COUNT(*) as total_faces,
              COUNT(insightface_confidence) as has_detection_conf,
              COUNT(recognition_confidence) as has_recognition_conf,
              AVG(insightface_confidence) as avg_detection,
              AVG(recognition_confidence) as avg_recognition,
              MIN(insightface_confidence) as min_detection,
              MAX(insightface_confidence) as max_detection,
              MIN(recognition_confidence) as min_recognition,
              MAX(recognition_confidence) as max_recognition
            FROM photo_faces
            WHERE person_id IS NOT NULL
        """)
        
        print(f"   Total faces: {stats['total_faces']}")
        print(f"   Has insightface_confidence: {stats['has_detection_conf']}")
        print(f"   Has recognition_confidence: {stats['has_recognition_conf']}")
        if stats['avg_detection']:
            print(f"   Avg detection confidence: {stats['avg_detection']:.3f} (range: {stats['min_detection']:.3f} - {stats['max_detection']:.3f})")
        if stats['avg_recognition']:
            print(f"   Avg recognition confidence: {stats['avg_recognition']:.3f} (range: {stats['min_recognition']:.3f} - {stats['max_recognition']:.3f})")
        
        # Check 6: Invalid confidence values (outside 0-1 range)
        print("\n[6/6] Checking for invalid confidence values...")
        invalid_detection = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE insightface_confidence IS NOT NULL 
              AND (insightface_confidence < 0 OR insightface_confidence > 1)
        """)
        
        invalid_recognition = await conn.fetchval("""
            SELECT COUNT(*)
            FROM photo_faces 
            WHERE recognition_confidence IS NOT NULL 
              AND (recognition_confidence < 0 OR recognition_confidence > 1)
        """)
        
        print(f"   Invalid insightface_confidence: {invalid_detection}")
        print(f"   Invalid recognition_confidence: {invalid_recognition}")
        
        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        total_issues = verified_wrong + count + auto_without_conf + invalid_detection + invalid_recognition
        
        if total_issues == 0:
            print("ALL CHECKS PASSED!")
        else:
            print(f"Total issues found: {total_issues}")
            print()
            print("Issues breakdown:")
            if verified_wrong > 0:
                print(f"  - Verified faces without recognition_confidence=1.0: {verified_wrong}")
            if count > 0:
                print(f"  - Faces with person_id but no recognition_confidence: {count}")
            if auto_without_conf > 0:
                print(f"  - Auto-recognized faces without recognition_confidence: {auto_without_conf}")
            if invalid_detection > 0:
                print(f"  - Invalid insightface_confidence values: {invalid_detection}")
            if invalid_recognition > 0:
                print(f"  - Invalid recognition_confidence values: {invalid_recognition}")
        
        print("=" * 60)
        
    finally:
        await conn.close()

if __name__ == '__main__':
    asyncio.run(check_confidence_fields())
