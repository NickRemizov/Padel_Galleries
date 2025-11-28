import asyncio
import os
import asyncpg

async def check_gallery_data():
    """Check if gallery images exist in database."""
    
    database_url = os.getenv("POSTGRES_URL")
    if not database_url:
        print("ERROR: POSTGRES_URL not set")
        return
    
    conn = await asyncpg.connect(database_url)
    
    try:
        # Check total galleries
        total_galleries = await conn.fetchval("SELECT COUNT(*) FROM galleries")
        print(f"\nTotal galleries: {total_galleries}")
        
        # Check total gallery images
        total_images = await conn.fetchval("SELECT COUNT(*) FROM gallery_images")
        print(f"Total gallery images: {total_images}")
        
        # Check specific gallery
        gallery_id = "76556eed-cc34-4c1c-832b-60ffc4f52ef3"
        images = await conn.fetch(
            "SELECT id, gallery_id, image_url, display_order FROM gallery_images WHERE gallery_id = $1",
            gallery_id
        )
        print(f"\nImages in gallery {gallery_id}: {len(images)}")
        for img in images[:5]:  # Show first 5
            print(f"  - {img['id']}: order={img['display_order']}, url={img['image_url'][:50]}...")
        
        # Check galleries with images
        galleries_with_images = await conn.fetch("""
            SELECT g.id, g.title, COUNT(gi.id) as image_count
            FROM galleries g
            LEFT JOIN gallery_images gi ON g.id = gi.gallery_id
            GROUP BY g.id, g.title
            ORDER BY image_count DESC
            LIMIT 10
        """)
        
        print(f"\nTop 10 galleries by image count:")
        for g in galleries_with_images:
            print(f"  - {g['title']}: {g['image_count']} images")
    
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_gallery_data())
