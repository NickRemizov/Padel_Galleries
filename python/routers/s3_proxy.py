"""
S3/MinIO Proxy Router
Проксирует запросы к MinIO для безопасного доступа к файлам
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
import os

router = APIRouter(tags=["S3 Proxy"])

# MinIO конфигурация
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9200")

@router.get("/{path:path}")
async def proxy_s3_file(request: Request):
    """
    Проксирует запросы к MinIO
    
    URL: /api/s3-proxy/galleries/2025/11/image.jpg
    -> MinIO: http://localhost:9200/galleries/2025/11/image.jpg
    """
    try:
        raw_path = request.scope.get('raw_path', b'').decode('utf-8')
        # Убираем prefix /api/s3-proxy/
        path = raw_path.replace("/api/s3-proxy/", "", 1)
        
        minio_url = f"{MINIO_ENDPOINT}/{path}"
        
        print(f"[S3 Proxy] Raw path: {raw_path}")
        print(f"[S3 Proxy] Path to MinIO: {path}")
        print(f"[S3 Proxy] MinIO URL: {minio_url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(minio_url, follow_redirects=True)
            
            print(f"[S3 Proxy] MinIO response: {response.status_code}")
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"File not found: {path}")
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"MinIO error: {response.status_code}"
                )
            
            # Определяем content-type по расширению
            content_type = "application/octet-stream"
            path_lower = path.lower()
            if path_lower.endswith(".jpg") or path_lower.endswith(".jpeg"):
                content_type = "image/jpeg"
            elif path_lower.endswith(".png"):
                content_type = "image/png"
            elif path_lower.endswith(".gif"):
                content_type = "image/gif"
            elif path_lower.endswith(".webp"):
                content_type = "image/webp"
            elif path_lower.endswith(".svg"):
                content_type = "image/svg+xml"
            
            return StreamingResponse(
                iter([response.content]),
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=31536000",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"MinIO connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
