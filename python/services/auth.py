from fastapi import HTTPException, Security, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

security = HTTPBearer()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change_this_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 часа

API_SECRET_KEY = os.getenv("API_SECRET_KEY")


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Проверка API ключа для server-to-server запросов"""
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=500, 
            detail="API_SECRET_KEY not configured on server"
        )
    
    if not x_api_key:
        raise HTTPException(
            status_code=401, 
            detail="X-API-Key header required"
        )
    
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(
            status_code=403, 
            detail="Invalid API key"
        )
    
    return {"authenticated": True, "method": "api_key"}


async def verify_google_token(token: str) -> dict:
    """Проверка Google OAuth токена"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Неверный Google токен")
            
            token_info = response.json()
            
            # Проверяем что токен для нашего приложения
            if token_info.get("aud") != GOOGLE_CLIENT_ID:
                raise HTTPException(status_code=401, detail="Токен не для этого приложения")
            
            return token_info
    
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Ошибка проверки токена: {str(e)}")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Создание JWT токена"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Получение текущего пользователя из JWT токена"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        
        if user_email is None:
            raise HTTPException(status_code=401, detail="Неверный токен")
        
        return {"email": user_email, "name": payload.get("name")}
    
    except JWTError:
        raise HTTPException(status_code=401, detail="Неверный токен")


# Опциональная аутентификация (для публичных endpoint'ов)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[dict]:
    """Опциональная аутентификация - не требует токен"""
    if credentials is None:
        return None
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        
        if user_email is None:
            return None
        
        return {"email": user_email, "name": payload.get("name"), "sub": user_email}
    except:
        return None
