from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
import sys
import os

from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        # Initialize MongoDB connection
        self.client = AsyncIOMotorClient(settings.MONGODB_URI)
        self.db = self.client.get_database()
        self.users_collection = self.db.get_collection("users")
        
    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password):
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.AUTH_JWT_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.AUTH_JWT_SECRET.get_secret_value(), 
            algorithm=settings.AUTH_JWT_ALGORITHM
        )
        return encoded_jwt

    async def get_user_by_email(self, email: str):
        return await self.users_collection.find_one({"email": email})

    async def create_user(self, name: str, email: str, password: str, location: Optional[str] = None):
        user_dict = {
            "name": name,
            "email": email,
            "hashed_password": self.get_password_hash(password),
            "location": location,
            "created_at": datetime.utcnow()
        }
        result = await self.users_collection.insert_one(user_dict)
        user_dict["_id"] = str(result.inserted_id)
        return user_dict

_auth_service = None

def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
