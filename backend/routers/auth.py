from fastapi import APIRouter, Depends, HTTPException, status, Header
from pydantic import BaseModel
from typing import Optional
from jose import JWTError, jwt
import sys

from config import settings
from services.auth_service import get_auth_service, AuthService

router = APIRouter(prefix="/api/auth", tags=["auth"])

class UserSignup(BaseModel):
    name: str
    email: str
    password: str
    location: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

async def get_current_user(authorization: Optional[str] = Header(None), auth_svc: AuthService = Depends(get_auth_service)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not authorization or not authorization.startswith("Bearer "):
        raise credentials_exception
        
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, settings.AUTH_JWT_SECRET.get_secret_value(), algorithms=[settings.AUTH_JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = await auth_svc.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    
    # Return safe user dict
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "location": user.get("location")
    }

@router.post("/signup", response_model=TokenResponse)
async def signup(user: UserSignup, auth_svc: AuthService = Depends(get_auth_service)):
    existing = await auth_svc.get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    new_user = await auth_svc.create_user(
        name=user.name, 
        email=user.email, 
        password=user.password, 
        location=user.location
    )
    
    access_token = auth_svc.create_access_token(data={"sub": user.email})
    safe_user = {
        "id": str(new_user["_id"]),
        "name": new_user["name"],
        "email": new_user["email"],
        "location": new_user.get("location")
    }
    
    return {"access_token": access_token, "token_type": "bearer", "user": safe_user}

@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin, auth_svc: AuthService = Depends(get_auth_service)):
    db_user = await auth_svc.get_user_by_email(user.email)
    if not db_user or not auth_svc.verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = auth_svc.create_access_token(data={"sub": user.email})
    safe_user = {
        "id": str(db_user["_id"]),
        "name": db_user["name"],
        "email": db_user["email"],
        "location": db_user.get("location")
    }
    
    return {"access_token": access_token, "token_type": "bearer", "user": safe_user}

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user
