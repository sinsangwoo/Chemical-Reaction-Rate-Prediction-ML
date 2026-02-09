"""Authentication and authorization for Chemical Reaction ML API.

Implements:
- User registration and login
- JWT token generation and validation
- Password hashing with bcrypt
- API key authentication
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import secrets
import os

from api.database import get_db, User, APIKey

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer for token authentication
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Bcrypt hash
    
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password with bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        Bcrypt hash
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token.
    
    Args:
        data: Token payload
        expires_delta: Optional expiration time
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password.
    
    Args:
        db: Database session
        email: User email
        password: Plain text password
    
    Returns:
        User if authenticated, None otherwise
    """
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Authorization header
        db: Database session
    
    Returns:
        Current user
    
    Raises:
        HTTPException: If token invalid or user not found
    """
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_access_token(token)
    
    if payload is None:
        raise credentials_exception
    
    user_id: int = payload.get("sub")
    
    if user_id is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user.
    
    Args:
        current_user: Current user from token
    
    Returns:
        Active user
    
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user


def create_api_key(db: Session, user_id: int, name: str) -> str:
    """Create new API key for user.
    
    Args:
        db: Database session
        user_id: User ID
        name: Friendly name for key
    
    Returns:
        Generated API key (plain text, only shown once)
    """
    # Generate random key
    key = secrets.token_urlsafe(32)
    
    # Hash key for storage
    hashed_key = get_password_hash(key)
    
    # Create API key record
    api_key = APIKey(
        user_id=user_id,
        key=hashed_key,
        name=name
    )
    
    db.add(api_key)
    db.commit()
    
    # Return plain key (only time it's visible)
    return key


def validate_api_key(db: Session, key: str) -> Optional[User]:
    """Validate API key and return associated user.
    
    Args:
        db: Database session
        key: Plain text API key
    
    Returns:
        User if key valid, None otherwise
    """
    # Get all active API keys
    api_keys = db.query(APIKey).filter(APIKey.is_active == True).all()
    
    for api_key in api_keys:
        if verify_password(key, api_key.key):
            # Update last used timestamp
            api_key.last_used_at = datetime.utcnow()
            db.commit()
            
            # Return user
            user = db.query(User).filter(User.id == api_key.user_id).first()
            return user
    
    return None
