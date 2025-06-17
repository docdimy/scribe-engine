"""
Sicherheits- und Authentifizierungsmodule
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
security_scheme = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationError(Exception):
    """Custom exception for authentication failures."""
    pass


class SecurityManager:
    """Zentrale Sicherheitsverwaltung"""
    
    def __init__(self):
        self.secret_key = settings.api_secret_key
        self.algorithm = settings.token_algorithm
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Erstellt einen JWT Access Token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifiziert einen JWT Token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def hash_api_key(self, api_key: str) -> str:
        """Erstellt einen Hash für API-Key Logging (für Audit-Zwecke)"""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    def generate_request_id(self) -> str:
        """Generiert eine eindeutige Request-ID"""
        return secrets.token_urlsafe(16)
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validiert einen API-Key (vereinfacht für Demo)"""
        # In Produktion: Validierung gegen Datenbank
        # Für Demo: Akzeptiere jeden nicht-leeren API-Key
        return bool(api_key and len(api_key.strip()) > 0)


# Global security manager instance
security_manager = SecurityManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security_scheme)) -> Dict[str, Any]:
    """Dependency für authentifizierte Anfragen"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Versuche JWT Token zu verifizieren
    token_payload = security_manager.verify_token(credentials.credentials)
    if token_payload:
        return token_payload
    
    # Fallback: API-Key Validierung
    if security_manager.validate_api_key(credentials.credentials):
        return {
            "api_key_hash": security_manager.hash_api_key(credentials.credentials),
            "auth_type": "api_key"
        }
    
    raise credentials_exception


def hash_password(password: str) -> str:
    """Hasht ein Passwort"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifiziert ein Passwort gegen seinen Hash"""
    return pwd_context.verify(plain_password, hashed_password)


class DataEncryption:
    """Encryption-at-Rest für sensitive Daten"""
    
    @staticmethod
    def encrypt_data(data: bytes) -> bytes:
        """Verschlüsselt Daten (vereinfacht für Demo)"""
        # In Produktion: Echte Verschlüsselung mit AES
        return data
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes) -> bytes:
        """Entschlüsselt Daten (vereinfacht für Demo)"""
        # In Produktion: Echte Entschlüsselung
        return encrypted_data
    
    @staticmethod
    def secure_delete(data: Any) -> None:
        """Sicheres Löschen von Daten aus dem Speicher"""
        # In Produktion: Überschreiben des Speichers
        del data 