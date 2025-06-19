"""
Central configuration for the Scribe Engine Service
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class OutputFormat(str, Enum):
    JSON = "json"
    XML = "xml"
    FHIR = "fhir"


class FHIRBundleType(str, Enum):
    DOCUMENT = "document"
    TRANSACTION = "transaction"


class ModelName(str, Enum):
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class STTModel(str, Enum):
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"
    ASSEMBLYAI_UNIVERSAL = "assemblyai-universal"
    # Future: Local Whisper with custom speaker diarization
    LOCAL_WHISPER = "local-whisper"  # For future implementation


class Settings(BaseSettings):
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Configuration
    api_title: str = Field(default="Scribe Engine API")
    api_description: str = Field(default="Medical Audio Transcription and Analysis Service")
    api_version: str = Field(default="1.0.0")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=3001)
    api_secret_key: str = Field(..., env="API_SECRET_KEY")
    
    # Domain Configuration
    base_domain: str = Field(default="numediq.de")
    api_base_url: str = Field(default="https://api.numediq.de")
    frontend_url: str = Field(default="https://app.numediq.de")
    
    # External Service APIs
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    assemblyai_api_key: str = Field(..., env="ASSEMBLYAI_API_KEY")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=10)
    rate_limit_window: int = Field(default=60)  # seconds
    
    # Audio Processing Limits
    max_audio_duration: int = Field(default=600)  # seconds
    max_file_size_mb: int = Field(default=50)
    supported_audio_formats: List[str] = Field(
        default=["audio/mpeg", "audio/wav", "audio/mp4", "audio/m4a", "audio/ogg", "audio/webm"]
    )
    
    # Timeouts and Retries
    stt_timeout: int = Field(default=60)
    llm_timeout: int = Field(default=60)
    max_retries: int = Field(default=3)
    
    # Language Support
    supported_languages: List[str] = Field(
        default=["en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "auto"]
    )
    
    # LLM Configuration
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=2000)
    default_llm_model: str = Field(default="gpt-4.1-nano")
    
    # STT Configuration
    default_stt_model: str = Field(default="gpt-4o-mini-transcribe")
    enable_local_whisper: bool = Field(default=False)  # Future feature flag
    local_whisper_model_path: str = Field(default="/models/whisper-large-v3")  # Future
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000", 
            "http://localhost:8080",
            "https://app.numediq.de",
            "https://numediq.de",
            "https://www.numediq.de"
        ]
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(default=["GET", "POST"])
    cors_allow_headers: List[str] = Field(default=["*"])
    
    # Security
    token_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    jwt_issuer: str = Field(default="numediq.de")
    data_encryption_key: str = Field(..., env="DATA_ENCRYPTION_KEY")
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")
    
    # Audit Logging
    audit_log_enabled: bool = Field(default=True)
    audit_log_retention_days: int = Field(default=90)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 