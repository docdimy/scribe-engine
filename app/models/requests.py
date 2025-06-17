"""
Pydantic Models for API Requests
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
from app.config import OutputFormat, ModelName


class TranscribeRequest(BaseModel):
    """Request Model for Audio Transcription"""
    
    diarization: bool = Field(
        default=False,
        description="Enable speaker diarization (uses AssemblyAI)"
    )
    
    specialty: Optional[str] = Field(
        default=None,
        description="Medical specialty (e.g. 'cardiology', 'neurology')",
        max_length=100
    )
    
    conversation_type: Optional[str] = Field(
        default="consultation",
        description="Type of conversation (e.g. 'consultation', 'discharge', 'notes')",
        max_length=100
    )
    
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON,
        description="Desired output format"
    )
    
    language: str = Field(
        default="auto",
        description="Language as ISO 639-1 code or 'auto' for automatic detection",
        max_length=5
    )
    
    model: ModelName = Field(
        default=ModelName.GPT_4O_MINI,
        description="LLM model for analysis"
    )
    
    @validator('language')
    def validate_language(cls, v):
        from app.config import settings
        if v not in settings.supported_languages:
            raise ValueError(f"Language '{v}' not supported. Available languages: {settings.supported_languages}")
        return v.lower()
    
    @validator('specialty')
    def validate_specialty(cls, v):
        if v and len(v.strip()) == 0:
            return None
        return v.strip().lower() if v else None
    
    @validator('conversation_type')
    def validate_conversation_type(cls, v):
        if v and len(v.strip()) == 0:
            return "consultation"
        return v.strip().lower() if v else "consultation"


class HealthCheckRequest(BaseModel):
    """Request Model for Health Check"""
    detailed: bool = Field(
        default=False,
        description="Request detailed health information"
    ) 