"""
Pydantic Models for API Requests
"""

from typing import Optional
from pydantic import BaseModel, Field


class HealthCheckRequest(BaseModel):
    """Request Model for Health Check"""
    detailed: bool = Field(
        default=False,
        description="Request detailed health information"
    )

class JobCreationResponse(BaseModel):
    """Response model for endpoints that start a background job."""
    job_id: str = Field(description="Unique identifier for the created job.") 