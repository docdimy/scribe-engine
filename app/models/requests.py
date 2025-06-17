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