"""
Scribe Engine - FastAPI Main Application
"""

import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.config import settings, OutputFormat, ModelName, STTModel
from app.core.logging import setup_logging, get_logger, audit_logger
from app.core.security import get_current_user, security_manager, AuthenticationError
from app.models.requests import TranscribeRequest, validate_query_parameters
from app.models.responses import (
    TranscribeResponse, HealthCheckResponse, ErrorResponse, 
    RateLimitResponse, TranscriptionResult, AnalysisResult
)
from app.services.audio_processor import AudioProcessor
from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.fhir_service import FHIRService
from app.core import metrics

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
audio_processing_duration = Histogram('audio_processing_duration_seconds', 'Audio processing duration')

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Service instances
audio_processor = AudioProcessor()
stt_service = STTService()
llm_service = LLMService()
fhir_service = FHIRService()

# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Scribe Engine starting...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API Version: {settings.api_version}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Scribe Engine shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
    debug=settings.debug,
    servers=[
        {"url": "http://localhost:3001", "description": "Local development"},
        {"url": settings.api_base_url, "description": "Production API"},
    ]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# Middleware for request tracking and metrics
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Request tracking and Prometheus metrics"""
    
    start_time = time.time()
    request_id = security_manager.generate_request_id()
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        
        # Update metrics
        duration = time.time() - start_time
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        request_duration.observe(duration)
        
        # Set response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{duration:.3f}s"
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        logger.error(f"Request {request_id} failed: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An internal error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            headers={"X-Request-ID": request_id}
        )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Service health check"""
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.api_version,
        uptime_seconds=int(time.time())  # Simplified
    )


@app.get("/ready", response_model=HealthCheckResponse)
async def readiness_check():
    """Service readiness check"""
    
    # TODO: Check external dependencies (OpenAI, AssemblyAI, Redis)
    details = {
        "openai": "connected",
        "assemblyai": "connected", 
        "redis": "connected"
    }
    
    return HealthCheckResponse(
        status="ready",
        timestamp=datetime.utcnow(),
        version=settings.api_version,
        uptime_seconds=int(time.time()),
        details=details
    )


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Main endpoint for audio transcription
@app.post(
    "/v1/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    diarization: bool = False,
    specialty: str = "general",
    conversation_type: str = "consultation",
    output_format: OutputFormat = OutputFormat.JSON,
    language: str = "auto",
    model: str = "gpt-4.1-nano",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Transcribe and analyze audio data
    
    - **diarization**: Enable speaker diarization (AssemblyAI)
    - **specialty**: Medical specialty
    - **conversation_type**: Type of consultation
    - **output_format**: Output format (json, xml, fhir)
    - **language**: Language (ISO 639-1 or auto)
    - **model**: LLM model for analysis
    """
    
    start_time = time.time()
    request_id = request.state.request_id
    
    try:
        # Authenticate user
        user_info = await get_current_user(credentials)
        logger.info(f"Request {request_id} authenticated for user: {user_info.get('sub', 'unknown')}")
        
        # Validate query parameters
        validated_params = validate_query_parameters(
            diarization=diarization,
            specialty=specialty,
            conversation_type=conversation_type,
            output_format=output_format,
            language=language,
            model=model
        )
        
        # Process audio file
        logger.info(f"Processing audio file for request {request_id}")
        
        # Validate audio file
        audio_data = await audio_processor.validate_audio_file(audio_file)
        
        # Extract metadata
        metadata = await audio_processor.extract_metadata(audio_data)
        logger.info(f"Audio metadata: {metadata}")
        
        # Speech-to-text
        logger.info(f"Starting STT for request {request_id}")
        start_time = time.time()
        
        if validated_params.diarization:
            # Use AssemblyAI for diarization
            transcript_result = await stt_service.transcribe_with_assemblyai(
                audio_data=audio_data,
                language=validated_params.language,
                diarization=True
            )
        else:
            # Use OpenAI Whisper (default)
            transcript_result = await stt_service.transcribe_with_openai(
                audio_data=audio_data,
                language=validated_params.language
            )
        
        stt_duration = time.time() - start_time
        logger.info(f"STT completed in {stt_duration:.2f}s for request {request_id}")
        
        # Medical analysis with LLM
        logger.info(f"Starting LLM analysis for request {request_id}")
        start_time = time.time()
        
        analysis_result = await llm_service.analyze_medical_content(
            transcript=transcript_result.full_text,
            specialty=validated_params.specialty,
            conversation_type=validated_params.conversation_type,
            model=validated_params.model
        )
        
        llm_duration = time.time() - start_time
        logger.info(f"LLM analysis completed in {llm_duration:.2f}s for request {request_id}")
        
        # Format output based on requested format
        if validated_params.output_format == OutputFormat.FHIR:
            # Generate FHIR Bundle
            fhir_data = await fhir_service.create_fhir_bundle(
                transcript=transcript_result,
                analysis=analysis_result,
                request_id=request_id,
                specialty=validated_params.specialty,
                conversation_type=validated_params.conversation_type
            )
            
            response_data = {
                "request_id": request_id,
                "format": "fhir",
                "data": fhir_data
            }
        else:
            # Standard JSON/XML response
            response_data = {
                "request_id": request_id,
                "transcript": transcript_result.dict(),
                "analysis": analysis_result.dict(),
                "metadata": {
                    "processing_time": stt_duration + llm_duration,
                    "specialty": validated_params.specialty,
                    "conversation_type": validated_params.conversation_type,
                    "diarization_used": validated_params.diarization,
                    "model_used": validated_params.model,
                    "language": validated_params.language,
                    "audio_metadata": metadata
                }
            }
        
        # Log completion for audit trail
        background_tasks.add_task(
            log_request,
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            status_code=200,
            duration=stt_duration + llm_duration,
            user_agent=request.headers.get("user-agent"),
            ip_address=get_remote_address(request),
            user_id=user_info.get("sub"),
            additional_data={
                "audio_duration": metadata.get("duration"),
                "transcript_length": len(transcript_result.full_text),
                "specialty": validated_params.specialty
            }
        )
        
        logger.info(f"Request {request_id} completed successfully")
        
        return TranscriptionResponse(**response_data)
        
    except AuthenticationError as e:
        logger.warning(f"Authentication failed for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    
    except ValueError as e:
        logger.warning(f"Validation error for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error for request {request_id}: {e}")
        
        # Log error for audit trail
        background_tasks.add_task(
            log_error,
            request_id=request_id,
            error=str(e),
            method=request.method,
            url=str(request.url)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )


def _create_xml_output(transcript: TranscriptionResult, analysis: AnalysisResult) -> str:
    """Create XML output for transcript and analysis"""
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<medical_consultation>
    <transcript>
        <full_text><![CDATA[{transcript.full_text}]]></full_text>
        <language>{transcript.language_detected or 'unknown'}</language>
        <duration>{transcript.duration or 0}</duration>
        <confidence>{transcript.confidence or 0}</confidence>
        <segments>
            {"".join([
                f"""<segment>
                    <text><![CDATA[{seg.text}]]></text>
                    <start_time>{seg.start_time or 0}</start_time>
                    <end_time>{seg.end_time or 0}</end_time>
                    <speaker>{seg.speaker or 'unknown'}</speaker>
                    <confidence>{seg.confidence or 0}</confidence>
                </segment>"""
                for seg in transcript.segments
            ])}
        </segments>
    </transcript>
    <analysis>
        <summary><![CDATA[{analysis.summary}]]></summary>
        <diagnosis><![CDATA[{analysis.diagnosis or ''}]]></diagnosis>
        <treatment><![CDATA[{analysis.treatment or ''}]]></treatment>
        <medication><![CDATA[{analysis.medication or ''}]]></medication>
        <follow_up><![CDATA[{analysis.follow_up or ''}]]></follow_up>
        <specialty_notes><![CDATA[{analysis.specialty_notes or ''}]]></specialty_notes>
        <icd10_codes>
            {"".join([f"<code>{code}</code>" for code in (analysis.icd10_codes or [])])}
        </icd10_codes>
    </analysis>
</medical_consultation>"""
    
    return xml_content


# Override rate limit handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit error handler"""
    
    response = RateLimitResponse(
        message="Too many requests. Please try again later.",
        retry_after=int(exc.retry_after),
        limit=settings.rate_limit_requests,
        window=settings.rate_limit_window,
        timestamp=datetime.utcnow()
    )
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=response.dict(),
        headers={"Retry-After": str(int(exc.retry_after))}
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Unhandled error in request {request_id}: {exc}")
    logger.error(f"Stacktrace: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat()
        },
        headers={"X-Request-ID": request_id}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development"
    ) 