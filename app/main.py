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
from app.models.responses import (
    ScribeResponse, HealthCheckResponse, ErrorResponse, 
    RateLimitResponse, TranscriptionResult, AnalysisResult
)
from app.services.audio_processor import AudioProcessor
from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.fhir_service import FHIRService

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
    response_model=ScribeResponse,
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
    audio_file: UploadFile = File(...),
    user_info: dict = Depends(get_current_user),
    # Use Enums for automatic validation
    output_format: OutputFormat = OutputFormat.JSON,
    model: ModelName = ModelName.GPT_4_1_NANO,
    stt_model: STTModel = STTModel.GPT_4O_MINI_TRANSCRIBE,
    # Optional parameters with validation
    diarization: bool = False,
    specialty: str = "general",
    conversation_type: str = "consultation",
    language: str = "auto"
):
    """
    Transcribe and analyze audio data
    
    - **diarization**: Enable speaker diarization (AssemblyAI)
    - **specialty**: Medical specialty
    - **conversation_type**: Type of consultation
    - **output_format**: Output format (json, xml, fhir)
    - **language**: Language (ISO 639-1 or auto)
    - **model**: LLM model for analysis
    - **stt_model**: STT model for transcription
    """
    
    start_time = time.time()
    request_id = request.state.request_id
    
    try:
        logger.info(f"Request {request_id} authenticated for user: {user_info.get('sub', 'api_key')}")
        
        # Parameters are now validated by FastAPI using Enums
        
        # Process audio file
        with audio_processing_duration.time():
            processed_audio_path, duration, file_metadata = await audio_processor.process_and_save_audio(
                file=audio_file, 
                max_duration=settings.max_audio_duration, 
                max_size_mb=settings.max_file_size_mb,
                supported_types=settings.supported_audio_types
            )
        
        logger.info(f"Request {request_id}: Audio processed in {time.time() - start_time:.2f}s. Duration: {duration:.2f}s")
        
        # Perform STT
        transcript = await stt_service.transcribe(
            file_path=processed_audio_path,
            stt_model=stt_model,
            language=language,
            diarization=diarization
        )
        
        logger.info(f"Request {request_id}: STT completed. Language: {transcript.language_detected}")
        
        # Perform LLM analysis
        analysis = await llm_service.analyze(
            transcript=transcript.full_text,
            model=model,
            specialty=specialty,
            conversation_type=conversation_type
        )
        
        logger.info(f"Request {request_id}: LLM analysis completed.")
        
        # Generate final response based on output format
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response_data = {
            "request_id": request_id,
            "timestamp": datetime.utcnow(),
            "transcript": transcript,
            "analysis": analysis,
            "output_format": output_format,
            "processing_time_ms": processing_time_ms
        }
        
        # Handle different output formats
        if output_format == OutputFormat.FHIR:
            fhir_bundle = await fhir_service.create_fhir_bundle(
                transcript=transcript,
                analysis=analysis,
                patient_id="example-patient-id" # Placeholder
            )
            response_data["fhir_bundle"] = fhir_bundle
            
        elif output_format == OutputFormat.XML:
            xml_content = _create_xml_output(transcript, analysis)
            # For simplicity, returning it in the JSON payload. 
            # Could also return a pure XML response.
            response_data["xml_content"] = xml_content
            
        # Clean up temporary file
        audio_processor.cleanup(processed_audio_path)
        
        return ScribeResponse(**response_data)

    except AuthenticationError as e:
        logger.warning(f"Request {request_id} failed authentication: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions from services (e.g., audio validation)
        raise
        
    except Exception as e:
        logger.error(f"Request {request_id} failed with an unexpected error: {e}", exc_info=True)
        # Clean up in case of failure
        if 'processed_audio_path' in locals() and processed_audio_path:
            audio_processor.cleanup(processed_audio_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during transcription."
        )


def _create_xml_output(transcript: TranscriptionResult, analysis: AnalysisResult) -> str:
    """Create XML output for transcript and analysis"""
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<medical_consultation>
    <transcript>
        <full_text><![CDATA[{transcript.full_text}]]></full_text>
        <language>{transcript.language_detected or 'unknown'}</language>
        <confidence>{transcript.confidence or 0}</confidence>
        <segments>
            {"".join([
                f"<segment>"
                f"<text><![CDATA[{seg.text}]]></text>"
                f"<start_time>{seg.start_time or 0}</start_time>"
                f"<end_time>{seg.end_time or 0}</end_time>"
                f"<speaker>{seg.speaker or 'unknown'}</speaker>"
                f"<confidence>{seg.confidence or 0}</confidence>"
                f"</segment>"
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