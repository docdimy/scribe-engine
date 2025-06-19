"""
Scribe Engine - FastAPI Main Application
"""

import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
import httpx

from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, FileResponse

from app.config import settings, OutputFormat, ModelName, STTModel, FHIRBundleType
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


# --- Dependency Status Checks ---
async def check_openai_status() -> (str, str):
    """Checks the status of the OpenAI API."""
    try:
        client = httpx.AsyncClient()
        response = await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {settings.openai_api_key}"})
        if response.status_code == 200:
            return "ok", "OpenAI API is reachable."
        return "error", f"OpenAI API returned status {response.status_code}."
    except Exception as e:
        return "error", f"Failed to connect to OpenAI API: {e}"

async def check_assemblyai_status() -> (str, str):
    """Checks the status of the AssemblyAI API."""
    try:
        client = httpx.AsyncClient()
        headers = {"authorization": settings.assemblyai_api_key}
        # Using a simple endpoint like /v2/transcript with a limit of 1 to check connectivity
        response = await client.get("https://api.assemblyai.com/v2/transcript?limit=1", headers=headers)
        if 200 <= response.status_code < 300:
            return "ok", "AssemblyAI API is reachable."
        return "error", f"AssemblyAI API returned status {response.status_code}."
    except Exception as e:
        return "error", f"Failed to connect to AssemblyAI API: {e}"

# --------------------------------


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

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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


# Middleware for security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    if "Strict-Transport-Security" not in response.headers:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if "X-Content-Type-Options" not in response.headers:
        response.headers["X-Content-Type-Options"] = "nosniff"
    if "X-Frame-Options" not in response.headers:
        response.headers["X-Frame-Options"] = "DENY"
    # A basic Content Security Policy. 
    # 'unsafe-inline' is needed for the styles and scripts in index.html.
    if "Content-Security-Policy" not in response.headers:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "object-src 'none'"
        )
    return response


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


@app.get("/ready")
async def readiness_check(request: Request):
    """
    Checks if the service and its dependencies are ready to accept traffic.
    Returns 200 OK if all checks pass, otherwise 503 Service Unavailable.
    """
    checks = {
        "openai": check_openai_status(),
        "assemblyai": check_assemblyai_status(),
    }
    
    results = await asyncio.gather(*checks.values())
    
    details = {}
    all_ok = True
    for (name, (status, message)) in zip(checks.keys(), results):
        details[name] = {"status": status, "message": message}
        if status != "ok":
            all_ok = False
            
    response_data = {
        "status": "ready" if all_ok else "unavailable",
        "timestamp": datetime.utcnow(),
        "version": settings.api_version,
        "details": details
    }
    
    if all_ok:
        return JSONResponse(status_code=status.HTTP_200_OK, content=response_data)
    else:
        logger.warning(f"Readiness check failed: {details}")
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response_data)


# Serve main page
@app.get("/", include_in_schema=False)
async def read_index():
    """Serviert die statische Test-Webseite"""
    return FileResponse('app/static/index.html')


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
    audio_file: UploadFile = File(..., alias="file"),
    user_info: dict = Depends(get_current_user),
    # Use Enums for automatic validation
    output_format: OutputFormat = OutputFormat.JSON,
    model: ModelName = ModelName.GPT_4_1_NANO,
    # Optional parameters with validation
    diarization: bool = False,
    specialty: str = "general",
    conversation_type: str = "consultation",
    language: str = "auto",
    output_language: Optional[str] = None,
    fhir_bundle_type: Optional[FHIRBundleType] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Transcribe and analyze audio data
    """
    
    start_time = time.time()
    request_id = request.state.request_id
    
    # Validate parameter combination
    if fhir_bundle_type and output_format != OutputFormat.FHIR:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'fhir_bundle_type' parameter is only applicable when 'output_format' is 'fhir'."
        )
    
    processed_audio_path = None
    try:
        logger.info(f"Request {request_id} authenticated for user: {user_info.get('sub', 'unknown')}")
        
        # 1. Process audio file
        with audio_processing_duration.time():
            processed_audio_path = await audio_processor.process_and_save_audio(
                file=audio_file, 
                specialty=specialty
            )
        
        logger.info(f"Request {request_id}: Audio processed and saved to {processed_audio_path}")
        
        # 2. Determine STT provider and model
        if diarization:
            stt_provider = "assemblyai"
            stt_model = STTModel.ASSEMBLYAI_UNIVERSAL.value
        else:
            stt_provider = "openai"
            stt_model = STTModel.GPT_4O_MINI_TRANSCRIBE.value

        # 3. Transcribe audio
        logger.info(f"Starting transcription with {stt_provider}, language: {language}, diarization: {diarization}")
        transcript = await stt_service.transcribe(
            request_id=request_id,
            file_path=processed_audio_path,
            stt_provider=stt_provider,
            stt_model=stt_model,
            language=language,
            diarization=diarization,
            stt_prompt=None  # Placeholder for future implementation
        )
        
        # Set output language to detected language if not provided
        final_output_language = output_language or transcript.language_detected or "en"

        # 4. Analyze transcript with LLM
        logger.info(f"Request {request_id}: Starting LLM analysis with model: {model} -> output lang: {final_output_language}")
        analysis = await llm_service.analyze(
            transcript=transcript.text,
            model=model.value,
            specialty=specialty,
            conversation_type=conversation_type,
            output_language=final_output_language
        )
        
        # 5. Handle different output formats
        fhir_bundle = None
        xml_content = None
        
        if output_format == OutputFormat.FHIR:
            logger.info(f"Request {request_id}: Generating FHIR bundle")
            fhir_bundle = await fhir_service.create_fhir_bundle(
                transcript=transcript,
                analysis=analysis,
                request_id=request_id,
                specialty=specialty,
                conversation_type=conversation_type,
                bundle_type=(fhir_bundle_type.value if fhir_bundle_type else FHIRBundleType.DOCUMENT.value)
            )
        elif output_format == OutputFormat.XML:
            xml_content = _create_xml_output(transcript, analysis)
            
        # 6. Clean up temporary file
        background_tasks.add_task(audio_processor.cleanup, processed_audio_path)
        
        # 7. Create final response
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        response_data = ScribeResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            transcript=transcript,
            analysis=analysis,
            output_format=output_format,
            processing_time_ms=processing_time_ms,
            fhir_bundle=fhir_bundle,
            xml_content=xml_content,
        )
        return response_data

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
        if processed_audio_path:
            # Use background task for cleanup here as well to avoid blocking
            background_tasks.add_task(audio_processor.cleanup, processed_audio_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during transcription."
        )


def _create_xml_output(transcript: TranscriptionResult, analysis: AnalysisResult) -> str:
    """Create XML output for transcript and analysis"""
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<medical_consultation>
    <transcript>
        <full_text><![CDATA[{transcript.text}]]></full_text>
        <language>{transcript.language_detected or 'unknown'}</language>
        <segments>
            {"".join([
                f"<segment>"
                f"<text><![CDATA[{seg.text}]]></text>"
                f"<start>{seg.start}</start>"
                f"<end>{seg.end}</end>"
                f"<speaker>{seg.speaker or 'U'}</speaker>"
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
        content=response.model_dump(by_alias=True),
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