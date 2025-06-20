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
import os
import json

from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File, Form, BackgroundTasks
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

from app.config import settings, OutputFormat, ModelName, FHIRBundleType
from app.core.logging import setup_logging, get_logger, audit_logger
from app.core.security import get_current_user, security_manager, AuthenticationError
from app.models.responses import (
    ScribeResponse, HealthCheckResponse, ErrorResponse, 
    RateLimitResponse, TranscriptionResult, AnalysisResult
)
from app.services.audio_processor import AudioProcessor
from app.services.stt_service import STTService, process_and_save_audio
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

# Mount static files directory only in development
if settings.environment == "development":
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
    
    # Add request ID and start time to state
    request.state.request_id = request_id
    request.state.start_time = start_time
    
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
    # OpenAI status is checked by LLM service when needed.
    # We only check AssemblyAI here as it's the core STT dependency.
    checks = {
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


# Serve index.html only in development
if settings.environment == "development":
    @app.get("/", include_in_schema=False)
    async def read_index():
        """Serves the index.html file."""
        return FileResponse("app/static/index.html")


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
    background_tasks: BackgroundTasks,
    user_info: dict = Depends(get_current_user),
    audio_file: UploadFile = File(..., alias="file"),
    # Form fields
    output_format: OutputFormat = Form(OutputFormat.JSON),
    model: ModelName = Form(ModelName.GPT_4_1_NANO),
    diarization: bool = Form(False),
    specialty: str = Form("general"),
    conversation_type: str = Form("consultation"),
    language: str = Form("auto"),
    output_language: Optional[str] = Form(None),
    fhir_bundle_type: Optional[FHIRBundleType] = Form(None)
):
    """
    This endpoint receives an audio file and configuration, transcribes the audio,
    analyzes the content with an LLM, and returns the result in the specified format.
    """
    request_id = request.state.request_id
    temp_audio_file = None
    
    # --- Input Validation ---
    if language != "auto" and language not in settings.supported_languages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported input language '{language}'. Supported are: {', '.join(settings.supported_languages)}"
        )
    if output_language and output_language not in settings.supported_languages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported output language '{output_language}'. Supported are: {', '.join(settings.supported_languages)}"
        )
    
    try:
        # --- 1. Save and encrypt audio file ---
        # This returns an open temporary file that must be closed later
        temp_audio_file = await process_and_save_audio(audio_file, specialty)
        file_path = temp_audio_file.name

        # --- 2. Transcription ---
        logger.info(f"[{request_id}] Starting transcription with AssemblyAI...")
        
        transcript_result = await stt_service.transcribe(
            request_id=request_id,
            file_path=file_path,
            diarization=diarization,
            language=language,
        )
        
        # Add the deletion of the transcript to background tasks
        if transcript_result.provider_transcript_id:
            background_tasks.add_task(
                stt_service.delete_transcript, transcript_result.provider_transcript_id
            )

        # Determine the language for the LLM analysis output.
        # Priority: 1. User-specified output language.
        #           2. Detected language from STT.
        #           3. Fallback to English.
        if output_language:
            llm_output_lang = output_language
        elif transcript_result.language_detected:
            llm_output_lang = transcript_result.language_detected
        else:
            logger.warning(f"[{request_id}] Could not determine language. Defaulting LLM output to English.")
            llm_output_lang = "en"
        
        # --- 3. Analysis ---
        logger.info(f"[{request_id}] Starting analysis with LLM...")
        analysis_result = await llm_service.analyze(
            transcript=transcript_result.full_text,
            model=model.value,
            specialty=specialty,
            conversation_type=conversation_type,
            output_language=llm_output_lang
        )
        
        logger.info(f"[{request_id}] LLM analysis completed successfully.")

        # --- 4. Format the output ---
        fhir_bundle = None
        if output_format == OutputFormat.FHIR:
            fhir_bundle = await fhir_service.create_fhir_bundle(
                request_id=request_id,
                transcript=transcript_result, 
                analysis=analysis_result,
                specialty=specialty,
                conversation_type=conversation_type,
                bundle_type=fhir_bundle_type
            )
            logger.info(f"[{request_id}] Analysis formatted to FHIR.")

        # --- 5. Create final response ---
        processing_time_ms = int((time.time() - request.state.start_time) * 1000)

        response_data = ScribeResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            transcript=transcript_result,
            analysis=analysis_result,
            output_format=output_format,
            processing_time_ms=processing_time_ms
        )

        if output_format == OutputFormat.FHIR:
            response_data.fhir_bundle = fhir_bundle
            # Use model_dump_json to handle complex nested models like FHIR resources
            # Then load it back to a dict for JSONResponse to handle correctly.
            return JSONResponse(content=json.loads(response_data.model_dump_json(exclude_none=True)))
        elif output_format == OutputFormat.XML:
            xml_content = _create_xml_output(request_id, transcript_result, analysis_result)
            response_data.xml_content = xml_content
            return PlainTextResponse(content=xml_content, media_type="application/xml")

        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id} failed with an unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        )
    finally:
        if temp_audio_file:
            try:
                temp_audio_file.close()
                os.remove(temp_audio_file.name)
                logger.info(f"Request {request_id}: Cleaned up temporary file {temp_audio_file.name}")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to cleanup temporary file {temp_audio_file.name}: {e}", exc_info=True)


def _create_xml_output(request_id: str, transcript: TranscriptionResult, analysis: AnalysisResult) -> str:
    """Create XML output for transcript and analysis"""
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ScribeResult>
    <request_id>{request_id}</request_id>
    <transcript>
        <full_text><![CDATA[{transcript.full_text}]]></full_text>
        <segments>
            {"".join(f'<segment speaker="{s.speaker or ""}"><start>{s.start_time}</start><end>{s.end_time}</end><text><![CDATA[{s.text}]]></text></segment>' for s in transcript.segments)}
        </segments>
    </transcript>
    <analysis>
        <summary><![CDATA[{analysis.summary}]]></summary>
        <diagnosis><![CDATA[{analysis.diagnosis}]]></diagnosis>
        <treatment><![CDATA[{analysis.treatment}]]></treatment>
        <medication><![CDATA[{analysis.medication}]]></medication>
        <follow_up><![CDATA[{analysis.follow_up}]]></follow_up>
        <specialty_notes><![CDATA[{analysis.specialty_notes}]]></specialty_notes>
        <icd10_codes>
            {"".join(f'<code>{code}</code>' for code in analysis.icd10_codes or [])}
        </icd10_codes>
    </analysis>
</ScribeResult>
"""


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