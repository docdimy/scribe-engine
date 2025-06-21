"""
Scribe Engine - FastAPI Main Application
"""

import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio
import httpx
import os
import json

from fastapi import FastAPI, HTTPException, Request, Depends, status, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
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
from app.models.requests import JobCreationResponse
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

# SSE Job Queues
job_status_queues: Dict[str, asyncio.Queue] = {}

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
        response = await client.get("https://api.eu.assemblyai.com/v2/transcript?limit=1", headers=headers)
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


async def run_transcription_pipeline(
    request: Request,
    background_tasks: BackgroundTasks,
    job_id: str,
    queue: asyncio.Queue,
    user_info: dict,
    audio_data: bytes,
    content_type: str,
    output_format: OutputFormat,
    model: ModelName,
    diarization: bool,
    specialty: str,
    conversation_type: str,
    language: str,
    output_language: Optional[str],
    fhir_bundle_type: Optional[FHIRBundleType]
):
    """
    The full transcription and analysis pipeline, designed to run in the background.
    It pushes status updates to a queue for SSE.
    """
    request_id = request.state.request_id
    temp_audio_file = None

    try:
        # --- 1. Save and encrypt audio file (can be slow on some filesystems) ---
        # Send a message FIRST, so the user doesn't see a long "Connecting..." message.
        await queue.put(json.dumps({"status": "processing", "message": "Processing uploaded audio...", "progress": 10}))
        await asyncio.sleep(0.01) # Allow the message to be sent before blocking I/O

        temp_audio_file = await process_and_save_audio(audio_data, content_type, specialty)
        file_path = temp_audio_file.name

        # --- 2. Transcription (slow operation) ---
        logger.info(f"[{request_id}] Starting transcription with AssemblyAI...")
        await queue.put(json.dumps({"status": "processing", "message": "Starting transcription... (this may take a moment)", "progress": 20}))
        await asyncio.sleep(0.01) # Force event to be sent before blocking
        
        # Get the raw transcript object from AssemblyAI
        raw_transcript = await stt_service.transcribe(
            request_id=request_id,
            file_path=file_path,
            diarization=diarization,
            language=language,
        )

        # --- 2.5. Speaker Labeling with LeMUR (optional step) ---
        speaker_map = {}
        if diarization and raw_transcript.utterances:
            await queue.put(json.dumps({"status": "processing", "message": "Identifying speaker roles (Provider/Patient)...", "progress": 82}))
            await asyncio.sleep(0.01) # Allow message to be sent
            speaker_map = await llm_service.label_speakers_with_lemur(
                raw_transcript, 
                specialty=specialty, 
                conversation_type=conversation_type
            )

        # Convert the raw transcript to our internal model, applying the new speaker labels
        transcript_result = stt_service.create_transcription_result(raw_transcript, speaker_map)
        
        await queue.put(json.dumps({"status": "processing", "message": f"Transcription complete. {len(transcript_result.full_text)} characters recognized.", "progress": 85}))
        
        # Add the deletion of the transcript to background tasks
        if transcript_result.provider_transcript_id:
            background_tasks.add_task(
                stt_service.delete_transcript, transcript_result.provider_transcript_id
            )

        # Determine the language for the LLM analysis output.
        if output_language:
            llm_output_lang = output_language
        elif transcript_result.language_detected:
            llm_output_lang = transcript_result.language_detected
        else:
            logger.warning(f"[{request_id}] Could not determine language. Defaulting LLM output to English.")
            llm_output_lang = "en"
        
        # --- 3. Analysis (slow-ish operation) ---
        logger.info(f"[{request_id}] Starting analysis with LLM...")
        await queue.put(json.dumps({"status": "processing", "message": "Starting medical analysis...", "progress": 90}))
        await asyncio.sleep(0.01) # Force event to be sent before blocking
        analysis_result = await llm_service.analyze(
            transcript=transcript_result.full_text,
            model=model.value,
            specialty=specialty,
            conversation_type=conversation_type,
            output_language=llm_output_lang
        )
        await queue.put(json.dumps({"status": "processing", "message": "Analysis complete. Formatting result...", "progress": 95}))
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
            processing_time_ms=processing_time_ms,
            fhir_bundle=fhir_bundle
        )
        
        final_json = json.loads(response_data.model_dump_json(exclude_none=True))
        await queue.put(json.dumps({"status": "complete", "data": final_json, "progress": 100}))
        
    except Exception as e:
        logger.error(f"Request {request_id} failed in background pipeline: {e}", exc_info=True)
        error_detail = "An unexpected internal error occurred during processing."
        if isinstance(e, HTTPException):
            error_detail = e.detail
        await queue.put(json.dumps({"status": "error", "message": error_detail, "progress": 100}))
    finally:
        await queue.put("__END__")
        # Cleanup
        if temp_audio_file:
            try:
                temp_audio_file.close()
                os.remove(temp_audio_file.name)
                logger.info(f"Request {request_id}: Cleaned up temporary file {temp_audio_file.name}")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to cleanup temporary file {temp_audio_file.name}: {e}", exc_info=True)


# Main endpoint for audio transcription
@app.post(
    "/v1/transcribe",
    response_model=JobCreationResponse,
    status_code=status.HTTP_202_ACCEPTED,
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
    This endpoint receives an audio file and configuration, starts the processing
    in the background and returns a job ID to track the status via SSE.
    """
    request_id = request.state.request_id
    
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

    # Read the audio file into memory immediately
    audio_data = await audio_file.read()
    content_type = audio_file.content_type

    job_id = str(uuid.uuid4())
    job_status_queues[job_id] = asyncio.Queue()

    background_tasks.add_task(
        run_transcription_pipeline,
        request=request,
        background_tasks=background_tasks,
        job_id=job_id,
        queue=job_status_queues[job_id],
        user_info=user_info,
        audio_data=audio_data,
        content_type=content_type,
        output_format=output_format,
        model=model,
        diarization=diarization,
        specialty=specialty,
        conversation_type=conversation_type,
        language=language,
        output_language=output_language,
        fhir_bundle_type=fhir_bundle_type,
    )

    return JobCreationResponse(job_id=job_id)

async def sse_event_generator(request: Request, job_id: str) -> AsyncGenerator[str, None]:
    """Yields server-sent events for a given job ID."""
    queue = job_status_queues.get(job_id)
    if not queue:
        # This part will not be reached if we raise HTTPException before,
        # but it's good practice.
        logger.warning(f"SSE generator started for non-existent job_id: {job_id}")
        return

    try:
        while True:
            if await request.is_disconnected():
                logger.info(f"Client disconnected from SSE stream for job_id: {job_id}")
                break

            try:
                message = await asyncio.wait_for(queue.get(), timeout=30)
                if message == "__END__":
                    logger.info(f"SSE stream finished for job_id: {job_id}")
                    break
                
                yield f"data: {message}\n\n"
                queue.task_done()
            except asyncio.TimeoutError:
                # Send a keep-alive comment to prevent client/proxy timeouts
                yield ": keep-alive\n\n"

    except asyncio.CancelledError:
        logger.info(f"SSE generator cancelled for job_id: {job_id}")
    finally:
        # Clean up the queue once the client has disconnected or the job is done
        if job_id in job_status_queues:
            # Empty the queue to unblock the background task if it's still putting things
            while not job_status_queues[job_id].empty():
                job_status_queues[job_id].get_nowait()
                job_status_queues[job_id].task_done()
            del job_status_queues[job_id]
            logger.info(f"Cleaned up queue for job_id: {job_id}")

@app.get("/v1/transcribe/status/{job_id}")
async def get_transcription_status(request: Request, job_id: str):
    """
    Endpoint to get real-time status updates for a transcription job using SSE.
    """
    if job_id not in job_status_queues:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job ID not found or already completed.")
    
    return StreamingResponse(
        sse_event_generator(request, job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )

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