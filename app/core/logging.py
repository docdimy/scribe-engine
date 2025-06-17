"""
Strukturiertes Logging Setup für Scribe Engine
"""

import sys
import structlog
from datetime import datetime
from typing import Dict, Any
from app.config import settings


def setup_logging():
    """Konfiguriert strukturiertes Logging"""
    
    # Timestamper für konsistente Zeitstempel
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    
    # Processor-Chain definieren
    processors = [
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if settings.environment == "development":
        # Development: Colored console output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:
        # Production: JSON output
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    
    # Structlog konfigurieren
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, settings.log_level.upper(), 20)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """Erstellt einen konfigurierten Logger"""
    return structlog.get_logger(name or __name__)


class AuditLogger:
    """Spezieller Logger für Audit-Events"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_api_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        api_key_hash: str = None,
        user_agent: str = None,
        ip_address: str = None,
        **kwargs
    ):
        """Loggt API-Anfragen für Audit-Zwecke"""
        self.logger.info(
            "api_request",
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            api_key_hash=api_key_hash,
            user_agent=user_agent,
            ip_address=ip_address,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_audio_processing(
        self,
        request_id: str,
        audio_duration: float,
        audio_size_bytes: int,
        language: str,
        model_used: str,
        processing_time_ms: int,
        **kwargs
    ):
        """Loggt Audio-Verarbeitungsevents"""
        self.logger.info(
            "audio_processing",
            request_id=request_id,
            audio_duration=audio_duration,
            audio_size_bytes=audio_size_bytes,
            language=language,
            model_used=model_used,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_external_api_call(
        self,
        request_id: str,
        service: str,
        endpoint: str,
        response_status: int,
        response_time_ms: int,
        **kwargs
    ):
        """Loggt Calls zu externen APIs"""
        self.logger.info(
            "external_api_call",
            request_id=request_id,
            service=service,
            endpoint=endpoint,
            response_status=response_status,
            response_time_ms=response_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        stack_trace: str = None,
        **kwargs
    ):
        """Loggt Fehler-Events"""
        self.logger.error(
            "error_event",
            request_id=request_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )


# Global audit logger instance
audit_logger = AuditLogger() 