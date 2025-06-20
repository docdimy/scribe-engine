"""
Pydantic Models für API Responses
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from app.config import OutputFormat


class TranscriptSegment(BaseModel):
    """Einzelnes Transkript-Segment mit optionaler Sprecher-Info"""
    text: str = Field(description="Transkribierter Text")
    start_time: Optional[float] = Field(default=None, description="Startzeit in Sekunden")
    end_time: Optional[float] = Field(default=None, description="Endzeit in Sekunden")
    speaker: Optional[str] = Field(default=None, description="Sprecher-ID (nur bei Diarisierung)")
    confidence: Optional[float] = Field(default=None, description="Konfidenz-Score (0.0-1.0)")


class TranscriptionResult(BaseModel):
    """Vollständiges Transkriptionsergebnis"""
    provider_transcript_id: Optional[str] = Field(default=None, description="Eindeutige ID des Transkripts beim STT-Provider (z.B. AssemblyAI)")
    full_text: str = Field(description="Vollständiger transkribierter Text")
    segments: List[TranscriptSegment] = Field(
        default=[],
        description="Einzelne Transkript-Segmente"
    )
    language_detected: Optional[str] = Field(default=None, description="Erkannte Sprache (ISO 639-1)")
    confidence: Optional[float] = Field(default=None, description="Durchschnittliche Konfidenz")
    duration: Optional[float] = Field(default=None, description="Audio-Dauer in Sekunden")


class AnalysisResult(BaseModel):
    """LLM-Analyseergebnis"""
    summary: str = Field(description="Zusammenfassung der Konsultation")
    diagnosis: Optional[str] = Field(description="Diagnose oder Verdachtsdiagnose")
    treatment: Optional[str] = Field(description="Behandlungsempfehlungen")
    medication: Optional[str] = Field(description="Medikamentöse Therapie")
    follow_up: Optional[str] = Field(description="Nachsorge-Empfehlungen")
    specialty_notes: Optional[str] = Field(description="Fachspezifische Notizen")
    icd10_codes: Optional[List[str]] = Field(description="Relevante ICD-10 Codes")
    
    
class ScribeResponse(BaseModel):
    """Hauptantwort für Transkriptions- und Analyse-Requests"""
    request_id: str = Field(description="Eindeutige Request-ID")
    timestamp: datetime = Field(description="Verarbeitungszeitpunkt")
    transcript: TranscriptionResult = Field(description="Transkriptionsergebnis")
    analysis: AnalysisResult = Field(description="LLM-Analyseergebnis")
    output_format: OutputFormat = Field(description="Verwendetes Ausgabeformat")
    processing_time_ms: int = Field(description="Verarbeitungszeit in Millisekunden")
    
    # FHIR-spezifische Felder (nur wenn output_format=fhir)
    fhir_bundle: Optional[Dict[str, Any]] = Field(
        default=None,
        description="FHIR R4 Bundle (nur bei FHIR-Output)"
    )
    
    # XML-spezifische Felder (nur wenn output_format=xml)
    xml_content: Optional[str] = Field(
        default=None,
        description="XML-Repräsentation (nur bei XML-Output)"
    )


class HealthCheckResponse(BaseModel):
    """Health Check Response"""
    status: str = Field(description="Service Status (healthy/unhealthy)")
    timestamp: datetime = Field(description="Check-Zeitpunkt")
    version: str = Field(description="Service-Version")
    uptime_seconds: int = Field(description="Uptime in Sekunden")
    
    # Detaillierte Informationen (optional)
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detaillierte Gesundheitsinformationen"
    )


class ErrorResponse(BaseModel):
    """Standardisierte Fehlerantwort"""
    error: str = Field(description="Fehlertyp")
    message: str = Field(description="Fehlerbeschreibung")
    details: Optional[Dict[str, Any]] = Field(description="Zusätzliche Fehlerdetails")
    request_id: Optional[str] = Field(description="Request-ID für Debugging")
    timestamp: datetime = Field(description="Fehlerzeitpunkt")


class RateLimitResponse(BaseModel):
    """Rate Limit Exceeded Response"""
    error: str = Field(default="rate_limit_exceeded")
    message: str = Field(description="Rate Limit Fehlermeldung")
    retry_after: int = Field(description="Sekunden bis zum nächsten Versuch")
    limit: int = Field(description="Request-Limit")
    window: int = Field(description="Zeitfenster in Sekunden")
    timestamp: datetime = Field(description="Fehlerzeitpunkt") 