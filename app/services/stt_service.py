"""
Speech-to-Text Service
Supports OpenAI and AssemblyAI for transcription with diarization
"""

import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any
import httpx
from openai import AsyncOpenAI
import assemblyai as aai
from app.config import settings
from app.models.responses import TranscriptSegment, TranscriptionResult, SpeakerSegment
from app.core.logging import get_logger, audit_logger
from app.core.security import DataEncryption
import io
import json

logger = get_logger(__name__)


class STTService:
    """Speech-to-Text service with multiple provider support"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        aai.settings.api_key = settings.assemblyai_api_key
        self.assemblyai_client = aai.TranscriptionConfig
        self.assemblyai_headers = {
            "authorization": settings.assemblyai_api_key,
            "content-type": "application/json"
        }
        
        # Future: Local Whisper configuration
        # self.local_whisper_enabled = settings.enable_local_whisper
        # self.whisper_model_path = settings.local_whisper_model_path
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: str,
        diarization: bool = False,
        request_id: str = None
    ) -> TranscriptionResult:
        """
        Haupteinstiegspunkt für Audio-Transkription
        
        Args:
            audio_data: Audio-Daten als Bytes
            language: Sprache (ISO 639-1 oder "auto")
            diarization: Sprecher-Diarisierung aktivieren
            request_id: Request-ID für Logging
        """
        
        try:
            if diarization:
                # AssemblyAI für Diarisierung verwenden
                result = await self._transcribe_with_assemblyai(
                    audio_data, language, request_id
                )
            else:
                # OpenAI für Standard-Transkription
                result = await self._transcribe_with_openai(
                    audio_data, language, request_id
                )
            
            # Audit-Logging
            if request_id:
                audit_logger.log_external_api_call(
                    request_id=request_id,
                    service="stt",
                    endpoint="transcribe",
                    response_status=200,
                    response_time_ms=0  # Wird vom Aufrufer gesetzt
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Transkription fehlgeschlagen: {e}")
            if request_id:
                audit_logger.log_error(
                    request_id=request_id,
                    error_type="stt_error",
                    error_message=str(e)
                )
            raise
    
    async def _transcribe_with_openai(
        self,
        audio_data: bytes,
        language: str,
        request_id: str = None
    ) -> TranscriptionResult:
        """Transkription mit OpenAI Whisper"""
        
        # Temporäre Datei für OpenAI erstellen
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # OpenAI API aufrufen mit Retry-Logic
            transcript_response = await self._openai_transcribe_with_retry(
                temp_file_path, language, request_id
            )
            
            # Response verarbeiten
            full_text = transcript_response.text
            detected_language = getattr(transcript_response, 'language', None)
            
            # Segment-basierte Antwort erstellen (OpenAI liefert normalerweise nur Text)
            segments = [
                TranscriptSegment(
                    text=full_text,
                    start_time=None,
                    end_time=None,
                    speaker=None,
                    confidence=None
                )
            ]
            
            result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language_detected=detected_language,
                confidence=None,
                duration=None
            )
            
            logger.info(f"OpenAI Transkription erfolgreich: {len(full_text)} Zeichen")
            return result
            
        finally:
            # Temporäre Datei sicher löschen
            try:
                os.unlink(temp_file_path)
                DataEncryption.secure_delete(audio_data)
            except:
                pass
    
    async def _transcribe_with_assemblyai(
        self,
        audio_data: bytes,
        language: str,
        request_id: str = None
    ) -> TranscriptionResult:
        """Transkription mit AssemblyAI (mit Diarisierung)"""
        
        # Temporäre Datei für AssemblyAI erstellen
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # AssemblyAI Konfiguration
            config = aai.TranscriptionConfig(
                speaker_labels=True,  # Diarisierung aktivieren
                language_code=language if language != "auto" else None,
                auto_chapters=False,
                summarization=False,
                sentiment_analysis=False
            )
            
            # AssemblyAI API aufrufen mit Retry-Logic
            transcript = await self._assemblyai_transcribe_with_retry(
                temp_file_path, config, request_id
            )
            
            # Response verarbeiten
            segments = []
            full_text_parts = []
            
            if hasattr(transcript, 'utterances') and transcript.utterances:
                for utterance in transcript.utterances:
                    segment = TranscriptSegment(
                        text=utterance.text,
                        start_time=utterance.start / 1000.0,  # ms zu s
                        end_time=utterance.end / 1000.0,
                        speaker=f"Speaker {utterance.speaker}",
                        confidence=utterance.confidence
                    )
                    segments.append(segment)
                    full_text_parts.append(f"[{segment.speaker}] {segment.text}")
            else:
                # Fallback ohne Diarisierung
                full_text_parts.append(transcript.text)
                segments.append(
                    TranscriptSegment(
                        text=transcript.text,
                        start_time=None,
                        end_time=None,
                        speaker=None,
                        confidence=transcript.confidence
                    )
                )
            
            full_text = "\n".join(full_text_parts)
            
            result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language_detected=getattr(transcript, 'language_code', language),
                confidence=getattr(transcript, 'confidence', None),
                duration=getattr(transcript, 'audio_duration', None)
            )
            
            logger.info(f"AssemblyAI Transkription erfolgreich: {len(segments)} Segmente")
            return result
            
        finally:
            # Temporäre Datei sicher löschen
            try:
                os.unlink(temp_file_path)
                DataEncryption.secure_delete(audio_data)
            except:
                pass
    
    async def _openai_transcribe_with_retry(
        self,
        file_path: str,
        language: str,
        request_id: str = None,
        max_retries: int = None
    ):
        """OpenAI API mit Retry-Logic"""
        
        max_retries = max_retries or settings.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                with open(file_path, "rb") as audio_file:
                    response = await self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language if language != "auto" else None,
                        response_format="json"
                    )
                    return response
                    
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                # Exponential Backoff
                wait_time = 2 ** attempt
                logger.warning(f"OpenAI API Versuch {attempt + 1} fehlgeschlagen, warte {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def _assemblyai_transcribe_with_retry(
        self,
        file_path: str,
        config: aai.TranscriptionConfig,
        request_id: str = None,
        max_retries: int = None
    ):
        """AssemblyAI API mit Retry-Logic"""
        
        max_retries = max_retries or settings.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(file_path)
                
                # Auf Completion warten
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"AssemblyAI Fehler: {transcript.error}")
                
                return transcript
                
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                # Exponential Backoff
                wait_time = 2 ** attempt
                logger.warning(f"AssemblyAI API Versuch {attempt + 1} fehlgeschlagen, warte {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def transcribe_with_openai(
        self,
        audio_data: bytes,
        language: str = "auto",
        model: str = "gpt-4o-mini-transcribe"  # Updated model name
    ) -> TranscriptionResult:
        """
        Transcribe audio using OpenAI's gpt-4o-mini-transcribe model
        
        Args:
            audio_data: Audio file data
            language: Language code (auto for detection)
            model: Model name (gpt-4o-mini-transcribe)
            
        Returns:
            TranscriptionResult with text and metadata
        """
        
        try:
            # Prepare the audio file for OpenAI API
            files = {
                "file": ("audio.wav", io.BytesIO(audio_data), "audio/wav"),
                "model": (None, model),
                "response_format": (None, "verbose_json"),
                "timestamp_granularities[]": (None, "word")
            }
            
            if language != "auto":
                files["language"] = (None, language)
            
            # Make request to OpenAI API
            async with httpx.AsyncClient(timeout=settings.stt_timeout) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}"
                    },
                    files=files
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    raise Exception(f"OpenAI transcription failed: {response.text}")
                
                result = response.json()
                
                # Parse OpenAI response
                full_text = result.get("text", "")
                language_detected = result.get("language")
                duration = result.get("duration")
                
                # Extract word-level timestamps if available
                words = result.get("words", [])
                
                logger.info(f"OpenAI transcription completed: {len(full_text)} characters")
                
                return TranscriptionResult(
                    full_text=full_text,
                    language_detected=language_detected,
                    duration=duration,
                    word_timestamps=words,
                    confidence_score=None,  # OpenAI doesn't provide overall confidence
                    speakers=None,  # No diarization in OpenAI
                    provider="openai",
                    model_used=model
                )
                
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise
    
    async def transcribe_with_assemblyai(
        self,
        audio_data: bytes,
        language: str = "auto",
        diarization: bool = True,
        model: str = "assemblyai-universal"  # Updated model reference
    ) -> TranscriptionResult:
        """
        Transcribe audio using AssemblyAI Universal model with speaker diarization
        
        Args:
            audio_data: Audio file data
            language: Language code
            diarization: Enable speaker diarization
            model: Model reference (assemblyai-universal)
            
        Returns:
            TranscriptionResult with diarization if enabled
        """
        
        try:
            # Step 1: Upload audio file
            async with httpx.AsyncClient(timeout=60) as client:
                upload_response = await client.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers={"authorization": settings.assemblyai_api_key},
                    data=audio_data
                )
                
                if upload_response.status_code != 200:
                    raise Exception(f"AssemblyAI upload failed: {upload_response.text}")
                
                upload_url = upload_response.json()["upload_url"]
            
            # Step 2: Submit transcription request with Universal model
            transcription_config = {
                "audio_url": upload_url,
                "speaker_labels": diarization,
                "punctuate": True,
                "format_text": True,
                "language_detection": language == "auto",
                "speech_model": "universal"  # Use Universal model
            }
            
            if language != "auto":
                transcription_config["language_code"] = language
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://api.assemblyai.com/v2/transcript",
                    headers=self.assemblyai_headers,
                    json=transcription_config
                )
                
                if response.status_code != 200:
                    raise Exception(f"AssemblyAI transcription request failed: {response.text}")
                
                transcript_id = response.json()["id"]
            
            # Step 3: Poll for completion
            transcript_result = await self._poll_assemblyai_result(transcript_id)
            
            # Parse AssemblyAI response
            full_text = transcript_result.get("text", "")
            language_detected = transcript_result.get("language_code")
            confidence = transcript_result.get("confidence")
            
            # Parse speaker segments if diarization was enabled
            speakers = None
            if diarization and transcript_result.get("utterances"):
                speakers = []
                for utterance in transcript_result["utterances"]:
                    speakers.append(SpeakerSegment(
                        speaker=f"Speaker {utterance['speaker']}",
                        text=utterance["text"],
                        start_time=utterance["start"] / 1000.0,  # Convert ms to seconds
                        end_time=utterance["end"] / 1000.0,
                        confidence=utterance.get("confidence")
                    ))
            
            logger.info(f"AssemblyAI transcription completed: {len(full_text)} characters, {len(speakers) if speakers else 0} speaker segments")
            
            return TranscriptionResult(
                full_text=full_text,
                language_detected=language_detected,
                confidence_score=confidence,
                speakers=speakers,
                provider="assemblyai",
                model_used="universal"  # AssemblyAI Universal model
            )
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}")
            raise
    
    async def _poll_assemblyai_result(self, transcript_id: str) -> Dict[str, Any]:
        """Poll AssemblyAI for transcription completion"""
        
        max_attempts = 60  # 5 minutes with 5-second intervals
        attempt = 0
        
        async with httpx.AsyncClient() as client:
            while attempt < max_attempts:
                response = await client.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=self.assemblyai_headers
                )
                
                if response.status_code != 200:
                    raise Exception(f"AssemblyAI polling failed: {response.text}")
                
                result = response.json()
                status = result.get("status")
                
                if status == "completed":
                    return result
                elif status == "error":
                    error_message = result.get("error", "Unknown error")
                    raise Exception(f"AssemblyAI transcription failed: {error_message}")
                
                # Wait before next poll
                await asyncio.sleep(5)
                attempt += 1
        
        raise Exception("AssemblyAI transcription timeout")
    
    # Future method for local Whisper implementation
    async def transcribe_with_local_whisper(
        self,
        audio_data: bytes,
        language: str = "auto",
        diarization: bool = True
    ) -> TranscriptionResult:
        """
        Future implementation: Transcribe audio using local Whisper installation
        with custom speaker diarization pipeline
        
        This method will be implemented when migrating to local infrastructure.
        Expected features:
        - Local Whisper large-v3 model
        - Custom speaker diarization using pyannote or similar
        - Faster processing without API limits
        - Better privacy (no external API calls)
        """
        raise NotImplementedError("Local Whisper implementation planned for future release")
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available STT models
        
        Returns:
            Dictionary with provider names and available models
        """
        return {
            "openai": ["gpt-4o-mini-transcribe"],
            "assemblyai": ["universal"],
            "local": ["whisper-large-v3"] if settings.enable_local_whisper else []
        } 