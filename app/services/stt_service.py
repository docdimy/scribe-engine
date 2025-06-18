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
from app.models.responses import TranscriptSegment, TranscriptionResult
from app.core.logging import get_logger, audit_logger
from app.core.security import DataEncryption
import io
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = get_logger(__name__)

# Define which exceptions should trigger a retry
retryable_exceptions = (
    httpx.TimeoutException,
    aai.errors.AssemblyAIError, # General AssemblyAI errors
    # Add other transient exceptions if needed
)

# Define retry condition for OpenAI server errors (5xx)
def is_openai_server_error(exception):
    """Return True if the exception is an OpenAI 5xx error"""
    from openai import APIStatusError
    return isinstance(exception, APIStatusError) and exception.status_code >= 500


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
    
    async def transcribe(
        self,
        file_path: str,
        language: str,
        diarization: bool = False
    ) -> TranscriptionResult:
        """
        Main entry point for audio transcription.
        Selects the STT provider based on the diarization parameter.
        """
        logger.info(f"Starting transcription. Language: {language}, Diarization: {diarization}")

        if diarization:
            logger.info("Diarization requested, using AssemblyAI.")
            return await self._transcribe_with_assemblyai(file_path, language, diarization=True)
        else:
            logger.info("No diarization, using OpenAI.")
            return await self._transcribe_with_openai(file_path, language)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=(retry_if_exception_type(retryable_exceptions) | retry_if_exception_type(is_openai_server_error))
    )
    async def _transcribe_with_openai(
        self,
        file_path: str,
        language: str
    ) -> TranscriptionResult:
        """Transcription with OpenAI Whisper"""
        try:
            transcript_response = await self._openai_transcribe_with_retry(
                file_path, language
            )
            
            full_text = transcript_response.text
            
            segments = [
                TranscriptSegment(text=full_text)
            ]
            
            result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language_detected=getattr(transcript_response, 'language', language)
            )
            
            logger.info(f"OpenAI transcription successful: {len(full_text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}", exc_info=True)
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(retryable_exceptions)
    )
    async def _transcribe_with_assemblyai(
        self,
        file_path: str,
        language: str,
        diarization: bool = True
    ) -> TranscriptionResult:
        """Transcription with AssemblyAI (with diarization)"""
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=diarization,
                language_code=language if language != "auto" else None
            )
            
            transcript = await self._assemblyai_transcribe_with_retry(
                file_path, config
            )
            
            segments = []
            full_text_parts = []
            
            if diarization and hasattr(transcript, 'utterances') and transcript.utterances:
                for utterance in transcript.utterances:
                    segment = TranscriptSegment(
                        text=utterance.text,
                        start_time=utterance.start / 1000.0,
                        end_time=utterance.end / 1000.0,
                        speaker=f"Speaker {utterance.speaker}",
                        confidence=utterance.confidence
                    )
                    segments.append(segment)
                    full_text_parts.append(f"[{segment.speaker}] {segment.text}")
                full_text = "\n".join(full_text_parts)
            else:
                full_text = transcript.text
                segments.append(
                    TranscriptSegment(
                        text=transcript.text,
                        confidence=transcript.confidence
                    )
                )
            
            result = TranscriptionResult(
                full_text=full_text,
                segments=segments,
                language_detected=getattr(transcript, 'language_code', language),
                confidence=getattr(transcript, 'confidence', None),
                duration=getattr(transcript, 'audio_duration', None)
            )
            
            logger.info(f"AssemblyAI transcription successful: {len(segments)} segments")
            return result
            
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {e}", exc_info=True)
            raise

    async def _openai_transcribe_with_retry(
        self,
        file_path: str,
        language: str,
        max_retries: int = None
    ):
        """OpenAI API with Retry-Logic"""
        
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
                logger.warning(f"OpenAI API attempt {attempt + 1} failed, waiting {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
    
    async def _assemblyai_transcribe_with_retry(
        self,
        file_path: str,
        config: aai.TranscriptionConfig,
        max_retries: int = None
    ):
        """AssemblyAI API with Retry-Logic"""
        
        max_retries = max_retries or settings.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                transcriber = aai.Transcriber(config=config)
                transcript = transcriber.transcribe(file_path)
                
                # Wait for completion
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"AssemblyAI error: {transcript.error}")
                
                return transcript
                
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                # Exponential Backoff
                wait_time = 2 ** attempt
                logger.warning(f"AssemblyAI API attempt {attempt + 1} failed, waiting {wait_time}s: {e}")
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