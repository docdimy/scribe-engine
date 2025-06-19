"""
Speech-to-Text Service
Supports OpenAI and AssemblyAI for transcription with diarization
"""

import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any
import httpx
import openai
import assemblyai as aai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile, HTTPException, status
import librosa
import soundfile as sf
import io

# Try to import the specific error class, fall back if it doesn't exist
try:
    from assemblyai.errors import AssemblyAIError
except ImportError:
    AssemblyAIError = Exception

from app.config import settings
from app.models.responses import TranscriptSegment, TranscriptionResult
from app.core.logging import get_logger, audit_logger
from app.core.security import data_encryption

logger = get_logger(__name__)

# Configure external clients
openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
if settings.assemblyai_api_key:
    aai.settings.api_key = settings.assemblyai_api_key

# Define retryable exceptions
RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.NetworkError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError, AssemblyAIError)

class STTService:
    """Service for Speech-to-Text transcription."""

    def _delete_temp_file(self, file_path: str):
        """Safely deletes a temporary file."""
        try:
            os.remove(file_path)
            logger.info(f"Successfully deleted temporary file: {file_path}")
        except OSError as e:
            logger.error(f"Error deleting temporary file {file_path}: {e}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(f"Retrying STT API call, attempt {retry_state.attempt_number}...")
    )
    async def _transcribe_with_openai(
        self, file_path: str, language: str, stt_model: str, stt_prompt: Optional[str]
    ) -> TranscriptionResult:
        """Transcribes audio using OpenAI's Whisper API after decrypting."""
        logger.info(f"Starting transcription with OpenAI for file: {file_path}")
        decrypted_audio = None
        try:
            with open(file_path, "rb") as f:
                encrypted_data = f.read()
            
            logger.info("Decrypting audio data for transcription.")
            decrypted_audio = data_encryption.decrypt_data(encrypted_data)
            logger.info("Audio data successfully decrypted.")

            audio_file = io.BytesIO(decrypted_audio)
            # Provide a filename for the API, helps with format detection.
            # The actual format is determined by the bytes, but filename is good practice.
            audio_file.name = "audio.webm" 

            transcript = await openai_client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                response_format="verbose_json",
                language=language if language != "auto" else None,
                prompt=stt_prompt
            )

            segments = [
                TranscriptSegment(start=seg['start'], end=seg['end'], text=seg['text'])
                for seg in transcript.segments
            ]
            
            result = TranscriptionResult(
                text=transcript.text, 
                segments=segments,
                language_detected=getattr(transcript, 'language', language)
            )
            logger.info(f"OpenAI transcription successful: {len(transcript.text)} characters")
            return result
        except openai.APIError as e:
            logger.error(f"OpenAI API error during transcription: {e}", exc_info=True)
            raise
        finally:
            if decrypted_audio:
                data_encryption.secure_delete(decrypted_audio)
            self._delete_temp_file(file_path)
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(f"Retrying STT API call, attempt {retry_state.attempt_number}...")
    )
    async def _transcribe_with_assemblyai(
        self, file_path: str, diarization: bool, language: str
    ) -> TranscriptionResult:
        """Transcribes audio using AssemblyAI's API after decrypting."""
        logger.info(f"Starting transcription with AssemblyAI for file: {file_path}")
        decrypted_audio = None
        try:
            with open(file_path, "rb") as f:
                encrypted_data = f.read()

            logger.info("Decrypting audio data for transcription.")
            decrypted_audio = data_encryption.decrypt_data(encrypted_data)
            logger.info("Audio data successfully decrypted.")
            
            config = aai.TranscriptionConfig(
                speaker_labels=diarization,
                language_code=language if language != "auto" else None,
            )
            transcriber = aai.Transcriber(config=config)
            
            # AssemblyAI SDK can take raw bytes
            transcript = transcriber.transcribe(decrypted_audio)

            if transcript.status == aai.TranscriptStatus.error:
                raise AssemblyAIError(f"AssemblyAI transcription failed: {transcript.error}")

            if diarization and transcript.utterances:
                 segments = [
                    TranscriptSegment(start=utt.start/1000.0, end=utt.end/1000.0, text=utt.text, speaker=utt.speaker)
                    for utt in transcript.utterances
                ]
            else:
                segments = [
                    TranscriptSegment(start=word.start / 1000.0, end=word.end / 1000.0, text=word.text)
                    for word in transcript.words
                ]

            result = TranscriptionResult(
                text=transcript.text,
                segments=segments,
                language_detected=transcript.language_code if transcript.language_code else language
            )
            logger.info(f"AssemblyAI transcription successful: {len(transcript.text)} characters")
            return result
        except Exception as e:
            logger.error(f"AssemblyAI error during transcription: {e}", exc_info=True)
            raise
        finally:
            if decrypted_audio:
                data_encryption.secure_delete(decrypted_audio)
            self._delete_temp_file(file_path)

    async def transcribe(
        self,
        request_id: str,
        file_path: str,
        stt_provider: str,
        stt_model: str,
        diarization: bool,
        language: str,
        stt_prompt: Optional[str],
    ) -> TranscriptionResult:
        """
        Routes the transcription request to the appropriate provider and model.
        """
        audit_logger.log_transcription_request(
            request_id=request_id,
            provider=stt_provider,
            model=stt_model,
            diarization=diarization,
            language=language
        )

        if stt_provider == "openai":
            return await self._transcribe_with_openai(file_path, language, stt_model, stt_prompt)
        elif stt_provider == "assemblyai":
            if not settings.assemblyai_api_key:
                raise ValueError("AssemblyAI API key is not configured.")
            return await self._transcribe_with_assemblyai(file_path, diarization, language)
        else:
            # This part is for a potential local model. Not implemented.
            logger.error(f"Unsupported STT provider: {stt_provider}")
            raise ValueError(f"Unsupported STT provider: {stt_provider}")

    def get_available_models(self) -> dict:
        """Returns a dictionary of available models per provider."""
        return {
            "openai": ["whisper-1"], 
            "assemblyai": ["default"], 
        }

async def process_and_save_audio(file: UploadFile, specialty: str) -> str:
    """
    Processes the uploaded audio file and saves it temporarily.
    - Validates audio format and content.
    - Encrypts the audio data before saving.
    - Returns the path to the encrypted temporary file.
    """
    logger.info("Starting audio processing...")

    # Validate file type
    if file.content_type not in settings.supported_audio_formats:
        logger.warning(
            f"Unsupported audio format: {file.content_type}. Supported: {settings.supported_audio_formats}"
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: {file.content_type}. Please use one of {settings.supported_audio_formats}",
        )

    try:
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No audio data received.",
            )

        logger.info(f"Received {len(audio_data)} bytes of audio data.")

        # --- Encryption Step ---
        logger.info("Encrypting audio data for secure storage.")
        encrypted_data = data_encryption.encrypt_data(audio_data)
        logger.info("Audio data successfully encrypted.")


        # Create a temporary file to store the encrypted audio
        with NamedTemporaryFile(delete=False, suffix=".enc") as temp_file:
            temp_file.write(encrypted_data)
            temp_file_path = temp_file.name
            logger.info(f"Encrypted audio saved temporarily to {temp_file_path}")

        return temp_file_path

    except Exception as e:
        logger.error(f"Error during audio processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio file.",
        )