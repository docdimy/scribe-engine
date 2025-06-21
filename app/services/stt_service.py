"""
Speech-to-Text Service
Uses AssemblyAI for transcription with diarization support.
"""

import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any
import httpx
import assemblyai as aai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile, HTTPException, status
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

# Configure AssemblyAI client
if settings.assemblyai_api_key:
    aai.settings.api_key = settings.assemblyai_api_key
    aai.settings.base_url = settings.assemblyai_api_base_url
    logger.info(f"Configured AssemblyAI client to use base URL: {settings.assemblyai_api_base_url}")

# Define retryable exceptions for network issues or server errors
RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.NetworkError, AssemblyAIError)


class STTService:
    """Service for Speech-to-Text transcription using AssemblyAI."""

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=lambda retry_state: logger.warning(f"Retrying AssemblyAI API call, attempt {retry_state.attempt_number}...")
    )
    async def transcribe(
        self,
        request_id: str,
        file_path: str,
        diarization: bool,
        language: str,
        stt_prompt: Optional[str] = None # Added for future use with AssemblyAI prompting
    ) -> aai.Transcript:
        """
        Transcribes audio using AssemblyAI's API and returns the raw transcript object.
        The audio file at file_path is expected to be encrypted.
        """
        if not settings.assemblyai_api_key:
            raise ValueError("AssemblyAI API key is not configured.")
            
        logger.info(f"Starting transcription with AssemblyAI for file: {file_path}")
        decrypted_audio = None
        
        audit_logger.log_transcription_request(
            request_id=request_id,
            provider="assemblyai",
            model="universal (default)", # Using the default recommended model from AssemblyAI
            diarization=diarization,
            language=language
        )

        try:
            with open(file_path, "rb") as f:
                encrypted_data = f.read()

            logger.info("Decrypting audio data for transcription.")
            decrypted_audio = data_encryption.decrypt_data(encrypted_data)
            logger.info("Audio data successfully decrypted.")
            
            config_params = {"speaker_labels": diarization}
            
            if language == "auto":
                config_params["language_detection"] = True
            else:
                config_params["language_code"] = language

            # Future: Add prompt support when needed
            # if stt_prompt:
            #     config_params["prompt"] = stt_prompt

            config = aai.TranscriptionConfig(**config_params)
            transcriber = aai.Transcriber(config=config)
            
            transcript = transcriber.transcribe(decrypted_audio)

            if transcript.status == aai.TranscriptStatus.error:
                logger.error(f"AssemblyAI transcription failed: {transcript.error}")
                raise AssemblyAIError(f"AssemblyAI transcription failed: {transcript.error}")

            # Log the link between our request ID and AssemblyAI's transcript ID for traceability
            logger.info(f"[{request_id}] AssemblyAI transcript created with ID: {transcript.id}")
            logger.info(f"AssemblyAI transcription successful for ID {transcript.id}: {len(transcript.text or '')} characters")
            
            return transcript
        except Exception as e:
            logger.error(f"Error during AssemblyAI transcription: {e}", exc_info=True)
            raise
        finally:
            if decrypted_audio:
                # Ensure the decrypted data (in-memory bytes) is securely cleared
                data_encryption.secure_delete(decrypted_audio)

    def create_transcription_result(
        self, 
        transcript: aai.Transcript, 
        speaker_map: Optional[Dict[str, str]] = None
    ) -> TranscriptionResult:
        """
        Converts a raw AssemblyAI transcript into our internal TranscriptionResult model,
        applying speaker labels from the speaker_map if provided.
        """
        speaker_map = speaker_map or {}
        
        if transcript.config.speaker_labels and transcript.utterances:
            segments = [
                TranscriptSegment(
                    start=utt.start/1000.0, 
                    end=utt.end/1000.0, 
                    text=utt.text, 
                    speaker=speaker_map.get(utt.speaker, utt.speaker) # Use mapped speaker, fallback to original
                )
                for utt in transcript.utterances
            ]
        else:
            # Fallback for non-diarized text (though this path is less likely if we enable LeMUR only with diarization)
            segments = [
                TranscriptSegment(start=word.start / 1000.0, end=word.end / 1000.0, text=word.text)
                for word in transcript.words
            ]

        # Correctly determine the detected language from the transcript's config.
        detected_language_val = None
        if transcript.config and transcript.config.language_code:
            lang_code = transcript.config.language_code
            if hasattr(lang_code, 'value'):
                detected_language_val = lang_code.value
            else:
                detected_language_val = str(lang_code)

        return TranscriptionResult(
            provider_transcript_id=transcript.id,
            full_text=transcript.text or "",
            segments=segments,
            language_detected=detected_language_val
        )

    async def delete_transcript(self, transcript_id: str):
        """
        Deletes a transcript from AssemblyAI's servers.
        This is a 'fire-and-forget' operation from the user's perspective.
        """
        if not transcript_id:
            return
        
        logger.info(f"Initiating deletion of AssemblyAI transcript ID: {transcript_id}")
        try:
            # Use the static method `delete_by_id` and run it in a thread
            # to avoid blocking the async event loop.
            await asyncio.to_thread(aai.Transcript.delete_by_id, transcript_id)
            logger.info(f"Successfully deleted AssemblyAI transcript ID: {transcript_id}")
        except Exception as e:
            # We log the error but do not re-raise it, as this is a background task.
            logger.error(f"Failed to delete AssemblyAI transcript ID {transcript_id}: {e}", exc_info=True)


async def process_and_save_audio(audio_data: bytes, content_type: str, specialty: str) -> NamedTemporaryFile:
    """
    Processes uploaded audio data and saves it to a temporary file.
    - Validates audio format and content.
    - Encrypts the audio data before saving.
    - Returns the open NamedTemporaryFile object.
    """
    logger.info("Starting audio processing...")

    if content_type not in settings.supported_audio_formats:
        logger.warning(f"Unsupported audio format: {content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: {content_type}",
        )
    
    temp_file = None
    try:
        if not audio_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No audio data.")

        logger.info(f"Received {len(audio_data)} bytes of audio data.")

        logger.info("Encrypting audio data for secure storage.")
        encrypted_data = data_encryption.encrypt_data(audio_data)
        logger.info("Audio data successfully encrypted.")

        temp_file = NamedTemporaryFile(delete=False, suffix=".enc")
        temp_file.write(encrypted_data)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        logger.info(f"Encrypted audio saved temporarily to {temp_file.name}")
        
        return temp_file

    except Exception as e:
        if temp_file:
            temp_file.close()
            os.remove(temp_file.name)
        logger.error(f"Error during audio processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process audio file.",
        )