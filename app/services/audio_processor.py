"""
Audio Processing und Validierung
"""

import asyncio
import tempfile
import os
from typing import Tuple, Optional, Dict, Any
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.mp4 import MP4
from fastapi import HTTPException, status, UploadFile
from app.config import settings
from app.core.logging import get_logger
from app.core.security import data_encryption

logger = get_logger(__name__)


class AudioProcessor:
    """Audio-Validierung und Verarbeitung"""

    async def process_and_save_audio(self, file: "UploadFile", specialty: str) -> str:
        """
        Processes the uploaded audio file and saves it temporarily.
        - Validates audio format and content.
        - Converts to a standard format (if necessary).
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".enc") as temp_file:
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

    async def cleanup(self, file_path: Optional[str]):
        """Safely delete the temporary file."""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except OSError as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Maps content type to file extension."""
        return {
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/mp4": ".mp4",
            "audio/m4a": ".m4a",
            "audio/ogg": ".ogg",
            "audio/webm": ".webm",
        }.get(content_type, ".tmp")

    def _detect_content_type_from_data(self, audio_data: bytes, filename: Optional[str]) -> str:
        """Detects Content-Type based on file signature or filename."""
        # Check for MP4/M4A first, as 'ftyp' can be a few bytes in
        if b'ftyp' in audio_data[4:12]:
            logger.info("Detected content type: audio/mp4 (ftyp signature)")
            return "audio/mp4"

        signatures = {
            b'ID3': "audio/mpeg",      # MP3 with ID3 Tag
            b'\xff\xfb': "audio/mpeg",  # MP3 frame
            b'\xff\xf3': "audio/mpeg",  # MP3 frame
            b'\xff\xf2': "audio/mpeg",  # MP3 frame
            b'RIFF': "audio/wav",      # WAV
            b'OggS': "audio/ogg",      # OGG
        }

        for signature, detected_type in signatures.items():
            if audio_data.startswith(signature):
                logger.info(f"Detected content type: {detected_type} (signature)")
                return detected_type

        # Fallback to filename extension if detection fails
        if filename:
            ext_map = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.mp4': 'audio/mp4',
                '.ogg': 'audio/ogg',
            }
            _, ext = os.path.splitext(filename)
            if ext.lower() in ext_map:
                logger.info(f"Guessed content type from filename: {ext_map[ext.lower()]}")
                return ext_map[ext.lower()]

        logger.warning("Could not detect specific audio type. Falling back to 'application/octet-stream'.")
        return "application/octet-stream"

    def _extract_metadata(self, file_path: str, content_type: str) -> Tuple[float, Dict[str, Any]]:
        """Extracts duration and other metadata using mutagen."""
        try:
            audio = MutagenFile(file_path)
            if audio is None:
                raise ValueError("Could not load audio file with mutagen.")
            
            duration = audio.info.length if hasattr(audio.info, 'length') else 0.0
            
            metadata = {
                "duration_seconds": duration,
                "bitrate": getattr(audio.info, 'bitrate', None),
                "sample_rate": getattr(audio.info, 'sample_rate', None),
                "channels": getattr(audio.info, 'channels', None),
                "content_type": content_type
            }
            return float(duration), metadata
        except Exception as e:
            logger.warning(f"Could not extract metadata using mutagen: {e}")
            # Fallback if mutagen fails
            return 0.0, {"content_type": content_type}
    
    @staticmethod
    def detect_content_type(audio_data: bytes) -> str:
        """Erkennt Content-Type basierend auf Datei-Signature"""
        
        # Datei-Signaturen für verschiedene Audio-Formate
        signatures = {
            b'ID3': "audio/mpeg",  # MP3 mit ID3 Tag
            b'\xff\xfb': "audio/mpeg",  # MP3
            b'\xff\xf3': "audio/mpeg",  # MP3
            b'\xff\xf2': "audio/mpeg",  # MP3
            b'RIFF': "audio/wav",  # WAV
            b'ftyp': "audio/mp4",  # MP4/M4A
            b'OggS': "audio/ogg",  # OGG
        }
        
        # Prüfe erste Bytes
        for signature, content_type in signatures.items():
            if audio_data.startswith(signature):
                return content_type
            # Prüfe auch nach ein paar Bytes (für MP4/M4A)
            if signature in audio_data[:12]:
                return content_type
        
        # Fallback
        return "audio/mpeg" 