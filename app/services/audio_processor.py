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
from fastapi import HTTPException, status
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Audio-Validierung und Verarbeitung"""

    async def process_and_save_audio(
        self,
        file: "UploadFile",
        max_duration: int,
        max_size_mb: int,
        supported_types: list[str],
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Validates audio, saves it to a temporary file, and returns its path and metadata.
        """
        audio_data = await file.read()
        content_type = file.content_type

        # 1. Validate Content-Type
        if content_type not in supported_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio format '{content_type}'. Supported: {supported_types}",
            )

        # 2. Validate File Size
        size_mb = len(audio_data) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio file too large ({size_mb:.1f}MB). Maximum: {max_size_mb}MB",
            )

        # 3. Create temp file and extract duration
        extension = self._get_extension_from_content_type(content_type)
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        try:
            duration, metadata = self._extract_metadata(temp_file_path, content_type)

            # 4. Validate Duration
            if duration > max_duration:
                self.cleanup(temp_file_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio duration too long ({duration:.1f}s). Maximum: {max_duration}s",
                )
            
            logger.info(f"Audio validated and saved to temp file: {temp_file_path}")
            return temp_file_path, duration, metadata

        except Exception as e:
            self.cleanup(temp_file_path) # Ensure cleanup on error
            logger.error(f"Failed to process audio file: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read or process audio file: {str(e)}",
            )
            
    def cleanup(self, file_path: Optional[str]):
        """Safely delete a temporary file."""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except OSError as e:
                logger.error(f"Error cleaning up file {file_path}: {e}")

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get a file extension from a MIME type."""
        return {
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/mp4": ".mp4",
            "audio/m4a": ".m4a",
            "audio/ogg": ".ogg",
            "audio/webm": ".webm",
        }.get(content_type, ".tmp")

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