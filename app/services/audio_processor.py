"""
Audio Processing und Validierung
"""

import asyncio
from typing import Tuple, Optional
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.mp4 import MP4
from fastapi import HTTPException, status
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Audio-Validierung und Metadaten-Extraktion"""
    
    @staticmethod
    async def validate_audio_content(audio_data: bytes, content_type: str) -> Tuple[float, bool]:
        """
        Validiert Audio-Inhalt und extrahiert Metadaten
        
        Returns:
            Tuple[duration_seconds, is_valid]
        """
        try:
            # Content-Type Validierung
            if content_type not in settings.supported_audio_types:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Audio-Format '{content_type}' wird nicht unterstützt. "
                           f"Unterstützte Formate: {settings.supported_audio_types}"
                )
            
            # Dateigröße prüfen
            size_mb = len(audio_data) / (1024 * 1024)
            if size_mb > settings.max_file_size_mb:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio-Datei zu groß ({size_mb:.1f}MB). "
                           f"Maximum: {settings.max_file_size_mb}MB"
                )
            
            # Audio-Dauer extrahieren
            duration = await AudioProcessor._extract_duration(audio_data, content_type)
            
            # Dauer-Validierung
            if duration > settings.max_audio_duration:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio-Dauer zu lang ({duration:.1f}s). "
                           f"Maximum: {settings.max_audio_duration}s"
                )
            
            logger.info(
                f"Audio validiert: {duration:.1f}s, {size_mb:.1f}MB, {content_type}"
            )
            
            return duration, True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio-Validierung fehlgeschlagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio-Validierung fehlgeschlagen: {str(e)}"
            )
    
    @staticmethod
    async def _extract_duration(audio_data: bytes, content_type: str) -> float:
        """Extrahiert die Audio-Dauer aus den Metadaten"""
        
        # Temporäre Datei erstellen für Mutagen
        import tempfile
        import os
        
        # Dateierweiterung basierend auf Content-Type bestimmen
        extension_map = {
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/mp4": ".m4a",
            "audio/m4a": ".m4a",
            "audio/ogg": ".ogg"
        }
        
        extension = extension_map.get(content_type, ".tmp")
        
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Metadaten mit Mutagen extrahieren
            audio_file = MutagenFile(temp_file_path)
            
            if audio_file is None:
                raise ValueError("Ungültige Audio-Datei oder nicht unterstütztes Format")
            
            duration = getattr(audio_file, 'info', None)
            if duration and hasattr(duration, 'length'):
                return float(duration.length)
            
            # Fallback: Versuch verschiedene Formate
            duration = AudioProcessor._try_specific_formats(temp_file_path, content_type)
            if duration:
                return duration
            
            raise ValueError("Konnte Audio-Dauer nicht bestimmen")
            
        finally:
            # Temporäre Datei löschen
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    @staticmethod
    def _try_specific_formats(file_path: str, content_type: str) -> Optional[float]:
        """Versucht format-spezifische Dauer-Extraktion"""
        
        try:
            if content_type == "audio/mpeg":
                audio = MP3(file_path)
                return float(audio.info.length) if hasattr(audio.info, 'length') else None
                
            elif content_type == "audio/wav":
                audio = WAVE(file_path)
                return float(audio.info.length) if hasattr(audio.info, 'length') else None
                
            elif content_type in ["audio/mp4", "audio/m4a"]:
                audio = MP4(file_path)
                return float(audio.info.length) if hasattr(audio.info, 'length') else None
                
        except Exception as e:
            logger.warning(f"Format-spezifische Dauer-Extraktion fehlgeschlagen: {e}")
            return None
        
        return None
    
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