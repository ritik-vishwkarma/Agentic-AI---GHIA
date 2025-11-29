"""
Text-to-Speech Service for Hindi
Provides voice feedback to patients in their language.
"""
import io
import logging
from typing import Optional
import base64

logger = logging.getLogger(__name__)

# Check for gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS not installed. Run: pip install gTTS")


class TTSService:
    """
    Text-to-Speech service for patient feedback.
    Supports Hindi and other Indian languages.
    """
    
    LANGUAGE_CODES = {
        "hi": "hi",  # Hindi
        "bn": "bn",  # Bengali
        "gu": "gu",  # Gujarati
        "mr": "mr",  # Marathi
        "ta": "ta",  # Tamil
        "te": "te",  # Telugu
        "en": "en",  # English
    }
    
    def __init__(self):
        self.available = GTTS_AVAILABLE
    
    def generate_audio(
        self,
        text: str,
        language: str = "hi"
    ) -> Optional[bytes]:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            language: Language code (hi, bn, etc.)
            
        Returns:
            MP3 audio bytes or None if failed
        """
        if not self.available:
            logger.warning("TTS not available")
            return None
        
        try:
            lang_code = self.LANGUAGE_CODES.get(language, "hi")
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.read()
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    def generate_audio_base64(
        self,
        text: str,
        language: str = "hi"
    ) -> Optional[str]:
        """
        Generate audio and return as base64 string for web embedding.
        """
        audio_bytes = self.generate_audio(text, language)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None
    
    def generate_patient_feedback(
        self,
        risk_level: str,
        recommended_action: str,
        language: str = "hi"
    ) -> Optional[bytes]:
        """
        Generate audio feedback for patient based on intake result.
        """
        if language == "hi":
            if risk_level == "urgent":
                message = f"आपकी स्थिति गंभीर है। कृपया तुरंत डॉक्टर से मिलें। {recommended_action}"
            elif risk_level == "moderate":
                message = f"आपको जल्द डॉक्टर से मिलना चाहिए। {recommended_action}"
            else:
                message = f"आपकी स्थिति ठीक है। {recommended_action}"
        else:
            if risk_level == "urgent":
                message = f"Your condition needs immediate attention. Please see a doctor now. {recommended_action}"
            elif risk_level == "moderate":
                message = f"You should see a doctor soon. {recommended_action}"
            else:
                message = f"Your condition is stable. {recommended_action}"
        
        return self.generate_audio(message, language)


# Singleton
_tts_service: Optional[TTSService] = None

def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service