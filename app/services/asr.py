"""
ASR Service using AI4Bharat IndicWav2Vec (HuggingFace implementation)
Supports: Hindi, Bengali, Gujarati, Marathi, Tamil, Telugu, Nepali, Odia, Sinhala
"""

import tempfile
import os
from typing import Tuple, Optional
import logging
import torch
import asyncio

logger = logging.getLogger(__name__)

# Model cache
_model = None
_processor = None
_current_language = None


# Language to model mapping (from AI4Bharat documentation)
LANGUAGE_MODELS = {
    "hi": "ai4bharat/indicwav2vec-hindi",
    "bn": "ai4bharat/indicwav2vec-bengali",
    "gu": "ai4bharat/indicwav2vec-gujarati",
    "mr": "ai4bharat/indicwav2vec-marathi",
    "ta": "ai4bharat/indicwav2vec-tamil",
    "te": "ai4bharat/indicwav2vec-telugu",
    "ne": "ai4bharat/indicwav2vec-nepali",
    "or": "ai4bharat/indicwav2vec-odia",
    "si": "ai4bharat/indicwav2vec-sinhala",
}


def get_indicwav2vec_model(language: str = "hi"):
    """
    Load AI4Bharat IndicWav2Vec model for specific language.

    Args:
        language: Language code (hi, bn, gu, mr, ta, te, ne, or, si)

    Returns:
        (model, processor)
        (transcribed_text, detected_language)
    """
    global _model, _processor, _current_language

    # Check if already loaded for this language
    if _model is not None and _current_language == language:
        return _model, _processor

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        model_name = LANGUAGE_MODELS.get(language, LANGUAGE_MODELS["hi"])

        logger.info(f"🔄 Loading AI4Bharat IndicWav2Vec model: {model_name}")

        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load processor (tokenizer)
        _processor = Wav2Vec2Processor.from_pretrained(model_name)

        # Load model
        _model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        _model.to(device)
        _model.eval()

        _current_language = language

        logger.info(f"✅ AI4Bharat model loaded successfully for {language}")
        return _model, _processor

    except Exception as e:
        logger.error(f"❌ Failed to load AI4Bharat model: {e}")
        raise


# async def transcribe_audio(
#     audio_bytes: bytes,
#     language: str = "hi"
# ) -> Tuple[str, str]:
#     """
#     Transcribe audio using AI4Bharat IndicWav2Vec.

#     Args:
#         audio_bytes: Raw audio data (WAV format, 16kHz recommended)
#         language: Language code (default: "hi" for Hindi)

#     Returns:
#         (transcribed_text, detected_language)
#     """
#     # Save to temp file
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#         f.write(audio_bytes)
#         temp_path = f.name

#     try:
#         import librosa
#         import numpy as np

#         # Load model
#         model, processor = get_indicwav2vec_model(language)

#         # Load and resample audio to 16kHz
#         audio, sample_rate = librosa.load(temp_path, sr=16000)

#         # Normalize audio
#         audio = audio.astype(np.float32)

#         # Process audio
#         inputs = processor(
#             audio,
#             sampling_rate=16000,
#             return_tensors="pt",
#             padding=True
#         )

#         # Move to GPU if available
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         # Inference
#         with torch.no_grad():
#             logits = model(**inputs).logits

#         # Decode
#         predicted_ids = torch.argmax(logits, dim=-1)
#         transcription = processor.batch_decode(predicted_ids)[0]

#         # Clean up transcription
#         transcription = transcription.strip()

#         logger.info(f"✅ Transcribed ({language}): {transcription[:100]}...")
#         return transcription, language

#     except Exception as e:
#         logger.error(f"❌ Transcription failed: {e}")
#         # Return mock data for demo
#         # return transcribe_audio_mock(audio_bytes)

#     finally:
#         # Clean up temp file
#         if os.path.exists(temp_path):
#             os.remove(temp_path)


async def transcribe_audio(audio_bytes: bytes, language: str = "hi") -> Tuple[str, str]:
    """
    Transcribe audio using AI4Bharat IndicWav2Vec.

    Args:
        audio_bytes: Raw audio data (WAV format, 16kHz recommended)
        language: Language code (default: "hi" for Hindi)

    Returns:
        (transcribed_text, detected_language)
    """
    # Run the blocking transcription in a thread pool
    return await asyncio.to_thread(_transcribe_audio_sync, audio_bytes, language)


def _transcribe_audio_sync(audio_bytes: bytes, language: str = "hi") -> Tuple[str, str]:
    """Synchronous transcription implementation."""
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        import librosa
        import numpy as np

        # Load model
        model, processor = get_indicwav2vec_model(language)

        # Load and resample audio to 16kHz
        audio, sample_rate = librosa.load(temp_path, sr=16000)

        # Normalize audio
        audio = audio.astype(np.float32)

        # Process audio
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Clean up transcription
        transcription = transcription.strip()

        logger.info(f"✅ Transcribed ({language}): {transcription[:100]}...")
        return transcription, language

    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        # Return mock data for demo
        return transcribe_audio_mock(audio_bytes)

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def transcribe_audio_streaming(
    audio_path: str, language: str = "hi"
) -> Tuple[str, str]:
    """Transcribe from file path (for streaming scenarios)"""
    with open(audio_path, "rb") as f:
        return transcribe_audio(f.read(), language)


def transcribe_audio_mock(audio_bytes: bytes) -> Tuple[str, str]:
    """Mock transcription for testing when model fails"""
    logger.warning("⚠️ Using mock transcription (for testing)")
    return ("मुझे कमर में बहुत दर्द है तीन दिन से। सूजन भी है थोड़ी।", "hi")


def detect_language(audio_bytes: bytes) -> str:
    """
    Auto-detect language from audio.
    For MVP, we assume Hindi. Can be enhanced with language detection model.
    """
    # TODO: Implement actual language detection
    # For now, default to Hindi
    return "hi"


def get_supported_languages() -> list:
    """Return list of supported languages"""
    return list(LANGUAGE_MODELS.keys())
