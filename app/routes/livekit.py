"""LiveKit API routes for real-time audio"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import uuid
import asyncio

from app.services.livekit import get_livekit_service, LIVEKIT_AVAILABLE
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateRoomRequest(BaseModel):
    patient_id: Optional[str] = None
    phc_id: str = "phc-1"
    language: str = "hi"


class RoomTokenResponse(BaseModel):
    room_name: str
    access_token: str
    livekit_url: str
    patient_id: str


class LiveKitStatusResponse(BaseModel):
    available: bool
    configured: bool
    active_rooms: int
    message: str


# Store room languages and tasks
_room_languages: dict = {}
_bot_tasks: dict = {}


@router.get("/status", response_model=LiveKitStatusResponse)
async def get_livekit_status():
    """Check if LiveKit is available and configured"""
    service = get_livekit_service()
    configured = service.is_configured()
    
    if not LIVEKIT_AVAILABLE:
        message = "LiveKit package not installed. Run: pip install livekit livekit-api"
    elif not configured:
        message = "LiveKit not configured. Add LIVEKIT_API_KEY and LIVEKIT_API_SECRET to .env"
    else:
        message = "LiveKit is ready for real-time audio"
    
    return {
        "available": LIVEKIT_AVAILABLE,
        "configured": configured,
        "active_rooms": len(service.get_active_rooms()),
        "message": message
    }


@router.post("/create-room", response_model=RoomTokenResponse)
async def create_patient_room(request: CreateRoomRequest):
    """Create a LiveKit room for patient audio intake."""
    service = get_livekit_service()
    settings = get_settings()
    
    if not service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="LiveKit not configured"
        )
    
    try:
        patient_id = request.patient_id or f"patient-{uuid.uuid4().hex[:8]}"
        room_name = f"{request.phc_id}-{patient_id}"
        
        # Store language for this room
        _room_languages[room_name] = request.language
        
        token = await service.create_room_token(
            room_name=room_name,
            participant_name=patient_id,
            is_bot=False
        )
        
        logger.info(f"✅ Created room: {room_name} (language: {request.language})")
        
        return {
            "room_name": room_name,
            "access_token": token,
            "livekit_url": settings.livekit_url,
            "patient_id": patient_id
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to create room: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-bot/{room_name}")
async def start_ghia_bot(room_name: str):
    """Start GHIA bot to listen to patient audio in a room."""
    service = get_livekit_service()
    
    if not service.is_configured():
        raise HTTPException(status_code=503, detail="LiveKit not configured")
    
    # Get language for this room
    language = _room_languages.get(room_name, "hi")
    
    async def process_audio(audio_data: bytes, room: str):
        """Process audio when patient finishes speaking"""
        try:
            from app.agents.orchestrator import run_agent_pipeline
            from app.services.asr import transcribe_audio
            
            logger.info(f"🎵 Processing LiveKit audio: {len(audio_data)} bytes from room {room}")
            
            if len(audio_data) < 1000:
                logger.warning("Audio too short, skipping")
                return
            
            # Convert raw audio to WAV
            wav_audio = service.create_wav_from_raw(
                audio_data,
                sample_rate=48000,  # LiveKit default
                channels=1,
                sample_width=2
            )
            
            logger.info(f"📦 Converted to WAV: {len(wav_audio)} bytes")
            
            # Transcribe
            text, lang = transcribe_audio(wav_audio, language)
            
            # Check if we got real transcription
            if not text or "मुझे कमर में" in text:  # Mock response check
                logger.warning("Got mock transcription - audio may be silent or corrupted")
                return
            
            logger.info(f"📝 LiveKit transcription: {text[:100]}...")
            
            # Run agent pipeline
            result = await run_agent_pipeline(text, lang)
            
            logger.info(f"✅ LiveKit intake complete:")
            logger.info(f"   Chief Complaint: {result.get('chief_complaint')}")
            logger.info(f"   Risk Level: {result.get('risk_level')}")
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Cancel existing bot for this room if any
    if room_name in _bot_tasks:
        old_task = _bot_tasks[room_name]
        if not old_task.done():
            old_task.cancel()
    
    # Start bot as asyncio task (not BackgroundTask)
    logger.info(f"🤖 Creating bot task for room: {room_name}")
    
    task = asyncio.create_task(
        service.start_bot_listener(room_name, process_audio)
    )
    _bot_tasks[room_name] = task
    
    # Add callback to cleanup when done
    def cleanup_task(t):
        _bot_tasks.pop(room_name, None)
        _room_languages.pop(room_name, None)
        if t.exception():
            logger.error(f"Bot task error: {t.exception()}")
    
    task.add_done_callback(cleanup_task)
    
    logger.info(f"🤖 Bot task started for room: {room_name}")
    
    return {
        "message": f"GHIA bot starting in room: {room_name}",
        "status": "connecting",
        "language": language
    }


@router.delete("/room/{room_name}")
async def close_room(room_name: str):
    """Close a room and cleanup resources"""
    # Cancel bot task if running
    if room_name in _bot_tasks:
        task = _bot_tasks[room_name]
        if not task.done():
            task.cancel()
        _bot_tasks.pop(room_name, None)
    
    _room_languages.pop(room_name, None)
    
    return {"message": f"Room {room_name} cleanup initiated"}


@router.get("/rooms")
async def list_rooms():
    """List all active rooms"""
    service = get_livekit_service()
    return {
        "rooms": list(service.get_active_rooms().keys()),
        "bot_tasks": list(_bot_tasks.keys())
    }