"""
LiveKit service for real-time audio streaming.
Handles room creation, token generation, and audio processing.
"""
import asyncio
import logging
from typing import Optional, Dict, Callable, Any
from datetime import timedelta
import io
import wave
import struct

from app.config import get_settings

logger = logging.getLogger(__name__)

# Check if livekit packages are available
try:
    from livekit import api, rtc
    LIVEKIT_AVAILABLE = True
    logger.info("✅ LiveKit packages loaded")
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    logger.warning(f"⚠️ LiveKit not available: {e}")
    logger.warning("Install with: pip install livekit livekit-api")


class LiveKitService:
    """Service for managing LiveKit rooms and audio processing"""
    
    def __init__(self):
        self.settings = get_settings()
        self._active_rooms: Dict[str, dict] = {}
        self._audio_buffers: Dict[str, bytearray] = {}
        self._room_connections: Dict[str, Any] = {}
    
    def is_configured(self) -> bool:
        """Check if LiveKit is properly configured"""
        if not LIVEKIT_AVAILABLE:
            return False
        return bool(
            self.settings.livekit_api_key and 
            self.settings.livekit_api_secret and
            self.settings.livekit_url
        )
    
    async def create_room_token(
        self,
        room_name: str,
        participant_name: str,
        is_bot: bool = False
    ) -> str:
        """
        Create access token for a participant to join a room.
        """
        if not LIVEKIT_AVAILABLE:
            raise RuntimeError("LiveKit not installed")
        
        if not self.is_configured():
            raise RuntimeError("LiveKit not configured")
        
        # Create token with appropriate permissions
        token = api.AccessToken(
            self.settings.livekit_api_key,
            self.settings.livekit_api_secret
        )
        
        token.with_identity(participant_name)
        token.with_name(participant_name)
        
        # Grant permissions
        grant = api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        token.with_grants(grant)
        
        # Set expiry (1 hour)
        token.with_ttl(timedelta(hours=1))
        
        jwt_token = token.to_jwt()
        
        # Track room
        self._active_rooms[room_name] = {
            "participants": [participant_name],
            "created_at": asyncio.get_event_loop().time()
        }
        
        logger.info(f"✅ Created token for {participant_name} in room {room_name}")
        return jwt_token
    
    def get_active_rooms(self) -> Dict[str, dict]:
        """Get all active rooms"""
        return self._active_rooms.copy()
    
    async def start_bot_listener(
        self,
        room_name: str,
        on_audio_complete: Callable[[bytes, str], Any]
    ):
        """
        Start a bot that listens to audio in a room.
        When the patient disconnects, processes the collected audio.
        """
        if not LIVEKIT_AVAILABLE:
            logger.error("LiveKit not available")
            return
        
        logger.info(f"🤖 Starting bot listener for room: {room_name}")
        
        # Initialize audio buffer for this room
        self._audio_buffers[room_name] = bytearray()
        
        try:
            # Create room connection
            room = rtc.Room()
            self._room_connections[room_name] = room
            
            # Track state
            patient_connected = False
            audio_track = None
            
            @room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant
            ):
                nonlocal patient_connected, audio_track
                
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    patient_connected = True
                    audio_track = track
                    logger.info(f"🎤 Subscribed to audio from: {participant.identity}")
                    
                    # Start receiving audio frames
                    asyncio.create_task(
                        self._process_audio_stream(room_name, track)
                    )
            
            @room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                nonlocal patient_connected
                
                logger.info(f"👋 Participant disconnected: {participant.identity}")
                
                if patient_connected and not participant.identity.startswith("bot-"):
                    patient_connected = False
                    
                    # Process collected audio
                    audio_data = bytes(self._audio_buffers.get(room_name, b''))
                    
                    if len(audio_data) > 1000:
                        logger.info(f"🎵 Processing {len(audio_data)} bytes of audio from room {room_name}")
                        asyncio.create_task(on_audio_complete(audio_data, room_name))
                    else:
                        logger.warning(f"⚠️ Audio too short: {len(audio_data)} bytes")
                    
                    # Cleanup
                    self._audio_buffers.pop(room_name, None)
            
            @room.on("disconnected")
            def on_disconnected():
                logger.info(f"🔌 Bot disconnected from room: {room_name}")
                self._room_connections.pop(room_name, None)
                self._audio_buffers.pop(room_name, None)
            
            # Create bot token
            bot_token = await self.create_room_token(
                room_name=room_name,
                participant_name=f"bot-{room_name}",
                is_bot=True
            )
            
            # Connect to room
            logger.info(f"🔗 Bot connecting to: {self.settings.livekit_url}")
            await room.connect(self.settings.livekit_url, bot_token)
            logger.info(f"✅ Bot connected to room: {room_name}")
            
            # Keep connection alive until room is empty or timeout
            timeout = 300  # 5 minutes max
            start_time = asyncio.get_event_loop().time()
            
            while room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                await asyncio.sleep(1)
                
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.info(f"⏰ Room {room_name} timeout, disconnecting bot")
                    break
                
                # Check if room is empty (only bot left)
                if len(room.remote_participants) == 0 and patient_connected:
                    logger.info(f"📭 Room {room_name} is empty, processing audio...")
                    await asyncio.sleep(2)  # Wait for final audio
                    break
            
            # Disconnect
            await room.disconnect()
            
        except Exception as e:
            logger.error(f"❌ Bot listener error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            self._room_connections.pop(room_name, None)
            self._audio_buffers.pop(room_name, None)
            self._active_rooms.pop(room_name, None)
    
    async def _process_audio_stream(self, room_name: str, track: rtc.Track):
        """Process incoming audio frames from a track"""
        try:
            audio_stream = rtc.AudioStream(track)
            
            async for frame_event in audio_stream:
                frame = frame_event.frame
                
                # Convert audio frame to bytes
                audio_bytes = self._frame_to_bytes(frame)
                
                # Append to buffer
                if room_name in self._audio_buffers:
                    self._audio_buffers[room_name].extend(audio_bytes)
                
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
    
    def _frame_to_bytes(self, frame: rtc.AudioFrame) -> bytes:
        """Convert LiveKit AudioFrame to raw bytes"""
        try:
            # Get raw audio data
            data = frame.data
            
            # Convert to bytes if needed
            if isinstance(data, memoryview):
                return bytes(data)
            elif isinstance(data, bytes):
                return data
            else:
                # Convert samples to bytes
                samples = list(data)
                return struct.pack(f'{len(samples)}h', *samples)
                
        except Exception as e:
            logger.error(f"Frame conversion error: {e}")
            return b''
    
    def create_wav_from_raw(
        self, 
        raw_audio: bytes, 
        sample_rate: int = 48000,
        channels: int = 1,
        sample_width: int = 2
    ) -> bytes:
        """Convert raw PCM audio to WAV format"""
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
        
        wav_buffer.seek(0)
        return wav_buffer.read()


# Singleton instance
_livekit_service: Optional[LiveKitService] = None


def get_livekit_service() -> LiveKitService:
    """Get or create LiveKit service instance"""
    global _livekit_service
    if _livekit_service is None:
        _livekit_service = LiveKitService()
    return _livekit_service