"""Sarvam AI Speech-to-Text service using raw WebSockets with reconnection support.

This module provides a streaming Speech-to-Text service using Sarvam AI's WebSocket API
with direct WebSocket connection management (not using the SDK). This enables automatic
reconnection on connection failures, similar to the TTS service implementation.
"""

import asyncio
import base64
import json
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.stt_service import STTService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam, you need to `pip install pipecat-ai[sarvam]`.")
    raise Exception(f"Missing module: {e}")


def language_to_sarvam_language(language: Language) -> str:
    """Convert a Language enum to Sarvam's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Sarvam language code string.
    """
    # Mapping of pipecat Language enum to Sarvam language codes
    LANGUAGE_MAP = {
        Language.BN_IN: "bn-IN",
        Language.GU_IN: "gu-IN",
        Language.HI_IN: "hi-IN",
        Language.KN_IN: "kn-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
        Language.TA_IN: "ta-IN",
        Language.TE_IN: "te-IN",
        Language.PA_IN: "pa-IN",
        Language.OR_IN: "od-IN",
        Language.EN_IN: "en-IN",
        Language.AS_IN: "as-IN",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class SarvamSTTWebsocketService(STTService, WebsocketService):
    """Sarvam speech-to-text service using raw WebSockets with auto-reconnection.

    Provides real-time speech recognition using Sarvam's WebSocket API with
    automatic reconnection on connection failures.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam STT service.

        Parameters:
            language: Target language for transcription. Defaults to None (required for saarika models).
            prompt: Optional prompt to guide translation style/context for STT-Translate models.
                   Only applicable to saaras (STT-Translate) models. Defaults to None.
            vad_signals: Enable VAD signals in response. Defaults to True.
            high_vad_sensitivity: Enable high VAD (Voice Activity Detection) sensitivity. Defaults to None.
        """

        language: Optional[Language] = None
        prompt: Optional[str] = None
        vad_signals: bool = True
        high_vad_sensitivity: bool = None

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saarika:v2.5",
        sample_rate: Optional[int] = None,
        input_audio_codec: str = "wav",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription.
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            input_audio_codec: Audio codec/format of the input file. Defaults to "wav".
            params: Configuration parameters for Sarvam STT service.
            **kwargs: Additional arguments passed to the parent services.
        """
        params = params or SarvamSTTWebsocketService.InputParams()

        # Validate that saaras models don't accept language parameter
        if "saaras" in model.lower():
            if params.language is not None:
                raise ValueError(
                    f"Model '{model}' does not accept language parameter. "
                    "STT-Translate models auto-detect language."
                )

        # Validate that saarika models don't accept prompt parameter
        if "saarika" in model.lower():
            if params.prompt is not None:
                raise ValueError(
                    f"Model '{model}' does not accept prompt parameter. "
                    "Prompts are only supported for STT-Translate models"
                )

        STTService.__init__(self, sample_rate=sample_rate, **kwargs)
        WebsocketService.__init__(self, reconnect_on_error=True, **kwargs)

        self.set_model_name(model)
        self._api_key = api_key
        self._model = model
        self._language_code: Optional[Language] = params.language
        
        # For saarika models, default to "unknown" if language is not provided
        if params.language:
            self._language_string = language_to_sarvam_language(params.language)
        elif "saarika" in model.lower():
            self._language_string = "unknown"
        else:
            self._language_string = None
            
        self._prompt = params.prompt
        self._vad_signals = params.vad_signals
        self._high_vad_sensitivity = params.high_vad_sensitivity
        self._input_audio_codec = input_audio_codec

        # WebSocket connection state
        self._websocket = None
        self._receive_task = None
        self._keepalive_task = None
        
        # SDK headers for identification
        self._sdk_headers = sdk_headers()
        
        logger.info(f"Sarvam STT WebSocket initialized with headers: {self._sdk_headers}")

    def language_to_service_language(self, language: Language) -> str:
        """Convert pipecat Language enum to Sarvam's language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The Sarvam language code string.
        """
        return language_to_sarvam_language(language)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish WebSocket connection."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close WebSocket connection."""
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close WebSocket connection."""
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self):
        """Establish WebSocket connection to Sarvam STT API."""
        await self._connect_websocket()
        await self._start_receive_task()
        await self._start_keepalive_task()

    async def _disconnect(self):
        """Close WebSocket connection and cleanup tasks."""
        await self._stop_keepalive_task()
        await self._stop_receive_task()
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to the Sarvam STT WebSocket endpoint."""
        try:
            # Build WebSocket URL with query parameters
            base_url = "wss://api.sarvam.ai/speech-to-text/ws"
            params = []
            
            if self._language_string:
                params.append(f"language-code={self._language_string}")
            if self._model:
                params.append(f"model={self._model}")
            if self._input_audio_codec:
                params.append(f"input_audio_codec={self._input_audio_codec}")
            if self.sample_rate:
                params.append(f"sample_rate={self.sample_rate}")
            if self._high_vad_sensitivity is not None:
                params.append(f"high_vad_sensitivity={'true' if self._high_vad_sensitivity else 'false'}")
            if self._vad_signals is not None:
                params.append(f"vad_signals={'true' if self._vad_signals else 'false'}")
            
            url = f"{base_url}?{'&'.join(params)}"
            
            # Prepare headers
            headers = {
                "Api-Subscription-Key": self._api_key,
            }
            headers.update(self._sdk_headers)
            
            logger.debug(f"Connecting to Sarvam STT WebSocket: {url}")
            self._websocket = await websocket_connect(url, additional_headers=headers)
            logger.info("Connected to Sarvam STT WebSocket successfully")
            
            await self._call_event_handler("on_connected")
            
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam STT WebSocket: {e}")
            await self._handle_error(f"Connection failed: {e}")
            raise

    async def _disconnect_websocket(self):
        """Disconnect from the Sarvam STT WebSocket."""
        if self._websocket:
            try:
                logger.debug("Disconnecting from Sarvam STT WebSocket")
                if self._websocket.state == State.OPEN:
                    await self._websocket.close()
                self._websocket = None
                await self._call_event_handler("on_disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting from Sarvam STT WebSocket: {e}")

    async def _start_receive_task(self):
        """Start the task to receive messages from WebSocket."""
        if not self._receive_task:
            self._receive_task = asyncio.create_task(self._receive_messages())

    async def _stop_receive_task(self):
        """Stop the receive task."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

    async def _start_keepalive_task(self):
        """Start the keepalive task to prevent connection timeout."""
        if not self._keepalive_task:
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _stop_keepalive_task(self):
        """Stop the keepalive task."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

    async def _keepalive_loop(self):
        """Send periodic ping messages to keep connection alive."""
        try:
            while True:
                await asyncio.sleep(10)  # Ping every 10 seconds (faster than TTS's 20s)
                if self._websocket and self._websocket.state == State.OPEN:
                    try:
                        await self._websocket.ping()
                        logger.debug("Sent keepalive ping to Sarvam STT")
                    except Exception as e:
                        logger.warning(f"Failed to send keepalive ping: {e}")
        except asyncio.CancelledError:
            pass

    async def _receive_messages(self):
        """Receive and process messages from the WebSocket."""
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message) if isinstance(message, str) else message
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Sarvam STT WebSocket connection closed: {e}")
            await self._handle_error(f"Connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            await self._handle_error(f"Receive error: {e}")

    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket messages.
        
        Args:
            data: Parsed JSON message from Sarvam API
        """
        message_type = data.get("type")
        
        if message_type == "data":
            # Transcription result
            transcript = data.get("data", {}).get("transcript", "")
            if transcript:
                logger.debug(f"Received transcription: {transcript}")
                await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601()))
                await self._stop_metrics()
                
        elif message_type == "events":
            # VAD signal
            event_data = data.get("data", {})
            signal_type = event_data.get("signal_type")
            
            if signal_type == "START_SPEECH":
                logger.debug("User started speaking (VAD)")
                await self.push_frame(UserStartedSpeakingFrame())
                await self._start_metrics()
            elif signal_type == "STOP_SPEECH":
                logger.debug("User stopped speaking (VAD)")
                await self.push_frame(UserStoppedSpeakingFrame())
                
        else:
            logger.debug(f"Received unknown message type: {message_type}")

    async def _handle_error(self, error: str):
        """Handle errors and trigger reconnection if enabled.
        
        Args:
            error: Error message
        """
        error_frame = ErrorFrame(error=f"Error sending audio to Sarvam: {error}", fatal=False)
        await self._report_error(error_frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames.

        Handles VAD frames for TTFB tracking when using Pipecat's VAD
        instead of Sarvam's built-in VAD.
        """
        await super().process_frame(frame, direction)

        # Only handle VAD frames when not using Sarvam's VAD signals
        if not self._vad_signals:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                await self._start_metrics()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._send_flush()

    @traced_stt
    async def run_stt(self, audio: bytes) -> None:
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe
        """
        if not self._websocket or self._websocket.state != State.OPEN:
            logger.warning("WebSocket not connected, skipping audio send")
            return

        try:
            # Encode audio to base64
            audio_b64 = base64.b64encode(audio).decode("utf-8")
            
            # Prepare message
            message = {
                "audio": {
                    "data": audio_b64,
                    "sample_rate": self.sample_rate,
                    "encoding": f"audio/{self._input_audio_codec}"
                }
            }
            
            # Send to WebSocket
            await self._websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending audio to Sarvam: {e}")
            await self._handle_error(str(e))

    async def _send_flush(self):
        """Send flush signal to force finalize partial transcriptions."""
        if not self._websocket or self._websocket.state != State.OPEN:
            return

        try:
            flush_message = {"flush": True}
            await self._websocket.send(json.dumps(flush_message))
            logger.debug("Sent flush signal to Sarvam STT")
        except Exception as e:
            logger.error(f"Error sending flush signal: {e}")

    async def _start_metrics(self):
        """Start TTFB metrics tracking."""
        if self.can_generate_metrics():
            await self.start_ttfb_metrics()

    async def _stop_metrics(self):
        """Stop TTFB metrics tracking."""
        if self.can_generate_metrics():
            await self.stop_ttfb_metrics()
