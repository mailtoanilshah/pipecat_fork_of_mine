"""Sarvam AI Speech-to-Text service using raw WebSockets with reconnection support.

This module provides a streaming Speech-to-Text service using Sarvam AI's WebSocket API
with direct WebSocket connection management (not using the SDK). This enables automatic
reconnection on connection failures.
"""

import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

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
from pipecat.services.stt_service import WebsocketSTTService
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


class SarvamSTTWebsocketService(WebsocketSTTService):
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

        super().__init__(sample_rate=sample_rate, reconnect_on_error=True, **kwargs)

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
        self._receive_task = None
        
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
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close WebSocket connection."""
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish WebSocket connection to Sarvam STT API."""
        await super()._connect()
        await self._connect_websocket()
        
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close WebSocket connection and cleanup tasks."""
        await super()._disconnect()
        
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
            
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


    async def _receive_messages(self):
        """Receive and process messages from the WebSocket.
        
        This is called by _receive_task_handler from WebsocketService base class.
        """
        if not self._websocket:
            return
            
        async for message in self._websocket:
            try:
                data = json.loads(message) if isinstance(message, str) else message
                await self._handle_message(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Sarvam STT message: {e}")
            except Exception as e:
                logger.error(f"Error handling Sarvam STT message: {e}")
                raise

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
                await self.stop_ttfb_metrics()
                await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601()))
                
        elif message_type == "events":
            # VAD signal
            event_data = data.get("data", {})
            signal_type = event_data.get("signal_type")
            
            if signal_type == "START_SPEECH":
                logger.debug("User started speaking (VAD)")
                await self.push_frame(UserStartedSpeakingFrame())
                await self.start_ttfb_metrics()
            elif signal_type == "END_SPEECH":
                logger.debug("User stopped speaking (VAD)")
                await self.push_frame(UserStoppedSpeakingFrame())
                
        else:
            logger.debug(f"Received unknown message type: {message_type}")


    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames.

        Handles VAD frames for TTFB tracking when using Pipecat's VAD
        instead of Sarvam's built-in VAD.
        """
        await super().process_frame(frame, direction)

        # Only handle VAD frames when not using Sarvam's VAD signals
        if not self._vad_signals:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                await self.start_ttfb_metrics()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._send_flush()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe
            
        Yields:
            None (transcription results come via WebSocket callbacks)
        """
        if not self._websocket or self._websocket.state != State.OPEN:
            logger.warning("Sarvam STT WebSocket not connected, skipping audio send")
            yield None
            return

        try:
            # Encode audio to base64
            audio_b64 = base64.b64encode(audio).decode("utf-8")
            
            # Prepare message per Sarvam API spec
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
            logger.error(f"Error sending audio to Sarvam STT: {e}")
            yield ErrorFrame(error=f"Error sending audio to Sarvam: {e}", fatal=False)
            return
            
        yield None

    async def _send_flush(self):
        """Send flush signal to force finalize partial transcriptions."""
        if not self._websocket or self._websocket.state != State.OPEN:
            return

        try:
            # Per Sarvam API spec
            flush_message = {"type": "flush"}
            await self._websocket.send(json.dumps(flush_message))
            logger.debug("Sent flush signal to Sarvam STT")
        except Exception as e:
            logger.error(f"Error sending flush signal to Sarvam STT: {e}")

