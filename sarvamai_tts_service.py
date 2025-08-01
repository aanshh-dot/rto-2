#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI text-to-speech service implementation."""

import asyncio
import base64
import json
from typing import AsyncGenerator, Optional
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    CancelFrame,
    EndFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import (
    WebsocketTTSService,
)

from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
import websockets


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Sarvam AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Sarvam AI language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.BN: "bn-IN",  # Bengali
        Language.EN: "en-IN",  # English (India)
        Language.GU: "gu-IN",  # Gujarati
        Language.HI: "hi-IN",  # Hindi
        Language.KN: "kn-IN",  # Kannada
        Language.ML: "ml-IN",  # Malayalam
        Language.MR: "mr-IN",  # Marathi
        Language.OR: "od-IN",  # Odia
        Language.PA: "pa-IN",  # Punjabi
        Language.TA: "ta-IN",  # Tamil
        Language.TE: "te-IN",  # Telugu
    }

    return LANGUAGE_MAP.get(language)


class SarvamWebsocketTTSService(WebsocketTTSService):
    """
    Minimalist TTS service for Sarvam AI WebSocket endpoint using Pipecat base class.

    Supports streaming text-to-speech with buffering and real-time audio generation.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "bulbul:v2",
        target_language_code: str = "hi-IN",
        pitch: float = 0.0,
        pace: float = 1.0,
        speaker: str = "anushka",
        loudness: float = 1.0,
        speech_sample_rate: int = 24000,
        enable_preprocessing: bool = False,
        min_buffer_size: int = 30,
        max_chunk_length: int = 200,
        output_audio_codec: str = "linear16",
        output_audio_bitrate: str = "128k",
        **kwargs,
    ):
        # Initialize parent class first
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            sample_rate=speech_sample_rate,
            **kwargs,
        )

        # WebSocket endpoint URL
        self._websocket_url = f"wss://api.sarvam.ai/text-to-speech/ws?model={model}"
        self._api_key = api_key
        self.set_model_name(model)
        self.set_voice(speaker)
        # Configuration parameters
        self._config = {
            "target_language_code": target_language_code,
            "pitch": pitch,
            "pace": pace,
            "speaker": speaker,
            "loudness": loudness,
            "speech_sample_rate": speech_sample_rate,
            "enable_preprocessing": enable_preprocessing,
            "min_buffer_size": min_buffer_size,
            "max_chunk_length": max_chunk_length,
            "output_audio_codec": output_audio_codec,
            "output_audio_bitrate": output_audio_bitrate,
        }
        # self._websocket = None
        self._started = False
        self._cumulative_time = 0

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam TTS service.

        Args:
            frame: The end frame.
        """
        logger.debug("getting stop frame")
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam TTS service.

        Args:
            frame: The cancel frame.
        """
        logger.debug("getting cancel frame")

        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending stop command."""
        if self._websocket:
            msg = {"type": "flush"}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False

    async def _flush_and_reconnect(self):
        """Flush current synthesis and reconnect WebSocket to clear stale requests."""
        try:
            logger.debug("Flushing and reconnecting func")
            if self._websocket:
                # Send flush message if supported
                msg = {"type": "flush"}
                await self._websocket.send(json.dumps(msg))

            # Disconnect and reconnect to clear any pending synthesis
            logger.debug("flush wala disconnect")
            await self._disconnect()
            await self._connect()

        except Exception as e:
            logger.error(f"Error during TTS flush and reconnect: {e}")

    async def _connect(self):
        """Connect to Sarvam WebSocket and start background tasks."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket and clean up tasks."""
        logger.debug("in _disconnect func")
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Sarvam API."""
        try:
            if self._websocket and not self._websocket.closed:
                logger.debug("Webscoket already connected")
                return

            logger.debug("Connecting to Sarvam TTS Websocket")
            subprotocols = [f"api-subscription-key.{self._api_key}"]
            self._websocket = await websockets.connect(
                self._websocket_url,
                subprotocols=subprotocols,
                ping_interval=None,  # We'll handle pings manually
                ping_timeout=10,
                close_timeout=10,
            )
            logger.debug("Connected to Sarvam TTS Websocket")
            await self._send_config()

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _send_config(self):
        """Send initial configuration message."""
        if not self._websocket:
            raise Exception("WebSocket not connected")

        config_message = {"type": "config", "data": self._config}

        try:
            await self._websocket.send(json.dumps(config_message))
            logger.debug("Configuration sent successfully")
        except Exception as e:
            logger.error(f"Failed to send config: {str(e)}")
            await self.push_frame(ErrorFrame(f"Failed to send config: {str(e)}"))
            raise

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Sarvam")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._started = False
            self._websocket = None

    async def _receive_messages(self):
        """Receive and process messages from Sarvam WebSocket."""
        async for message in WatchdogAsyncIterator(
            self._websocket, manager=self.task_manager
        ):
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("type") == "audio":
                    # Check for interruption before processing audio
                    await self.stop_ttfb_metrics()
                    audio = base64.b64decode(msg["data"]["audio"])
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    logger.debug("pushing audio frames from tts to output")
                    await self.push_frame(frame)
                elif msg.get("type") == "error":
                    error_msg = msg["data"]["message"]
                    logger.error(f"TTS Error: {error_msg}")

                    # If it's a timeout error, the connection might need to be reset
                    if (
                        "too long" in error_msg.lower()
                        or "timeout" in error_msg.lower()
                    ):
                        logger.warning(
                            "Connection timeout detected, service may need restart"
                        )

                    await self.push_frame(ErrorFrame(f"TTS Error: {error_msg}"))

                elif msg.get("type") == "pong":
                    logger.debug("Received pong message")

    async def _keepalive_task_handler(self):
        """Handle keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10 if self.task_manager.task_watchdog_enabled else 9
        while True:
            self.reset_watchdog()
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive message to maintain connection."""
        if self._websocket:
            # Send empty text for keepalive
            msg = {"type": "ping"}
            await self._websocket.send(json.dumps(msg))

    async def _send_text(self, text: str):
        """Send text to Neuphonic WebSocket for synthesis."""
        if self._websocket:
            msg = {"type": "text", "data": {"text": text}}
            logger.debug(f"Sending text to websocket: {msg}")
            await self._websocket.send(json.dumps(msg))
            msg = {"type": "flush"}
            await self._websocket.send(json.dumps(msg))


    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Neuphonic's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """

        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0
                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                logger.debug("error wala disconnect")
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
