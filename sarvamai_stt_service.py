import asyncio
import base64
from enum import StrEnum
import json
from typing import Literal, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    StartInterruptionFrame,
    VADUserStoppedSpeakingFrame,
    StopInterruptionFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pydantic import BaseModel

try:
    import httpx
    import websockets
    from sarvamai import AsyncSarvamAI
    from sarvamai.speech_to_text_translate_streaming.socket_client import (
        AsyncSpeechToTextTranslateStreamingSocketClient,
    )
    from sarvamai.speech_to_text_streaming.socket_client import (
        AsyncSpeechToTextStreamingSocketClient,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Sarvam, you need to `pip install sarvamai websockets httpx`."
    )
    raise Exception(f"Missing module: {e}")


class TranscriptionMetrics(BaseModel):
    audio_duration: float
    processing_latency: float


class TranscriptionData(BaseModel):
    request_id: str
    transcript: str
    language_code: Optional[str]
    metrics: Optional[TranscriptionMetrics] = None
    is_final: Optional[bool] = None


class TranscriptionResponse(BaseModel):
    type: Literal["data"]
    data: TranscriptionData


class VADSignal(StrEnum):
    START = "START_SPEECH"
    END = "END_SPEECH"


class EventData(BaseModel):
    signal_type: VADSignal
    occured_at: float


class EventResponse(BaseModel):
    type: Literal["events"]
    data: EventData


class SarvamSpeechToTextTranslateService(STTService):
    """Sarvam speech-to-text service.

    Provides real-time speech recognition using Sarvam's WebSocket API.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saaras:v2.5",
        language_code: str = "hi-IN",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription.
            language_code: Language code for transcription (e.g., "hi-IN", "kn-IN").
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        super().__init__(sample_rate=sample_rate or 16000, **kwargs)

        self.set_model_name(model)
        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        self._websocket = None
        self._listening_task = None

        self._bot_speaking = False  # Track bot speaking state

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the Sarvam model and reconnect.

        Args:
            model: The Sarvam model name to use.
        """
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._model = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        """Set the recognition language and reconnect.

        Args:
            language: The language to use for speech recognition.
        """
        # Map pipecat Language enum to Sarvam language codes
        language_mapping = {
            Language.HI: "hi-IN",
            Language.KN: "kn-IN",
            Language.EN: "en-IN",
            Language.EN_US: "en-IN",
            Language.EN_IN: "en-IN",
            # Add more mappings as needed
        }

        language_code = language_mapping.get(language, "hi-IN")
        logger.info(f"Switching STT language to: [{language_code}]")
        self._language_code = language_code
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the Sarvam STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes):
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if self._websocket:
            # Convert audio bytes to base64 for Sarvam API
            audio_base64 = base64.b64encode(audio).decode("utf-8")

            try:
                message = {
                    "audio": {
                        "data": audio_base64,
                        "encoding": "audio/wav",
                        "sample_rate": self.sample_rate,
                    }
                }
                await self._websocket_connection.send(json.dumps(message))

            except Exception as e:
                logger.error(f"Error sending audio to Sarvam: {e}")

        yield None

    async def _connect(self):
        """Connect to Sarvam WebSocket API directly."""
        logger.debug("Connecting to Sarvam")

        try:
            # Build WebSocket URL and headers manually
            ws_url = (
                self._client._client_wrapper.get_environment().production
                + "/speech-to-text-translate/ws"
            )

            # Add query parameters
            query_params = httpx.QueryParams()
            query_params = query_params.add("model", self._model)
            query_params = query_params.add("vad_signals", True)

            # Add language_code if needed by the API
            # query_params = query_params.add("language_code", self._language_code)

            ws_url = ws_url + f"?{query_params}"

            # Get headers
            headers = self._client._client_wrapper.get_headers()
            headers["Api-Subscription-Key"] = self._api_key

            # Connect to WebSocket directly
            self._websocket_connection = await websockets.connect(
                ws_url, extra_headers=headers
            )

            # Create the socket client wrapper
            self._websocket = AsyncSpeechToTextTranslateStreamingSocketClient(
                websocket=self._websocket_connection
            )

            # Start listening for messages
            self._listening_task = asyncio.create_task(self._listen_for_messages())

            logger.info("Connected to Sarvam successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Sarvam: {e}")
            self._websocket = None

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket API."""
        if self._listening_task:
            self._listening_task.cancel()
            try:
                await self._listening_task
            except asyncio.CancelledError:
                pass
            self._listening_task = None

        if self._websocket_connection:
            try:
                await self._websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                print("Disconnected from Sarvam WebSocket")
                self._websocket_connection = None
                self._websocket = None

    async def _listen_for_messages(self):
        """Listen for messages from Sarvam WebSocket."""
        try:
            while self._websocket:
                try:
                    message = await self._websocket_connection.recv()
                    response = json.loads(message)
                    await self._handle_response(response)

                    # response = await self._websocket.recv()
                    # await self._handle_response(response)
                except Exception as e:
                    logger.error(f"Error receiving message from Sarvam: {e}")
                    # Attempt to reconnect
                    await self._disconnect()
                    await self._connect()
                    break
        except asyncio.CancelledError:
            logger.debug("Message listening cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in message listener: {e}")

    async def _handle_response(self, response):
        """Handle transcription response from Sarvam.

        Args:
            response: The response object from Sarvam WebSocket.
        """
        print("response: ", response)
        try:
            if response["type"] == "events":
                parsed = EventResponse(**response)
                signal = parsed.data.signal_type
                timestamp = parsed.data.occured_at
                print(f"Signal: {signal}, Occurred at: {timestamp}")
                if signal == VADSignal.START:
                    logger.debug("User started speaking ")
                    await self.push_frame(UserStartedSpeakingFrame())
                    await self.push_frame(VADUserStartedSpeakingFrame())
                    await self.push_frame(StartInterruptionFrame())

            elif response["type"] == "data":
                await self.stop_ttfb_metrics()
                parsed = TranscriptionResponse(**response)
                transcript = parsed.data.transcript
                language_code = parsed.data.language_code
                language = "en-IN"
                await self.push_frame(UserStoppedSpeakingFrame())
                await self.push_frame(VADUserStoppedSpeakingFrame())
                await self.push_frame(StopInterruptionFrame())

                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=response,
                    )
                )
                logger.debug("Text frame from STT pushed to queue")

                # await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()

        except Exception as e:
            logger.error(f"Error handling Sarvam response: {e}")

    def _map_language_code_to_enum(self, language_code: str) -> Language:
        """Map Sarvam language code (e.g., "hi-IN") to pipecat Language enum."""
        logger.debug(f"Audio coming is as {language_code}")
        mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-US": Language.EN_US,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
        }
        return mapping.get(language_code, Language.HI_IN)

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass


class SarvamSpeechToTextService(STTService):
    """Sarvam speech-to-text transcription service.

    Provides real-time speech recognition using Sarvam's WebSocket API for transcription only.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saarika:v2.5",
        language_code: str = "hi-IN",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Sarvam STT transcription service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription.
            language_code: Language code for transcription (e.g., "hi-IN", "kn-IN").
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        super().__init__(sample_rate=sample_rate or 16000, **kwargs)

        self.set_model_name(model)
        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        self._websocket = None
        self._listening_task = None
        self._bot_speaking = False  # Track bot speaking state

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the Sarvam model and reconnect.

        Args:
            model: The Sarvam model name to use.
        """
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._model = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        """Set the recognition language and reconnect.

        Args:
            language: The language to use for speech recognition.
        """
        # Map pipecat Language enum to Sarvam language codes
        language_mapping = {
            Language.HI: "hi-IN",
            Language.KN: "kn-IN",
            Language.EN: "en-IN",
            Language.EN_US: "en-IN",
            Language.EN_IN: "en-IN",
            # Add more mappings as needed
        }

        language_code = language_mapping.get(language, "hi-IN")
        logger.info(f"Switching STT language to: [{language_code}]")
        self._language_code = language_code
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the Sarvam STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes):
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if self._websocket:
            # Convert audio bytes to base64 for Sarvam API
            audio_base64 = base64.b64encode(audio).decode("utf-8")

            try:
                await self._websocket.transcribe(
                    audio=audio_base64,
                    encoding="audio/wav",
                    sample_rate=self.sample_rate,
                )
            except Exception as e:
                logger.error(f"Error sending audio to Sarvam: {e}")

        yield None

    async def _connect(self):
        """Connect to Sarvam WebSocket API directly."""
        logger.debug("Connecting to Sarvam")

        try:
            # Build WebSocket URL and headers manually for speech-to-text streaming
            ws_url = (
                self._client._client_wrapper.get_environment().production
                + "/speech-to-text/ws"
            )

            # Add query parameters
            query_params = httpx.QueryParams()
            query_params = query_params.add("model", self._model)
            query_params = query_params.add("language-code", self._language_code)
            query_params = query_params.add("vad_signals", True)

            ws_url = ws_url + f"?{query_params}"

            # Get headers
            headers = self._client._client_wrapper.get_headers()
            headers["Api-Subscription-Key"] = self._api_key

            # Connect to WebSocket directly
            self._websocket_connection = await websockets.connect(
                ws_url, extra_headers=headers
            )

            # Create the socket client wrapper
            self._websocket = AsyncSpeechToTextStreamingSocketClient(
                websocket=self._websocket_connection
            )

            # Start listening for messages
            self._listening_task = asyncio.create_task(self._listen_for_messages())

            logger.info("Connected to Sarvam transcription service successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Sarvam: {e}")
            self._websocket = None

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket API."""
        if self._listening_task:
            self._listening_task.cancel()
            try:
                await self._listening_task
            except asyncio.CancelledError:
                pass
            self._listening_task = None

        if self._websocket_connection:
            try:
                await self._websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                print("Disconnected from Sarvam WebSocket")
                self._websocket_connection = None
                self._websocket = None

    async def _listen_for_messages(self):
        """Listen for messages from Sarvam WebSocket."""
        try:
            while self._websocket:
                try:
                    message = await self._websocket_connection.recv()
                    response = json.loads(message)
                    await self._handle_response(response)

                    # response = await self._websocket.recv()
                    # await self._handle_response(response)
                except Exception as e:
                    logger.error(f"Error receiving message from Sarvam: {e}")
                    # Attempt to reconnect
                    await self._disconnect()
                    await self._connect()
                    break
        except asyncio.CancelledError:
            logger.debug("Message listening cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in message listener: {e}")

    async def _handle_response(self, response):
        """Handle transcription response from Sarvam.

        Args:
            response: The response object from Sarvam WebSocket.
        """
        print("response: ", response)
        try:
            if response["type"] == "events":
                parsed = EventResponse(**response)
                signal = parsed.data.signal_type
                timestamp = parsed.data.occured_at
                print(f"Signal: {signal}, Occurred at: {timestamp}")
                if signal == VADSignal.START:
                    logger.debug("User started speaking ")

                    await self.push_frame(UserStartedSpeakingFrame())
                    await self.push_frame(VADUserStartedSpeakingFrame())
                    await self.push_frame(StartInterruptionFrame())

            elif response["type"] == "data":
                # Stop TTFB metrics on first response
                parsed = TranscriptionResponse(**response)
                await self.stop_ttfb_metrics()

                # Extract transcription data based on Sarvam response structure
                transcript = parsed.data.transcript
                is_final = getattr(parsed, "is_final", True)

                # Map language code back to Language enum
                language = "en-IN"

                await self.push_frame(UserStoppedSpeakingFrame())
                await self.push_frame(VADUserStoppedSpeakingFrame())
                await self.push_frame(StopInterruptionFrame())
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=response,
                    )
                )
                logger.error("PUSHED FRAME")
                await self.push_frame(UserStoppedSpeakingFrame(emulated=False))

                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
        except Exception as e:
            logger.error(f"Error handling Sarvam response: {e}")

    def _map_language_code_to_enum(self, language_code: str) -> Language:
        """Map Sarvam language code to pipecat Language enum."""
        mapping = {
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "en-IN": Language.EN_IN,
        }
        return mapping.get(language_code, Language.HI)

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass
