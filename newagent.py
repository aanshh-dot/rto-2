# newagent.py  –  Pipecat-0.0.76  ✦  Groq token streaming + working pre-warm
import asyncio, logging, os
from dotenv import load_dotenv

from pipecat.frames.frames import Frame
from pipecat.utils.time import time_now_iso8601
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner  import PipelineRunner
from pipecat.pipeline.task    import PipelineTask
from pipecat.transports.local.audio import (
    LocalAudioTransport, LocalAudioTransportParams
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.groq.llm import GroqLLMService

from sarvamai_stt_service  import SarvamSpeechToTextTranslateService   # yours
from sarvamai_tts_service  import SarvamWebsocketTTSService           # yours


from pipecat.services.openai.llm import (
    OpenAILLMService,
    OpenAIUserContextAggregator,   # ← correct import for 0.0.76
    OpenAIAssistantContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext as StreamingContext,
)

load_dotenv(override=True)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY") or "sk_xxx"
GROQ_API_KEY   = os.getenv("GROQ_API_KEYY")

logging.basicConfig()
logger = logging.getLogger("pipecat")
logger.setLevel(logging.INFO)

class StreamPassthrough(FrameProcessor):
    """Let every LLM chunk continue downstream immediately."""
    async def on_frame(self, frame, direction: FrameDirection):
        # Just relay the frame as-is
        await self.push_frame(frame, direction)

# ──────────── 1. minimal “partial token” frame ────────────
class LLMPartialFrame(Frame):
    """Single token / chunk streamed from the LLM."""
    def __init__(self, text: str):
        super().__init__(time_now_iso8601())
        self.text = text

# ──────────── 2. streaming context for 0.0.76 ────────────
class StreamingContext(OpenAILLMContext):
    """
    Emit LLMPartialFrame for every token chunk so downstream TTS
    can speak while the model is still generating.
    """
    async def on_partial_response(self, token: str, **_kw):
        await self.push_frame(LLMPartialFrame(token))

# ──────────── 3. Groq pre-warmer (fixed) ────────────
async def prewarm_groq(llm: GroqLLMService):
    try:
        await llm._client.chat.completions.create(
                model       = llm.model_name,
                messages    = [{"role": "user", "content": "Say Something!"}],
                max_tokens  = 1,
                temperature = 0.0,
                stream      = False,
        )
        logger.warning("Groq pre-warm complete")
    except Exception as exc:
        logger.warning(f"Groq pre-warm failed: {exc}")

# ──────────── 4. main ────────────
async def main():
    logger.info("Starting low-latency voice agent …")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = SarvamSpeechToTextTranslateService(
        api_key=SARVAM_API_KEY, model="saaras:v2.5", language_code="en-IN"
    )

    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
        stream=True                       # Groq streams tokens
    )

    system_prompt = ("You are a helpful conversational assistant. "
                     "Reply in ≤2 short sentences.")

    ctx      = StreamingContext([{"role": "system", "content": system_prompt}])

    # Create context aggregators with fast timeout for speed
    from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams

    ctx_agg = llm.create_context_aggregator(
        ctx,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05)  # Fast aggregation
    )

    tts = SarvamWebsocketTTSService(
        api_key=SARVAM_API_KEY,
        model="bulbul:v2",
        speaker="anushka",
        output_audio_codec="linear16",
    )

    stream_passthrough = StreamPassthrough()

    pipeline = Pipeline([
        transport.input(),
        stt,
        ctx_agg.user(),        # transcripts → LLM
        llm,                   # Groq streaming
        #stream_passthrough,    # <-- new: let chunks fly
        tts,                   # chunks → audio
        transport.output(),
        ctx_agg.assistant(),   # save only *final* assistant message
    ])
    task   = PipelineTask(pipeline)
    runner = PipelineRunner()

    await prewarm_groq(llm)              # warm TLS + model

    @transport.event_handler("on_client_connected")
    async def _(_, __):
        await task.queue_frames([ctx_agg.user().get_context_frame()])

    await runner.run(task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline stopped manually.")
