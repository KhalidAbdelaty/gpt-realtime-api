"""
GPT-Realtime-2 Family Demo
A polished Streamlit demo with three real working tabs:

1. Transcription: gpt-realtime-whisper via WebSocket
2. Translation: gpt-realtime-translate via WebSocket
3. Voice Assistant: gpt-realtime-2 via WebSocket

Requirements:
    pip install streamlit websocket-client python-dotenv numpy

Usage:
    streamlit run demo_app.py

Deployment note:
    For a public app, put OPENAI_API_KEY in Streamlit Secrets.
    Do not hardcode your API key in this file.
"""

import os
import io
import json
import time
import wave
import base64
import threading
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import websocket
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
TRANSLATION_MODEL = "gpt-realtime-translate"
VOICE_AGENT_MODEL = "gpt-realtime-2"


st.set_page_config(
    page_title="GPT-Realtime-2 Demo",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

APP_CSS = """
<style>
:root {
    --bg-main: #F7FAFF;
    --bg-card: rgba(255, 255, 255, 0.82);
    --bg-card-solid: #FFFFFF;
    --text-main: #172033;
    --text-muted: #65728A;
    --accent: #6C8CFF;
    --accent-soft: #EAF0FF;
    --accent-green: #2EC4B6;
    --accent-purple: #8E7DFF;
    --border: rgba(108, 140, 255, 0.18);
    --shadow: 0 18px 55px rgba(31, 45, 77, 0.10);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(108, 140, 255, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(46, 196, 182, 0.14), transparent 30%),
        linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 70%);
    color: var(--text-main);
}

.block-container {
    max-width: 1060px;
    padding-top: 2.3rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.95), rgba(244,248,255,0.96));
    border-right: 1px solid var(--border);
}

.hero {
    padding: 2rem 2rem 1.7rem 2rem;
    border-radius: 30px;
    border: 1px solid var(--border);
    background:
        linear-gradient(135deg, rgba(255,255,255,0.95), rgba(239,245,255,0.9));
    box-shadow: var(--shadow);
    margin-bottom: 1.4rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: .02em;
    color: #3653B8;
    background: var(--accent-soft);
    border: 1px solid rgba(108, 140, 255, 0.22);
    padding: .38rem .72rem;
    border-radius: 999px;
    margin-bottom: .85rem;
}

.hero h1 {
    font-size: 2.4rem;
    line-height: 1.1;
    margin: 0 0 .75rem 0;
    color: var(--text-main);
}

.hero p {
    color: var(--text-muted);
    font-size: 1.02rem;
    line-height: 1.7;
    max-width: 760px;
    margin: 0;
}

.soft-card {
    padding: 1.1rem 1.15rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    box-shadow: 0 12px 35px rgba(31, 45, 77, 0.065);
    margin-bottom: 1rem;
}

.soft-card h3 {
    margin-top: 0;
    margin-bottom: .4rem;
    color: var(--text-main);
}

.soft-card p {
    color: var(--text-muted);
    margin-bottom: 0;
    line-height: 1.6;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .35rem .7rem;
    border-radius: 999px;
    background: #ECFFF9;
    color: #087F73;
    border: 1px solid rgba(46,196,182,.22);
    font-weight: 700;
    font-size: .82rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .5rem;
    background: rgba(255,255,255,.62);
    padding: .45rem;
    border-radius: 18px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 14px;
    padding: .65rem 1rem;
    color: var(--text-muted);
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #EEF3FF, #F3FBFA);
    color: #2540A8;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(108, 140, 255, .28);
    background: linear-gradient(135deg, #6C8CFF, #8E7DFF);
    color: white;
    font-weight: 800;
    padding: .65rem 1.2rem;
    box-shadow: 0 10px 24px rgba(108, 140, 255, .22);
}

.stButton > button:hover {
    border: 1px solid rgba(108, 140, 255, .45);
    filter: brightness(1.02);
}

.stButton > button:disabled {
    background: #EEF2F7;
    color: #98A2B3;
    box-shadow: none;
}

[data-testid="stAudioInput"] {
    border-radius: 22px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,.7);
    padding: .85rem;
}

[data-testid="stAlert"] {
    border-radius: 18px;
}

hr {
    border: none;
    height: 1px;
    background: rgba(108, 140, 255, .15);
    margin: 1.4rem 0;
}

.small-muted {
    color: var(--text-muted);
    font-size: .9rem;
    line-height: 1.6;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Secrets and API key
# ---------------------------------------------------------------------

def get_api_key_from_server() -> str:
    """Read API key from Streamlit Secrets first, then environment."""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
        if secret_key:
            return secret_key
    except Exception:
        pass

    return os.environ.get("OPENAI_API_KEY", "")


with st.sidebar:
    st.header("Configuration")

    server_api_key = get_api_key_from_server()

    if server_api_key:
        st.success("API key loaded from server secrets or environment.")
        api_key = server_api_key
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="For public deployment, use Streamlit Secrets instead of typing the key here.",
        )

    st.divider()

    st.markdown(
        """
        **Models**

        - Transcription: `gpt-realtime-whisper`
        - Translation: `gpt-realtime-translate`
        - Voice agent: `gpt-realtime-2`
        """
    )

    st.caption(
        "For deployment, add OPENAI_API_KEY in Streamlit Secrets. "
        "Never hardcode your key in the app."
    )


if not api_key:
    st.warning("Add your OpenAI API key in the sidebar or Streamlit Secrets to start.")
    st.stop()


# ---------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------

def browser_audio_to_pcm16(audio_bytes: bytes) -> Tuple[bytes, float]:
    """
    Convert Streamlit browser-recorded WAV audio to 24 kHz mono PCM16.

    Returns:
        pcm16_bytes: raw PCM16 mono audio
        duration_s: audio duration after conversion
    """
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        samples = (samples - 128) << 8
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16)
    elif sample_width == 4:
        samples_32 = np.frombuffer(frames, dtype=np.int32)
        samples = (samples_32 / 2147483648.0 * 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    if len(samples) == 0:
        raise ValueError("The recorded audio is empty.")

    duration_s = len(samples) / source_rate

    if source_rate != SAMPLE_RATE:
        target_len = max(1, int(duration_s * SAMPLE_RATE))
        old_x = np.linspace(0, duration_s, num=len(samples), endpoint=False)
        new_x = np.linspace(0, duration_s, num=target_len, endpoint=False)
        samples = np.interp(new_x, old_x, samples).astype(np.int16)

    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return samples.tobytes(), len(samples) / SAMPLE_RATE


def pcm16_to_b64(pcm16_bytes: bytes) -> str:
    """Encode raw PCM16 bytes as base64."""
    return base64.b64encode(pcm16_bytes).decode("utf-8")


def pcm16_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 audio in a WAV container so Streamlit can play it."""
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)

    return buffer.getvalue()


def safe_json_loads(message: str) -> Dict:
    """Parse a WebSocket message safely."""
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {"type": "invalid_json", "raw": message}


def error_message_from_event(event: Dict) -> str:
    """Extract a clean error message from an OpenAI Realtime error event."""
    error = event.get("error", {})
    message = error.get("message") or event.get("message") or "Unknown API error"
    code = error.get("code")
    if code:
        return f"{code}: {message}"
    return message


def trim_leading_silence(
    pcm16_bytes: bytes,
    threshold_rms: float = 200.0,
    window_ms: int = 50,
    keep_lookback_ms: int = 50,
) -> Tuple[bytes, int]:
    """
    Detect and trim leading silence from PCM16 mono audio.

    Returns the trimmed bytes and how many bytes were stripped from the front.
    The translation model emits silent/filler audio at the start of its output
    stream while it is still 'listening', which makes Streamlit playback start
    with several seconds of dead air. We strip that prefix here.
    """
    if not pcm16_bytes:
        return pcm16_bytes, 0

    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm16_bytes, 0

    window_samples = max(1, int(SAMPLE_RATE * (window_ms / 1000.0)))
    n_windows = len(samples) // window_samples
    if n_windows == 0:
        return pcm16_bytes, 0

    chopped = samples[:n_windows * window_samples].reshape(n_windows, window_samples)
    rms_per_window = np.sqrt(np.mean(chopped ** 2, axis=1))

    above = np.where(rms_per_window > threshold_rms)[0]
    if len(above) == 0:
        return pcm16_bytes, 0

    first_speech_window = above[0]
    lookback_windows = max(0, keep_lookback_ms // window_ms)
    first_speech_window = max(0, first_speech_window - lookback_windows)
    start_sample = first_speech_window * window_samples
    start_byte = start_sample * SAMPLE_WIDTH_BYTES
    trimmed = pcm16_bytes[start_byte:]
    return trimmed, start_byte


# ---------------------------------------------------------------------
# Realtime WebSocket functions
# ---------------------------------------------------------------------

def run_transcription_session(pcm16_bytes: bytes, language: str = "en") -> str:
    """
    Send one recorded clip to gpt-realtime-whisper and return the transcript.
    """
    result = {
        "transcript": "",
        "error": None,
    }

    audio_b64 = pcm16_to_b64(pcm16_bytes)

    def on_open(ws):
        session_update = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE,
                        },
                        "transcription": {
                            "model": TRANSCRIPTION_MODEL,
                            "language": language,
                        },
                        "turn_detection": None,
                    }
                },
            },
        }

        ws.send(json.dumps(session_update))
        ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))
        ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
        }))

    def on_message(ws, message):
        event = safe_json_loads(message)
        event_type = event.get("type", "")

        if event_type == "conversation.item.input_audio_transcription.delta":
            result["transcript"] += event.get("delta", "")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            result["transcript"] = event.get("transcript", result["transcript"])
            ws.close()

        elif event_type == "error":
            result["error"] = error_message_from_event(event)
            ws.close()

    def on_error(ws, error):
        result["error"] = str(error)

    app = websocket.WebSocketApp(
        "wss://api.openai.com/v1/realtime?intent=transcription",
        header=[f"Authorization: Bearer {api_key}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )

    watchdog = threading.Timer(35, app.close)
    watchdog.start()
    app.run_forever(ping_interval=20, ping_timeout=10)
    watchdog.cancel()

    if result["error"]:
        raise RuntimeError(result["error"])

    return result["transcript"].strip()


def run_translation_session(
    pcm16_bytes: bytes,
    target_language: str,
    duration_s: float,
) -> Dict[str, object]:
    """
    Send one recorded clip to gpt-realtime-translate.

    Closes when a 'done' event arrives, or after an idle timeout once
    audio chunks stop arriving. Never closes on a fixed sleep.

    Returns:
        source: source transcript
        translated: translated transcript
        audio_wav: translated speech as WAV bytes, if returned
    """
    result = {
        "source": "",
        "translated": "",
        "audio_pcm16": bytearray(),
        "error": None,
    }

    IDLE_TIMEOUT_S = 5.0
    MAX_WAIT_S = max(45.0, duration_s + 30.0)

    done_event = threading.Event()
    last_activity: Dict[str, float] = {"t": 0.0}
    got_output: Dict[str, bool] = {"value": False}

    def mark():
        last_activity["t"] = time.monotonic()
        got_output["value"] = True

    # The translation endpoint expects a continuous live stream, not a single
    # large append. We send the clip as 100 ms chunks at real-time pace
    # (mirroring test2_translation.py) and follow it with a short silent
    # tail to give VAD a clean end-of-speech cue.
    CHUNK_MS = 100
    SILENCE_TAIL_S = 1.5
    BYTES_PER_SAMPLE = SAMPLE_WIDTH_BYTES * CHANNELS
    CHUNK_BYTES = int(SAMPLE_RATE * (CHUNK_MS / 1000.0)) * BYTES_PER_SAMPLE
    silence_tail_pcm16 = b"\x00\x00" * int(SAMPLE_RATE * SILENCE_TAIL_S)
    streamed_payload = pcm16_bytes + silence_tail_pcm16

    def stream_input_audio(ws_ref):
        try:
            for start_idx in range(0, len(streamed_payload), CHUNK_BYTES):
                if done_event.is_set():
                    break
                if not (ws_ref.sock and ws_ref.sock.connected):
                    break
                chunk = streamed_payload[start_idx:start_idx + CHUNK_BYTES]
                if not chunk:
                    break
                ws_ref.send(json.dumps({
                    "type": "session.input_audio_buffer.append",
                    "audio": pcm16_to_b64(chunk),
                }))
                time.sleep(CHUNK_MS / 1000.0)
        except Exception:
            pass

    def on_open(ws):
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "audio": {
                    "output": {"language": target_language},
                    "input": {"transcription": {"model": TRANSCRIPTION_MODEL}},
                }
            },
        }))
        threading.Thread(target=stream_input_audio, args=(ws,), daemon=True).start()

    def on_message(ws, message):
        event = safe_json_loads(message)
        event_type = event.get("type", "")

        if event_type == "session.input_transcript.delta":
            result["source"] += event.get("delta", "")
            mark()

        elif event_type == "session.output_transcript.delta":
            result["translated"] += event.get("delta", "")
            mark()

        elif event_type == "session.output_audio.delta":
            audio_piece = event.get("audio") or event.get("delta")
            if audio_piece:
                result["audio_pcm16"].extend(base64.b64decode(audio_piece))
                mark()

        elif event_type == "response.done":
            done_event.set()
            ws.close()

        elif event_type == "error":
            result["error"] = error_message_from_event(event)
            done_event.set()
            ws.close()

    def on_error(ws, error):
        result["error"] = str(error)
        done_event.set()

    app = websocket.WebSocketApp(
        f"wss://api.openai.com/v1/realtime/translations?model={TRANSLATION_MODEL}",
        header=[f"Authorization: Bearer {api_key}"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )

    ws_thread = threading.Thread(target=lambda: app.run_forever(), daemon=True)
    ws_thread.start()

    start = time.monotonic()
    while ws_thread.is_alive():
        now = time.monotonic()
        if done_event.is_set():
            break
        if got_output["value"] and now - last_activity["t"] > IDLE_TIMEOUT_S:
            app.close()
            break
        if now - start > MAX_WAIT_S:
            app.close()
            break
        time.sleep(0.1)

    ws_thread.join(timeout=2)

    if result["error"]:
        raise RuntimeError(result["error"])

    audio_wav = None
    if result["audio_pcm16"]:
        # Strip the leading silence the model emits during its initial
        # listening period before it actually starts translating; otherwise
        # playback begins with several seconds of dead air.
        trimmed_pcm16, _ = trim_leading_silence(bytes(result["audio_pcm16"]))
        audio_wav = pcm16_to_wav_bytes(trimmed_pcm16)

    return {
        "source": result["source"].strip(),
        "translated": result["translated"].strip(),
        "audio_wav": audio_wav,
    }


def format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 8) -> str:
    """Format recent chat history for a new Realtime session."""
    if not history:
        return "No previous turns."

    recent = history[-max_turns:]
    lines = []

    for item in recent:
        role = item.get("role", "unknown")
        content = item.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No previous turns."


def run_voice_agent_session(
    pcm16_bytes: bytes,
    duration_s: float,
    voice: str,
    style: str,
    history: List[Dict[str, str]],
) -> Dict[str, object]:
    """
    Send one user voice clip to gpt-realtime-2 and return assistant text and audio.
    This is a real Realtime WebSocket call, not a Chat API fallback.
    """
    result = {
        "user_transcript": "",
        "assistant_transcript": "",
        "assistant_audio_pcm16": bytearray(),
        "error": None,
    }

    audio_b64 = pcm16_to_b64(pcm16_bytes)
    history_text = format_history_for_prompt(history)
    response_requested = {"value": False}

    instructions = f"""
You are a friendly voice assistant in a public demo.

Style:
{style}

Rules:
- Keep replies natural and concise.
- Do not say you can listen to audio files later.
- You are receiving the user's current voice message directly.
- If the user asks for live internet search, say that this demo version does not browse the web.
- Do not invent facts about real people.
- If the user wants to end the conversation, say a short goodbye.

Recent conversation:
{history_text}
""".strip()

    wait_s = max(18.0, min(45.0, duration_s + 25.0))

    def on_open(ws):
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": VOICE_AGENT_MODEL,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE,
                        },
                        "transcription": {
                            "model": TRANSCRIPTION_MODEL,
                        },
                        "turn_detection": None,
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE,
                        },
                        "voice": voice,
                    },
                },
                "instructions": instructions,
                "reasoning": {
                    "effort": "low",
                },
            },
        }

        ws.send(json.dumps(session_update))

        ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))

        ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
        }))

    def on_message(ws, message):
        event = safe_json_loads(message)
        event_type = event.get("type", "")

        if event_type == "conversation.item.input_audio_transcription.completed":
            result["user_transcript"] = event.get("transcript", "").strip()
            if not response_requested["value"]:
                response_requested["value"] = True
                ws.send(json.dumps({
                    "type": "response.create",
                }))

        elif event_type in ("response.audio.delta", "response.output_audio.delta"):
            audio_piece = event.get("delta") or event.get("audio")
            if audio_piece:
                result["assistant_audio_pcm16"].extend(base64.b64decode(audio_piece))

        elif event_type in (
            "response.audio_transcript.delta",
            "response.output_audio_transcript.delta",
        ):
            result["assistant_transcript"] += event.get("delta", "")

        elif event_type == "response.done":
            ws.close()

        elif event_type == "error":
            message_text = error_message_from_event(event)

            # This error can happen in live barge-in scripts.
            # This Streamlit version does not call response.cancel,
            # but we ignore it safely if the server sends it.
            if "response_cancel_not_active" in message_text:
                return

            result["error"] = message_text
            ws.close()

    def on_error(ws, error):
        result["error"] = str(error)

    app = websocket.WebSocketApp(
        f"wss://api.openai.com/v1/realtime?model={VOICE_AGENT_MODEL}",
        header=[
            f"Authorization: Bearer {api_key}",
            "OpenAI-Safety-Identifier: streamlit-demo-user",
        ],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )

    watchdog = threading.Timer(wait_s, app.close)
    watchdog.start()
    app.run_forever(ping_interval=20, ping_timeout=10)
    watchdog.cancel()

    if result["error"]:
        raise RuntimeError(result["error"])

    audio_wav = None
    if result["assistant_audio_pcm16"]:
        audio_wav = pcm16_to_wav_bytes(bytes(result["assistant_audio_pcm16"]))

    return {
        "user_transcript": result["user_transcript"].strip(),
        "assistant_transcript": result["assistant_transcript"].strip(),
        "assistant_audio_wav": audio_wav,
    }


# ---------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------

if "voice_history" not in st.session_state:
    st.session_state.voice_history = []


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">🎙️ Realtime AI audio demo</div>
        <h1>GPT-Realtime-2 Family Demo</h1>
        <p>
            A clean, light, production-style Streamlit demo for transcription,
            translation, and a real voice assistant. Each tab sends actual audio
            to the Realtime API and returns real model output.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        """
        <div class="soft-card">
            <h3>Transcription</h3>
            <p>Speech to text with <b>gpt-realtime-whisper</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        """
        <div class="soft-card">
            <h3>Translation</h3>
            <p>Speech to translated text and audio.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        """
        <div class="soft-card">
            <h3>Voice agent</h3>
            <p>Real <b>gpt-realtime-2</b> voice response.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab1, tab2, tab3 = st.tabs([
    "📝 Transcription",
    "🌐 Translation",
    "🤖 Voice Assistant",
])


# ---------------------------------------------------------------------
# Tab 1: Transcription
# ---------------------------------------------------------------------

with tab1:
    st.subheader("Real transcription")
    st.markdown(
        """
        Record a short clip, then send it to `gpt-realtime-whisper`.
        The app prints only the final transcript.
        """
    )

    audio1 = st.audio_input("Record audio for transcription", key="audio_transcription")

    if audio1 is not None:
        st.audio(audio1.getvalue(), format="audio/wav")

    run_t1 = st.button(
        "Transcribe",
        key="run_transcription",
        disabled=audio1 is None,
    )

    if run_t1 and audio1 is not None:
        with st.spinner("Transcribing..."):
            try:
                pcm16, _ = browser_audio_to_pcm16(audio1.getvalue())
                transcript = run_transcription_session(pcm16, language="en")

                st.success("Done")
                st.markdown("### Transcript")
                st.write(transcript or "No speech detected.")

            except Exception as exc:
                st.error(f"Transcription failed: {exc}")


# ---------------------------------------------------------------------
# Tab 2: Translation
# ---------------------------------------------------------------------

with tab2:
    st.subheader("Real speech translation")
    st.markdown(
        """
        Record speech, choose the output language, and the app will return
        the source transcript, translated text, and translated audio if available.
        """
    )

    language_options = {
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh",
        "Portuguese": "pt",
        "Russian": "ru",
        "Hindi": "hi",
        "Italian": "it",
        "Indonesian": "id",
        "Vietnamese": "vi",
        "English": "en",
    }

    target_name = st.selectbox(
        "Target language",
        list(language_options.keys()),
        index=0,
    )
    target_code = language_options[target_name]

    audio2 = st.audio_input("Record audio for translation", key="audio_translation")

    if audio2 is not None:
        st.audio(audio2.getvalue(), format="audio/wav")

    run_t2 = st.button(
        "Translate",
        key="run_translation",
        disabled=audio2 is None,
    )

    if run_t2 and audio2 is not None:
        with st.spinner(f"Translating to {target_name}..."):
            try:
                pcm16, duration_s = browser_audio_to_pcm16(audio2.getvalue())
                translation = run_translation_session(
                    pcm16,
                    target_language=target_code,
                    duration_s=duration_s,
                )

                st.success("Done")

                left, right = st.columns(2)

                with left:
                    st.markdown("### Source")
                    st.write(translation["source"] or "No source transcript returned.")

                with right:
                    st.markdown(f"### {target_name}")
                    st.write(translation["translated"] or "No translation returned.")

                if translation["audio_wav"]:
                    st.markdown("### Translated audio")
                    st.audio(translation["audio_wav"], format="audio/wav")

            except Exception as exc:
                st.error(f"Translation failed: {exc}")

    st.caption(
        "This tab uses the actual translation WebSocket endpoint. "
        "For a clean demo, it processes one recorded clip at a time."
    )


# ---------------------------------------------------------------------
# Tab 3: Voice Assistant
# ---------------------------------------------------------------------

with tab3:
    st.subheader("Real voice assistant")
    st.markdown(
        """
        This tab uses `gpt-realtime-2` directly through WebSocket.
        Record your message, send it, then the assistant replies with text and audio.
        """
    )

    voice_col, style_col = st.columns([1, 2])

    with voice_col:
        selected_voice = st.selectbox(
            "Assistant voice",
            ["marin", "cedar", "alloy", "verse"],
            index=0,
        )

    with style_col:
        assistant_style = st.text_area(
            "Assistant style",
            value=(
                "Speak casually and warmly. Keep answers short. "
                "Use simple English unless the user speaks Arabic."
            ),
            height=90,
        )

    st.divider()

    audio3 = st.audio_input("Record your message to the assistant", key="audio_voice_agent")

    if audio3 is not None:
        st.audio(audio3.getvalue(), format="audio/wav")

    run_t3 = st.button(
        "Talk to assistant",
        key="run_voice_agent",
        disabled=audio3 is None,
    )

    if run_t3 and audio3 is not None:
        with st.spinner("The voice assistant is responding..."):
            try:
                pcm16, duration_s = browser_audio_to_pcm16(audio3.getvalue())

                reply = run_voice_agent_session(
                    pcm16_bytes=pcm16,
                    duration_s=duration_s,
                    voice=selected_voice,
                    style=assistant_style,
                    history=st.session_state.voice_history,
                )

                user_text = reply["user_transcript"] or "Voice message"
                assistant_text = reply["assistant_transcript"] or "I responded with audio."

                st.session_state.voice_history.append({
                    "role": "user",
                    "content": user_text,
                })

                st.session_state.voice_history.append({
                    "role": "assistant",
                    "content": assistant_text,
                    "audio_wav": reply["assistant_audio_wav"],
                })

            except Exception as exc:
                st.error(f"Voice assistant failed: {exc}")
                st.info(
                    "If the model returns an access error, check that your API key "
                    "has access to the Realtime models."
                )

    if st.session_state.voice_history:
        st.markdown("### Conversation")

        for item in st.session_state.voice_history:
            role = item["role"]

            with st.chat_message(role):
                st.write(item["content"])

                if role == "assistant" and item.get("audio_wav"):
                    st.audio(item["audio_wav"], format="audio/wav")

        clear_col, _ = st.columns([1, 3])

        with clear_col:
            if st.button("Clear chat"):
                st.session_state.voice_history = []
                st.rerun()

    st.caption(
        "This is a real turn-based voice demo for Streamlit. "
        "For live interruption and full-duplex browser audio, use a custom WebRTC frontend."
    )