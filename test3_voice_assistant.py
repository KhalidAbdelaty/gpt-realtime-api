"""
Test 3: Full-Duplex Voice Assistant with gpt-realtime-2

Requirements:
    pip install websocket-client sounddevice numpy python-dotenv

Usage:
    python test3_voice_assistant.py

Notes:
    - This version prevents the microphone from capturing the assistant's speaker output.
    - If you use headphones and want barge-in, set MUTE_MIC_DURING_ASSISTANT = False.
"""

import os
import json
import base64
import threading
import queue
import time

import websocket
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv


# =========================
# Environment
# =========================

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")


# =========================
# Audio settings
# =========================

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_DURATION_MS = 50
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

VOICE = "marin"

# Keep this True if you are using laptop speakers.
# Set to False only if you use headphones and want barge-in interruption.
MUTE_MIC_DURING_ASSISTANT = True

# Extra silence after assistant audio to avoid speaker echo tail.
ASSISTANT_COOLDOWN_MS = 800


# =========================
# Shared state
# =========================

audio_in_queue = queue.Queue(maxsize=500)
audio_out_queue = queue.Queue()

ws_app = None
ws_ready = False

stop_event = threading.Event()
send_lock = threading.Lock()

assistant_playback_active = threading.Event()

current_response_item_id = None
playback_position_ms = 0

response_active = False
pending_response_create = False
assistant_line_started = False

last_assistant_audio_time = 0.0


# =========================
# Helpers
# =========================

def safe_send(ws, payload):
    """Send a JSON event safely."""
    try:
        if ws and ws.sock and ws.sock.connected:
            with send_lock:
                ws.send(json.dumps(payload))
            return True
    except Exception as e:
        if not stop_event.is_set():
            print(f"\nSend error: {e}")
    return False


def should_mute_microphone():
    """Return True while assistant audio is playing or just finished."""
    if not MUTE_MIC_DURING_ASSISTANT:
        return False

    if assistant_playback_active.is_set():
        return True

    elapsed_ms = (time.monotonic() - last_assistant_audio_time) * 1000
    return elapsed_ms < ASSISTANT_COOLDOWN_MS


def reset_assistant_line():
    """End assistant printed line cleanly."""
    global assistant_line_started

    if assistant_line_started:
        print()
        assistant_line_started = False


# =========================
# WebSocket callbacks
# =========================

def on_open(ws):
    global ws_ready

    print("Connected. Configuring voice agent...")

    session_config = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime-2",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": SAMPLE_RATE
                    },
                    "noise_reduction": {
                        "type": "near_field"
                    },
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "en"
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "medium",

                        # We create the response manually after the user transcript is complete.
                        "create_response": False,

                        # If mic is muted during assistant playback, interruption is not needed.
                        # If using headphones and MUTE_MIC_DURING_ASSISTANT=False, this becomes useful.
                        "interrupt_response": not MUTE_MIC_DURING_ASSISTANT
                    }
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": SAMPLE_RATE
                    },
                    "voice": VOICE
                }
            },
            "instructions": (
                "You are a helpful voice assistant. "
                "Keep answers short, clear, and natural. "
                "Do not transcribe audio unless the user explicitly asks you to transcribe. "
                "If the user asks a normal question, answer the question directly."
            ),
            "reasoning": {
                "effort": "low"
            }
        }
    }

    safe_send(ws, session_config)

    ws_ready = True
    print("Voice assistant ready. Speak to start. Press Ctrl+C to quit.\n")

    threading.Thread(
        target=audio_input_sender,
        args=(ws,),
        daemon=True
    ).start()


def on_message(ws, message):
    global current_response_item_id
    global playback_position_ms
    global response_active
    global pending_response_create
    global assistant_line_started

    event = json.loads(message)
    event_type = event.get("type", "")

    if event_type == "session.created":
        session_id = event.get("session", {}).get("id", "unknown")
        print(f"Session id: {session_id}")

    elif event_type == "input_audio_buffer.speech_started":
        if not MUTE_MIC_DURING_ASSISTANT:
            handle_barge_in(ws)

    elif event_type == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcript", "").strip()

        if transcript:
            reset_assistant_line()
            print(f"\n[User] {transcript}")

            if not response_active and not pending_response_create:
                pending_response_create = True
                safe_send(ws, {
                    "type": "response.create"
                })

    elif event_type == "response.created":
        response_active = True
        pending_response_create = False
        playback_position_ms = 0
        assistant_line_started = False

    elif event_type == "response.output_item.added":
        item = event.get("item", {})
        if item.get("role") == "assistant":
            current_response_item_id = item.get("id")

    elif event_type == "response.output_audio.delta":
        audio_b64 = event.get("delta")

        if audio_b64:
            item_id = event.get("item_id")
            if item_id:
                current_response_item_id = item_id

            audio_bytes = base64.b64decode(audio_b64)
            audio_out_queue.put(audio_bytes)

    elif event_type == "response.output_audio_transcript.delta":
        delta = event.get("delta", "")

        if delta:
            if not assistant_line_started:
                print("\n[Assistant] ", end="", flush=True)
                assistant_line_started = True

            print(delta, end="", flush=True)

    elif event_type == "response.output_audio_transcript.done":
        reset_assistant_line()

    elif event_type == "response.done":
        response_active = False
        pending_response_create = False

    elif event_type == "error":
        error = event.get("error", {})
        code = error.get("code", "")

        # Ignore this harmless error if it ever appears.
        if code == "response_cancel_not_active":
            return

        reset_assistant_line()
        print(f"\n[ERROR] {json.dumps(event, indent=2)}")


def on_error(ws, error):
    if isinstance(error, KeyboardInterrupt):
        return

    if not stop_event.is_set():
        print(f"\nWebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    stop_event.set()
    audio_out_queue.put(None)
    print(f"\nConnection closed. Status: {close_status_code}, Message: {close_msg}")


# =========================
# Barge-in handling
# =========================

def handle_barge_in(ws):
    """
    Stop local assistant playback if the user interrupts.

    This is mainly useful with headphones.
    With laptop speakers, keep MUTE_MIC_DURING_ASSISTANT=True.
    """

    global playback_position_ms
    global current_response_item_id
    global response_active
    global pending_response_create

    while True:
        try:
            audio_out_queue.get_nowait()
        except queue.Empty:
            break

    if current_response_item_id and playback_position_ms > 0:
        safe_send(ws, {
            "type": "conversation.item.truncate",
            "item_id": current_response_item_id,
            "content_index": 0,
            "audio_end_ms": playback_position_ms
        })

    playback_position_ms = 0
    response_active = False
    pending_response_create = False


# =========================
# Audio input
# =========================

def audio_input_sender(ws):
    """Send microphone chunks to the Realtime session."""
    while not stop_event.is_set():
        try:
            pcm16 = audio_in_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_b64 = base64.b64encode(pcm16).decode("utf-8")

        ok = safe_send(ws, {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        })

        if not ok:
            break


def audio_input_callback(indata, frames, time_info, status):
    """Collect microphone audio and push it to a queue."""
    if not ws_ready or stop_event.is_set():
        return

    # This is the important echo-prevention line.
    # It stops the mic from sending assistant speaker audio back to the model.
    if should_mute_microphone():
        return

    audio = np.clip(indata[:, 0], -1, 1)
    pcm16 = (audio * 32767).astype(np.int16).tobytes()

    try:
        audio_in_queue.put_nowait(pcm16)
    except queue.Full:
        pass


# =========================
# Audio output
# =========================

def playback_loop():
    """Play assistant audio chunks and track playback position."""
    global playback_position_ms
    global last_assistant_audio_time

    try:
        with sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16"
        ) as stream:
            while not stop_event.is_set():
                chunk = audio_out_queue.get()

                if chunk is None:
                    break

                assistant_playback_active.set()

                stream.write(chunk)

                samples = len(chunk) / (2 * CHANNELS)
                chunk_ms = int((samples / SAMPLE_RATE) * 1000)

                playback_position_ms += chunk_ms
                last_assistant_audio_time = time.monotonic()

                if audio_out_queue.empty():
                    assistant_playback_active.clear()

    except Exception as e:
        print(f"\nPlayback error: {e}")


# =========================
# Main
# =========================

def main():
    global ws_app

    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2"

    headers = [
        f"Authorization: Bearer {OPENAI_API_KEY}",
        "OpenAI-Safety-Identifier: hashed-user-id"
    ]

    ws_app = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    playback_thread = threading.Thread(
        target=playback_loop,
        daemon=True
    )
    playback_thread.start()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_input_callback
        ):
            ws_app.run_forever(
                ping_interval=20,
                ping_timeout=10
            )

    except KeyboardInterrupt:
        stop_event.set()

        if ws_app:
            ws_app.close()

    audio_out_queue.put(None)
    playback_thread.join(timeout=2)


if __name__ == "__main__":
    main()