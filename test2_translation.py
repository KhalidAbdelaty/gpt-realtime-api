"""
Test 2: Real-Time Translation with gpt-realtime-translate

Requirements:
pip install websocket-client sounddevice numpy python-dotenv

Usage:
python test2_translation.py

Default:
English speech -> Spanish speech + transcript
"""

import os
import json
import base64
import queue
import threading

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

# Realtime translation works best with 200ms chunks over WebSocket.
CHUNK_DURATION_MS = 200
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Supported examples: es, pt, fr, ja, ru, zh, de, ko, hi, id, vi, it, en
TARGET_LANGUAGE = "es"

SHOW_SOURCE_TRANSCRIPT = True
SHOW_TRANSLATED_TRANSCRIPT = True
PLAY_TRANSLATED_AUDIO = True


# =========================
# Global state
# =========================

ws_app = None
ws_ready = False

audio_in_queue = queue.Queue(maxsize=500)
audio_out_queue = queue.Queue(maxsize=500)

stop_event = threading.Event()
send_lock = threading.Lock()
print_lock = threading.Lock()

last_print_stream = None


# =========================
# Helpers
# =========================

def safe_send(ws, payload):
    """Send WebSocket message safely from any thread."""
    try:
        if ws.sock and ws.sock.connected:
            with send_lock:
                ws.send(json.dumps(payload))
            return True
    except Exception:
        pass

    return False


def print_delta(label, delta):
    """Print streaming transcript deltas cleanly."""
    global last_print_stream

    if not delta:
        return

    with print_lock:
        if last_print_stream != label:
            if last_print_stream is not None:
                print()
            print(f"{label} ", end="", flush=True)
            last_print_stream = label

        print(delta, end="", flush=True)


# =========================
# Audio input sender
# =========================

def audio_sender(ws):
    """Send microphone audio chunks to the translation session."""
    while not stop_event.is_set():
        try:
            pcm16 = audio_in_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_b64 = base64.b64encode(pcm16).decode("utf-8")

        ok = safe_send(ws, {
            "type": "session.input_audio_buffer.append",
            "audio": audio_b64
        })

        if not ok:
            break


# =========================
# Audio playback
# =========================

def playback_loop():
    """Play translated audio chunks as they arrive."""
    if not PLAY_TRANSLATED_AUDIO:
        return

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

                stream.write(chunk)

    except Exception as e:
        print(f"\nPlayback error: {e}")


# =========================
# WebSocket callbacks
# =========================

def on_open(ws):
    global ws_ready

    print(f"Connected. Configuring English -> {TARGET_LANGUAGE} translation...")

    session_config = {
        "type": "session.update",
        "session": {
            "audio": {
                "input": {
                    "transcription": {
                        "model": "gpt-realtime-whisper"
                    },
                    "noise_reduction": {
                        "type": "near_field"
                    }
                },
                "output": {
                    "language": TARGET_LANGUAGE
                }
            }
        }
    }

    safe_send(ws, session_config)
    ws_ready = True

    print("Speak in English. Translated output will play automatically.\n")

    threading.Thread(target=audio_sender, args=(ws,), daemon=True).start()


def on_message(ws, message):
    event = json.loads(message)
    event_type = event.get("type", "")

    if event_type == "session.created":
        session_id = event.get("session", {}).get("id", "unknown")
        print(f"Session created: {session_id}")

    elif event_type == "session.updated":
        pass

    elif event_type == "session.output_audio.delta":
        # Important: the translated audio is in event["delta"], not event["audio"].
        audio_b64 = event.get("delta")

        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            audio_out_queue.put(audio_bytes)

    elif event_type == "session.output_transcript.delta":
        if SHOW_TRANSLATED_TRANSCRIPT:
            print_delta(f"[{TARGET_LANGUAGE.upper()}]", event.get("delta", ""))

    elif event_type == "session.input_transcript.delta":
        if SHOW_SOURCE_TRANSCRIPT:
            print_delta("[EN]", event.get("delta", ""))

    elif event_type == "error":
        print(f"\n[ERROR] {json.dumps(event, indent=2)}")


def on_error(ws, error):
    if isinstance(error, KeyboardInterrupt):
        return

    print(f"\nWebSocket error: {repr(error)}")


def on_close(ws, close_status_code, close_msg):
    stop_event.set()
    audio_out_queue.put(None)
    print(f"\nConnection closed. Status: {close_status_code}, Message: {close_msg}")


# =========================
# Microphone callback
# =========================

def audio_input_callback(indata, frames, time_info, status):
    """Collect microphone audio quickly and push it to a queue."""
    if not ws_ready:
        return

    # Keep this callback very light to avoid input overflow.
    audio = indata[:, 0]
    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    try:
        audio_in_queue.put_nowait(pcm16)
    except queue.Full:
        pass


# =========================
# Main
# =========================

def main():
    global ws_app

    url = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"

    headers = [
        f"Authorization: Bearer {OPENAI_API_KEY}",
        "OpenAI-Safety-Identifier: hashed-user-id",
    ]

    ws_app = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    playback_thread = threading.Thread(target=playback_loop, daemon=True)
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
            # Gracefully close the translation session.
            safe_send(ws_app, {"type": "session.close"})
            ws_app.close()

    audio_out_queue.put(None)
    playback_thread.join(timeout=2)


if __name__ == "__main__":
    main()