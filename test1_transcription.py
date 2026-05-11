"""
Test 1: Real-Time Audio Transcription with gpt-realtime-whisper

Requirements:
pip install websocket-client sounddevice numpy python-dotenv

Usage:
python test1_transcription.py
"""

import os
import json
import base64
import queue
import threading
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
CHUNK_DURATION_MS = 100
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Based on your logs:
# silence was around 0.000014
# speech reached 0.0015, 0.006, 0.012
SILENCE_RMS_THRESHOLD = 0.0004

COMMIT_INTERVAL_SECONDS = 3

# Each chunk is 100ms.
# 8 chunks means around 0.8 seconds of accepted speech.
MIN_CHUNKS_BEFORE_COMMIT = 8
MIN_RMS_BEFORE_COMMIT = 0.0008


# =========================
# Global state
# =========================

ws_app = None
ws_ready = False

audio_queue = queue.Queue(maxsize=500)
stop_event = threading.Event()
sent_audio_since_commit = threading.Event()

send_lock = threading.Lock()
stats_lock = threading.Lock()

chunks_since_commit = 0
max_rms_since_commit = 0.0


# =========================
# Helpers
# =========================

def safe_send(ws, payload):
    """Send a WebSocket message safely from multiple threads."""
    try:
        if ws.sock and ws.sock.connected:
            with send_lock:
                ws.send(json.dumps(payload))
            return True
    except Exception:
        pass

    return False


def audio_sender(ws):
    """Send queued microphone chunks to OpenAI."""
    global chunks_since_commit, max_rms_since_commit

    while not stop_event.is_set():
        try:
            pcm16, rms = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_b64 = base64.b64encode(pcm16).decode("utf-8")

        ok = safe_send(ws, {
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        })

        if not ok:
            break

        with stats_lock:
            chunks_since_commit += 1
            max_rms_since_commit = max(max_rms_since_commit, rms)

        sent_audio_since_commit.set()


def periodic_commit(ws):
    """Commit only when enough real speech was collected."""
    global chunks_since_commit, max_rms_since_commit

    while not stop_event.is_set():
        time.sleep(COMMIT_INTERVAL_SECONDS)

        if not sent_audio_since_commit.is_set():
            continue

        with stats_lock:
            chunks = chunks_since_commit
            max_rms = max_rms_since_commit

        # If the captured audio is too short or too weak, clear it instead of transcribing noise.
        if chunks < MIN_CHUNKS_BEFORE_COMMIT or max_rms < MIN_RMS_BEFORE_COMMIT:
            safe_send(ws, {
                "type": "input_audio_buffer.clear"
            })

            with stats_lock:
                chunks_since_commit = 0
                max_rms_since_commit = 0.0

            sent_audio_since_commit.clear()
            continue

        ok = safe_send(ws, {
            "type": "input_audio_buffer.commit"
        })

        if not ok:
            break

        with stats_lock:
            chunks_since_commit = 0
            max_rms_since_commit = 0.0

        sent_audio_since_commit.clear()


# =========================
# WebSocket callbacks
# =========================

def on_open(ws):
    global ws_ready

    print("Connected. Configuring transcription session...")

    session_config = {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": SAMPLE_RATE
                    },
                    "transcription": {
                        "model": "gpt-realtime-whisper",
                        "language": "en"
                    },
                    "turn_detection": None
                }
            }
        }
    }

    safe_send(ws, session_config)
    ws_ready = True

    print("Listening... Speak normally. Press Ctrl+C to stop.\n")

    threading.Thread(target=audio_sender, args=(ws,), daemon=True).start()
    threading.Thread(target=periodic_commit, args=(ws,), daemon=True).start()


def on_message(ws, message):
    event = json.loads(message)
    event_type = event.get("type", "")

    if event_type in ("transcription_session.created", "session.created"):
        session_id = (
            event.get("session")
            or event.get("transcription_session")
            or {}
        ).get("id", "unknown")

        print(f"Session created: {session_id}")

    elif event_type == "conversation.item.input_audio_transcription.completed":
        transcript = event.get("transcript", "").strip()

        if transcript:
            print(f"[DONE] {transcript}")

    elif event_type == "error":
        error_code = event.get("error", {}).get("code", "")

        # Ignore empty commits silently.
        if error_code == "input_audio_buffer_commit_empty":
            return

        print(f"[ERROR] {json.dumps(event, indent=2)}")


def on_error(ws, error):
    # Suppress normal Ctrl+C interruption noise.
    if isinstance(error, KeyboardInterrupt):
        return

    print(f"WebSocket error: {repr(error)}")


def on_close(ws, close_status_code, close_msg):
    stop_event.set()


# =========================
# Microphone callback
# =========================

def audio_callback(indata, frames, time_info, status):
    """Collect microphone audio quickly and push it to the queue."""
    if not ws_ready:
        return

    audio = indata[:, 0]
    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < SILENCE_RMS_THRESHOLD:
        return

    pcm16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

    try:
        audio_queue.put_nowait((pcm16, rms))
    except queue.Full:
        pass


# =========================
# Main
# =========================

def main():
    global ws_app

    url = "wss://api.openai.com/v1/realtime?intent=transcription"

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

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            ws_app.run_forever(
                ping_interval=20,
                ping_timeout=10
            )

    except KeyboardInterrupt:
        stop_event.set()

        if ws_app:
            ws_app.close()


if __name__ == "__main__":
    main()