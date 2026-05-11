# GPT-Realtime-2 API — Python Code Examples

Companion code for the DataCamp tutorial:
**[GPT-Realtime-2 API Tutorial: Three Tests, Three Verdicts](https://www.datacamp.com/tutorial/gpt-realtime-2-api)**

Four scripts that demonstrate all three models in the GPT-Realtime family:

| File | Model | What it does |
|------|-------|--------------|
| `test1_transcription.py` | `gpt-realtime-whisper` | Live microphone transcription via WebSocket |
| `test2_translation.py` | `gpt-realtime-translate` | Real-time speech translation with audio playback |
| `test3_voice_assistant.py` | `gpt-realtime-2` | Full-duplex voice assistant with barge-in support |
| `demo_app.py` | All three | Streamlit app combining all three models in one UI |

---

## Requirements

Python 3.9 or newer.

```bash
pip install websocket-client sounddevice numpy python-dotenv
```

For the Streamlit demo only:

```bash
pip install streamlit
```

On macOS, install PortAudio first: `brew install portaudio`  
On Linux: `sudo apt install portaudio19-dev`

---

## Setup

Create a `.env` file in the same folder as the scripts:

```
OPENAI_API_KEY=sk-...
```

> **Note:** These models require at least **Tier 1** API access. The Free tier is not supported.

---

## Running the scripts

**Transcription** — speak into your microphone, get live text:
```bash
python test1_transcription.py
```

**Translation** — speak English, hear Spanish (or any supported language):
```bash
python test2_translation.py
```
Change `TARGET_LANGUAGE` at the top of the file to any supported ISO 639 code (`fr`, `de`, `ja`, `ar`, etc.).

**Voice assistant** — full-duplex conversation with `gpt-realtime-2`:
```bash
python test3_voice_assistant.py
```
> If using laptop speakers (not headphones), keep `MUTE_MIC_DURING_ASSISTANT = True` to prevent the microphone from picking up the assistant's output.

**Streamlit demo** — all three models in a browser UI:
```bash
streamlit run demo_app.py
```

---

## Key technical notes

- Audio format: **24 kHz mono PCM16**, base64-encoded
- `gpt-realtime-whisper` does **not** support `server_vad`; the scripts use manual `input_audio_buffer.commit` on a timed interval
- `gpt-realtime-translate` uses a dedicated `/v1/realtime/translations` endpoint and `session.input_audio_buffer.append` (note the `session.` prefix)
- `gpt-realtime-2` assistant audio arrives as `response.output_audio.delta` (not `response.audio.delta`)

---

## License

MIT
