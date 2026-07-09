# DTBot — Offline Edition: Setup Guide

This turns your cloud DTBot (Groq STT/LLM + edge-tts) into a fully
offline pipeline: **whisper.cpp → Qwen2.5-3B (llama.cpp) → Piper**,
with **OpenWakeWord** replacing the old polling-based wake detection.

Nothing in `dtbot_offline.py` calls the internet anymore. The RAG index
(TF-IDF over your PDFs) was already local and needs no changes.

---

## 1. Install Python dependencies

```bash
pip install pywhispercpp openwakeword piper-tts \
            sounddevice soundfile pygame requests \
            python-dotenv fitz PyMuPDF scikit-learn numpy
```

`pywhispercpp` builds whisper.cpp from source on first install — make
sure you have a C++ compiler (`build-essential` on Ubuntu).

```bash
sudo apt install build-essential espeak espeak-data libespeak-dev
```

## 2. Whisper.cpp model (STT)

No manual download needed — `pywhispercpp` fetches the ggml `base`
model automatically the first time `WhisperModel("base")` runs, and
caches it locally. If you want to pin a specific file (e.g. one you
already downloaded), set:

```bash
# .env
WHISPER_MODEL=/path/to/ggml-base.bin
WHISPER_THREADS=4
```

## 3. llama.cpp server + Qwen2.5-3B (LLM)

This runs as a **separate, always-on process** — start it before
launching the bot, and leave it running:

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF   # CPU-only; drop this flag if you have a CUDA GPU
cmake --build llama.cpp/build --config Release -j --target llama-server

# Download the Q4_K_M GGUF from Hugging Face:
#   huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF \
#     qwen2.5-3b-instruct-q4_k_m.gguf --local-dir models

./llama.cpp/build/bin/llama-server \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  -c 2048 --port 8080
```

Verify it's up: `curl http://127.0.0.1:8080/health`

In `.env`:
```bash
LLAMA_SERVER_URL=http://127.0.0.1:8080
```

**Tip:** run `llama-server` under a process supervisor (systemd,
`supervisord`, or just `tmux`) so it survives reboots and restarts
automatically if it crashes — DTBot itself does not manage this
process for you.

## 4. Piper voices (TTS)

```bash
pip install piper-tts
```

Download voice models from the [Piper voices
page](https://github.com/rhasspy/piper/blob/master/VOICES.md) — you
need both the `.onnx` and matching `.onnx.json` for each language:

```bash
# English
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json

# Hindi — community-trained Hindi voices are more limited than English;
# check the voices page for the current best option and adjust the
# .env path below accordingly.
hi_IN-<voice>-medium.onnx
hi_IN-<voice>-medium.onnx.json
```

In `.env`:
```bash
PIPER_BIN=piper
PIPER_MODEL_EN=/path/to/en_US-lessac-medium.onnx
PIPER_MODEL_HI=/path/to/hi_IN-<voice>-medium.onnx
```

espeak remains installed as the **last-resort fallback** if Piper
itself ever fails on a given utterance (missing voice file, corrupted
model, etc.) — it doesn't need any extra configuration.

## 5. OpenWakeWord (wake word)

```bash
pip install openwakeword
python -c "import openwakeword; openwakeword.utils.download_models()"
```

By default the bot uses the pretrained `hey_jarvis` model. To use a
different pretrained word or train your own (see the main offline
guide from earlier in this conversation for the training-notebook
route), set:

```bash
# .env
WAKE_MODEL_NAME=hey_jarvis        # used when WAKE_MODEL_PATH is empty
WAKE_MODEL_PATH=                  # or: /path/to/your_custom_wakeword.onnx
WAKE_THRESHOLD=0.5
WAKE_INFERENCE_FRAMEWORK=onnx     # or "tflite"
```

Test detection standalone before wiring it into the full bot — run
openWakeWord's own `examples/detect_from_microphone.py` first to tune
`WAKE_THRESHOLD` for your room/mic before trusting it inside DTBot.

## 6. .env summary

```bash
# STT
WHISPER_MODEL=base
WHISPER_THREADS=4

# LLM
LLAMA_SERVER_URL=http://127.0.0.1:8080

# TTS
PIPER_BIN=piper
PIPER_MODEL_EN=/path/to/en_US-lessac-medium.onnx
PIPER_MODEL_HI=/path/to/hi_IN-voice-medium.onnx

# Wake word
WAKE_MODEL_NAME=hey_jarvis
WAKE_MODEL_PATH=
WAKE_THRESHOLD=0.5
WAKE_INFERENCE_FRAMEWORK=onnx

# Misc (unchanged from the cloud build)
MIC_NAME=
LOG_LEVEL=INFO
ENABLE_WEB_FALLBACK=false
```

`GROQ_API_KEY` is no longer needed — delete it from `.env` if present.

## 7. Startup order

1. Start `llama-server` and confirm `/health` returns 200.
2. Run `python dtbot_offline.py`. The startup banner will show you the
   live status of whisper.cpp, the llama-server connection, Piper, and
   the wake-word model — check it before assuming something's broken.
3. Say your wake word ("hey jarvis" by default). The bot replies in
   Hindi/Hinglish by default ("Haan, mein sun raha hoon"), then listens
   for your actual question.

## 8. Known trade-offs vs. the cloud build

- **Latency**: expect the first sentence of a reply noticeably later
  than the cloud version, especially on modest CPUs — Qwen2.5-3B at
  4–8 tok/s is much slower than a cloud-hosted 20B model. Keep
  `MAX_TOKENS` short (already 200 in this build) and consider
  streaming Piper sentence-by-sentence as a future optimization (see
  the earlier message in this conversation for that pattern).
- **Hindi TTS quality**: Piper's Hindi voice selection is currently
  much smaller than its English lineup — audition a few before
  committing, or fall back to espeak's Hindi voice if none sound
  acceptable.
- **Web fallback is off by default** (`ENABLE_WEB_FALLBACK=false`) —
  questions outside both PDFs and the LLM's own knowledge will get a
  best-effort general-knowledge answer instead of a live web search.
