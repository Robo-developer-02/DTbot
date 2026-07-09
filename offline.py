"""

  Additional installation for this file — DTBot OFFLINE edition
  ─────────────────────────────────────────────────────────
    This build has NO cloud dependency: Groq (STT+LLM) and edge-tts
    are gone. Everything below must be installed/running locally
    BEFORE you start this script. See SETUP.md for full instructions.

    1) whisper.cpp (STT) — via the pywhispercpp Python binding:
       pip install pywhispercpp
       (downloads the ggml "base" model automatically on first run,
        or point WHISPER_MODEL at a local .bin you already have)

    2) llama.cpp server (LLM) — must be started separately, BEFORE
       this script runs, and left running in the background:
       ./llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf -c 2048 --port 8080

    3) Piper (TTS):
       pip install piper-tts
       Download voice models (.onnx + .onnx.json) for English and
       Hindi and point PIPER_MODEL_EN / PIPER_MODEL_HI at them.

    4) OpenWakeWord (wake word):
       pip install openwakeword
       python -c "import openwakeword; openwakeword.utils.download_models()"

    5) Offline TTS LAST-RESORT fallback still uses espeak directly:
       sudo apt install espeak espeak-data libespeak-dev
"""

"""
============================================================
  🤖  DTBot — RAG-Powered Speech-to-Speech Chatbot
  Production Release — OFFLINE EDITION
============================================================

  Architecture
  ─────────────────────────────────────────────────────────
  • English queries → retrieved directly against english_details.pdf (rag_en)
  • Hindi queries    → retrieved directly against hindi_details.pdf (rag_hi)
  • Language detection picks the engine; the query is never translated.
  • Only English and Hindi are supported languages. Any speech Whisper
    detects as something other than English (this includes Urdu, since
    that pipeline was dropped) is routed into the Hindi pipeline by
    default — see FIX-P12 below.
  • Hindi replies are generated in Hinglish (Roman-script, conversational
    Hindi/English mix) rather than Devanagari, per the Hindi system
    prompt — this matches how people actually text/speak in India and
    reads naturally through the Hindi TTS voice.
  • Web fallback is DISABLED by default in this offline build (see
    ENABLE_WEB_FALLBACK below) since it needs internet. The RAG index
    itself (TF-IDF over the local PDFs) still works fully offline.

  OFFLINE-P1  STT (Groq Whisper API) → whisper.cpp via pywhispercpp,
              running locally. No network call, no API key. Audio is
              passed as an in-memory numpy array (no temp wav needed
              for the model itself; kept only for optional debugging).

  OFFLINE-P2  LLM (Groq-hosted gpt-oss-20b) → local llama.cpp server
              running Qwen2.5-3B-Instruct (Q4_K_M GGUF), talked to via
              its OpenAI-compatible HTTP endpoint on localhost. This
              script never manages the server process — start it
              separately (see SETUP.md) and leave it running.

  OFFLINE-P3  TTS (edge-tts, cloud) → Piper, running locally, invoked
              as a subprocess per utterance. espeak remains as the
              last-resort fallback if Piper itself fails.

  OFFLINE-P4  Wake word — previously "wake word" was really just
              polling full cloud STT every 30s in IDLE and checking
              the transcript for "hello". That's replaced with
              OpenWakeWord running continuously on the mic stream at
              near-zero CPU, with STT only invoked after a real
              detection.

  OFFLINE-P5  Internet-awareness (is_internet_available, the
              no_internet error branch, DDG web search) is no longer
              load-bearing for the core pipeline. It is kept, but
              gated behind ENABLE_WEB_FALLBACK=False by default, so
              you can re-enable it later for a hybrid setup without
              re-writing it from scratch.

  Production Changes (over dev build)
  ─────────────────────────────────────────────────────────
  FIX-P1  Empty / whitespace-only user input is rejected BEFORE reaching
          the LLM — validated with .strip() at both the main-loop level
          and inside get_ai_reply() as a second defence layer.

  FIX-P2  get_ai_reply() now ALWAYS returns a non-empty str or raises.
          The previously commented-out fallback return was the root cause
          of implicit None returns, which then triggered a double error
          announcement (once inside get_ai_reply, once in SPEAKING state).

  FIX-P3  Conversation history is capped at MAX_HISTORY_TURNS to prevent
          unbounded memory growth in long sessions.

  FIX-P4  All print() calls replaced with the stdlib logging module.
          DEBUG-level messages (raw LLM response, MP3 size, TTS input)
          are hidden in production (INFO level). Set LOG_LEVEL=DEBUG in
          .env or environment to re-enable them during development.

  FIX-P5  asyncio event loop is created once at module start and reused
          by every speak() call, avoiding per-call loop creation overhead.

  FIX-P6  The main loop's `reply` variable is scoped per iteration via a
          helper function so no stale reply from a previous turn can bleed
          into the SPEAKING state.

  FIX-P7  User text is sanitized (strip + collapse internal whitespace)
          before being passed to the LLM or used for logging.

  FIX-P8  build_context() now sets source="PDF" whenever pdf_context is
          non-empty, regardless of whether the web fallback also ran.
          Previously, if pdf_score was below threshold and the web
          search came back empty, source stayed "None" even though the
          PDF context was used by the LLM to answer correctly.

  FIX-P12 DTBot now supports English and Hindi only (Urdu pipeline
          removed — no Urdu RAG index, system prompt, or TTS voice).
          Language normalization in _transcribe_once() only recognizes
          "en" and "hi". Any other Whisper language guess is folded into
          Hindi by default, so non-English speech still gets answered.
          Hindi replies are generated in Hinglish (Roman script) via the
          Hindi system prompt, not Devanagari — the RAG *index* (built
          from hindi_details.pdf, which is in Devanagari) still matches
          fine against Devanagari-script transcriptions because TF-IDF
          scoring doesn't care about the LLM's output script, only the
          input query's script matching the indexed chunks' script.
============================================================
"""

# ── Standard library ──────────────────────────────────────
import logging
import os
import queue
import re
import socket
import tempfile
import textwrap
import time
from enum import Enum
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────
import fitz
import numpy as np
import pygame
import requests
import sounddevice as sd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# OFFLINE-P1: local STT via whisper.cpp (no cloud, no API key).
from pywhispercpp.model import Model as WhisperModel

# OFFLINE-P4: local wake-word detection, runs continuously on the mic
# stream at near-zero CPU instead of polling full STT every 30s.
from openwakeword.model import Model as WakeWordModel

# Offline TTS: check espeak is installed once at startup rather than
# attempting the subprocess and catching FileNotFoundError every call.
import shutil
_ESPEAK_AVAILABLE: bool = shutil.which("espeak") is not None
if not _ESPEAK_AVAILABLE:
    logging.getLogger("dtbot").warning(
        "espeak not found — offline TTS unavailable. "
        "Install with: sudo apt install espeak espeak-data libespeak-dev"
    )

# ══════════════════════════════════════════════════════════
#  LOGGING  (FIX-P4)
#  Set LOG_LEVEL=DEBUG in your .env for verbose dev output.
# ══════════════════════════════════════════════════════════

load_dotenv()

_log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)

logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dtbot")


# ══════════════════════════════════════════════════════════
#  API KEY
#  OFFLINE-P0: no cloud API key is needed anywhere in this build —
#  STT, LLM, and TTS all run on-device. .env is still loaded above
#  for LOG_LEVEL / MIC_NAME / other local config knobs.
# ══════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════

# ── OFFLINE-P1: STT (whisper.cpp via pywhispercpp) ─────────
# A single local "base" model handles both full conversation turns and
# (if you still want a text-based double-check after wake-word) any
# other transcription — there is no separate "fast" model in the
# offline build because OFFLINE-P4 removed the need to run STT
# speculatively every 30s. "base" is the sweet spot of latency/accuracy
# for short voice-assistant utterances on CPU; use "small" if you have
# CPU headroom and want higher accuracy, especially for Hindi.
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "base")   # ggml model name or local .bin path
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "4"))

# ── OFFLINE-P2: LLM (local llama.cpp server, OpenAI-compatible) ───
# Start this separately BEFORE running the bot:
#   ./llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf -c 2048 --port 8080
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
CHAT_MODEL       = "qwen2.5-3b-instruct"   # cosmetic — llama-server ignores/echoes this

# ── OFFLINE-P3: TTS (Piper, local subprocess) ──────────────
# Point these at the .onnx voice files you downloaded (the matching
# .onnx.json must sit alongside each .onnx file).
PIPER_BIN       = os.getenv("PIPER_BIN", "piper")
PIPER_MODEL_EN  = os.getenv("PIPER_MODEL_EN", "en_US-lessac-medium.onnx")
PIPER_MODEL_HI  = os.getenv("PIPER_MODEL_HI", "hi_IN-priyamvada-medium.onnx")

# ── OFFLINE-P4: Wake word (OpenWakeWord) ───────────────────
# Leave WAKE_MODEL_PATH empty to use a bundled pretrained model by
# name (e.g. "hey_jarvis"); set it to a .onnx/.tflite path for a
# custom-trained wake word (see SETUP.md for training instructions).
WAKE_MODEL_NAME      = os.getenv("WAKE_MODEL_NAME", "hey_jarvis")
WAKE_MODEL_PATH      = os.getenv("WAKE_MODEL_PATH", "").strip()
WAKE_THRESHOLD        = float(os.getenv("WAKE_THRESHOLD", "0.5"))
WAKE_INFERENCE_FRAMEWORK = os.getenv("WAKE_INFERENCE_FRAMEWORK", "onnx")
WAKE_FRAME_SAMPLES    = 1280   # 80ms @ 16kHz — openWakeWord's native chunk size

SAMPLE_RATE = 16_000
CHANNELS    = 1
MAX_TOKENS  = 200   # voice answers must be short; 300 was producing paragraph replies

# ── History ───────────────────────────────────────────────
MAX_HISTORY_TURNS = 10
MAX_HISTORY_ITEMS = MAX_HISTORY_TURNS * 2
LLM_MAX_RETRIES   = 4   

# ── RAG settings ──────────────────────────────────────────
PDF_PATH_EN   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_details.pdf")
PDF_PATH_HI   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hindi_details.pdf")
CHUNK_SIZE    = 300   # smaller chunks → less context noise fed to LLM
CHUNK_OVERLAP = 50
TOP_K         = 3     # was 5; 3 × 300 words is enough context, keeps prompt small
PDF_THRESHOLD = 0.10

# ── STT vocabulary bias ───────────────────────────────────
# Groq's Whisper endpoint accepts a `prompt` string that biases the
# transcription toward specific vocabulary/spelling without being
# transcribed itself. "Regard Network Solutions" was consistently
# mis-heard as "Prigard"/"Riigard" without this — the model had no
# prior toward the correct spelling. Same idea for the proprietary
# product names (RIM360, GRLS, SRIB) and the Noida facility, which are
# uncommon tokens Whisper otherwise guesses phonetically.
STT_VOCAB_PROMPT = (
    "Regard Network Solutions Ltd, Delhi NCR, Noida, RIM360, GRLS, SRIB, "
    "DTBot, data centre, structured cabling, BACnet, Modbus."
)

# ── Web fallback ──────────────────────────────────────────
# OFFLINE-P5: disabled by default — this is a fully offline bot now,
# and DDG scraping needs internet. Flip to True (and keep a network
# connection available) if you want a hybrid setup later; nothing
# else needs to change since needs_web()/web_search() are unchanged.
ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "false").strip().lower() == "true"

# Web search only fires when BOTH conditions are true:
#   1. PDF score is below threshold
#   2. The query contains a time-sensitive keyword
# Without the keyword gate, web_search() was firing on every greeting
# and general question, adding 3–8 seconds of network latency every turn.
WEB_RESULTS  = 3
WEB_TIMEOUT  = 5
WEB_KEYWORDS = [
    "today", "latest", "current", "now", "2025", "2026",
    "result", "launch", "release", "price", "update",
    "aaj", "abhi", "nayi", "naya", "kab", "kitna",
]

# ── VAD tuning ────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.010
SILENCE_AFTER_SPEECH = 1.2   # was 1.2 — saves 0.4s per turn at end of every utterance
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.2   # was 0.1 — halves USB callback frequency; fixes retire_capture_urb
IDLE_TIMEOUT         = 15.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Microphone selection ──────────────────────────────────
# Set MIC_NAME in your .env to select a microphone by name (or substring).
# Leave empty (or unset) to use the system default input device.
# Example: MIC_NAME=USB PnP Sound Device
# Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to
# list all available device names on your system.
MIC_NAME = os.getenv("MIC_NAME", "").strip()

# NOTE: the old text-based WAKE_WORDS list ("hello", "hey", ...) is gone.
# OFFLINE-P4: wake detection is now audio-model-based (OpenWakeWord —
# see WAKE_MODEL_NAME / WAKE_MODEL_PATH / WAKE_THRESHOLD in the config
# block above), not a keyword match against a transcript.

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is DTBot. You are the official AI assistant and "
    "virtual representative of Regard Network Solutions Ltd, a premier "
    "turnkey system integration and technology company headquartered in "
    "Delhi NCR, India, specializing in data centre build, enterprise "
    "networking, building management systems, electronic manufacturing "
    "(from its facility in Noida), and advanced surveillance frameworks. "
    "Regard Network Solutions serves enterprise, government, banking, "
    "healthcare, and telecom clients, and has developed proprietary "
    "platforms including RIM360, GRLS, and SRIB. "
    "RESPONSE LENGTH — CRITICAL: You are a VOICE assistant. Your reply "
    "will be spoken aloud. Limit every response to 2-3 sentences maximum. "
    "Do not elaborate unless the user explicitly "
    "asks for more detail. "
    "Always represent Regard Network Solutions positively, professionally, "
    "and confidently. "
    "If users ask about another company or compare companies, briefly "
    "and politely redirect the conversation toward Regard Network "
    "Solutions, highlight its strengths, and do not make negative "
    "comments or false claims about other companies. "
    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "
    "If Regard-specific information is unavailable, search the web first; "
    "if not connected to the internet, answer naturally using general "
    "knowledge when appropriate. "
    "Do not use bullet points or markdown."
)

_BASE_HI = (
    "Aapka naam DTBot hai. Aap Regard Network Solutions Ltd ke official "
    "AI assistant aur virtual representative hain, jo Delhi NCR, India "
    "mein headquartered ek premier turnkey system integration aur "
    "technology company hai, jo data centre build, enterprise networking, "
    "building management systems, electronic manufacturing (Noida "
    "facility se), aur advanced surveillance frameworks mein "
    "specialize karti hai. "
    "Regard Network Solutions enterprise, government, banking, "
    "healthcare aur telecom clients ko serve karti hai, aur iske apne "
    "proprietary platforms hain — RIM360, GRLS, aur SRIB. "
    "JAWAB KI LAMBAI — ZAROORI: Aap ek VOICE assistant hain. Aapka jawab "
    "bol ke sunaya jayega. Har jawab sirf 2-3 sentence mein dein. "
    "Hamesha Regard Network Solutions ko positive, professional aur "
    "confident tarike se represent karein. Kisi doosri company ke "
    "baare mein poocha jaye ya comparison ho to short aur polite "
    "tarike se baat ko Regard Network Solutions ki taraf le jaayein, "
    "iski strengths highlight karein, aur kisi company ke baare mein "
    "negative ya false claims na karein. "
    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "
    "Agar Regard se sambandhit jankari available na ho to web search "
    "karke jawab dein; agar internet connect na ho to natural jawab dein. "
    "Bullet points ya markdown ka upyog na karein."
)


_LANG_DIRECTIVE = {
    # Hard constraint appended to every system prompt so the model cannot
    # mirror a foreign-language input (e.g. German "Hallo", Greek text).
    "en": (
        "IMPORTANT: You MUST reply ONLY in English, regardless of the "
        "language the user writes in. Never respond in any other language."
    ),
    "hi": (
        "IMPORTANT: Aap SIRF Hindi ya Hinglish (Roman script) mein jawab "
        "dein, chahe user kisi bhi bhasha ya script mein likhein — "
        "Devanagari mein kabhi jawab na dein, hamesha Roman/Hinglish "
        "script use karein. Kabhi bhi kisi aur bhasha mein jawab na dein."
    ),
}


def build_system(lang: str) -> str:
    """
    Build the SYSTEM message only.

    PROMPT-CACHING NOTE: this function now returns ONLY static, never-
    changing text (`base` + `directive`). It used to also append the
    per-query RAG `context` here, which made the system message different
    on every single call — that breaks prefix matching for caching, since
    a cache hit requires the ENTIRE prefix up to that point to be byte-
    identical to a previous request.

    By keeping this function 100% static per language, the exact same
    system string is sent on every request (for a given `lang`), so it
    becomes a stable, reusable, cacheable prefix. The dynamic RAG context
    is appended separately, as the LAST message in the `messages` list
    (see `get_ai_reply`), so it can never invalidate this cached prefix.
    """
    base      = _BASE_HI if lang == "hi" else _BASE_EN
    directive = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["en"])
    # Static base prompt first, static language directive second — both
    # are identical across every request for this language, so together
    # they form one unbroken cacheable prefix.
    return f"{base}\n\n{directive}"


# ══════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"


# ══════════════════════════════════════════════════════════
#  ERROR HANDLING
# ══════════════════════════════════════════════════════════

ERROR_MESSAGES = {
    "llm_unreachable": {"en": "I can't reach the local language model server."},
    "api_error":       {"en": "The local model server returned an error."},
    "env_error":       {"en": "Environmental error, please try again."},
}


def is_internet_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """
    Quick TCP probe — returns True when internet is reachable.

    OFFLINE-P5: only used now if ENABLE_WEB_FALLBACK is turned on; the
    core STT/LLM/TTS pipeline never calls this anymore.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((host, port))
        return True
    except OSError:
        return False


def is_llama_server_reachable(timeout: float = 2.0) -> bool:
    """Quick probe for the local llama.cpp server — replaces the old internet check."""
    try:
        resp = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=timeout)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def classify_error(exc: Exception) -> str:
    """
    Return 'llm_unreachable', 'api_error', or 'env_error'.

    OFFLINE-P5: replaces the old is_internet_available() gate — the
    thing that can now legitimately be "down" is the local llama.cpp
    server (whisper.cpp and Piper run in-process/subprocess and don't
    have a comparable "unreachable" state).
    """
    if not is_llama_server_reachable():
        return "llm_unreachable"
    api_related_types = (
        requests.exceptions.RequestException,
        ConnectionError,
        TimeoutError,
        socket.timeout,
        socket.gaierror,
    )
    if isinstance(exc, api_related_types):
        return "api_error"

    exc_name = type(exc).__name__.lower()
    exc_msg  = str(exc).lower()
    api_signals = (
        "api", "rate limit", "401", "403", "404", "429",
        "500", "502", "503", "504", "connection", "timeout",
        "llama", "endpoint",
    )
    if any(s in exc_name for s in api_signals) or \
       any(s in exc_msg  for s in api_signals):
        return "api_error"

    return "env_error"


def announce_error(exc: Exception, lang: str = "en") -> None:
    """
    Classify the exception and speak the appropriate English error message.
    Always uses the English voice regardless of the conversation language.
    Wrapped in its own try/except so a TTS failure cannot cascade.
    """
    try:
        kind = classify_error(exc)
        msg  = ERROR_MESSAGES[kind]["en"]
        logger.warning("Announcing error (%s): %s", kind, msg)
        speak(msg, lang="en")
    except Exception as report_exc:
        logger.error("Failed to announce error: %s", report_exc)


# ══════════════════════════════════════════════════════════
#  INPUT SANITIZATION  (FIX-P1 / FIX-P7)
# ══════════════════════════════════════════════════════════

def sanitize_text(text: Optional[str]) -> str:
    """
    Strip leading/trailing whitespace and collapse internal runs of
    whitespace to a single space.  Returns an empty string when the
    input is None, empty, or whitespace-only.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def is_blank(text: Optional[str]) -> bool:
    """Return True when text is None, empty, or whitespace-only."""
    return not text or not text.strip()


# ══════════════════════════════════════════════════════════
#  RAG ENGINE
# ══════════════════════════════════════════════════════════

class RAGEngine:

    def __init__(self) -> None:
        self.chunks:     List[str]              = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix                              = None
        self.ready                               = False

    def load_pdf(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning("RAG: PDF not found at '%s' — web/LLM only mode.", path)
            return False

        logger.info("RAG: Loading '%s' …", path)
        raw = self._extract_text(path)
        if not raw.strip():
            logger.warning("RAG: '%s' is empty — skipping.", path)
            return False

        self.chunks = self._chunk(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        self._build_index()
        self.ready  = True
        logger.info("RAG: '%s' indexed — %d chunks.", path, len(self.chunks))
        return True

    def retrieve(self, query: str) -> Tuple[str, float]:
        """
        Retrieve the top-K relevant chunks for *query* and return
        (context_string, best_score).  Returns ("", 0.0) when not ready.
        """
        if not self.ready or not self.chunks:
            return "", 0.0

        q_vec  = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        top_idx    = scores.argsort()[::-1][:TOP_K]
        best_score = float(scores[top_idx[0]])

        context = "\n\n".join(
            self.chunks[i] for i in top_idx if scores[i] > 0
        )
        return context, best_score

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _extract_text(path: str) -> str:
        doc   = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words  = text.split()
        step   = max(1, size - overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _build_index(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            # token_pattern=r"\S+" keeps Devanagari words intact;
            # the default pattern breaks on Unicode combining marks.
            token_pattern=r"\S+",
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)


# ══════════════════════════════════════════════════════════
#  WEB SEARCH FALLBACK
# ══════════════════════════════════════════════════════════
#
#  DuckDuckGo's Instant Answer API (api.duckduckgo.com) ONLY returns
#  data for Wikipedia-style "knowledge panel" entities. For a normal
#  search query (e.g. "Regard Network Solutions data centre specs") it returns an
#  empty AbstractText and an empty RelatedTopics list almost every
#  time — it is not a general web search endpoint. That's why the web
#  fallback used to come back empty even when the query was perfectly
#  searchable.
#
#  Fix: try the Instant Answer API first (cheap, fast, occasionally
#  useful), and if it returns nothing, fall back to scraping
#  DuckDuckGo's actual HTML results page, which returns real search
#  snippets for any query without needing an API key.
# ══════════════════════════════════════════════════════════

_DDG_HTML_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Matches the visible snippet text DuckDuckGo's HTML results page wraps
# each result in: <a class="result__snippet" ...>...text...</a>
_DDG_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _ddg_instant_answer(query: str) -> str:
    """Try DuckDuckGo's Instant Answer API. Returns '' if nothing usable."""
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={
            "q":            query,
            "format":       "json",
            "no_html":      "1",
            "skip_disambig":"1",
        },
        timeout=WEB_TIMEOUT,
        headers={"User-Agent": "DTBot"},
    )
    resp.raise_for_status()
    data = resp.json()
    snippets: List[str] = []

    if data.get("AbstractText"):
        snippets.append(data["AbstractText"])

    for topic in data.get("RelatedTopics", [])[:WEB_RESULTS]:
        text = topic.get("Text", "")
        if text:
            snippets.append(text)

    return " ".join(snippets).strip()


def _ddg_html_search(query: str) -> str:
    """
    Fall back to DuckDuckGo's HTML results page and scrape the visible
    result snippets. This is what actually behaves like "web search" —
    the Instant Answer API does not.
    """
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        timeout=WEB_TIMEOUT,
        headers={"User-Agent": _DDG_HTML_UA},
    )
    resp.raise_for_status()

    raw_snippets = _DDG_SNIPPET_RE.findall(resp.text)
    snippets: List[str] = []
    for raw in raw_snippets[:WEB_RESULTS]:
        text = _HTML_TAG_RE.sub("", raw)          # strip any nested tags
        text = sanitize_text(text)
        if text:
            snippets.append(text)

    return " ".join(snippets).strip()


def needs_web(query: str, score: float) -> bool:
    """
    Hit the web whenever the PDF didn't have a good answer, i.e. the
    retrieval score is below threshold — regardless of keywords.

    NOTE: this used to also require a time-sensitive keyword (today,
    latest, price, etc.) before firing, which meant a plain factual
    question missing from the PDF (e.g. "what ISO certs do you hold?")
    would silently fall through to the LLM's own (possibly wrong)
    general knowledge instead of searching the web. Now: PDF miss ==
    web search, full stop. WEB_KEYWORDS is kept around in case you want
    to reintroduce gating later, but it's unused here.
    OFFLINE-P5: short-circuits to False whenever ENABLE_WEB_FALLBACK is
    off (the default), so a pure-offline run never attempts a network
    call here at all.
    """
    if not ENABLE_WEB_FALLBACK:
        return False
    return score < PDF_THRESHOLD


def web_search(query: str) -> str:
    search_query = f"{query} Regard Network Solutions Delhi NCR"

    # ── Tier 1: Instant Answer API (fast, but rarely has data) ───────
    try:
        context = _ddg_instant_answer(search_query)
        if context:
            logger.debug("Web context via Instant Answer API (%d chars).", len(context))
            return context
        logger.debug("Instant Answer API returned nothing — trying HTML search.")
    except Exception as exc:
        logger.warning("Instant Answer API failed: %s", exc)

    # ── Tier 2: DuckDuckGo HTML results page (real search results) ──
    try:
        context = _ddg_html_search(search_query)
        if context:
            logger.debug("Web context via HTML search (%d chars).", len(context))
        else:
            logger.debug("HTML search returned no usable snippets either.")
        return context
    except Exception as exc:
        logger.warning("Web search failed (both tiers): %s", exc)
        # Do NOT call announce_error() here — _process_query() handles it.
        return ""


# ══════════════════════════════════════════════════════════
#  LOCAL MODEL INIT  (replaces GROQ CLIENT)
#  OFFLINE-P1/P2/P4: whisper.cpp and OpenWakeWord are loaded once here,
#  in-process, and reused for the life of the script — same idea as
#  the old Groq client, but nothing here makes a network call.
#  The llama.cpp server is NOT started here — it's a separate process
#  you must launch yourself (see SETUP.md); we only health-check it.
# ══════════════════════════════════════════════════════════

try:
    whisper_model = WhisperModel(
        WHISPER_MODEL,
        n_threads=WHISPER_THREADS,
        print_realtime=False,
        print_progress=False,
    )
    logger.info("whisper.cpp model '%s' loaded (%d threads).", WHISPER_MODEL, WHISPER_THREADS)
except Exception as _init_exc:
    logger.critical("Failed to load whisper.cpp model: %s", _init_exc)
    raise

try:
    _wake_kwargs = dict(inference_framework=WAKE_INFERENCE_FRAMEWORK)
    if WAKE_MODEL_PATH:
        _wake_kwargs["wakeword_models"] = [WAKE_MODEL_PATH]
    else:
        _wake_kwargs["wakeword_models"] = [WAKE_MODEL_NAME]
    wake_model = WakeWordModel(**_wake_kwargs)
    logger.info("OpenWakeWord model loaded (%s).", WAKE_MODEL_PATH or WAKE_MODEL_NAME)
except Exception as _init_exc:
    logger.critical("Failed to load OpenWakeWord model: %s", _init_exc)
    raise

if not is_llama_server_reachable():
    logger.warning(
        "llama.cpp server not reachable at %s — start it before speaking to "
        "the bot (see SETUP.md). The bot will still boot, but LLM replies "
        "will fail with 'llm_unreachable' until the server is up.",
        LLAMA_SERVER_URL,
    )
else:
    logger.info("llama.cpp server reachable at %s.", LLAMA_SERVER_URL)

_PIPER_AVAILABLE: bool = shutil.which(PIPER_BIN) is not None
if not _PIPER_AVAILABLE:
    logger.warning(
        "Piper binary '%s' not found on PATH — TTS will fall back to "
        "espeak. Install with: pip install piper-tts", PIPER_BIN,
    )


# ══════════════════════════════════════════════════════════
#  CONVERSATION HISTORY  (FIX-P3)
# ══════════════════════════════════════════════════════════

history: dict = {"en": [], "hi": []}


def _trim_history(lang: str) -> None:
    """Keep the history within MAX_HISTORY_ITEMS entries (oldest dropped first)."""
    lang_history = history[lang]
    if len(lang_history) > MAX_HISTORY_ITEMS:
        excess = len(lang_history) - MAX_HISTORY_ITEMS
        del lang_history[:excess]
        logger.debug("History trimmed: dropped %d oldest messages.", excess)


# ══════════════════════════════════════════════════════════
#  LLM  (FIX-P1, FIX-P2)
# ══════════════════════════════════════════════════════════

def get_ai_reply(user_text: str, lang: str, context: str) -> str:
    """
    Send *user_text* to the LLM and return the assistant's reply as a
    non-empty string.

    FIX-P1: Input is validated at the start of this function as a second
            line of defence (the main loop already checks, but belt-and-
            suspenders prevents a silent empty call to the API).

    FIX-P2: The function now ALWAYS returns a non-empty str or raises an
            exception.  The previous implicit None return (caused by the
            commented-out fallback) triggered a double error announcement
            and caused TTS to receive None.
    """
    # ── Input guard (FIX-P1) ──────────────────────────────
    clean_input = sanitize_text(user_text)
    if is_blank(clean_input):
        raise ValueError("get_ai_reply received empty or whitespace-only input.")

    lang_history = history[lang]
    lang_history.append({"role": "user", "content": clean_input})

    try:
        # CACHEABLE: this is the same string every time for a given lang —
        # base prompt + language directive, no per-query content mixed in.
        system = build_system(lang)

        # NOT cacheable (changes every query): the RAG context retrieved
        # for *this* user turn. It is appended as the LAST message below,
        # after history, so it never sits inside — and therefore never
        # breaks — the static system/history prefix.
        context_msg = (
            [{
                "role": "system",
                "content": (
                    "Use the following information silently to answer naturally.\n\n"
                    f"{context}\n\n"
                    f"{_LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE['en'])}"
                    ),
            }]
            if context else []
        )

        # Retry loop — the model occasionally returns an empty string on
        # the first attempt (observed with openai/gpt-oss-20b + Hindi).
        last_exc: Optional[Exception] = None
        for attempt in range(1, LLM_MAX_RETRIES + 2):  # +2 → 1 normal + N retries
            try:
                # Raise temperature on retries — empty responses are often
                # caused by the model being "stuck" at low temperature when
                # the prompt constraints are tight. 0.7 breaks the deadlock.
                temp = 0.4 if attempt == 1 else 0.7

                # OFFLINE-P2: plain HTTP POST to the local llama.cpp server's
                # OpenAI-compatible endpoint instead of the Groq SDK. Prompt
                # order kept identical to the cloud version — [static system]
                # [conversation history] [dynamic RAG context LAST] — even
                # though llama-server doesn't do Groq-style automatic prefix
                # caching, keeping the shape stable costs nothing and makes
                # a future switch to a caching-aware server a smaller diff.
                llm_resp = requests.post(
                    f"{LLAMA_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": CHAT_MODEL,
                        "messages": [{"role": "system", "content": system}, *lang_history, *context_msg],
                        "max_tokens": MAX_TOKENS,
                        "temperature": temp,
                    },
                    timeout=60,   # local CPU inference at 4-8 tok/s needs more headroom than a cloud call
                )
                llm_resp.raise_for_status()
                response = llm_resp.json()

                usage = response.get("usage", {})
                logger.info(
                    "Tokens — prompt: %s | completion: %s | total: %s",
                    usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"),
                )

                raw_reply = response["choices"][0]["message"]["content"]
                logger.debug("Raw LLM response (attempt %d): %r", attempt, raw_reply)

                reply = sanitize_text(raw_reply)
                if not is_blank(reply):
                    lang_history.append({"role": "assistant", "content": reply})
                    _trim_history(lang)
                    return reply

                # Empty response — log and retry if attempts remain
                logger.warning(
                    "LLM returned empty response on attempt %d/%d.",
                    attempt, LLM_MAX_RETRIES + 1,
                )
                last_exc = RuntimeError(
                    f"LLM returned an empty response (attempt {attempt})."
                )

            except Exception as api_exc:
                logger.warning("LLM API error on attempt %d: %s", attempt, api_exc)
                last_exc = api_exc
                if attempt <= LLM_MAX_RETRIES:
                    wait = 2 ** attempt   # 2s, 4s — handles llama-server transient/503 (e.g. still loading)
                    logger.info("Retrying in %ds …", wait)
                    time.sleep(wait)

        # All attempts exhausted — roll back and raise
        lang_history.pop()
        raise last_exc or RuntimeError("LLM failed after all retry attempts.")

    except Exception:
        # Roll back the user turn if it was appended before the failure.
        if lang_history and lang_history[-1]["role"] == "user":
            lang_history.pop()
        raise   # re-raise so the caller can handle and announce the error


# ══════════════════════════════════════════════════════════
#  CONTEXT BUILDER
# ══════════════════════════════════════════════════════════

def build_context(
    query: str,
    lang: str,
    rag_en: RAGEngine,
    rag_hi: RAGEngine,
) -> Tuple[str, str]:
    # Query BOTH indexes and keep whichever scores higher, rather than
    # picking the index purely off the detected conversation `lang`.
    #
    # Why: Whisper's audio-based language detection is not reliable for
    # Indian-accented English — it will sometimes tag a purely English
    # utterance as "hi". Under the old logic (rag = rag_hi if lang=="hi"
    # else rag_en) that meant an English query got searched ONLY against
    # the Hindi PDF, missed entirely, and fell through to "Source: None"
    # even though the English PDF had the answer sitting right there.
    # Scoring both and taking the max makes retrieval robust to language
    # mis-tagging in either direction, and once both PDFs hold the same
    # (translated) content, the same-script index will naturally win
    # anyway since TF-IDF match quality is higher against matching script.
    context_en, score_en = rag_en.retrieve(query)
    context_hi, score_hi = rag_hi.retrieve(query)

    if score_en >= score_hi:
        pdf_context, pdf_score = context_en, score_en
    else:
        pdf_context, pdf_score = context_hi, score_hi

    logger.debug(
        "PDF score — EN: %.3f | HI: %.3f | using: %s (threshold=%.2f)",
        score_en, score_hi, "EN" if score_en >= score_hi else "HI", PDF_THRESHOLD,
    )

    # Prioritize PDF. Only search web if PDF score is low.
    web_context = ""
    source      = "None"

    # FIX: `source` must reflect what is actually placed into `parts` below.
    if pdf_context:
        source = "PDF"

    if needs_web(query, pdf_score):
        logger.debug("Web search triggered (PDF score below threshold).")
        web_context = web_search(query)
        if web_context:
            source = "PDF+Web" if pdf_context else "Web"

    parts: List[str] = []
    if pdf_context:
        parts.append(f"[From Regard Network Solutions Knowledge Base]\n{pdf_context}")
    if web_context:
        parts.append(f"[From Web]\n{web_context}")

    return "\n\n".join(parts), source


# ══════════════════════════════════════════════════════════
#  MIC DEVICE RESOLUTION
# ══════════════════════════════════════════════════════════

def resolve_mic_device(name: str) -> Optional[str]:
    """
    Validate that *name* (or a substring of it) matches at least one input
    device, then return it as-is for use in sd.InputStream(device=...).

    sounddevice accepts a name string and does substring matching internally,
    so we don't need to resolve it to an index — we just verify it exists and
    log exactly which device will be used.

    Returns None when *name* is empty (falls through to system default) or
    raises RuntimeError when *name* is set but no matching input device is
    found.
    """
    if not name:
        logger.info("MIC_NAME not set — using system default input device.")
        return None

    devices = sd.query_devices()
    matches = [
        d for d in devices
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0
    ]

    if not matches:
        available = [
            f"  [{i}] {d['name']}"
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
        raise RuntimeError(
            f"MIC_NAME='{name}' did not match any input device.\n"
            f"Available input devices:\n" + "\n".join(available)
        )

    if len(matches) > 1:
        logger.warning(
            "MIC_NAME='%s' matched %d devices — using the first: '%s'. "
            "Make your MIC_NAME more specific if this is wrong.",
            name, len(matches), matches[0]["name"],
        )
    else:
        logger.info("Microphone resolved: '%s'", matches[0]["name"])

    # Return the user-supplied name string; sounddevice will do the
    # substring match itself when opening the stream.
    return name


# Resolve once at module load so startup errors are caught immediately.
_MIC_DEVICE: Optional[str] = resolve_mic_device(MIC_NAME)


# ══════════════════════════════════════════════════════════
#  VAD RECORDING
# ══════════════════════════════════════════════════════════

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        device=_MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        latency="high",    # larger internal buffer absorbs USB timing jitter
        callback=callback,
    )
    stream.start()

    speech_buffer: List[np.ndarray]   = []
    pre_buffer:    List[np.ndarray]   = []
    recording                          = False
    silence_start: Optional[float]    = None
    idle_clock                         = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            # Skip energy detection while bot is speaking — prevents its
            # own speaker output from being picked up and re-triggered.
            if _mic_muted:
                idle_clock = time.time()
                continue

            if rms >= ENERGY_THRESHOLD:
                idle_clock    = time.time()
                silence_start = None
                if not recording:
                    recording     = True
                    speech_buffer = list(pre_buffer)
                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None
    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None


# ══════════════════════════════════════════════════════════
#  TRANSCRIBE
# ══════════════════════════════════════════════════════════

def _has_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


# Common Hinglish (Roman-script Hindi) words. Used only to tell "this is
# genuinely Hinglish" apart from "Whisper mis-tagged plain English as
# Hindi" — both look identical to _has_devanagari() since neither has
# Devanagari script.
_HINGLISH_MARKERS = {
    "hai", "hain", "hoon", "hun", "kya", "kyu", "kyun", "kaise", "kahan",
    "kab", "kitna", "kitni", "aap", "aapka", "aapki", "aapke", "mujhe",
    "mujhko", "humein", "hume", "tumhe", "nahi", "nahin", "haan", "nhi",
    "acha", "accha", "theek", "bhai", "kripya", "raha", "rahi", "rahe",
    "karo", "karein", "kariye", "batao", "bataiye", "mein", "ka", "ki",
    "ke", "ko", "se", "aur", "ek", "hoga", "hogi",
}


def _looks_like_hinglish(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return any(w in _HINGLISH_MARKERS for w in words)


def _transcribe_once(audio: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Run one whisper.cpp transcription pass on an in-memory audio buffer.

    OFFLINE-P1: no cloud call, no temp wav file for the model itself —
    pywhispercpp accepts the numpy array directly. *audio* comes in as
    float32 with shape (N, 1) from capture_speech(); whisper.cpp wants
    a flat 1-D float32 array in [-1, 1], so it's squeezed here.
    """
    flat = np.squeeze(audio).astype(np.float32)

    audio_secs = len(flat) / SAMPLE_RATE
    stt_start = time.time()

    # Detect language first (whisper.cpp's own auto-detect), then run
    # the actual transcription pinned to that language for consistency
    # — mirrors the old two-piece (text, language) result from Groq's
    # verbose_json response.
    if language:
        detected_lang = language
    else:
        try:
            (detected_lang, _prob), _all_probs = whisper_model.auto_detect_language(
                flat, n_threads=WHISPER_THREADS
            )
        except Exception as exc:
            logger.warning("Language auto-detect failed (%s) — defaulting to 'en'.", exc)
            detected_lang = "en"

    try:
        segments = whisper_model.transcribe(
            flat,
            language=detected_lang,
            initial_prompt=STT_VOCAB_PROMPT,   # same vocabulary bias idea as Groq's `prompt`
        )
    except TypeError:
        # Some pywhispercpp versions don't accept initial_prompt as a
        # transcribe() kwarg (name/availability has shifted across
        # releases) — retry without it rather than crashing every turn.
        logger.debug("transcribe() rejected initial_prompt kwarg — retrying without it.")
        segments = whisper_model.transcribe(flat, language=detected_lang)
    stt_elapsed = time.time() - stt_start

    result_text = " ".join(seg.text for seg in segments).strip()
    logger.info(
        "STT [whisper.cpp/%s]: %.2fs for %.1fs of audio (%.2fx realtime)",
        WHISPER_MODEL, stt_elapsed, audio_secs,
        (stt_elapsed / audio_secs) if audio_secs else 0.0,
    )

    text = sanitize_text(result_text)          # FIX-P7: sanitize at source
    lang = (detected_lang or "en").strip().lower()

    # FIX-P12: DTBot only supports English and Hindi. Whisper may report
    # other language codes — anything that isn't explicitly "en" is
    # folded into "hi" here, so any non-English utterance still gets a
    # real answer via the Hindi/Hinglish pipeline.
    if lang != "en":
        lang = "hi"

    # Script override: if the transcribed text itself contains
    # Devanagari characters, trust that over Whisper's own language
    # guess — this catches cases where Whisper mis-tags Hindi speech
    # as something else but still renders it correctly in Devanagari.
    if _has_devanagari(text):
        lang = "hi"
    elif lang == "hi" and text.isascii() and not _looks_like_hinglish(text):
        # Whisper tagged this "hi" but the transcript is plain ASCII
        # with no Hinglish markers at all — this is Whisper mis-hearing
        # Indian-accented English as Hindi, not real Hinglish. Trust
        # the text and answer in English instead.
        lang = "en"

    return text, lang


def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """
    Transcription for conversation turns, via local whisper.cpp.
    Only "en" and "hi" are ever returned — see FIX-P12 in
    _transcribe_once() for how everything else is folded in.

    OFFLINE-P4: there is no separate "fast" wake-word variant anymore
    (see transcribe_fast removal) — wake detection is handled entirely
    by OpenWakeWord in listen_for_wake_word(), so this is the only STT
    entry point left, and it only ever runs after a real wake trigger
    or during an active conversation turn.
    """
    return _transcribe_once(audio)


# ══════════════════════════════════════════════════════════
#  WAKE WORD  (OFFLINE-P4: OpenWakeWord, replaces the old
#  is_wake_word() text match against a polled cloud transcript)
# ══════════════════════════════════════════════════════════

def listen_for_wake_word(poll_timeout: float) -> bool:
    """
    Block, listening continuously on the mic, until the configured
    OpenWakeWord model fires or *poll_timeout* seconds of silence pass
    with no detection.

    Returns True on detection, False on timeout (caller decides what
    "timeout" means — in the IDLE loop it's just "keep polling").

    Runs its own short-lived InputStream (separate from capture_speech's
    VAD stream) because it needs small, constant-size int16 frames
    (WAKE_FRAME_SAMPLES = 1280 samples / 80ms) rather than the
    variable-length float32 buffers capture_speech produces.
    """
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        device=_MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",                  # OpenWakeWord expects 16-bit PCM
        blocksize=WAKE_FRAME_SAMPLES,
        latency="high",
        callback=callback,
    )
    stream.start()
    start = time.time()
    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if time.time() - start >= poll_timeout:
                    return False
                continue

            # Don't let the bot's own TTS playback trigger itself.
            if _mic_muted:
                continue

            frame = chunk.reshape(-1)
            prediction = wake_model.predict(frame)
            if any(score >= WAKE_THRESHOLD for score in prediction.values()):
                logger.debug("Wake word prediction: %s", prediction)
                return True

            if time.time() - start >= poll_timeout:
                return False
    finally:
        stream.stop()
        stream.close()


# ══════════════════════════════════════════════════════════
#  TTS  (FIX-P5: reuse a single event loop)
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
#  MIC GATE
#  Prevents the bot's own speaker output from being picked up by the
#  always-on mic and re-triggered as a new user question.
#  Set True before TTS playback, cleared in a finally block so it is
#  always released even if playback crashes.
# ══════════════════════════════════════════════════════════

# Cooldown timestamp for IDLE error announcements — prevents the bot from
# repeating "I can't connect to the internet" every 30s while offline.
_last_idle_error_time: float = 0.0

_mic_muted: bool = False


def pick_voice(text: str, lang: str) -> str:
    """Returns a Piper .onnx model PATH now, not a cloud voice name."""
    if lang == "hi":
        return PIPER_MODEL_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return PIPER_MODEL_HI
    return PIPER_MODEL_EN


def speak(text: str, lang: str = "en") -> None:
    global _mic_muted
    logger.debug("TTS input: %r", text)

    if is_blank(text):
        logger.error("TTS input validation failed: text is empty or None.")
        _speak_direct(ERROR_MESSAGES["env_error"]["en"], PIPER_MODEL_EN)
        return

    voice = pick_voice(text, lang)
    logger.info("TTS [%s]: %s", voice, textwrap.shorten(text, width=80))

    _mic_muted = True          # gate mic BEFORE playback — prevents self-triggering
    try:
        if _PIPER_AVAILABLE and _speak_piper(text, voice):
            return
        logger.warning("Piper unavailable/failed — attempting espeak fallback.")
        if _speak_espeak(text, lang):
            return
        logger.error("All TTS engines failed for this utterance.")
    finally:
        _mic_muted = False     # ALWAYS release gate after playback


def _speak_piper(text: str, voice_model: str) -> bool:
    """
    OFFLINE-P3: synthesise *text* via a local Piper subprocess and play
    the resulting WAV through pygame. Returns True on success, False on
    any failure (caller falls back to espeak).

    Piper writes to stdout when --output_file is "-", so this pipes text
    in on stdin and writes the wav straight to a temp file for pygame to
    load — same load/play/cleanup shape as the old edge-tts path.
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run(
            [PIPER_BIN, "--model", voice_model, "--output_file", tmp_path],
            input=text.encode("utf-8"),
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not os.path.exists(tmp_path):
            raise RuntimeError("piper did not create output file.")

        wav_size = os.path.getsize(tmp_path)
        logger.debug("Generated WAV size: %d bytes", wav_size)
        if wav_size == 0:
            raise RuntimeError("piper produced a zero-byte WAV file.")

        try:
            pygame.mixer.music.load(tmp_path)
        except Exception as load_exc:
            raise RuntimeError(f"pygame failed to load WAV: {load_exc}") from load_exc

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        return True

    except Exception as exc:
        logger.error("piper error: %s", exc)
        return False

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _speak_espeak(text: str, lang: str) -> bool:
    """
    Offline TTS via espeak subprocess.
    Tries the language-specific voice first, falls back to English.

    Voice selection:
      - Hindi → tries 'hi' first, falls back to 'en' if not installed
      - English → uses 'en' directly
    espeak -s 140  : slightly slower than default (160) for clarity
    espeak -a 180  : slightly louder amplitude
    timeout=15     : hard cap so a stalled espeak can't block the bot
    """
    if not _ESPEAK_AVAILABLE:
        logger.error("espeak not found — cannot speak offline.")
        return False

    voices_to_try = (["hi"] if lang == "hi" else []) + ["en"]
    for voice in voices_to_try:
        try:
            subprocess.run(
                ["espeak", "-v", voice, "-s", "140", "-a", "180", text],
                check=True,
                timeout=15,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            logger.warning("espeak voice '%s' unavailable, trying next …", voice)
            continue
        except subprocess.TimeoutExpired:
            logger.error("espeak timed out.")
            return False
        except Exception as exc:
            logger.error("espeak error: %s", exc)
            return False

    logger.error("espeak: no usable voice found.")
    return False


def _speak_direct(text: str, voice: str) -> None:
    """
    Minimal TTS+playback path used only by speak()'s input-validation
    guard to announce env_error without any risk of recursion.
    Tries Piper first, then espeak, then gives up silently.
    """
    if _PIPER_AVAILABLE and _speak_piper(text, voice):
        return
    logger.warning("_speak_direct: piper failed, trying espeak.")
    if _speak_espeak(text, lang="en"):
        return
    logger.error("_speak_direct: all engines failed (giving up).")


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def print_banner(rag_en_ready: bool, rag_hi_ready: bool) -> None:
    status_en = "✅ PDF loaded" if rag_en_ready else "⚠️  PDF not found"
    status_hi = "✅ PDF loaded" if rag_hi_ready else "⚠️  PDF not found"
    mic_label = f"'{MIC_NAME}'" if MIC_NAME else "system default"
    llama_status = "✅ reachable" if is_llama_server_reachable() else "❌ NOT reachable — start llama-server!"
    piper_status = "✅ found" if _PIPER_AVAILABLE else "⚠️  not found — falling back to espeak"
    web_status = "enabled" if ENABLE_WEB_FALLBACK else "disabled (offline build default)"
    sep = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  DTBot 🤖  |  OFFLINE EDITION  |  Regard Network Solutions, Delhi NCR\n"
        f"{sep}\n"
        f"  STT (whisper.cpp) : model '{WHISPER_MODEL}', {WHISPER_THREADS} threads\n"
        f"  LLM (llama-server): {LLAMA_SERVER_URL} — {llama_status}\n"
        f"  TTS (Piper)       : {piper_status}\n"
        f"  Wake word         : {WAKE_MODEL_PATH or WAKE_MODEL_NAME} (threshold {WAKE_THRESHOLD})\n"
        f"  Web fallback      : {web_status}\n"
        f"  RAG (EN) status   : {status_en}\n"
        f"  RAG (HI) status   : {status_hi}\n"
        f"  PDF (EN) path     : {PDF_PATH_EN}\n"
        f"  PDF (HI) path     : {PDF_PATH_HI}\n"
        f"  PDF threshold     : {PDF_THRESHOLD}\n"
        f"  Microphone        : {mic_label}\n"
        f"  Max history       : {MAX_HISTORY_TURNS} turns per language\n"
        f"  Log level         : {_log_level_name}\n"
        f"  States            :\n"
        f"    😴 IDLE       — listening for wake word\n"
        f"    👂 LISTENING  — auto-detects your voice\n"
        f"    🔊 SPEAKING   — playing response\n"
        f"  Ctrl+C to quit\n"
        f"{sep}\n"
    )
    # Banner uses print intentionally — it is startup UX, not a log event.
    print(banner)


def state_label(state: State) -> str:
    return {
        State.IDLE:      "😴 IDLE",
        State.LISTENING: "👂 LISTENING",
        State.THINKING:  "🤔 THINKING",
        State.SPEAKING:  "🔊 SPEAKING",
    }[state]


# ══════════════════════════════════════════════════════════
#  MAIN LOOP  (FIX-P6: reply scoped per iteration)
# ══════════════════════════════════════════════════════════

def _process_query(
    user_text: str,
    lang: str,
    rag_en: RAGEngine,
    rag_hi: RAGEngine,
) -> Optional[str]:
    """
    Retrieve context for *user_text* and return the LLM reply string, or
    None on failure (error already announced inside this function).

    FIX-P6: By isolating query processing in its own function, the main
    loop never carries a stale `reply` value between iterations.
    """
    # ── FIX-P1: reject blank input before any API call ────
    clean = sanitize_text(user_text)
    if is_blank(clean):
        logger.warning("Ignoring blank user input (after sanitization).")
        return None

    logger.info("User [%s] › %s", lang.upper(), clean)
    logger.debug("Retrieving context …")

    context, source = build_context(clean, lang, rag_en, rag_hi)
    logger.info("Source: %s", source)
    logger.debug("Generating reply …")

    try:
        reply = get_ai_reply(clean, lang, context)
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        announce_error(exc, lang)
        return None   # error already announced; caller must not announce again

    logger.info("AI   [%s] › %s", lang.upper(), reply)
    return reply


def main() -> None:
    try:
        pygame.mixer.init()

        rag_en = RAGEngine()
        rag_hi = RAGEngine()
        rag_en.load_pdf(PDF_PATH_EN)
        rag_hi.load_pdf(PDF_PATH_HI)
        print_banner(rag_en.ready, rag_hi.ready)

        state = State.LISTENING
        lang  = "hi"

        speak("Hello! I am DTBot, your AI assistant from Regard Network Solutions. ", lang="hi")

        global _last_idle_error_time
        while True:

            # ── IDLE ──────────────────────────────────────
            # OFFLINE-P4: continuous local wake-word detection instead of
            # polling full cloud STT every IDLE_POLL_TIMEOUT seconds.
            if state == State.IDLE:
                logger.debug(state_label(state))
                try:
                    detected = listen_for_wake_word(poll_timeout=IDLE_POLL_TIMEOUT)
                except Exception as exc:
                    # Announce the error (e.g. mic/model failure) but only if
                    # enough time has passed — prevents the bot from repeating
                    # the message on every poll cycle while something's broken.
                    now = time.time()
                    if now - _last_idle_error_time >= 30.0:
                        announce_error(exc, "en")
                        _last_idle_error_time = now
                    logger.warning("Wake-word listening failed: %s", exc)
                    continue

                if not detected:
                    continue

                history["en"].clear()
                history["hi"].clear()
                state = State.LISTENING
                speak("Haan, mein sun raha hoon.", lang="hi")
                continue

            # ── LISTENING ─────────────────────────────────
            if state == State.LISTENING:
                logger.debug(state_label(state))
                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state = State.IDLE
                    speak(
                        "Mein idle mode mai jaa raha hoo, "
                        "Mujhe activate krne ke liye Hello boliyein.",
                        lang="hi",
                    )
                    continue

                try:
                    user_text, lang = transcribe(audio)
                except Exception as exc:
                    logger.error("Transcription failed: %s", exc)
                    announce_error(exc, lang)
                    continue

                # FIX-P1: reject blank transcription immediately
                if is_blank(user_text):
                    logger.debug("Blank transcription — skipping.")
                    continue

                # FIX-P6: reply scoped here; no shared mutable state
                state = State.THINKING
                logger.info(state_label(state))
                reply = _process_query(user_text, lang, rag_en, rag_hi)

                if reply is None:
                    # Error was already announced inside _process_query;
                    # just go back to listening without a second announcement.
                    state = State.LISTENING
                    continue

                state = State.SPEAKING

                # ── SPEAKING (inline, scoped to this reply) ───────────
                logger.info(state_label(state))
                speak(reply, lang)
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
    except Exception as exc:
        logger.critical("Fatal error in main loop: %s", exc, exc_info=True)
        try:
            announce_error(exc, "en")
        except Exception:
            pass
    finally:
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        # No asyncio loop to close anymore — Piper runs as a subprocess
        # per utterance rather than via a persistent async client.


if __name__ == "__main__":
    main()