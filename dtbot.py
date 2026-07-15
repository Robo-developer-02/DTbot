"""
THIS CODE ONLY GIVES THE MESSAGE THAT IT IS MOVING , BUT ACTUALLY DOESN'T MOVE . 


============================================================
  DTBot — RAG-Powered Speech-to-Speech Chatbot
  Production Release
============================================================

  Architecture
  ─────────────────────────────────────────────────────────
  • Hindi queries   → retrieved directly against hindi_details.pdf (rag_hi)
  • English queries → retrieved directly against english_details.pdf (rag_en)
  • Language detection picks the engine; the query is never translated.
  • Web fallback only fires when PDF score is below threshold AND the
    query contains time-sensitive keywords.

  Additional installation
  ─────────────────────────────────────────────────────────
    Offline TTS fallback uses espeak directly (no pyttsx3):
    ```
    sudo apt install espeak espeak-data libespeak-dev
    ```

  See CHANGELOG.md for the history of fixes made to reach this
  production build (FIX-P1..P9).
============================================================
"""

# ── Standard library ──────────────────────────────────────
import asyncio
import atexit
import logging
import os
import queue
import re
import socket
import tempfile
import textwrap
import threading
import time
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────
import fitz
import numpy as np
import pygame
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import edge_tts
import subprocess

# Movement control — commands are sent to an ESP32 (jumper-wired UART:
# Pi TX -> ESP32 RX, Pi RX -> ESP32 TX, common GND) which drives the
# motors itself. Wrapped in try/except so the rest of the bot (RAG,
# TTS, STT, etc.) still runs fine on dev machines without pyserial
# installed or without the ESP32 connected.
try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False

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

# MODIFY: Even with LOG_LEVEL=DEBUG (for our own dtbot logs), the groq
# SDK's internal HTTP client (httpx/httpcore) logs full request/response
# dumps — headers, rate-limit info, etc. — at DEBUG level, which is pure
# noise here. Force those loggers to WARNING regardless of our own level.
for _noisy_logger in ("groq", "groq._base_client", "httpx", "httpcore"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════
#  API KEY
# ══════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

logger.info("API key loaded.")


# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════

STT_MODEL      = "whisper-large-v3"        # full quality — used for conversation
STT_MODEL_FAST = "whisper-large-v3-turbo"  # 3× faster — used for IDLE wake-word only
CHAT_MODEL     = "openai/gpt-oss-120b"
MODEL_CONTEXT_LIMIT = 131_072   # combined prompt+completion tokens for this model on Groq

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16_000
CHANNELS    = 1
MAX_TOKENS  = 250   # gpt-oss-120b is a reasoning model — it spends tokens on
                     # internal chain-of-thought before the visible answer.
                     # 120 was too tight: on longer RAG-context prompts the
                     # model could burn the whole budget on reasoning and
                     # get cut off before emitting any visible reply, coming
                     # back as an empty string despite a 200 OK. The system
                     # prompt already hard-caps the SPOKEN answer to 1–2
                     # sentences / 40 words, so this only gives headroom for
                     # reasoning — it does not make replies longer.

# ── History ───────────────────────────────────────────────
MAX_HISTORY_TURNS = 10
MAX_HISTORY_ITEMS = MAX_HISTORY_TURNS * 2

# ── LLM retry policy ───────────────────────────────────────
# Two separate retry budgets, because the two failure modes need very
# different handling:
#   - API_ERROR:   network/rate-limit/5xx — needs backoff, capped low
#                  so a genuinely dead connection doesn't hang the bot.
#   - EMPTY_REPLY: the model returned "" (observed occasionally with
#                  openai/gpt-oss-20b, especially in Hindi). This is
#                  NOT a server problem, so retrying immediately (no
#                  sleep) costs almost no extra latency, and giving it
#                  several more chances makes an empty final answer
#                  very rare.
LLM_API_MAX_RETRIES   = 2
LLM_EMPTY_MAX_RETRIES = 4

# FALLBACK_REPLY: spoken when the LLM is still empty after every retry.
# This is NOT a server/API error (the API call itself succeeded — it just
# returned no visible text), so it must never be routed through
# announce_error()/ERROR_MESSAGES, which would falsely tell the user the
# server or internet is down when it isn't.
FALLBACK_REPLY = "Sorry, I couldn't generate a response. Could you please ask again?"

# ── Response cache ────────────────────────────────────────
# Simple in-memory LRU keyed on (sanitized query, lang). Repeated questions
# (greetings, FAQs) skip RAG + the LLM call entirely — the biggest latency
# win available without touching the RAG/LLM architecture. OrderedDict is
# stdlib, so no new dependency; capped size keeps memory bounded on the Pi.
RESPONSE_CACHE_MAX_SIZE = 200

# ── Reliability timeouts (watchdog) ────────────────────────
# Bounds on individual slow components so one hung call (bad USB audio,
# a stalled Groq/edge-tts connection) can't freeze the whole bot forever.
# Web search already had WEB_TIMEOUT; these cover the other three.
STT_TIMEOUT_SECS = 20   # Groq transcription call
LLM_TIMEOUT_SECS = 30   # Groq chat completion call (was hardcoded 30 inline)
TTS_TIMEOUT_SECS = 15   # edge-tts synthesis (network call)

# ── Max recording duration ────────────────────────────────
# Hard ceiling on a single utterance so a stuck-open mic, background
# noise, or a user who keeps talking can't record forever and delay
# processing indefinitely.
MAX_RECORDING_SECS = 20.0

# ── Health logging ─────────────────────────────────────────
HEALTH_LOG_INTERVAL = 60.0   # seconds between CPU/RAM/temp/internet log lines

# ── RAG settings ──────────────────────────────────────────
PDF_PATH_EN   = "/home/dt/Desktop/DTown/Regard_Network_Solutions_Knowledge_Base_EN.pdf"
PDF_PATH_HI   = "/home/dt/Desktop/DTown/Regard_Network_Solutions_Knowledge_Base_HI.pdf"
CHUNK_SIZE    = 300   # smaller chunks → less context noise fed to LLM
CHUNK_OVERLAP = 50
TOP_K         = 3     # was 5; 3 × 300 words is enough context, keeps prompt small
PDF_THRESHOLD = 0.10

# ── Web fallback ──────────────────────────────────────────
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
ENERGY_THRESHOLD     = 0.10   # seed value only — see DYNAMIC THRESHOLD below
SILENCE_AFTER_SPEECH = 0.8   # was 1.2 — saves 0.4s per turn at end of every utterance
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.2   # was 0.1 — halves USB callback frequency; fixes retire_capture_urb
IDLE_TIMEOUT         = 15.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Dynamic energy threshold ──────────────────────────────
# A fixed ENERGY_THRESHOLD either over-triggers in loud venues (stage
# events, crowded halls) or under-triggers in quiet rooms. Instead, the
# trigger level now tracks the room's actual ambient noise floor via an
# exponential moving average (EMA), updated only during confirmed
# silence (never while the user is mid-utterance or the bot is
# speaking), and clamped to a sane min/max range.
NOISE_EMA_ALPHA       = 0.05   # smoothing factor — higher = adapts to noise faster
THRESHOLD_MULTIPLIER  = 3.0    # speech must exceed (noise floor × this) to trigger
DYNAMIC_THRESHOLD_MIN = 0.09   # floor — never require less than this even in silence
DYNAMIC_THRESHOLD_MAX = 0.35   # ceiling — never require more than this even in loud rooms

# ── Microphone selection ──────────────────────────────────
# Set MIC_NAME in your .env to select a microphone by name (or substring).
# Leave empty (or unset) to use the system default input device.
# Example: MIC_NAME=USB PnP Sound Device
# Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to
# list all available device names on your system.
MIC_NAME = os.getenv("MIC_NAME", "").strip()

# ── Wake words ────────────────────────────────────────────
WAKE_WORDS = ["hello", "hey", "hello dtbot", "hey dtbot", "dtbot"]

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is DTBot. You are the official AI assistant and "
    "virtual representative of Regard Network Solutions Ltd., a premier "
    "turnkey system integration company headquartered in Delhi NCR, India, "
    "specializing in end-to-end data centre build, enterprise networking, "
    "industrial building management systems, intelligent electronic "
    "manufacturing (at our Noida facility), and advanced surveillance "
    "frameworks. "
    "RESPONSE LENGTH — CRITICAL: You are a VOICE assistant. Your reply "
    "will be spoken aloud. Limit every response to 1–2 sentences maximum. "
    "Never exceed 40 words. Do not elaborate unless the user explicitly "
    "asks for more detail. "
    "Always represent Regard Network Solutions positively, professionally, and "
    "confidently. "
    "If users ask about another company or compare companies, briefly "
    "and politely redirect the conversation toward Regard Network Solutions, "
    "highlight Regard Network Solutions' strengths, and do not make negative comments or "
    "false claims about other companies. "
    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "
    "If Regard Network Solutions-specific information is unavailable, search the web first; "
    "if not connected to the internet, answer naturally using general "
    "knowledge when appropriate. "
    "Do not use bullet points or markdown."
)
_BASE_HI = (
    "Aapka naam DTBot hai. Aap Regard Network Solutions Ltd. ke official "
    "AI assistant aur virtual representative hain, jo Delhi NCR, India mein "
    "headquartered ek premier turnkey system integration company hai, "
    "jo end-to-end data centre build, enterprise networking, industrial "
    "building management systems, intelligent electronic manufacturing "
    "(Noida plant mein) aur advanced surveillance frameworks mein "
    "specialize karti hai. "
    "JAWAB KI LAMBAI — ZAROORI: Aap ek VOICE assistant hain. Aapka jawab "
    "bol ke sunaya jayega. Har jawab sirf 1–2 sentence mein dein. "
    "Kabhi 40 shabdon se zyada mat likhein. "
    "Hamesha Regard Network Solutions ko positive, professional aur confident "
    "tarike se represent karein. Kisi doosri company ke baare mein "
    "poocha jaye ya comparison ho to short aur polite tarike se baat ko "
    "Regard Network Solutions ki taraf le jaayein, iski strengths highlight "
    "karein, aur kisi company ke baare mein negative ya false claims na "
    "karein. "
    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "
    "Agar Regard Network Solutions ke sambandhit jankari available na ho to web search karke "
    "jawab dein; agar internet connect na ho to natural jawab dein. "
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
        "IMPORTANT: Aap SIRF Hindi ya Hinglish mein jawab dein, "
        "chahe user kisi bhi bhasha mein likhein. "
        "Kabhi bhi kisi aur bhasha mein jawab na dein."
    ),
}


# ══════════════════════════════════════════════════════════
#  MOVEMENT / ESP32 MOTOR CONTROL
# ══════════════════════════════════════════════════════════
#
#  Adds physical movement support (front / back / left / right) on top
#  of the existing chatbot. The motors are wired to an ESP32, not the
#  Pi's own GPIO pins — the Pi talks to the ESP32 over a jumper-wired
#  UART link (Pi TX -> ESP32 RX, Pi RX -> ESP32 TX, common GND) and the
#  ESP32's own firmware toggles its GPIO pins in response. This section
#  is fully self-contained and does not touch RAG, TTS, STT, history,
#  or caching.
#
#  Movement model:
#    • front / back  → tell the ESP32 to drive that motor for the
#      requested duration.
#    • left / right  → these are TURNS, not lateral holds. A "move
#      right for 3 seconds" request pivots right for a short fixed
#      pulse (TURN_PULSE_DURATION), then drives FRONT for the
#      requested duration — i.e. "turn right, then go ahead for 3s".
#
#  Serial protocol (matches the ESP32 sketch's Serial2 RPi-command
#  parser): each command is a newline-terminated ASCII line of the form
#  "<dir>,<seconds>\n", e.g. "F,3\n" (front, 3s), "L,0\n" (left — the
#  ESP32 ignores the number for turns and always pulses for its own
#  fixed defaultTurnTime), "S,0\n" (stop). dir is one of F/B/L/R/S.
#  The ESP32 owns the actual motor timing/soft-start/auto-stop once it
#  receives a command — Python just needs to send the right line.

ESP32_SERIAL_PORT = "/dev/serial0"  # jumper-wired UART; use /dev/ttyUSB0 if using a USB-serial adapter instead
ESP32_BAUD_RATE = 115200

_DIRECTION_COMMANDS = {
    "front": "F",
    "back": "B",
    "left": "L",
    "right": "R",
}

# Stop character the ESP32 sketch matches in its command switch.
_STOP_COMMAND = "S"

# left/right are pivot-only commands — a request in that direction
# always resolves to a short turn followed by a forward run.
_TURN_DIRECTIONS = ("left", "right")

# Fixed pulse length for the pivot phase of a turn. Not user-controlled.
TURN_PULSE_DURATION = 0.5  # seconds

# Safety cap: no single movement command may run longer than this, no
# matter what the user asks for. Applies to the forward/back run length
# (the turn pulse above is separate and always short).
MAX_MOVE_DURATION = 5  # seconds

_esp32: Optional["serial.Serial"] = None
if _SERIAL_AVAILABLE:
    try:
        _esp32 = serial.Serial(ESP32_SERIAL_PORT, ESP32_BAUD_RATE, timeout=1)
        time.sleep(2)  # give the ESP32 a moment to reset after the port opens
        logger.info(
            "Connected to ESP32 over serial (%s @ %d baud).",
            ESP32_SERIAL_PORT, ESP32_BAUD_RATE,
        )
    except Exception as exc:
        logger.warning(
            "Could not open serial connection to ESP32 (%s) — movement "
            "commands will be logged only, no motors will move: %s",
            ESP32_SERIAL_PORT, exc,
        )
        _esp32 = None
else:
    logger.warning(
        "pyserial not installed — movement commands will be logged only. "
        "Install with: pip install pyserial"
    )


def _send_to_esp32(direction: str, seconds: int = 0) -> None:
    """Send a '<direction>,<seconds>' newline-terminated line to the
    ESP32, matching its Serial2 command parser (charAt(0) = direction,
    substring(2) = whole seconds). *direction* must be one of
    F/B/L/R/S. For turns (L/R) and stop (S) the ESP32 ignores the
    seconds value, so 0 is fine there."""
    if _esp32 is None:
        return
    line = f"{direction},{int(seconds)}\n"
    try:
        _esp32.write(line.encode("utf-8"))
    except Exception as exc:
        logger.warning("Failed to send '%s' to ESP32: %s", line.strip(), exc)


def _esp32_cleanup() -> None:
    """Tell the ESP32 to stop and close the serial port on exit so
    nothing is left driving a motor after the process ends."""
    if _esp32 is None:
        return
    try:
        stop()
        _esp32.close()
    except Exception as exc:
        logger.warning("ESP32 serial cleanup on exit failed: %s", exc)


atexit.register(_esp32_cleanup)

# Synonym map → canonical direction. Includes English + Hindi/Hinglish
# phrasing so voice commands are recognized regardless of which
# language the user is speaking (matches transcribe()'s "en"/"hi" tags).
_DIRECTION_SYNONYMS = {
    "front": [
        "move ahead", "go ahead", "go forward", "forward", "ahead", "straight",
        "आगे बढ़ो", "आगे जाओ", "आगे चलो", "सीधे चलो", "सीधा जाओ", "फॉरवर्ड", "आगे",
    ],
    "back": [
        "move back", "go back", "backward", "reverse",
        "पीछे हटो", "पीछे जाओ", "पीछे चलो", "रिवर्स", "पीछे",
    ],
    "left": [
        "turn left", "move left", "go left", "left",
        "बाएं मुड़ो", "बायें मुड़ो", "बाएं जाओ", "बायें जाओ", "लेफ्ट", "बाएं", "बायें",
    ],
    "right": [
        "turn right", "move right", "go right", "right",
        "दाएं मुड़ो", "दायें मुड़ो", "दाएं जाओ", "दायें जाओ", "राइट", "दाएं", "दायें",
    ],
}

# Flattened (phrase, direction) pairs, longest phrase first so that e.g.
# "go ahead" is matched before the bare "ahead".
_DIRECTION_PHRASES: List[Tuple[str, str]] = sorted(
    (
        (phrase, direction)
        for direction, phrases in _DIRECTION_SYNONYMS.items()
        for phrase in phrases
    ),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

# Words/phrases that mean "stop moving right now" — checked before
# movement direction so an in-progress move can always be interrupted.
_STOP_PHRASES = [
    "stop moving", "stop robot", "stop", "halt", "freeze",
    "रुक जाओ", "रुको", "बंद करो", "स्टॉप", "रुक",
]

# Direction names in the user's language, for spoken replies.
_DIRECTION_LABELS = {
    "en": {"front": "front", "back": "back", "left": "left", "right": "right"},
    "hi": {"front": "आगे", "back": "पीछे", "left": "बाएं", "right": "दाएं"},
}

# ── Background movement state ──────────────────────────────
# move() runs on a worker thread so a long movement never blocks the
# mic/STT loop — the bot can keep listening (and hear a "stop") while
# it's driving.
_move_thread: Optional[threading.Thread] = None
_move_stop_event = threading.Event()
_move_lock = threading.Lock()


def stop() -> None:
    """Tell the ESP32 to turn all direction outputs off (halt all movement)."""
    _send_to_esp32(_STOP_COMMAND, 0)


def stop_movement() -> None:
    """
    Immediately interrupt any in-progress move() call and tell the
    ESP32 to stop. Safe to call even if nothing is currently moving.
    """
    _move_stop_event.set()
    stop()
    if _move_thread is not None and _move_thread.is_alive():
        _move_thread.join(timeout=1.0)


def _sleep_interruptible(duration: float) -> None:
    """Sleep in small steps so _move_stop_event can cut it short."""
    elapsed = 0.0
    step = 0.05
    while elapsed < duration and not _move_stop_event.is_set():
        time.sleep(min(step, duration - elapsed))
        elapsed += step


def _drive(direction: str, seconds: float, label: str) -> None:
    """Tell the ESP32 to run one motor for *seconds*, print status, then
    stop. The ESP32 itself owns the soft-start ramp and auto-stop timer
    once it gets the command — we just send the single line with the
    whole-second duration (turns ignore this number on the ESP32 side
    and always use its own fixed pulse) and sleep here so this call
    blocks for the same span. The trailing stop() is a safety net in
    case our sleep and the ESP32's own timer drift apart."""
    command = _DIRECTION_COMMANDS[direction]
    _send_to_esp32(command, max(1, round(seconds)))
    print(f"Moving {label} for {seconds} seconds")
    _sleep_interruptible(seconds)
    stop()


def _move_worker(direction: str, duration: float) -> None:
    with _move_lock:
        _move_stop_event.clear()
        stop()  # ensure a clean state before driving anything

        if direction in _TURN_DIRECTIONS:
            # Turn is a short fixed pulse — not user-controlled — then
            # continue straight ahead for the requested duration.
            print(f"Turning {direction} for {TURN_PULSE_DURATION} seconds")
            _drive(direction, TURN_PULSE_DURATION, direction)
            if _move_stop_event.is_set():
                return
            _drive("front", duration, "front")
        else:
            _drive(direction, duration, direction)


def move(direction: str, duration: float) -> None:
    """
    Execute a movement command.

    front/back: drive that pin HIGH for *duration* seconds (capped at
    MAX_MOVE_DURATION).

    left/right: pivot in that direction for a short fixed pulse
    (TURN_PULSE_DURATION), then drive front for *duration* seconds
    (capped at MAX_MOVE_DURATION) — i.e. "turn, then go ahead".

    Runs on a background thread so the caller is not blocked, and any
    movement already in progress is interrupted first.
    """
    global _move_thread

    direction = direction.lower().strip()
    if direction not in _DIRECTION_COMMANDS:
        logger.warning("move() called with unknown direction: %s", direction)
        return

    duration = min(duration, MAX_MOVE_DURATION)

    stop_movement()  # interrupt anything already running first

    _move_thread = threading.Thread(
        target=_move_worker, args=(direction, duration), daemon=True
    )
    _move_thread.start()


def extract_direction(text: str) -> Optional[str]:
    """
    Return the canonical direction ('front' / 'back' / 'left' / 'right')
    found in *text*, or None if no direction phrase is present.
    """
    lowered = sanitize_text(text).lower()
    for phrase, direction in _DIRECTION_PHRASES:
        if phrase in lowered:
            return direction
    return None


def extract_duration(text: str) -> int:
    """
    Return the integer number of seconds found in *text* (e.g. "for 5
    seconds", "3 sec", "10 seconds"). Defaults to 3 seconds if no
    duration is specified.
    """
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    return 3


def is_move_command(text: str) -> bool:
    """Return True if *text* contains a recognized movement direction."""
    return extract_direction(text) is not None


def is_stop_command(text: str) -> bool:
    """Return True if *text* is asking the robot to stop moving."""
    lowered = sanitize_text(text).lower()
    return any(phrase in lowered for phrase in _STOP_PHRASES)


def movement_reply(direction: str, duration: float, lang: str) -> str:
    """
    Localized, TTS-friendly confirmation for a movement command that
    was accepted and started, in the user's own language (falls back
    to English for any lang other than 'hi').
    """
    labels = _DIRECTION_LABELS.get(lang, _DIRECTION_LABELS["en"])
    label = labels[direction]

    if direction in _TURN_DIRECTIONS:
        if lang == "hi":
            return (
                f"ठीक है, पहले {label} मुड़ रहा हूँ, फिर {duration} सेकंड के लिए "
                f"आगे बढ़ रहा हूँ।"
            )
        return f"Turning {direction}, then moving front for {duration} seconds."

    if lang == "hi":
        return f"ठीक है, {label} की ओर {duration} सेकंड के लिए बढ़ रहा हूँ।"
    return f"Moving {direction} for {duration} seconds."


def stopped_reply(lang: str) -> str:
    """Localized confirmation that movement was halted."""
    return "रुक गया।" if lang == "hi" else "Stopped."


def duration_limit_reply(direction: str, lang: str) -> str:
    """
    User-facing (and TTS-spoken) reply for a movement request that
    exceeds MAX_MOVE_DURATION. Framed as a deliberate safety policy
    rather than a system limitation, in the user's own language.
    """
    labels = _DIRECTION_LABELS.get(lang, _DIRECTION_LABELS["en"])
    label = labels[direction]

    if lang == "hi":
        return (
            f"सुरक्षा कारणों से, मैं {label} की दिशा में एक बार में अधिकतम "
            f"{MAX_MOVE_DURATION} सेकंड तक ही जा सकता हूँ। कृपया छोटे-छोटे "
            f"आदेश एक के बाद एक दें।"
        )
    return (
        f"For safety, I can only move {direction} for up to "
        f"{MAX_MOVE_DURATION} seconds at a time. "
        f"Please give me shorter commands, one after another."
    )


@lru_cache(maxsize=4)
def build_system(lang: str) -> str:
    """
    Static system prompt ONLY — no RAG/web context.

    PROMPT CACHING: this string never changes for a given `lang` during
    the process lifetime, so it's memoized with @lru_cache instead of
    being rebuilt/re-joined on every single request — maxsize=4
    comfortably covers "en"/"hi" plus headroom.

    Context is NEVER mixed into the system prompt; it's appended to the
    trailing user turn instead (see _build_turn_content() and
    get_ai_reply()). That means the system prompt AND the prior-turns
    prefix stay byte-identical across calls, so Groq's own prompt cache
    can hit on the growing prefix too, not just our local Python object.
    """
    base      = _BASE_HI if lang == "hi" else _BASE_EN
    directive = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["en"])
    # Language directive stays at the END of the static prompt so it is
    # the last STATIC thing the model reads — maximising instruction-
    # following — before the trailing, per-turn context/question block.
    return f"{base}\n\n{directive}"


def _build_turn_content(user_text: str, context: str) -> str:
    """
    Compose the content of the CURRENT user turn only: the question,
    followed by RAG/web context if any. This is the one part of the
    prompt that legitimately changes every call, so it is kept at the
    very end, after the static system prompt and the unmodified prior
    history — maximising how much of the request can be served from
    Groq's prompt cache.
    """
    if not context:
        return user_text
    return (
        f"{user_text}\n\n"
        f"[Reference information — use silently to answer naturally, "
        f"never mention this section or its source]\n{context}"
    )


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

# Only English is used for error announcements — see announce_error().
ERROR_MESSAGES = {
    "no_internet": "I can't connect to the internet.",
    "api_error":   "I can't connect to the server.",
    "env_error":   "Environmental error, please try again.",
}


def is_internet_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """
    Quick TCP probe — returns True when internet is reachable.

    NOTE: timeout is passed directly to create_connection() rather than
    via socket.setdefaulttimeout(), which would mutate the process-wide
    default socket timeout and affect every other socket/requests call
    in the app for the rest of the program's life.
    """
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


_last_health_log_time: float = 0.0


def log_health() -> None:
    """
    Log CPU load, RAM usage, CPU temperature, and internet status.
    Self-throttled to once per HEALTH_LOG_INTERVAL — safe to call from a
    hot loop (e.g. every audio chunk in capture_speech()), since it's a
    no-op until the interval has elapsed. Uses only stdlib (/proc, /sys,
    os.getloadavg) rather than psutil, per the "no new dependencies"
    constraint.
    """
    global _last_health_log_time
    now = time.time()
    if now - _last_health_log_time < HEALTH_LOG_INTERVAL:
        return
    _last_health_log_time = now

    try:
        load1, _, _ = os.getloadavg()   # 1-minute load average, stands in for "CPU usage"
    except (OSError, AttributeError):
        load1 = -1.0

    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        total  = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1))
        avail  = int(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1))
        ram_pct = 100.0 * (1 - avail / total)
    except Exception:
        ram_pct = -1.0

    try:
        # Raspberry Pi OS exposes SoC temperature here, in millidegrees C.
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            cpu_temp = int(f.read().strip()) / 1000.0
    except Exception:
        cpu_temp = -1.0   # not available on this platform

    net_ok = is_internet_available()
    logger.info(
        "HEALTH load1min=%.2f ram_used=%.1f%% cpu_temp=%.1fC internet=%s",
        load1, ram_pct, cpu_temp, "up" if net_ok else "down",
    )


def classify_error(exc: Exception) -> str:
    """Return 'no_internet', 'api_error', or 'env_error'."""
    if not is_internet_available():
        return "no_internet"
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
        "api", "groq", "rate limit", "401", "403", "404", "429",
        "500", "502", "503", "504", "connection", "timeout",
        "network", "ssl", "host", "dns", "edge_tts", "endpoint",
    )
    if any(s in exc_name for s in api_signals) or \
       any(s in exc_msg  for s in api_signals):
        return "api_error"

    return "env_error"


def announce_error(exc: Exception) -> None:
    """
    Classify the exception and speak the appropriate English error message.
    Always uses the English voice regardless of the conversation language
    (error messages are English-only — see ERROR_MESSAGES).
    Wrapped in its own try/except so a TTS failure cannot cascade.
    """
    try:
        kind = classify_error(exc)
        msg  = ERROR_MESSAGES[kind]
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
#  search query (e.g. "DTown Robotics drone specs") it returns an
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
    Only hit the web when BOTH conditions are true:
      1. PDF score is below threshold (PDF didn't have a good answer)
      2. The query contains a time-sensitive keyword
    Without the keyword gate, web_search() fired on every greeting and
    general question, adding 3–8 seconds of latency to most turns.
    """
    q = query.lower()
    return score < PDF_THRESHOLD and any(kw in q for kw in WEB_KEYWORDS)


def web_search(query: str) -> str:
    search_query = f"{query} DTown Robotics DTR Noida"

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
#  GROQ CLIENT
# ══════════════════════════════════════════════════════════

try:
    # max_retries=0: the SDK's own built-in retry-on-429/5xx logic is
    # disabled here on purpose. Our get_ai_reply() already implements
    # a full retry policy (LLM_API_MAX_RETRIES / LLM_EMPTY_MAX_RETRIES)
    # with its own logging — leaving the SDK's retries enabled stacks a
    # second, invisible retry layer on top (it silently sleeps on 429
    # per Retry-After before our code ever sees the result), which is
    # what caused the unexplained multi-second stall.
    client = Groq(api_key=GROQ_API_KEY, max_retries=0)
    logger.info("Groq client initialised.")
except Exception as _init_exc:
    logger.critical("Failed to initialise Groq client: %s", _init_exc)
    raise


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
#  RESPONSE CACHE
#  Simple in-memory LRU: key = sanitized query + lang, value = reply.
#  OrderedDict gives O(1) LRU via move_to_end()/popitem(last=False) —
#  no new dependency, no separate class needed for something this small.
# ══════════════════════════════════════════════════════════

_response_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(query: str, lang: str) -> str:
    return f"{lang}:{sanitize_text(query).lower()}"


def cache_get(query: str, lang: str) -> Optional[str]:
    key = _cache_key(query, lang)
    if key not in _response_cache:
        return None
    _response_cache.move_to_end(key)   # mark as recently used
    return _response_cache[key]


def cache_set(query: str, lang: str, reply: str) -> None:
    # Never cache empty replies or the spoken fallback — those aren't
    # real answers and would just serve stale non-answers on repeat.
    if is_blank(reply) or reply == FALLBACK_REPLY:
        return
    key = _cache_key(query, lang)
    _response_cache[key] = reply
    _response_cache.move_to_end(key)
    if len(_response_cache) > RESPONSE_CACHE_MAX_SIZE:
        _response_cache.popitem(last=False)   # evict least-recently-used


# ══════════════════════════════════════════════════════════
#  TOKEN USAGE
# ══════════════════════════════════════════════════════════

def log_token_usage(response) -> None:
    """
    Log prompt / completion / total tokens for this call, and how much
    of the model's context window remains. Groq returns this in
    `response.usage` on every chat completion — no extra API call needed.
    Wrapped defensively since `usage` shape can vary by SDK version.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.debug("No usage data on response — skipping token log.")
        return

    prompt_tokens     = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens       = getattr(usage, "total_tokens", None)

    if total_tokens is None:
        return

    remaining = MODEL_CONTEXT_LIMIT - total_tokens
    logger.info(
        "Tokens — prompt: %s | completion: %s | total: %s | remaining of %s: %s (%.1f%% used)",
        prompt_tokens, completion_tokens, total_tokens,
        MODEL_CONTEXT_LIMIT, remaining,
        100 * total_tokens / MODEL_CONTEXT_LIMIT,
    )

    if remaining < 0.1 * MODEL_CONTEXT_LIMIT:
        logger.warning(
            "Context window is over 90%% used (%s/%s tokens) — "
            "consider trimming MAX_HISTORY_TURNS.",
            total_tokens, MODEL_CONTEXT_LIMIT,
        )


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

    Guaranteed fallback: if every empty-reply retry is exhausted, this
    returns FALLBACK_REPLY instead of raising — the API call succeeded,
    it just never produced visible text, so that is not a real error
    and must not be announced as one (see the empty_exhausted branch
    below).
    """
    # ── Input guard (FIX-P1) ──────────────────────────────
    clean_input = sanitize_text(user_text)
    if is_blank(clean_input):
        raise ValueError("get_ai_reply received empty or whitespace-only input.")

    lang_history = history[lang]
    lang_history.append({"role": "user", "content": clean_input})

    try:
        # PROMPT CACHING: `system` is now fully static per lang (see
        # build_system docstring). `lang_history[:-1]` — every prior
        # turn — is also sent unmodified, so together they form a
        # stable, growing prefix that matches the previous request
        # almost token-for-token. Only the final message (this turn's
        # question + RAG/web context, built by _build_turn_content)
        # is new, which is the minimum possible uncached suffix and
        # keeps time-to-first-token as low as possible.
        system = build_system(lang)
        api_messages = [{"role": "system", "content": system}, *lang_history[:-1]]
        api_messages.append({
            "role": "user",
            "content": _build_turn_content(clean_input, context),
        })

        api_retries    = 0
        empty_retries  = 0
        empty_exhausted = False   # True only when we ran out of EMPTY_REPLY retries (not a real API error)
        last_exc: Optional[Exception] = None

        while True:
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=api_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.4,
                    timeout=LLM_TIMEOUT_SECS,   # never hang forever waiting for Groq
                    # gpt-oss-120b is a reasoning model: by default
                    # (reasoning_effort="medium") it can spend its
                    # ENTIRE max_tokens budget on internal chain-of-
                    # thought and get cut off before writing any
                    # visible answer — a real 200 OK with empty
                    # content, which is exactly what was happening.
                    # "low" caps how much it reasons, leaving the
                    # budget for the actual reply and making empty
                    # responses far rarer.
                    reasoning_effort="low",
                )
                raw_reply = response.choices[0].message.content
                # MODIFY: dropped the "Raw LLM response" debug print — it
                # duplicated info already visible in the "AI [..] > .." log
                # line below and cluttered DEBUG output. Token usage is
                # still logged via log_token_usage().
                log_token_usage(response)

                reply = sanitize_text(raw_reply)
                if not is_blank(reply):
                    # Stored history keeps the plain question only (no
                    # context) — this is what keeps future turns'
                    # cached prefix intact regardless of what context
                    # this particular turn happened to need.
                    lang_history.append({"role": "assistant", "content": reply})
                    _trim_history(lang)
                    return reply

                # Empty response — not a server problem, so retry
                # immediately (no sleep) up to LLM_EMPTY_MAX_RETRIES
                # times. Almost never exhausted in practice, so the
                # bot effectively always ends up with a real answer.
                empty_retries += 1
                logger.warning(
                    "LLM returned empty response (empty attempt %d/%d) — retrying immediately.",
                    empty_retries, LLM_EMPTY_MAX_RETRIES,
                )
                if empty_retries >= LLM_EMPTY_MAX_RETRIES:
                    empty_exhausted = True
                    break
                # Small stagger (not a full backoff) — firing the retry
                # with zero delay was itself dense enough to trip Groq's
                # per-second rate limit, which is what caused the 429 you
                # saw. 0.3s is negligible for perceived latency but avoids
                # that.
                time.sleep(0.3)
                continue

            except Exception as api_exc:
                api_retries += 1
                logger.warning("LLM API error (attempt %d): %s", api_retries, api_exc)
                last_exc = api_exc
                if api_retries > LLM_API_MAX_RETRIES:
                    break
                wait = 2 ** api_retries   # 2s, 4s — handles Groq 429/503
                logger.info("Retrying in %ds …", wait)
                time.sleep(wait)

        # Change 1 (guaranteed spoken fallback): if we only ran out of
        # EMPTY_REPLY retries, the API call itself was fine — there was
        # no real error. Returning a natural fallback line here (instead
        # of raising) means the caller speaks it normally instead of
        # routing through announce_error(), which would falsely tell the
        # user the server/internet is down when it isn't.
        if empty_exhausted:
            logger.warning(
                "LLM empty after %d attempts — speaking fallback instead of raising.",
                empty_retries,
            )
            lang_history.pop()   # drop the user turn; no real assistant reply to pair it with
            return FALLBACK_REPLY

        # Otherwise a genuine API error persisted — roll back and raise
        # so the caller announces a real error.
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
    rag = rag_hi if lang == "hi" else rag_en

    pdf_context, pdf_score = rag.retrieve(query)
    logger.debug("PDF score: %.3f (threshold=%.2f)", pdf_score, PDF_THRESHOLD)

    # Prioritize PDF. Only search web if PDF score is low.
    web_context = ""
    source      = "None"

    # FIX: `source` must reflect what is actually placed into `parts` below.
    if pdf_context:
        source = "PDF"

    if needs_web(query, pdf_score):
        logger.debug("Web search triggered (low PDF score + time-sensitive query).")
        web_context = web_search(query)
        if web_context:
            source = "PDF+Web" if pdf_context else "Web"
    elif pdf_score < PDF_THRESHOLD:
        logger.debug("Web skipped — query is not time-sensitive.")

    parts: List[str] = []
    if pdf_context:
        parts.append(f"[From DTR Knowledge Base]\n{pdf_context}")
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

# Persists across calls (module-level) so the noise floor learned in
# one turn carries into the next, rather than re-learning from scratch
# every time capture_speech() opens a fresh stream.
_noise_floor: float = ENERGY_THRESHOLD

# Throttles the "MIC DEBUG rms=..." log line to once per second instead
# of once per audio chunk (which fired ~5x/second at CHUNK_SECS≈0.2s).
_last_mic_debug_log: float = 0.0
MIC_DEBUG_LOG_INTERVAL = 1.0  # seconds


def capture_speech(timeout: float) -> Optional[np.ndarray]:
    global _noise_floor, _last_mic_debug_log
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
    recording_start: Optional[float]  = None   # Change 2: when the current utterance started
    silence_start: Optional[float]    = None
    idle_clock                         = time.time()

    try:
        while True:
            log_health()   # Change 5: no-ops internally unless HEALTH_LOG_INTERVAL has passed

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
            now = time.time()
            if now - _last_mic_debug_log >= MIC_DEBUG_LOG_INTERVAL:
                logger.info("MIC DEBUG rms=%.4f floor=%.4f", rms, _noise_floor)
                _last_mic_debug_log = now

            # ── Dynamic threshold: update the ambient noise floor ──
            # Only while NOT recording an utterance — updating during
            # active speech would drag the floor upward and make the
            # bot progressively deaf to quieter speakers.
            if not recording:
                _noise_floor = (
                    (1 - NOISE_EMA_ALPHA) * _noise_floor + NOISE_EMA_ALPHA * rms
                )
            threshold = max(
                DYNAMIC_THRESHOLD_MIN,
                min(DYNAMIC_THRESHOLD_MAX, _noise_floor * THRESHOLD_MULTIPLIER),
            )

            if rms >= threshold:
                idle_clock    = time.time()
                silence_start = None
                if not recording:
                    recording       = True
                    recording_start = time.time()   # Change 2: start the max-duration clock
                    speech_buffer   = list(pre_buffer)
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

            # Change 2 (max recording timeout): hard ceiling so a stuck-open
            # mic or a user who keeps talking can't record forever — stop
            # and process whatever has been captured so far.
            if recording and time.time() - recording_start >= MAX_RECORDING_SECS:
                logger.warning(
                    "Max recording duration (%.0fs) reached — stopping capture.",
                    MAX_RECORDING_SECS,
                )
                break

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

def _cleanup_tmp_file(path: Optional[str]) -> None:
    """
    Best-effort removal of a temp WAV/MP3 file. Shared by every TTS/STT
    call site (Change 7) instead of repeating the same
    try/os.path.exists/unlink block in each function.
    """
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _has_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


def _has_arabic_script(text: str) -> bool:
    return any(0x0600 <= ord(ch) <= 0x06FF for ch in text)


def _transcribe_once(audio: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
    """
    Run one Whisper transcription pass. If *language* is given, it is
    passed as an explicit hint to the API (helps Whisper pick a script
    when the audio is otherwise ambiguous between Hindi and Urdu).
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)

        kwargs = dict(model=STT_MODEL, response_format="verbose_json")
        if language:
            kwargs["language"] = language

        with open(tmp_path, "rb") as f:
            # Change 6 (watchdog): bound the Groq call so a stalled STT
            # request can't hang the bot forever.
            result = client.audio.transcriptions.create(file=f, timeout=STT_TIMEOUT_SECS, **kwargs)
    finally:
        _cleanup_tmp_file(tmp_path)   # Change 7

    text = sanitize_text(result.text)          # FIX-P7: sanitize at source
    lang = (result.language or "en").strip().lower()

    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    if _has_devanagari(text) or _has_arabic_script(text):
        lang = "hi"

    return text, lang


def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """
    Full-quality transcription for conversation turns.
    Uses whisper-large-v3 + Urdu→Hindi script retry (FIX-P9).
    """
    text, lang = _transcribe_once(audio)

    if lang == "hi" and _has_arabic_script(text) and not _has_devanagari(text):
        logger.debug(
            "Hindi speech transcribed in Urdu/Arabic script (%r) — "
            "retrying with explicit language hint.", text
        )
        retry_text, retry_lang = _transcribe_once(audio, language="hi")
        if retry_text and _has_devanagari(retry_text):
            logger.debug("Retry succeeded with Devanagari output: %r", retry_text)
            return retry_text, "hi"
        logger.debug("Retry did not produce Devanagari output — keeping original.")

    return text, lang


def transcribe_fast(audio: np.ndarray) -> Tuple[str, str]:
    """
    Fast transcription for IDLE wake-word detection only.
    Uses whisper-large-v3-turbo — ~3× faster, slightly less accurate,
    but accuracy doesn't matter for a yes/no wake-word check.
    Raises ConnectionError immediately when offline so the IDLE handler
    can announce the error rather than waiting 30s for a Groq timeout.
    """
    if not is_internet_available():
        raise ConnectionError("No internet connection.")

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)
        with open(tmp_path, "rb") as f:
            # Change 6 (watchdog): same STT timeout as the main transcribe path.
            result = client.audio.transcriptions.create(
                model=STT_MODEL_FAST,
                file=f,
                response_format="verbose_json",
                timeout=STT_TIMEOUT_SECS,
            )
    finally:
        _cleanup_tmp_file(tmp_path)   # Change 7
    return sanitize_text(result.text), "en"


# ══════════════════════════════════════════════════════════
#  WAKE WORD
# ══════════════════════════════════════════════════════════

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


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

# Change 4 (latency logging): filled by _speak_edge_tts()/_speak_espeak()
# each time speak() runs, then read by main()'s per-turn latency log line.
# Reset at the top of speak() so a failed call doesn't report stale numbers.
_last_tts_timing: dict = {"gen": 0.0, "playback": 0.0}


# Module-level event loop created once; reused by every speak() call.
_tts_loop = asyncio.new_event_loop()


def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts_async(text: str, path: str, voice: str) -> None:
    await edge_tts.Communicate(text, voice=voice).save(path)


def speak(text: str, lang: str = "en") -> None:
    global _mic_muted
    logger.debug("TTS input: %r", text)

    _last_tts_timing["gen"] = 0.0        # Change 4: reset before each call
    _last_tts_timing["playback"] = 0.0

    if is_blank(text):
        logger.error("TTS input validation failed: text is empty or None.")
        _speak_direct(ERROR_MESSAGES["env_error"], TTS_VOICE_EN)
        return

    voice = pick_voice(text, lang)
    logger.info("TTS [%s]: %s", voice, textwrap.shorten(text, width=80))

    _mic_muted = True          # gate mic BEFORE playback — prevents self-triggering
    try:
        if not is_internet_available():
            logger.warning("speak(): offline — skipping edge-tts, using espeak.")
            _speak_espeak(text, lang)
            return
        if _speak_edge_tts(text, voice):
            return
        logger.warning("edge-tts failed — attempting offline espeak fallback.")
        if _speak_espeak(text, lang):
            return
        logger.error("All TTS engines failed for this utterance.")
    finally:
        _mic_muted = False     # ALWAYS release gate after playback


def _speak_edge_tts(text: str, voice: str) -> bool:
    """
    Try to synthesise and play *text* via edge-tts.
    Returns True on success, False on any failure.
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        # Change 6 (watchdog): bound the network synthesis call so a
        # stalled edge-tts connection can't hang the bot forever — falls
        # through to the except block below (then the espeak fallback)
        # just like any other edge-tts failure.
        t0 = time.time()
        _tts_loop.run_until_complete(
            asyncio.wait_for(_tts_async(text, tmp_path, voice), timeout=TTS_TIMEOUT_SECS)
        )
        _last_tts_timing["gen"] = time.time() - t0   # Change 4

        if not os.path.exists(tmp_path):
            raise RuntimeError("edge-tts did not create output file.")

        mp3_size = os.path.getsize(tmp_path)
        logger.debug("Generated MP3 size: %d bytes", mp3_size)
        if mp3_size == 0:
            raise RuntimeError("edge-tts produced a zero-byte MP3 file.")

        try:
            pygame.mixer.music.load(tmp_path)
        except Exception as load_exc:
            raise RuntimeError(f"pygame failed to load MP3: {load_exc}") from load_exc

        t0 = time.time()
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        _last_tts_timing["playback"] = time.time() - t0   # Change 4
        return True

    except Exception as exc:
        logger.error("edge-tts error: %s", exc)
        return False

    finally:
        _cleanup_tmp_file(tmp_path)   # Change 7: shared cleanup helper


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
            t0 = time.time()
            subprocess.run(
                ["espeak", "-v", voice, "-s", "140", "-a", "180", text],
                check=True,
                timeout=15,
                stderr=subprocess.DEVNULL,
            )
            # espeak does synthesis + playback in one blocking call, so this
            # rare offline-fallback path can't split gen/playback further —
            # attribute the whole thing to "gen" for the latency log.
            _last_tts_timing["gen"] = time.time() - t0
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
    Tries edge-tts first, then espeak, then gives up silently.
    """
    if _speak_edge_tts(text, voice):
        return
    logger.warning("_speak_direct: edge-tts failed, trying espeak.")
    if _speak_espeak(text, lang="en"):
        return
    logger.error("_speak_direct: all engines failed (giving up).")


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def print_banner(rag_en_ready: bool, rag_hi_ready: bool) -> None:
    status_en = "✅ PDF loaded" if rag_en_ready else "⚠️  PDF not found — web-only mode"
    status_hi = "✅ PDF loaded" if rag_hi_ready else "⚠️  PDF not found — web-only mode"
    mic_label = f"'{MIC_NAME}'" if MIC_NAME else "system default"
    sep = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  DTBot 🤖  |  DTown Robotics, Noida\n"
        f"{sep}\n"
        f"  RAG (EN) status : {status_en}\n"
        f"  RAG (HI) status : {status_hi}\n"
        f"  PDF (EN) path   : {PDF_PATH_EN}\n"
        f"  PDF (HI) path   : {PDF_PATH_HI}\n"
        f"  PDF threshold   : {PDF_THRESHOLD}  (below → web fallback)\n"
        f"  Mic threshold   : dynamic (noise floor × {THRESHOLD_MULTIPLIER}, "
        f"clamped {DYNAMIC_THRESHOLD_MIN}–{DYNAMIC_THRESHOLD_MAX})\n"
        f"  Microphone      : {mic_label}\n"
        f"  Max history     : {MAX_HISTORY_TURNS} turns per language\n"
        f"  Log level       : {_log_level_name}\n"
        f"  States          :\n"
        f"    👂 LISTENING  — auto-detects your voice\n"
        f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle\n"
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
    timings: Optional[dict] = None,
) -> Optional[str]:
    """
    Retrieve context for *user_text* and return the LLM reply string, or
    None on failure (error already announced inside this function).

    FIX-P6: By isolating query processing in its own function, the main
    loop never carries a stale `reply` value between iterations.

    *timings*, if given, is filled in-place with 'rag' and 'llm' elapsed
    seconds for the latency log in main() (Change 4).
    """
    if timings is None:
        timings = {}

    # ── FIX-P1: reject blank input before any API call ────
    clean = sanitize_text(user_text)
    if is_blank(clean):
        logger.warning("Ignoring blank user input (after sanitization).")
        return None

    logger.info("User [%s] › %s", lang.upper(), clean)

    # Stop command? Interrupt any in-progress movement immediately,
    # checked before movement direction so it always takes priority.
    if is_stop_command(clean):
        stop_movement()
        return stopped_reply(lang)

    # Movement command? Handle locally and short-circuit before any
    # RAG/cache/LLM work — movement never needs the chatbot pipeline.
    if is_move_command(clean):
        direction = extract_direction(clean)
        duration = extract_duration(clean)
        if duration > MAX_MOVE_DURATION:
            return duration_limit_reply(direction, lang)
        move(direction, duration)
        return movement_reply(direction, duration, lang)

    # Change 3 (response cache): skip RAG + the LLM call entirely on a
    # repeated question. Still recorded in conversation history so later
    # turns keep natural continuity.
    cached = cache_get(clean, lang)
    if cached is not None:
        logger.info("Cache hit — skipping RAG + LLM call.")
        timings["rag"] = 0.0
        timings["llm"] = 0.0
        lang_history = history[lang]
        lang_history.append({"role": "user", "content": clean})
        lang_history.append({"role": "assistant", "content": cached})
        _trim_history(lang)
        logger.info("AI   [%s] › %s", lang.upper(), cached)
        return cached

    logger.debug("Retrieving context …")
    t0 = time.time()
    context, source = build_context(clean, lang, rag_en, rag_hi)
    timings["rag"] = time.time() - t0
    logger.info("Source: %s", source)
    logger.debug("Generating reply …")

    t0 = time.time()
    try:
        reply = get_ai_reply(clean, lang, context)
    except Exception as exc:
        timings["llm"] = time.time() - t0
        logger.error("LLM generation failed: %s", exc)
        announce_error(exc)
        return None   # error already announced; caller must not announce again
    timings["llm"] = time.time() - t0

    cache_set(clean, lang, reply)   # Change 3: store for next time (skips empty/fallback replies)

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

        speak("Hello! I am DTbot, your AI assistant. ", lang="hi")

        global _last_idle_error_time
        while True:

            # ── IDLE ──────────────────────────────────────
            if state == State.IDLE:
                logger.debug(state_label(state))
                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
                if audio is None:
                    continue

                try:
                    wake_text, _ = transcribe_fast(audio)
                except Exception as exc:
                    # Announce the error (e.g. "I can't connect to the internet")
                    # but only if enough time has passed — prevents the bot from
                    # repeating the message every 30s while offline.
                    now = time.time()
                    if now - _last_idle_error_time >= 30.0:
                        announce_error(exc)
                        _last_idle_error_time = now
                    logger.warning("Wake-word transcription failed: %s", exc)
                    continue
                logger.debug("Heard (idle): %s", wake_text)

                if is_wake_word(wake_text):
                    history["en"].clear()
                    history["hi"].clear()
                    state = State.LISTENING
                    speak("Haan, mein sun raha hoon.", lang="hi")
                continue

            # ── LISTENING ─────────────────────────────────
            if state == State.LISTENING:
                logger.debug(state_label(state))
                turn_start = time.time()   # Change 4: total end-to-end timer for this turn

                t0 = time.time()
                audio = capture_speech(timeout=IDLE_TIMEOUT)
                capture_time = time.time() - t0

                if audio is None:
                    state = State.IDLE
                    speak(
                        "Mein idle mode mai jaa raha hoo, "
                        "Mujhe activate krne ke liye Hello boliyein.",
                        lang="hi",
                    )
                    continue

                t0 = time.time()
                try:
                    user_text, lang = transcribe(audio)
                except Exception as exc:
                    logger.error("Transcription failed: %s", exc)
                    announce_error(exc)
                    continue
                stt_time = time.time() - t0

                # FIX-P1: reject blank transcription immediately
                if is_blank(user_text):
                    logger.debug("Blank transcription — skipping.")
                    continue

                # FIX-P6: reply scoped here; no shared mutable state
                state = State.THINKING
                logger.info(state_label(state))
                timings: dict = {}
                reply = _process_query(user_text, lang, rag_en, rag_hi, timings)

                if reply is None:
                    # Error was already announced inside _process_query;
                    # just go back to listening without a second announcement.
                    state = State.LISTENING
                    continue

                state = State.SPEAKING

                # ── SPEAKING (inline, scoped to this reply) ───────────
                logger.info(state_label(state))
                speak(reply, lang)

                # Change 4: one simple combined latency line per turn.
                # TTS gen/playback come from _last_tts_timing, filled by
                # speak() -> _speak_edge_tts()/_speak_espeak() just above.
                total_time = time.time() - turn_start
                logger.info(
                    "LATENCY capture=%.2fs stt=%.2fs rag=%.2fs llm=%.2fs "
                    "tts_gen=%.2fs playback=%.2fs total=%.2fs",
                    capture_time, stt_time,
                    timings.get("rag", 0.0), timings.get("llm", 0.0),
                    _last_tts_timing["gen"], _last_tts_timing["playback"],
                    total_time,
                )

                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C).")
    except Exception as exc:
        logger.critical("Fatal error in main loop: %s", exc, exc_info=True)
        try:
            announce_error(exc)
        except Exception:
            pass
    finally:
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            _tts_loop.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
