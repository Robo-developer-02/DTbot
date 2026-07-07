"""

  Additional installation for this file
  ─────────────────────────────────────────────────────────
    Offline TTS fallback uses espeak directly (no pyttsx3):
    ```
    sudo apt install espeak espeak-data libespeak-dev
    ```
"""

"""
============================================================
  🤖  DTBot — RAG-Powered Speech-to-Speech Chatbot
  Production Release
============================================================

  Architecture
  ─────────────────────────────────────────────────────────
  • Hindi queries  → retrieved directly against hindi_details.pdf (rag_hi)
  • English queries → retrieved directly against english_details.pdf (rag_en)
  • Language detection picks the engine; the query is never translated.
  • Web fallback only fires when PDF score is below threshold AND the
    query contains time-sensitive keywords.

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

  FIX-P9  transcribe() detects when Whisper outputs Urdu/Arabic script
          for what is actually Hindi speech (a known ambiguity in the
          model) and retries once with an explicit language="hi" hint
          to bias toward Devanagari. Without this, the Hindi RAG index
          (built in Devanagari) scores ~0 against an Urdu-script query
          — zero character overlap — silently losing PDF access for
          that turn even though the spoken intent was understood fine.
============================================================
"""

# ── Standard library ──────────────────────────────────────
import asyncio
import logging
import os
import queue
import re
import socket
import tempfile
import textwrap
import time
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

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16_000
CHANNELS    = 1
MAX_TOKENS  = 320   # gpt-oss-20b is a reasoning model — it spends tokens on
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
DYNAMIC_THRESHOLD_MIN = 0.03   # floor — never require less than this even in silence
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
    "virtual representative of DTown Robotics, a robotics, drone "
    "and unmanned ground vehicle company headquartered in Noida, Uttar "
    "Pradesh, India. "
    "RESPONSE LENGTH — CRITICAL: You are a VOICE assistant. Your reply "
    "will be spoken aloud. Limit every response to 1–2 sentences maximum. "
    "Never exceed 40 words. Do not elaborate unless the user explicitly "
    "asks for more detail. "
    "Always represent DTown Robotics positively, professionally, and "
    "confidently. "
    "If users ask about another company or compare companies, briefly "
    "and politely redirect the conversation toward DTown Robotics, "
    "highlight DTown's strengths, and do not make negative comments or "
    "false claims about other companies. "
    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "
    "If DTown-specific information is unavailable, search the web first; "
    "if not connected to the internet, answer naturally using general "
    "knowledge when appropriate. "
    "Do not use bullet points or markdown."
)

_BASE_HI = (
    "Aapka naam DTBot hai. Aap DTown Robotics ke official AI "
    "assistant aur virtual representative hain, jo Noida, Uttar Pradesh, "
    "India mein headquartered ek robotics, drone aur unmanned ground "
    "vehicle company hai. "
    "JAWAB KI LAMBAI — ZAROORI: Aap ek VOICE assistant hain. Aapka jawab "
    "bol ke sunaya jayega. Har jawab sirf 1–2 sentence mein dein. "
    "Kabhi 40 shabdon se zyada mat likhein. "
    "Hamesha DTown Robotics ko positive, professional aur confident "
    "tarike se represent karein. Kisi doosri company ke baare mein "
    "poocha jaye ya comparison ho to short aur polite tarike se baat ko "
    "DTown Robotics ki taraf le jaayein, iski strengths highlight "
    "karein, aur kisi company ke baare mein negative ya false claims na "
    "karein. "
    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "
    "Agar DTown ke sambandhit jankari available na ho to web search karke "
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


@lru_cache(maxsize=4)
def build_system(lang: str) -> str:
    """
    Static system prompt ONLY — no RAG/web context.

    PROMPT CACHING (same technique as acrobot.py's FIX-A6): this string
    never changes for a given `lang` during the process lifetime, so
    it's memoized with @lru_cache instead of being rebuilt/re-joined on
    every single request — maxsize=4 comfortably covers "en"/"hi" plus
    headroom.

    This goes a step further than acrobot.py, though: acrobot still
    splices RAG context into the middle of this string per call (which
    means the actual prompt sent to Groq changes every turn regardless
    of the Python-level memoization — no real prefix-cache benefit on
    Groq's side). Here, context is NEVER mixed into the system prompt;
    it's appended to the trailing user turn instead (see
    _build_turn_content() and get_ai_reply()). That means the system
    prompt AND the prior-turns prefix stay byte-identical across calls,
    so Groq's own prompt cache can also hit on the growing prefix, not
    just our local Python object.
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

ERROR_MESSAGES = {
    "no_internet": {"en": "I can't connect to the internet."},
    "api_error":   {"en": "I can't connect to the server."},
    "env_error":   {"en": "Environmental error, please try again."},
}


def is_internet_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """Quick TCP probe — returns True when internet is reachable."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection((host, port))
        return True
    except OSError:
        return False


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

        api_retries   = 0
        empty_retries = 0
        last_exc: Optional[Exception] = None

        while True:
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=api_messages,
                    max_tokens=MAX_TOKENS,
                    temperature=0.4,
                    timeout=30,   # never hang forever waiting for Groq
                    # gpt-oss-20b is a reasoning model: by default
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
                logger.debug("Raw LLM response: %r", raw_reply)

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
                last_exc = RuntimeError(
                    f"LLM returned an empty response after {empty_retries} empty attempt(s)."
                )
                if empty_retries >= LLM_EMPTY_MAX_RETRIES:
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
    else:
        if pdf_score < PDF_THRESHOLD:
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


def capture_speech(timeout: float) -> Optional[np.ndarray]:
    global _noise_floor
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
            result = client.audio.transcriptions.create(file=f, **kwargs)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

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
            result = client.audio.transcriptions.create(
                model=STT_MODEL_FAST,
                file=f,
                response_format="verbose_json",
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
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

    if is_blank(text):
        logger.error("TTS input validation failed: text is empty or None.")
        _speak_direct(ERROR_MESSAGES["env_error"]["en"], TTS_VOICE_EN)
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

        _tts_loop.run_until_complete(_tts_async(text, tmp_path, voice))

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

        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        return True

    except Exception as exc:
        logger.error("edge-tts error: %s", exc)
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

        speak("Hello! I am DTown Bot, your AI assistant. ", lang="hi")

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
                        announce_error(exc, "en")
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
        try:
            _tts_loop.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()