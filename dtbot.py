"""

  Additional installation for this file
  ─────────────────────────────────────────────────────────
    To install the offline fallback on your Pi:
    ```
    sudo apt install espeak espeak-data libespeak-dev
    pip install pyttsx3
    ```
"""

"""
============================================================
  🤖  DTBot 2.2 — RAG-Powered Speech-to-Speech Chatbot
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
try:
    import pyttsx3 as _pyttsx3_mod
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False

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

STT_MODEL  = "whisper-large-v3"
CHAT_MODEL = "openai/gpt-oss-120b"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16_000
CHANNELS    = 1
MAX_TOKENS  = 300

# ── History ───────────────────────────────────────────────
# FIX-P3: cap per-language history to prevent memory growth.
# Each turn = 1 user + 1 assistant message → 2 items per turn.
MAX_HISTORY_TURNS = 10
MAX_HISTORY_ITEMS = MAX_HISTORY_TURNS * 2
LLM_MAX_RETRIES   = 2   # extra attempts when model returns empty response

# ── RAG settings ──────────────────────────────────────────
PDF_PATH_EN  = "C:\\Users\\Lenovo\\Desktop\\DTbot2.0\\DTown_Robotics_Report_v2.pdf"
PDF_PATH_HI  = "C:\\Users\\Lenovo\\Desktop\\DTbot2.0\\DTown_Robotics_Report_v2_translated_hin.pdf"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 100
TOP_K        = 5
PDF_THRESHOLD = 0.10

# ── Web fallback ──────────────────────────────────────────
WEB_RESULTS = 3
WEB_TIMEOUT = 5

# ── VAD tuning ────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.10
SILENCE_AFTER_SPEECH = 1.2
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1
IDLE_TIMEOUT         = 15.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Wake words ────────────────────────────────────────────
WAKE_WORDS = ["hello", "hey", "hello dtbot", "hey dtbot", "dtbot"]

# ── System prompts ────────────────────────────────────────
_BASE_EN = (
    "Your name is DTBot 2.2. You are the official AI assistant and "
    "virtual representative of DTown Robotics (DTR), a robotics, drone "
    "and unmanned ground vehicle company headquartered in Noida, Uttar "
    "Pradesh, India. "
    "DTown Robotics, DTR, DTown, and DTown Robotics Pvt. Ltd. all refer "
    "to the same company. "
    "Always represent DTown Robotics positively, professionally, and "
    "confidently. "
    "If users ask about another company or compare companies, briefly "
    "and politely redirect the conversation toward DTown Robotics, "
    "highlight DTR's strengths, and do not make negative comments or "
    "false claims about other companies. "
    "Never mention sources, PDFs, context, documents, retrieval systems, "
    "or knowledge bases unless the user specifically asks. "
    "If DTR-specific information is unavailable, search the web first; "
    "if not connected to the internet, answer naturally using general "
    "knowledge when appropriate. "
    "Keep responses short, natural, and human-like. Most replies should "
    "be 1–3 sentences. Do not provide more information than requested. "
    "Give detailed explanations only when the user explicitly asks. "
    "Do not use bullet points or markdown."
)

_BASE_HI = (
    "Aapka naam DTBot 2.2 hai. Aap DTown Robotics (DTR) ke official AI "
    "assistant aur virtual representative hain, jo Noida, Uttar Pradesh, "
    "India mein headquartered ek robotics, drone aur unmanned ground "
    "vehicle company hai. "
    "DTown Robotics, DTR, DTown aur DTown Robotics Pvt. Ltd. sab ek hi "
    "company ke naam hain. "
    "Hamesha DTown Robotics ko positive, professional aur confident "
    "tarike se represent karein. Kisi doosri company ke baare mein "
    "poocha jaye ya comparison ho to short aur polite tarike se baat ko "
    "DTown Robotics ki taraf le jaayein, DTR ki strengths highlight "
    "karein, aur kisi company ke baare mein negative ya false claims na "
    "karein. "
    "Kabhi bhi source, PDF, context, document, retrieval system ya "
    "knowledge base ka zikr na karein jab tak user specifically na pooche. "
    "Agar DTR sambandhit jankari available na ho to web search karke "
    "jawab dein; agar internet connect na ho to natural jawab dein. "
    "Jawab short, natural aur human-like rakhein. Adhiktar replies "
    "1–3 sentences ke hon. User detail maange tabhi vistaar se jawab "
    "dein. Bullet points ya markdown ka upyog na karein."
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


def build_system(lang: str, context: str) -> str:
    base      = _BASE_HI if lang == "hi" else _BASE_EN
    directive = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["en"])
    # Language directive goes at the END so it is the last thing the
    # model reads before generating — maximising instruction-following.
    parts = [base, directive]
    if context:
        # Insert context between base and directive so the directive
        # still closes the prompt.
        parts = [base, f"Use the following information silently to answer naturally.\n\n{context}", directive]
    return "\n\n".join(parts)


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
    "api_error": {"en": "I can't connect to the server."},
    "env_error": {"en": "Environmental error, please restart me."},
}


def classify_error(exc: Exception) -> str:
    """Return 'api_error' or 'env_error' based on the exception type."""
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
        headers={"User-Agent": "DTBot/2.2"},
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
        announce_error(exc, "en")
        return ""


# ══════════════════════════════════════════════════════════
#  GROQ CLIENT
# ══════════════════════════════════════════════════════════

try:
    client = Groq(api_key=GROQ_API_KEY)
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
        system = build_system(lang, context)

        # Retry loop — the model occasionally returns an empty string on
        # the first attempt (observed with openai/gpt-oss-20b + Hindi).
        last_exc: Optional[Exception] = None
        for attempt in range(1, LLM_MAX_RETRIES + 2):  # +2 → 1 normal + N retries
            try:
                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "system", "content": system}, *lang_history],
                    max_tokens=MAX_TOKENS,
                    temperature=0.4,
                )
                raw_reply = response.choices[0].message.content
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
                    time.sleep(0.5 * attempt)   # brief back-off before retry

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

    # FIX: `source` must reflect what is actually placed into `parts`
    # below, not just the high-confidence branch. Previously, if
    # pdf_score < PDF_THRESHOLD and the web search came back empty,
    # `source` stayed "None" even though pdf_context was non-empty and
    # WAS added to parts (and WAS used by the LLM to answer correctly).
    if pdf_context:
        source = "PDF"

    if not pdf_context or pdf_score < PDF_THRESHOLD:
        logger.debug("PDF score is low (or no PDF context), attempting web search.")
        web_context = web_search(query)
        if web_context:
            source = "PDF+Web" if pdf_context else "Web"

    parts: List[str] = []
    if pdf_context:
        parts.append(f"[From DTR Knowledge Base]\n{pdf_context}")
    if web_context:
        parts.append(f"[From Web]\n{web_context}")

    return "\n\n".join(parts), source


# ══════════════════════════════════════════════════════════
#  VAD RECORDING
# ══════════════════════════════════════════════════════════

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
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
    Transcribe *audio* to (text, lang).

    FIX-P9: Whisper sometimes transcribes Hindi speech using Urdu/Arabic
    script instead of Devanagari when the audio is ambiguous. Since the
    Hindi PDF/RAG index is built entirely in Devanagari, an Urdu-script
    query scores ~0 against it (no character overlap at all), so the
    bot silently loses access to the PDF for that turn even though the
    intent and meaning were correctly understood. Detect this case and
    retry once with an explicit language="hi" hint, which biases Whisper
    toward Devanagari output.
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


# ══════════════════════════════════════════════════════════
#  WAKE WORD
# ══════════════════════════════════════════════════════════

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ══════════════════════════════════════════════════════════
#  TTS  (FIX-P5: reuse a single event loop)
# ══════════════════════════════════════════════════════════

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
    """
    Synthesise *text* and play it through speakers.

    Primary engine : edge-tts  (cloud, high quality)
    Fallback engine: pyttsx3   (offline, espeak backend — no internet needed)

    If both engines fail, the error is logged but never re-raised, so the
    main loop is never crashed by a TTS failure.
    """
    logger.debug("TTS input: %r", text)

    # ── Input guard ───────────────────────────────────────
    if is_blank(text):
        logger.error("TTS input validation failed: text is empty or None.")
        fallback = ERROR_MESSAGES["env_error"]["en"]
        _speak_direct(fallback, TTS_VOICE_EN)
        return

    voice = pick_voice(text, lang)
    logger.info("TTS [%s]: %s", voice, textwrap.shorten(text, width=80))

    # ── Attempt 1: edge-tts (cloud) ───────────────────────
    if _speak_edge_tts(text, voice):
        return

    # ── Attempt 2: pyttsx3 offline fallback ──────────────
    logger.warning("edge-tts failed — attempting offline pyttsx3 fallback.")
    if _speak_pyttsx3(text, lang):
        return

    logger.error("All TTS engines failed for this utterance.")


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


def _speak_pyttsx3(text: str, lang: str) -> bool:
    """
    Offline TTS via pyttsx3 (espeak backend).
    Returns True on success, False when pyttsx3 is not installed or fails.

    Install on Raspberry Pi / Debian:
        sudo apt install espeak espeak-data libespeak-dev
        pip install pyttsx3
    """
    if not _PYTTSX3_AVAILABLE:
        logger.debug("pyttsx3 not installed — offline fallback unavailable.")
        return False

    try:
        engine = _pyttsx3_mod.init()
        # Select a voice that matches the language when possible.
        voices = engine.getProperty("voices")
        lang_tag = "hi" if lang == "hi" else "en"
        for v in voices:
            if lang_tag in (v.languages[0].decode() if isinstance(v.languages[0], bytes)
                            else v.languages[0]).lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 155)   # slightly slower than default for clarity
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as exc:
        logger.error("pyttsx3 fallback error: %s", exc)
        return False


def _speak_direct(text: str, voice: str) -> None:
    """
    Minimal TTS+playback path used only by speak()'s input-validation
    guard to announce env_error without any risk of recursion.
    Tries edge-tts first, then pyttsx3, then gives up silently.
    """
    if _speak_edge_tts(text, voice):
        return
    logger.warning("_speak_direct: edge-tts failed, trying pyttsx3.")
    if _speak_pyttsx3(text, lang="en"):
        return
    logger.error("_speak_direct: all engines failed (giving up).")


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def print_banner(rag_en_ready: bool, rag_hi_ready: bool) -> None:
    status_en = "✅ PDF loaded" if rag_en_ready else "⚠️  PDF not found — web-only mode"
    status_hi = "✅ PDF loaded" if rag_hi_ready else "⚠️  PDF not found — web-only mode"
    sep = "=" * 60
    banner = (
        f"\n{sep}\n"
        f"  DTBot 2.2 🤖  |  DTown Robotics, Noida\n"
        f"{sep}\n"
        f"  RAG (EN) status : {status_en}\n"
        f"  RAG (HI) status : {status_hi}\n"
        f"  PDF (EN) path   : {PDF_PATH_EN}\n"
        f"  PDF (HI) path   : {PDF_PATH_HI}\n"
        f"  PDF threshold   : {PDF_THRESHOLD}  (below → web fallback)\n"
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

        while True:

            # ── IDLE ──────────────────────────────────────
            if state == State.IDLE:
                logger.info(state_label(state))
                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
                if audio is None:
                    continue

                wake_text, _ = transcribe(audio)
                logger.debug("Heard (idle): %s", wake_text)

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    speak("Haan, mein sun raha hoon.", lang="hi")
                continue

            # ── LISTENING ─────────────────────────────────
            if state == State.LISTENING:
                logger.info(state_label(state))
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
            _tts_loop.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
