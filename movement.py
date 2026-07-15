# THIS CODE IS RESPONSIBLE FOR THE MOVEMENT OF THE ROBOT

import serial
import time
import re
import threading
import logging
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import wave
import io

# ---------- Logging ----------
logging.basicConfig(
    filename='robot.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
log = logging.getLogger(__name__)

# ---------- Config ----------
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 115200
RECOG_LANGUAGE = 'en-IN'
SAMPLE_RATE = 16000
TURN_PULSE = 0.6

DEFAULT_DURATION = 3
MAX_DURATION = 5

DYNAMIC_THRESHOLD_MIN = 0.135
THRESHOLD_MULTIPLIER = 1.6
NOISE_EMA_ALPHA = 0.1

# ---------- Silence-based dynamic recording ----------
CHUNK_DURATION = 0.2          # seconds per audio chunk while monitoring
MAX_RECORD_DURATION = 8       # hard safety cap - never record longer than this
SILENCE_HANG_TIME = 0.8       # seconds of continuous silence after speech to stop
PRE_SPEECH_TIMEOUT = 4        # max seconds to wait for speech to start before giving up

noise_floor_rms = DYNAMIC_THRESHOLD_MIN / THRESHOLD_MULTIPLIER
stop_timer_flag = threading.Event()
shutdown_flag = threading.Event()

recognizer = sr.Recognizer()


# ---------- Resilient serial wrapper ----------
class SafeSerial:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        self.connect()

    def connect(self):
        while not shutdown_flag.is_set():
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=1)
                time.sleep(2)
                log.info(f"Serial connected on {self.port}")
                return
            except serial.SerialException as e:
                log.error(f"Serial connect failed: {e}. Retrying in 2s...")
                time.sleep(2)

    def write(self, data):
        with self.lock:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(data)
                    return True
            except (serial.SerialException, OSError) as e:
                log.error(f"Serial write failed: {e}. Reconnecting...")
                self.reconnect()
        return False

    def reconnect(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.connect()

    def close(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass


safe_ser = SafeSerial(SERIAL_PORT, BAUD_RATE)


def send_command(cmd):
    safe_ser.write(cmd.encode())
    log.info(f"Command sent: {cmd}")


# ---------- Command word lists (English + Hindi/Hinglish synonyms) ----------
FORWARD_WORDS = [
    "move forward", "go forward", "forward", "straight", "ahead",
    "aage", "aagey", "seedha", "seedhe", "aage badho", "aage chalo",
    "aage jao", "aage badhao", "chalo aage"
]
BACKWARD_WORDS = [
    "move backward", "go backward", "go back", "backward", "back", "reverse",
    "peeche", "peechay", "ulta", "peeche jao", "peeche chalo",
    "peeche aao", "wapas jao"
]
LEFT_WORDS = [
    "turn left", "left", "baen", "baayen", "baen taraf", "baen mudo",
    "left mudo", "left ghumo", "baen ghumo"
]
RIGHT_WORDS = [
    "turn right", "right", "dayen", "daayen", "dayen taraf", "dayen mudo",
    "right mudo", "right ghumo", "dayen ghumo"
]
STOP_WORDS = [
    "stop", "halt", "ruko", "ruk jao", "ruk jaiye", "band karo", "band kar do"
]


def matches_any(text, word_list):
    return any(w in text for w in word_list)


def calculate_rms(audio):
    normalized = audio.astype(np.float32) / 32768.0
    return np.sqrt(np.mean(np.square(normalized)))


def calibrate_noise_floor(seconds=1.5, samplerate=SAMPLE_RATE):
    global noise_floor_rms
    log.info(f"Calibrating ambient noise for {seconds}s...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate,
                    channels=1, dtype='int16')
    sd.wait()
    noise_floor_rms = calculate_rms(audio.flatten())
    log.info(f"Noise floor set to rms={noise_floor_rms:.4f}")


def record_until_silence(samplerate=SAMPLE_RATE):
    """
    Records audio dynamically:
    - Waits for speech to start (rms crosses dynamic_threshold)
    - Keeps recording while speech continues
    - Stops after SILENCE_HANG_TIME seconds of silence following speech
    - Hard-capped at MAX_RECORD_DURATION to avoid infinite recording
    Returns None if nobody spoke within PRE_SPEECH_TIMEOUT.
    """
    chunk_samples = int(CHUNK_DURATION * samplerate)
    chunks = []
    speech_started = False
    silence_time = 0.0
    total_time = 0.0
    waited_for_speech = 0.0

    log.info("Listening...")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            rms = calculate_rms(chunk)
            dynamic_threshold = max(DYNAMIC_THRESHOLD_MIN, noise_floor_rms * THRESHOLD_MULTIPLIER)

            if not speech_started:
                if rms >= dynamic_threshold:
                    speech_started = True
                    chunks.append(chunk)
                else:
                    waited_for_speech += CHUNK_DURATION
                    if waited_for_speech >= PRE_SPEECH_TIMEOUT:
                        return None
            else:
                chunks.append(chunk)
                total_time += CHUNK_DURATION

                if rms < dynamic_threshold:
                    silence_time += CHUNK_DURATION
                    if silence_time >= SILENCE_HANG_TIME:
                        break
                else:
                    silence_time = 0.0

                if total_time >= MAX_RECORD_DURATION:
                    log.info("Max record duration hit, stopping.")
                    break

    audio = np.concatenate(chunks)
    return audio


def audio_to_sr_data(audio, samplerate=SAMPLE_RATE):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    with sr.AudioFile(buf) as source:
        return recognizer.record(source)


def extract_duration(text):
    match = re.search(r'(\d+)\s*(second|seconds|sec|sekend)', text)
    if match:
        requested = int(match.group(1))
        return min(requested, MAX_DURATION)
    return DEFAULT_DURATION


def timed_stop(duration, my_flag):
    time.sleep(duration)
    if not my_flag.is_set():
        send_command('x')
        log.info(f"AUTO STOP after {duration} sec")


def turn_then_forward(turn_cmd, total_duration, my_flag):
    send_command(turn_cmd)
    time.sleep(TURN_PULSE)

    if my_flag.is_set():
        return

    send_command('w')

    remaining = max(total_duration - TURN_PULSE, 0)
    time.sleep(remaining)
    if not my_flag.is_set():
        send_command('x')
        log.info(f"AUTO STOP after turn+forward ({total_duration}s total)")


def process_command(text):
    global stop_timer_flag
    text = text.lower()
    log.info(f"Heard: {text}")

    stop_timer_flag.set()
    new_flag = threading.Event()
    duration = extract_duration(text)

    if matches_any(text, STOP_WORDS):
        send_command('x')
        return

    if matches_any(text, LEFT_WORDS):
        log.info(f"LEFT (turn then forward for {duration}s total)")
        stop_timer_flag = new_flag
        threading.Thread(target=turn_then_forward, args=('a', duration, new_flag), daemon=True).start()
        return

    if matches_any(text, RIGHT_WORDS):
        log.info(f"RIGHT (turn then forward for {duration}s total)")
        stop_timer_flag = new_flag
        threading.Thread(target=turn_then_forward, args=('d', duration, new_flag), daemon=True).start()
        return

    if matches_any(text, FORWARD_WORDS):
        send_command('w')
        log.info(f"FORWARD for {duration}s")
        stop_timer_flag = new_flag
        threading.Thread(target=timed_stop, args=(duration, new_flag), daemon=True).start()
        return

    if matches_any(text, BACKWARD_WORDS):
        send_command('s')
        log.info(f"BACKWARD for {duration}s")
        stop_timer_flag = new_flag
        threading.Thread(target=timed_stop, args=(duration, new_flag), daemon=True).start()
        return

    log.info("Command not recognized")


def main():
    log.info("Voice control started.")
    calibrate_noise_floor()

    try:
        while True:
            try:
                audio = record_until_silence()

                if audio is None:
                    # koi bola hi nahi is cycle me, seedha next cycle try karo
                    continue

                rms = calculate_rms(audio)
                log.info(f"Captured speech segment (rms={rms:.3f})")

                audio_data = audio_to_sr_data(audio)

                try:
                    text = recognizer.recognize_google(audio_data, language=RECOG_LANGUAGE)
                    process_command(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    log.error(f"Speech recognition service error: {e}")

            except Exception as e:
                log.error(f"Unexpected error in main loop: {e}")
                time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Shutting down (user interrupt)...")

    finally:
        shutdown_flag.set()
        send_command('x')
        time.sleep(0.3)
        safe_ser.close()
        log.info("Clean shutdown complete.")


if __name__ == "__main__":
    main()