"""
EMLII v3.5 — Self-Learning Edition
=====================================
Runs as a terminal chatbot OR a browser web app.

How EMLII now predicts responses (without manual teaching):
  1. Exact match   — checks data/mem/<md5> as before.
  2. Similarity    — tokenises the input, scores word-overlap (Jaccard) against
                     every input EMLII has ever seen, uses the best match's response.
  3. Pattern rules — regex/keyword rules for greetings, questions, etc.
  4. Teach fallback— only if all three layers fail (rare once EMLII has learned enough).

Every predicted response is saved to disk immediately, so the same input becomes
an exact match next time and the response is never generated twice.

Usage:
    python EMLII.py              → interactive mode picker
    python EMLII.py --terminal
    python EMLII.py --web
    python EMLII.py --web 8888
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, sys, re, json, hashlib, socket, threading, string, itertools, random
import getpass
import shutil, tempfile, zipfile
import importlib
import functools
import subprocess
import urllib.request, urllib.error
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher     # String similarity fallback for short inputs

try:
    from flask import Flask, Response, jsonify, request as flask_request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# EMLII_DATA_DIR / EMLII_TOOLS_DIR / EMLII_MODELS_DIR (set by the desktop app's
# Settings → Folders) relocate these folders; unset means the usual ./data etc.
_DATA = (os.environ.get("EMLII_DATA_DIR") or "data").replace("\\", "/").rstrip("/") or "data"
PATHS = {
    'mem':       f'{_DATA}/mem/',
    'history':   f'{_DATA}/history/',
    'files':     f'{_DATA}/files/',
    'downloads': f'{_DATA}/downloads/',
}
for _p in PATHS.values():
    Path(_p).mkdir(parents=True, exist_ok=True)

# Config and inputs index live at the data/ root, not inside mem/
CONFIG_FILE = f'{_DATA}/config.json'
INPUTS_FILE = f'{_DATA}/inputs.json'
# inputs.json maps { md5_hash: original_text }.
# We can't derive original text from a hash (one-way), so we store it separately.
# This is what makes similarity search possible.

KILL = {"exit", "leave", "quit", "q"}

BUILTIN_COMMANDS = {
    "help", "version", "date", "calculate", "hash", "crack",
    "encrypt", "ecrypt", "decrypt", "dirserver", "stopserver",
    "exit", "quit", "leave", "q",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL PLUGINS
# ═══════════════════════════════════════════════════════════════════════════════
# This is the extension point: new !commands can be added by dropping a file
# into tools/, with nothing in this file ever needing to change again. See
# tools/README.md for the full contract. Summary:
#
#   tools/mytool.py:
#       COMMAND = "mytool"                 # triggers "!mytool <arg>"
#       HELP    = "One line for !help"
#       def run(arg, ctx):
#           return {"type": "reply", "text": "..."}   # same result-dict shapes
#                                                       # EMLII.commands() returns
#
# TOOLS maps the lowercase command word to its loaded module.

TOOLS = {}          # {"mytool": <module>, ...} — populated by load_tools()
TOOLS_DIR = Path(os.environ.get("EMLII_TOOLS_DIR") or "tools")


def load_tools():
    """
    Import every *.py file in tools/ (skipping names starting with '_') and
    register the ones that expose COMMAND and a callable run(arg, ctx).

    A broken or malformed tool is reported and skipped rather than crashing
    EMLII — one bad plugin should never take down the whole app.
    """
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))

    for f in sorted(TOOLS_DIR.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f.stem)
            importlib.reload(mod)   # picks up edits without restarting the picker loop
        except Exception as e:
            print(f"[tools] failed to load {f.name}: {e}")
            continue

        cmd = getattr(mod, "COMMAND", None)
        if not cmd or not callable(getattr(mod, "run", None)):
            print(f"[tools] {f.name} is missing COMMAND or run() — skipped")
            continue

        cmd = cmd.lower().lstrip("!")
        if cmd in BUILTIN_COMMANDS:
            print(f"[tools] {f.name}: '!{cmd}' is a built-in command, skipped")
            continue
        TOOLS[cmd] = mod

    if TOOLS:
        print(f"[tools] loaded: {', '.join('!' + c for c in TOOLS)}")

# Minimum Jaccard word-overlap to accept a similarity match as a valid prediction.
# 0.30 means at least 30% of words must overlap. Raise to be stricter.
SIMILARITY_THRESHOLD = 0.22

# ── Layer 4: local AI fallback (Ollama) ─────────────────────────────────────
# Used only when similarity search and pattern rules both miss, as a last
# resort before asking the user to manually teach a response. Requires
# `ollama serve` running locally — if it's unreachable, this layer is
# silently skipped and the pipeline falls through to the teach prompt.
def _pref(name):
    """A one-line setting saved by the !ai command, in the data folder."""
    try:
        return (Path(_DATA) / name).read_text().strip()
    except OSError:
        return ""


def _ollama_url():
    """Which Ollama the AI layer talks to: EMLII_OLLAMA_URL, else `!ai url ...`, else this device."""
    u = (os.environ.get("EMLII_OLLAMA_URL") or _pref("ollama_url.txt") or "http://127.0.0.1:11434").strip().rstrip("/")
    return u if u.endswith("/api/generate") else u + "/api/generate"


OLLAMA_URL     = _ollama_url()
OLLAMA_MODEL   = _pref("ollama_model.txt") or "hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q5_K_M"  # fallback if models/ is empty (`!ai model <name>` changes it)
OLLAMA_TIMEOUT = int(os.environ.get("EMLII_OLLAMA_TIMEOUT") or 20)   # seconds — small model, but give it room on first load (slow phones: set EMLII_OLLAMA_TIMEOUT=90)
AI_ENABLED     = True

ACTIVE_MODEL = None  # set by load_models() if a local .gguf was found and registered
MODELS_DIR   = Path(os.environ.get("EMLII_MODELS_DIR") or "models")


def load_models():
    """
    Drop-in model loading, same idea as load_tools() for tools/: put a
    .gguf file in models/ and EMLII auto-registers it with the local Ollama
    server and switches Layer 4 over to it — no code or config changes.

    If models/ is empty, ACTIVE_MODEL stays None and query_ollama() falls
    back to the OLLAMA_MODEL constant (a model you've `ollama pull`ed).
    If more than one .gguf is dropped in, the most recently modified one
    wins. Requires the `ollama` CLI on PATH and `ollama serve` running —
    like with tools/, a failure here is reported and skipped, never fatal.
    """
    global ACTIVE_MODEL
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ggufs = sorted(MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ggufs:
        return

    newest = ggufs[0]
    name = "emlii-" + re.sub(r"[^a-z0-9._-]+", "-", newest.stem.lower()).strip("-")

    modelfile = MODELS_DIR / f".{newest.stem}.Modelfile"
    try:
        modelfile.write_text(f"FROM {newest.resolve().as_posix()}\n")   # absolute: works for any Models folder
        result = subprocess.run(
            ["ollama", "create", name, "-f", str(modelfile)],
            capture_output=True, text=True, timeout=180,
        )
    except FileNotFoundError:
        print("[models] found", newest.name, "but the 'ollama' CLI isn't on PATH — skipped")
        return
    except subprocess.TimeoutExpired:
        print(f"[models] registering {newest.name} with Ollama timed out — skipped")
        return

    if result.returncode != 0:
        print(f"[models] failed to register {newest.name}: {result.stderr.strip()}")
        return

    ACTIVE_MODEL = name
    print(f"[models] loaded: {newest.name} -> local Ollama model '{name}'")


# ── Offline voice (speech-to-text + text-to-speech) ──────────────────────────
# Voice mode in the web UI normally uses the browser's speech recognition, which
# in Chrome/Edge is a CLOUD service. For fully offline voice, EMLII can do both
# directions locally:
#
#   Speech → text : Vosk  (pip install vosk, then unzip a model into models/)
#                   e.g. vosk-model-small-en-us-0.15 from alphacephei.com/vosk/models
#                   Any folder under models/ containing am/ and conf/ is detected.
#   Text → speech : the browser's own OS voices when available (offline on
#                   Windows/macOS/Android/iOS), otherwise espeak-ng / espeak on
#                   Linux, or pyttsx3 (pip install pyttsx3) as a server-side fallback.
#
# Like load_tools() and load_models(), a missing piece is reported and skipped.
STT_MODEL  = None     # vosk.Model once loaded
STT_NAME   = None     # folder name, shown in the UI
STT_STATUS = "not checked"   # why offline input is unavailable (shown in the UI)
TTS_ENGINE = None     # "espeak-ng" | "espeak" | "pyttsx3" | None
_TTS_LOCK  = threading.Lock()


def _model_dirs():
    """Places to look for a Vosk model: next to this script AND in the current folder."""
    seen, out = set(), []
    for d in (Path(__file__).resolve().parent / "models", Path.cwd() / "models", MODELS_DIR.resolve()):
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def load_voice():
    global STT_MODEL, STT_NAME, STT_STATUS, TTS_ENGINE
    try:
        import vosk
        vosk.SetLogLevel(-1)
    except ImportError:
        vosk = None
        STT_STATUS = "the 'vosk' package is not installed for this Python"
        print(f"[voice] {STT_STATUS}.")
        print(f"[voice] install it with:  {sys.executable} -m pip install vosk")

    if vosk:
        roots = _model_dirs()
        # A model left as a .zip is unpacked automatically.
        for root in roots:
            if not root.is_dir():
                continue
            for z in root.glob("vosk*.zip"):
                target = root / z.stem
                if not target.exists():
                    try:
                        print(f"[voice] unpacking {z.name} …")
                        with zipfile.ZipFile(z) as zf:
                            zf.extractall(root)
                    except Exception as e:
                        print(f"[voice] could not unpack {z.name}: {e}")
        found = []
        for root in roots:
            if root.is_dir():
                found += [d for d in sorted(root.rglob("*")) if d.is_dir() and (d / "am").is_dir()]
        print("[voice] looked for a Vosk model in: " + ", ".join(str(r) for r in roots))
        if not found:
            STT_STATUS = "no Vosk model found (expected a folder containing am/ and conf/ inside models/)"
            print(f"[voice] {STT_STATUS}")
            print("[voice] download one from alphacephei.com/vosk/models and unzip it into the "
                  f"models folder next to EMLII.py: {roots[0]}")
        else:
            try:
                STT_MODEL = vosk.Model(str(found[0]))
                STT_NAME = found[0].name
                STT_STATUS = "ok"
                print(f"[voice] offline speech input: {STT_NAME}  ({found[0]})")
            except Exception as e:
                STT_STATUS = f"found a model at {found[0]} but it failed to load: {e}"
                print(f"[voice] {STT_STATUS}")

    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            TTS_ENGINE = exe
            break
    else:
        try:
            import pyttsx3  # noqa: F401
            TTS_ENGINE = "pyttsx3"
        except ImportError:
            pass
    print(f"[voice] server speech output: {TTS_ENGINE or 'none (browser voices only)'}")


def transcribe_pcm(data):
    """data = raw 16-bit mono little-endian PCM at 16 kHz. Returns the text."""
    import vosk
    rec = vosk.KaldiRecognizer(STT_MODEL, 16000)
    rec.AcceptWaveform(data)
    return (json.loads(rec.FinalResult()).get("text") or "").strip()


def synth_wav(text):
    """Render text to WAV bytes with the server-side engine, or None on failure."""
    text = (text or "").strip()[:600]
    if not text or not TTS_ENGINE:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with _TTS_LOCK:
            if TTS_ENGINE in ("espeak-ng", "espeak"):
                subprocess.run([TTS_ENGINE, "-w", tmp, "--", text],
                               check=True, capture_output=True, timeout=30)
            else:
                import pyttsx3
                eng = pyttsx3.init()
                eng.save_to_file(text, tmp)
                eng.runAndWait()
                eng.stop()
        with open(tmp, "rb") as f:
            return f.read() or None
    except Exception as e:
        print(f"[voice] speech synthesis failed: {e}")
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def query_ollama(prompt, system=None):
    """
    Call a local Ollama server for a one-shot completion.
    Returns the generated text, or None if Ollama is unreachable, the
    model isn't pulled, or the request times out — callers should treat
    None as "this layer had nothing to offer" and fall through.
    """
    if not AI_ENABLED:
        return None
    payload = {
        "model":  ACTIVE_MODEL or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        text = (data.get("response") or "").strip()
        return text or None
    except (urllib.error.URLError, socket.timeout, TimeoutError,
            json.JSONDecodeError, ConnectionError, OSError):
        return None

# ANSI colour codes (terminal mode only)
GREEN  = '\033[92m'
CYAN   = '\033[96m'
YELLOW = '\033[93m'
RED    = '\033[91m'
MUTED  = '\033[90m'
BOLD   = '\033[1m'
RESET  = '\033[0m'


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  (stores API key and other settings across runs)
# ═══════════════════════════════════════════════════════════════════════════════

def load_config():
    """
    Load settings from data/config.json.
    Returns an empty dict if the file doesn't exist or is malformed.
    """
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(updates):
    """
    Merge `updates` into the existing config and write to disk.
    Merging (not replacing) means a single !setkey call won't wipe other settings.
    """
    cfg = load_config()
    cfg.update(updates)
    Path(CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# INPUTS INDEX  (hash → original text, needed for similarity search)
# ═══════════════════════════════════════════════════════════════════════════════

def load_inputs():
    """
    Load the full hash→text index from disk.
    Returns {} on first run (file doesn't exist yet).
    """
    try:
        with open(INPUTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_input(h, text):
    """
    Record that hash `h` was produced by `text`.
    Called every time EMLII learns something (manual teach OR prediction save).
    This is what makes future similarity searches possible.
    """
    inputs = load_inputs()
    inputs[h] = text
    with open(INPUTS_FILE, 'w') as f:
        json.dump(inputs, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY ENGINE  (offline, no dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(text):
    """
    Break a sentence into a set of lowercase word tokens.
    re.findall(r'\w+', ...) extracts sequences of word characters (letters, digits, _).
    Using a set removes duplicates — we care about which words appear, not how often.

    Example:
        tokenize("Hello there, hello!") → {'hello', 'there'}
    """
    return set(re.findall(r'\w+', text.lower()))


def jaccard_similarity(set_a, set_b):
    """
    Compute Jaccard similarity between two word sets.

    Jaccard = |intersection| / |union|
    Range: 0.0 (no shared words) to 1.0 (identical word sets).

    Example:
        a = {'how', 'are', 'you'}
        b = {'how', 'are', 'things'}
        intersection = {'how', 'are'}    size = 2
        union        = {'how','are','you','things'}  size = 4
        Jaccard = 2/4 = 0.5
    """
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sequence_similarity(a, b):
    """
    Character-level similarity using difflib.SequenceMatcher.
    Better than Jaccard for very short inputs (1–2 words) where word-set overlap
    is too coarse.

    SequenceMatcher finds the longest common subsequence of characters.
    .ratio() returns 2 * matching_chars / total_chars.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_similar(text):
    """
    Search the inputs index for the most semantically similar past input.

    Strategy:
        - For inputs with 3+ words: use Jaccard (word overlap), which handles
          paraphrasing well ("how are you" vs "how r u doing").
        - For short inputs (1–2 words): fall back to sequence similarity,
          which catches typos and abbreviations ("helo" ≈ "hello").
        - Take the maximum of both scores for each candidate.

    Returns:
        (hash, score, source_text) if best score ≥ SIMILARITY_THRESHOLD
        (None, 0.0, None) otherwise.
    """
    inputs = load_inputs()
    if not inputs:
        return None, 0.0, None

    tokens   = tokenize(text)
    is_short = len(tokens) <= 2   # Switch metric for short inputs

    best_hash  = None
    best_score = 0.0
    best_text  = None

    for h, stored_text in inputs.items():
        stored_tokens = tokenize(stored_text)

        if is_short:
            score = sequence_similarity(text, stored_text)
        else:
            j = jaccard_similarity(tokens, stored_tokens)
            s = sequence_similarity(text, stored_text)
            score = max(j, s)   # Take the better of the two metrics

        if score > best_score:
            best_score = score
            best_hash  = h
            best_text  = stored_text

    if best_score >= SIMILARITY_THRESHOLD:
        return best_hash, best_score, best_text
    return None, best_score, None


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN RULES  (offline keyword/regex matching)
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry is (list_of_keywords_or_phrases, list_of_possible_responses).
# EMLII picks a random response from the matched bucket, adding variety.
PATTERNS = [
    (
        ['hello', 'hi', 'hey', 'sup', "what's up", 'howdy', 'yo', 'hiya'],
        [
            "Hey! What's on your mind?",
            "Hello! How can I help?",
            "Hi there! What would you like to talk about?",
        ]
    ),
    (
        ['how are you', 'how r u', 'how do you feel', 'you okay', 'you alright'],
        [
            "I'm EMLII — no feelings, but I'm always learning. What about you?",
            "Running fine! Every conversation makes me smarter.",
        ]
    ),
    (
        ['your name', 'who are you', 'what are you', 'introduce yourself'],
        [
            f"I'm EMLII — a self-learning chatbot that grows smarter with every conversation.",
            "EMLII. I learn from conversations and get better over time.",
        ]
    ),
    (
        ['what can you do', 'help me', 'your commands', 'what do you do', 'capabilities'],
        [
            "I can chat, hash text, crack hashes, encrypt messages, and serve files on your LAN. Type !help for the full list.",
        ]
    ),
    (
        ['thank', 'thanks', 'thx', 'ty', 'cheers'],
        ["You're welcome!", "Happy to help!", "Anytime."]
    ),
    (
        ['bye', 'goodbye', 'see you', 'later', 'cya', 'ttyl'],
        ["See you! Type !exit to end the session.", "Later! Come back anytime."]
    ),
    (
        ['good morning', 'morning'],
        ["Good morning! Ready to learn something today?"]
    ),
    (
        ['good night', 'goodnight', 'night'],
        ["Good night! See you next time."]
    ),
    (
        ['love you', 'i love you'],
        ["That's sweet. I'm a chatbot, but I appreciate the sentiment!"]
    ),
    (
        ['are you human', 'are you real', 'are you a bot', 'are you ai'],
        ["I'm EMLII — a learning chatbot, not a human. But I'm getting smarter every day."]
    ),
    (
        ['tell me a joke', 'joke', 'say something funny'],
        [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I would tell a UDP joke, but you might not get it.",
            "Why did the Python programmer get lost? They followed too many exceptions.",
        ]
    ),
    (
        ['what time is it', 'current time', 'what is the time'],
        # Evaluated at match time, not at definition time
        ['__TIME__']
    ),
]


def pattern_response(text):
    """
    Check the input against the PATTERNS table and return a matching response.

    Matching logic:
        - Lowercase the input.
        - Check if any keyword/phrase from a bucket is contained anywhere in the input.
        - If matched, return a random choice from that bucket's responses.

    '__TIME__' is a special token replaced with the actual current time.

    Returns a response string, or None if nothing matched.
    """
    text_lower = text.lower().strip()

    for keywords, responses in PATTERNS:
        if any(kw in text_lower for kw in keywords):
            response = random.choice(responses)
            if response == '__TIME__':
                response = f"It's {datetime.now().strftime('%H:%M')} on {datetime.now().strftime('%A, %d %B %Y')}."
            return response

    # Question words — generic fallback for unmatched questions
    question_starters = ('what', 'where', 'when', 'why', 'who', 'how',
                         'is ', 'are ', 'do ', 'can ', 'does ', 'did ')
    if any(text_lower.startswith(q) for q in question_starters):
        return random.choice([
            "Hmm, I'm still learning about that. The more we chat, the smarter I get!",
            "I don't have an answer for that yet, but keep talking to me — I'll learn.",
            "Not sure yet. Teach me by just telling me what you'd like me to say!",
        ])

    return None   # No pattern matched

#___________________________SPECIAL SECTION___________________________________
#Sips Coffee ==> if you're reading this your'e amongst the 35% who's still carrying the prgramming legacy  (:
#PS Tylan --EMLII Creator
#I made effort to comment what each function and import are for. Your'e welcome
#Sips Coffee ==> bye (;


# ═══════════════════════════════════════════════════════════════════════════════
# HASH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Common password dictionary ────────────────────────────────────────────────
# The crack function tries every word here before brute-forcing. Most real-world
# hash cracks succeed on this list because people use weak passwords.

WORDLIST = [
    "password", "123456", "qwerty", "admin", "letmein", "monkey", "dragon",
    "master", "sunshine", "princess", "welcome", "shadow", "superman",
    "michael", "football", "baseball", "abc123", "iloveyou", "trustno1",
    "passw0rd", "batman", "hello", "freedom", "login", "starwars", "access",
    "mustang", "hunter", "ranger", "joshua", "thomas", "robert", "daniel",
    "george", "jordan", "harley", "dakota", "cheese", "pepper", "butter",
    "cookie", "killer", "matrix", "buster", "ginger", "summer", "winter",
    "spring", "autumn", "purple", "orange", "yellow", "silver", "hacker123",
]


def do_hash(text, algo="md5"):
    """
    Hash a string with the specified algorithm and return the hex digest.

    Parameters:
        text  (str): The plaintext to hash.
        algo  (str): Algorithm name — 'md5', 'sha1', 'sha256', or 'sha512'.
                     Defaults to 'md5' if the name is unrecognised.

    Returns:
        str: Lowercase hex string, e.g. '5d41402abc4b2a76b9719d911017c592'

    How it works:
        Hash functions take any input and produce a fixed-length fingerprint.
        The same input always produces the same fingerprint (deterministic).
        You cannot reverse the fingerprint back to the input (one-way).
        Even changing one character completely changes the fingerprint.
    """
    # Map algorithm names to the actual hashlib constructor functions.
    # The functions are stored as values, NOT called yet (no parentheses).
    fns = {
        "md5":    hashlib.md5,
        "sha1":   hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }

    # .get(algo, hashlib.md5) returns the function if found, or MD5 as fallback.
    # (text.encode()) converts the Python string to UTF-8 bytes — hash functions
    # require bytes, not strings.
    # .hexdigest() converts the raw binary hash output to a readable hex string.
    return fns.get(algo, hashlib.md5)(text.encode()).hexdigest()


def hash_all(text):
    """
    Hash a string with all four supported algorithms at once.

    Returns:
        dict: { 'md5': '...', 'sha1': '...', 'sha256': '...', 'sha512': '...' }

    Used by the !hash command to show all digests in one go rather than
    making the user run the command four times.
    """
    # Dict comprehension: creates a dict by iterating over algorithm names
    # and calling do_hash for each one. Equivalent to a for-loop building a dict.
    return {algo: do_hash(text, algo) for algo in ["md5", "sha1", "sha256", "sha512"]}


def crack_hash(target, verbose=False):
    """
    Attempt to find the original text that produced the given hash.

    Works in two phases:
        1. Dictionary attack: tries every word in WORDLIST against MD5, SHA1, SHA256.
        2. Brute force: tries every combination of a-z and 0-9 up to 3 characters.

    Parameters:
        target  (str):  The hex hash string to crack.
        verbose (bool): If True, prints progress to the terminal (used in terminal mode).

    Returns:
        dict: { 'found': True,  'text': 'password', 'algo': 'md5' }
           or { 'found': False }

    Why only 3-char brute force:
        36 chars (a-z + 0-9), length 3 = 36^3 = 46,656 combinations × 3 algos
        = ~140,000 hash operations. Fast enough for a web request.
        Length 4 would be 1.7 million operations — too slow for a live response.
    """
    # Normalise: strip accidental spaces and convert to lowercase so
    # 'ABC123' and 'abc123' compare equal.
    target = target.strip().lower()

    algos = ["md5", "sha1", "sha256"]

    # ── Phase 1: Dictionary attack ──
    for word in WORDLIST:
        for algo in algos:
            if do_hash(word, algo) == target:
                # Found a match — return immediately without brute forcing
                return {"found": True, "text": word, "algo": algo}

    # ── Phase 2: Brute force ──
    # string.ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    # string.digits          = '0123456789'
    chars = string.ascii_lowercase + string.digits

    for length in range(1, 4):   # Try lengths 1-3
        for combo in itertools.product(chars, repeat=length):
            # itertools.product('ab', repeat=2) → ('a','a'),('a','b'),('b','a'),('b','b')
            # ''.join(combo) converts the tuple ('a','b') into the string 'ab'
            word = ''.join(combo)

            if verbose:
                # \r returns the cursor to the start of the line so the next
                # print overwrites it — creating an in-place progress display.
                print(f"\r{MUTED}Trying: {word:<5}{RESET}", end='', flush=True)

            for algo in algos:
                if do_hash(word, algo) == target:
                    if verbose:
                        print()   # Move to new line after the \r progress display
                    return {"found": True, "text": word, "algo": algo}

    if verbose:
        print()
    return {"found": False}


# ═══════════════════════════════════════════════════════════════════════════════
# ENCRYPTION
# ═══════════════════════════════════════════════════════════════════════════════

def caesar(text, shift):
    """
    Encrypt or decrypt text using a Caesar cipher (character rotation).

    A Caesar cipher shifts each letter by `shift` positions in the alphabet.
    ROT-13 is a Caesar cipher with shift=13 (the most common variant because
    applying it twice returns the original: 13+13=26=full alphabet).

    Parameters:
        text  (str): The text to transform.
        shift (int): How many positions to shift. Positive = encrypt, negative = decrypt.
                     Use shift=13 for ROT-13 encrypt, shift=-13 for ROT-13 decrypt.

    Returns:
        str: The transformed text (non-letters pass through unchanged).

    Example:
        caesar('hello', 13)  → 'uryyb'
        caesar('uryyb', -13) → 'hello'

    How the maths works:
        'a' has ASCII code 97. Subtracting base (97) gives position 0.
        Adding shift and taking % 26 keeps the result in range 0-25.
        Adding base back converts to an ASCII code. chr() converts to a character.
    """
    out = []
    for ch in text:
        if ch.isalpha():
            # ord() converts a character to its ASCII integer code.
            # Upper and lower case have different base codes (65 vs 97),
            # so we pick the right base to keep case intact.
            base = ord('A') if ch.isupper() else ord('a')

            # Subtract base → position in alphabet (0-25)
            # Add shift     → shift position
            # % 26          → wrap around (position 25 + 1 = position 0 = 'a')
            # Add base      → back to ASCII code
            # chr()         → back to character
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            # Numbers, spaces, punctuation — pass through unchanged
            out.append(ch)

    # Join the list of characters into a single string
    return ''.join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE SERVER  (from emserver.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ── File server state ─────────────────────────────────────────────────────────
# A single dict holds everything about the current file server session.
# 'httpd' stores the actual server object so stop_fs() can call .shutdown() on it.

fs = {
    "running":   False,
    "port":      8080,
    "directory": str(Path.home()),  # Default to home dir; Path.home() = ~ resolved
    "httpd":     None,
}

# A mutex (mutual exclusion lock) protects the fs dict.
# Flask and the file server run on different threads simultaneously.
# Without the lock, both could read/write fs at the same time → corrupted data.
# Only one thread can hold the lock at a time.
fs_lock = threading.Lock()


def get_lan_ip():
    """
    Detect the device's local network (LAN) IP address.

    Returns:
        str: The LAN IP, e.g. '192.168.1.42'. Falls back to '127.0.0.1'.

    How the trick works:
        A UDP socket is created (SOCK_DGRAM = UDP, connectionless protocol).
        Calling .connect() on a UDP socket does NOT send any data.
        It just makes the OS decide which network interface it would use
        to reach the given address. Once the OS has decided, .getsockname()
        returns the local IP of that interface — which is your LAN IP.
        The target address (10.255.255.255) is non-routable and unreachable;
        it's only used to trigger the OS's routing decision.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # AF_INET = IPv4 address family
        # SOCK_DGRAM = UDP (we don't need a real connection, just routing info)
        s.connect(("10.255.255.255", 1))  # No data sent; OS just picks an interface
        ip = s.getsockname()[0]           # [0] = IP address, [1] would be port
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Loopback fallback if no network available


def _run_fs(directory, port):
    """
    Start the HTTP file server and block until shutdown() is called.

    This function is designed to run inside a background thread.
    It blocks on serve_forever() — that's intentional because the calling
    thread (the background thread) has nothing else to do.

    Parameters:
        directory (str): The folder to serve.
        port      (int): The TCP port to listen on.
    """
    # Serve `directory` via the `directory=` keyword (supported since Python
    # 3.7) instead of os.chdir(). os.chdir() changes the *whole process's*
    # working directory, not just this thread's — every other relative path
    # in the app (data/mem/, data/history/, data/files/) would start
    # resolving inside whatever folder got served, silently redirecting
    # EMLII's own memory/history writes until the process restarts.
    SimpleHTTPRequestHandler.log_message = lambda *a: None  # silence request logging
    handler = functools.partial(SimpleHTTPRequestHandler, directory=directory)

    # ("", port) = listen on ALL network interfaces (empty string = 0.0.0.0).
    # This is what makes other devices on the LAN able to reach the server.
    httpd = ThreadingHTTPServer(("", port), handler)

    # Update shared state under the lock before serve_forever() blocks.
    with fs_lock:
        fs["httpd"]   = httpd
        fs["running"] = True

    # Block here, handling requests, until httpd.shutdown() is called
    # from another thread (i.e., when the user runs !stopserver).
    httpd.serve_forever()


def start_fs(directory, port):
    """
    Spawn a background thread that runs the file server.

    Parameters:
        directory (str): The folder to serve.
        port      (int): The TCP port to listen on.

    Why daemon=True:
        Daemon threads are killed automatically when the main program exits.
        Without it, the file server thread would keep the process alive even
        after you close the terminal or press Ctrl+C.
    """
    t = threading.Thread(
        target=_run_fs,         # The function the thread will call
        args=(directory, port), # Arguments to pass to _run_fs
        daemon=True,            # Die when the main thread dies
    )
    t.start()


def stop_fs():
    """
    Gracefully shut down the file server if it is running.

    httpd.shutdown() signals serve_forever() to exit its loop cleanly,
    allowing the background thread to finish naturally.
    The lock is held for the entire operation so no other thread reads
    stale state while the shutdown is in progress.
    """
    with fs_lock:
        if fs.get("httpd"):
            fs["httpd"].shutdown()  # Signals serve_forever() to stop
            fs["httpd"] = None
        fs["running"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# EMLII CORE
# ═══════════════════════════════════════════════════════════════════════════════

class EMLII:
    """
    Central chatbot logic shared by both terminal and web modes.

    process() now runs a 4-layer prediction pipeline before falling
    back to manual teaching. Every prediction is saved to disk so
    the same input becomes an exact match on the next call.

    Result dict 'type' values:
        reply         → known response (exact match)
        predicted     → generated response + source + confidence + hash
        needs_teaching→ all layers failed, ask user
        learned       → just saved a new response
        help / hash_result / crack_start / crypt_result / dirserver / kill
    """

    VERSION = "3.5"


    @staticmethod
    def process(userchat, username="User"):
        """
        Main entry point — routes input through the full pipeline.

        Layer order:
            1. Exact match        (instant, O(1) file lookup)
            2. Similarity search  (scans inputs.json, Jaccard + SequenceMatcher)
            3. Pattern rules      (keyword table, O(n) where n = pattern count)
            4. Local AI model     (Ollama / SmolLM2-360M, last resort before asking)
            5. Teach fallback     (asks user, saves result)
        """
        userchat = userchat.strip()
        if not userchat:
            return None

        if userchat[0] == '!':
            return EMLII.commands(userchat, username)

        # Compute hash once — used throughout the pipeline
        h = hashlib.md5(userchat.encode()).hexdigest()

        # ── Layer 1: Exact match ───────────────────────────────────────────────
        try:
            with open(PATHS['mem'] + h) as f:
                return {"type": "reply", "text": f.read()}
        except FileNotFoundError:
            pass   # Not found — fall through to prediction layers

        # ── Layer 2: Similarity search ────────────────────────────────────────
        similar_hash, score, matched_text = find_similar(userchat)
        if similar_hash:
            try:
                with open(PATHS['mem'] + similar_hash) as f:
                    response = f.read()
                # Save immediately so this becomes an exact match next time
                EMLII._persist(h, userchat, response)
                return {
                    "type":       "predicted",
                    "text":       response,
                    "source":     "similarity",
                    "confidence": round(score, 2),
                    "matched":    matched_text,   # What input it was similar to
                    "hash":       h,
                }
            except FileNotFoundError:
                pass   # Response file was deleted — fall through

        # ── Layer 3: Pattern rules ────────────────────────────────────────────
        pattern = pattern_response(userchat)
        if pattern:
            EMLII._persist(h, userchat, pattern)
            return {
                "type":       "predicted",
                "text":       pattern,
                "source":     "pattern",
                "confidence": 1.0,
                "hash":       h,
            }

        # ── Layer 4: Local AI model (Ollama) ──────────────────────────────────
        # Similarity + pattern both missed. Try a small local model before
        # giving up. Returns None (silently) if Ollama isn't running or the
        # model isn't pulled — never blocks the pipeline.
        ai_text = query_ollama(
            userchat,
            system="You are EMLII, a concise command-line assistant. "
                   "Reply in 1-3 short sentences.",
        )
        if ai_text:
            EMLII._persist(h, userchat, ai_text)
            return {
                "type":       "predicted",
                "text":       ai_text,
                "source":     "ai",
                "confidence": 0.5,   # Model doesn't give us a real confidence score
                "hash":       h,
            }

        # ── Layer 5: Teach fallback ───────────────────────────────────────────
        # All prediction layers failed — ask the user to manually teach EMLII.
        return {"type": "needs_teaching", "hash": h, "prompt": userchat}


    @staticmethod
    def _persist(h, original_text, response):
        """
        Save a response to disk AND record the original text in inputs.json.
        Called by every prediction layer that succeeds.

        Private helper (leading underscore = internal use only convention).
        Not a true Python private method — just a naming signal to readers.
        """
        # Save the response file
        with open(PATHS['mem'] + h, 'w') as f:
            f.write(response)
        # Save the hash→text mapping for future similarity searches
        save_input(h, original_text)


    @staticmethod
    def learn(h, response, original_text=None):
        """
        Manually save a response (called from the teach flow).

        Parameters:
            h             : MD5 hex of the original input
            response      : What EMLII should say
            original_text : The original input (stored for similarity search)
        """
        with open(PATHS['mem'] + h, 'w') as f:
            f.write(response)
        if original_text:
            save_input(h, original_text)
        return {"type": "learned", "text": response}


    @staticmethod
    def correct(h, new_response, original_text=None):
        """
        Overwrite an existing (predicted) response with a user correction.
        Same as learn() but semantically distinct — used when correcting a
        prediction rather than teaching something new.
        """
        return EMLII.learn(h, new_response, original_text)


    @staticmethod
    def commands(userchat, username="User"):
        """Parse and dispatch a !command string."""
        raw   = userchat[1:]
        parts = raw.split(None, 1)
        cmd   = parts[0].lower().strip()
        arg   = parts[1].strip() if len(parts) > 1 else ""


        if cmd == 'help':
            cmds = [
                ("!help",              "Show all commands"),
                ("!version",           "EMLII version info"),
                ("!date",              "Current date & time"),
                ("!calculate <expr>",  "Evaluate a math expression"),
                ("!hash <text>",       "MD5 / SHA1 / SHA256 / SHA512"),
                ("!crack <hash>",      "Dictionary + brute-force crack"),
                ("!encrypt <text>",    "ROT-13 Caesar cipher encrypt"),
                ("!decrypt <text>",    "ROT-13 Caesar cipher decrypt"),
                ("!dirserver [path]",  "Start LAN file server on port 8080"),
                ("!stopserver",        "Stop the file server"),
                ("!memory",            "Show how much EMLII has learned"),
                ("!exit / !quit",      "End session"),
            ]
            # plugins dropped into tools/ show up here too
            cmds += [(f"!{c}", getattr(m, "HELP", "")) for c, m in sorted(TOOLS.items())]
            return {"type": "help", "commands": cmds}


        elif cmd == 'version':
            return {"type": "reply", "text": f"EMLII v{EMLII.VERSION} — Self-Learning Edition"}


        elif cmd == 'date':
            return {"type": "reply", "text": f"Date: {datetime.now().ctime()}"}


        elif cmd == 'calculate':
            if not arg:
                return {"type": "reply", "text": "Usage: !calculate <expression>"}
            try:
                allowed = set('0123456789+-*/(). ')
                if not all(c in allowed for c in arg):
                    raise ValueError
                result = eval(arg, {"__builtins__": {}})
                return {"type": "reply", "text": f"{arg} = {result}"}
            except Exception:
                return {"type": "reply", "text": f"Could not evaluate: {arg}"}


        elif cmd == 'hash':
            if not arg:
                return {"type": "reply", "text": "Usage: !hash <text>"}
            return {"type": "hash_result", "input": arg, "results": hash_all(arg)}


        elif cmd == 'crack':
            if not arg:
                return {"type": "reply", "text": "Usage: !crack <hash>"}
            return {"type": "crack_start", "hash": arg}


        elif cmd in ('encrypt', 'ecrypt'):
            if not arg:
                return {"type": "reply", "text": "Usage: !encrypt <text>"}
            return {"type": "crypt_result", "original": arg,
                    "output": caesar(arg, 13), "mode": "Encrypt", "method": "ROT-13"}


        elif cmd == 'decrypt':
            if not arg:
                return {"type": "reply", "text": "Usage: !decrypt <text>"}
            return {"type": "crypt_result", "original": arg,
                    "output": caesar(arg, -13), "mode": "Decrypt", "method": "ROT-13"}


        elif cmd == 'dirserver':
            with fs_lock:
                if fs["running"]:
                    return {"type": "reply",
                            "text": f"Already running → http://{get_lan_ip()}:{fs['port']}"}
            directory = arg if arg else str(Path.home())
            if not os.path.isdir(directory):
                return {"type": "reply", "text": f"Directory not found: {directory}"}
            with fs_lock: fs["port"] = 8080; fs["directory"] = directory
            start_fs(directory, 8080)
            return {"type": "dirserver", "url": f"http://{get_lan_ip()}:8080",
                    "directory": directory}


        elif cmd == 'stopserver':
            stop_fs()
            return {"type": "reply", "text": "File server stopped."}


        # ── !memory — show learning statistics ────────────────────────────────
        elif cmd == 'memory':
            inputs  = load_inputs()
            mem_dir = PATHS['mem']
            # Count response files (excludes any non-file entries)
            resp_count = sum(1 for f in Path(mem_dir).iterdir() if f.is_file())
            cfg        = load_config()
            return {
                "type": "reply",
                "text": (
                    f"EMLII memory stats\n"
                    f"─────────────────────────\n"
                    f"Learned responses : {resp_count}\n"
                    f"Indexed inputs    : {len(inputs)}\n"
                    f"Similarity cutoff : {SIMILARITY_THRESHOLD}"
                )
            }


        # ── Tool plugins ──────────────────────────────────────────────────────
        elif cmd in TOOLS:
            ctx = {
                "PATHS":    PATHS,
                "username": username,
                "fs":       fs,            # read-only peek at the file-server state if a tool wants it
            }
            try:
                result = TOOLS[cmd].run(arg, ctx)
            except Exception as e:
                # A plugin raising an exception should never take EMLII down.
                return {"type": "reply", "text": f"'!{cmd}' failed: {e}"}
            if isinstance(result, dict) and result.get("type"):
                return result
            # Tools may return a plain string instead of a full result dict.
            return {"type": "reply", "text": str(result)}



        elif cmd in KILL or cmd == 'exit':
            return {"type": "kill"}


        else:
            return {"type": "reply",
                    "text": f"Unknown command: !{cmd}\nType !help for a list."}

#Sips Coffee (0<0)-~U3

# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL FRONTEND
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    print(f"""
{GREEN}{BOLD} ######  ##     ##  #       ###  ###
 #       # #   # #  #        #    #
 ######  #  # #  #  #        #    #
 #       #   #   #  #        #    #
 ######  #       #  ######  ###  ###
{RESET}
{GREEN} EMLII v{EMLII.VERSION} — Self-Learning Edition{RESET}
 {"─" * 42}
""")


def render_terminal(result, username):
    """
    Render a result dict as coloured terminal output.
    Returns True if the session should end (kill signal).

    'predicted' type: shows the response with a source badge, then offers
    the user a chance to correct it in-line. Pressing Enter accepts.
    """
    if not result:
        return False

    t = result.get('type')

    if t == 'reply':
        print(f"\n{GREEN}EMLII:{RESET} {result['text']}\n")


    elif t == 'predicted':
        # Show the source so the user knows how the response was generated
        source_badges = {
            "similarity": f"{CYAN}~similar input{RESET}",
            "pattern":    f"{CYAN}~pattern{RESET}",
            "ai":         f"{YELLOW}~ai model (SmolLM2){RESET}",
        }
        badge = source_badges.get(result.get('source', ''), f"{CYAN}~predicted{RESET}")
        conf  = int(result.get('confidence', 0) * 100)

        print(f"\n{GREEN}EMLII{RESET} {badge}{MUTED} ({conf}% confidence){RESET}")
        print(f"  {result['text']}\n")

        # Offer in-line correction. Empty input = accept.
        correction = input(f"{MUTED}  [Enter] to accept, or type a correction: {RESET}").strip()
        if correction:
            # User disagreed — overwrite with their correction
            EMLII.correct(result['hash'], correction)
            print(f"{MUTED}  Correction saved.{RESET}\n")
        # If correction is empty, do nothing — prediction already persisted


    elif t == 'needs_teaching':
        print(f"\n{YELLOW}EMLII:{RESET} I don't know how to respond to \"{result['prompt']}\".")
        response = input(f"{YELLOW}     → Teach me (or Enter to skip): {RESET}").strip()
        if response:
            EMLII.learn(result['hash'], response, result['prompt'])
            print(f"\n{GREEN}EMLII:{RESET} {response}\n")
        else:
            print(f"{MUTED}     Skipped.{RESET}\n")


    elif t == 'learned':
        print(f"\n{GREEN}EMLII:{RESET} {result['text']}\n")


    elif t == 'help':
        print(f"\n{MUTED}{'─' * 46}{RESET}")
        print(f"{GREEN} Commands:{RESET}")
        print(f"{MUTED}{'─' * 46}{RESET}")
        for cmd, desc in result['commands']:
            print(f"  {CYAN}{cmd:<28}{RESET}{MUTED}{desc}{RESET}")
        print(f"{MUTED}{'─' * 46}{RESET}\n")


    elif t == 'hash_result':
        print(f"\n{MUTED} Input: \"{result['input']}\"{RESET}")
        print(f"{MUTED}{'─' * 46}{RESET}")
        for algo, h in result['results'].items():
            print(f"  {MUTED}{algo.upper():<8}{RESET}{CYAN}{h}{RESET}")
        print(f"{MUTED}{'─' * 46}{RESET}\n")


    elif t == 'crack_start':
        print(f"\n{YELLOW} Cracking...{RESET}\n")
        d = crack_hash(result['hash'], verbose=True)
        if d['found']:
            print(f"\n{GREEN} Cracked!{RESET}  {MUTED}{d['algo'].upper()}{RESET} → {GREEN}{BOLD}{d['text']}{RESET}\n")
        else:
            print(f"\n{RED} Could not crack.{RESET}\n")


    elif t == 'crypt_result':
        print(f"\n{MUTED} {result['mode']} · {result['method']}{RESET}")
        print(f"  {MUTED}Input :{RESET} {result['original']}")
        print(f"  {MUTED}Output:{RESET} {YELLOW}{result['output']}{RESET}\n")


    elif t == 'dirserver':
        print(f"\n{GREEN} File server started!{RESET}")
        print(f"  {MUTED}Directory :{RESET} {result['directory']}")
        print(f"  {MUTED}LAN URL   :{RESET} {CYAN}{result['url']}{RESET}\n")


    elif t == 'kill':
        print(f"\n{MUTED} Goodbye, {username}.{RESET}\n")
        return True

    return False


def run_terminal():
    """Main terminal loop."""
    print_header()
    username = input(f"{GREEN}Enter your username: {RESET}").strip() or "User"
    print(f"\n{GREEN}EMLII:{RESET} Hello, {username}. I am EMLII v{EMLII.VERSION}.")
    print(f"{GREEN}EMLII:{RESET} I learn and predict — the more we chat, the smarter I get.\n")

    while True:
        try:
            user_input = input(f"{CYAN}{username} ▶ {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{MUTED} Session ended.{RESET}\n"); break

        if not user_input:
            continue

        with open(PATHS['history'] + 'chat.txt', 'a') as f:
            f.write(f"[{datetime.now().ctime()}] {username}: {user_input}\n")

        if render_terminal(EMLII.process(user_input, username), username):
            break


# ═══════════════════════════════════════════════════════════════════════════════
# WEB FRONTEND  (Flask)
# ═══════════════════════════════════════════════════════════════════════════════

if FLASK_AVAILABLE:

    app = Flask(__name__)


    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        """Receive message, run pipeline, return result dict as JSON."""
        data = flask_request.get_json(silent=True) or {}
        msg  = data.get('message', '').strip()
        user = data.get('username', 'User')
        if not msg:
            return jsonify({"type": "reply", "text": ""})
        with open(PATHS['history'] + 'chat.txt', 'a') as f:
            f.write(f"[{datetime.now().ctime()}] {user}: {msg}\n")
        return jsonify(EMLII.process(msg, user) or {"type": "reply", "text": ""})


    @app.route('/api/teach', methods=['POST'])
    def api_teach():
        """Save a manually taught response."""
        data          = flask_request.get_json(silent=True) or {}
        h             = data.get('hash', '')
        resp          = data.get('response', '').strip()
        original_text = data.get('original_text', '')
        if not h or not resp:
            return jsonify({"ok": False})
        result = EMLII.learn(h, resp, original_text or None)
        return jsonify({"ok": True, **result})


    @app.route('/api/correct', methods=['POST'])
    def api_correct():
        """
        Overwrite a predicted response with the user's correction.
        Called when the user clicks 'Correct this' on a predicted bubble.
        Identical to /api/teach but semantically separate.
        """
        data          = flask_request.get_json(silent=True) or {}
        h             = data.get('hash', '')
        new_resp      = data.get('response', '').strip()
        original_text = data.get('original_text', '')
        if not h or not new_resp:
            return jsonify({"ok": False})
        result = EMLII.correct(h, new_resp, original_text or None)
        return jsonify({"ok": True, **result})


    @app.route('/api/crack', methods=['POST'])
    def api_crack():
        """Run crack_hash synchronously; may take several seconds."""
        data   = flask_request.get_json(silent=True) or {}
        target = data.get('hash', '').strip()
        if not target:
            return jsonify({"found": False})
        return jsonify(crack_hash(target, verbose=False))


    @app.route('/api/plugins')
    def api_plugins():
        """
        Expose the live tools/*.py plugin registry to the web UI, so the
        plugin count badge and the help panel's plugin list always reflect
        what load_tools() actually registered — nothing hardcoded client-side.
        Also reports which Layer 4 model is active: a local models/*.gguf
        file (loaded by load_models()) if one was found, else the pulled
        OLLAMA_MODEL fallback.
        """
        return jsonify({
            "plugins": [
                [cmd, getattr(mod, "HELP", "")] for cmd, mod in sorted(TOOLS.items())
            ],
            "model": {
                "name":  ACTIVE_MODEL or OLLAMA_MODEL,
                "local": ACTIVE_MODEL is not None,
            },
        })


    @app.route('/api/voice_caps')
    def api_voice_caps():
        """What the server can do offline — the voice UI picks its mode from this."""
        return jsonify({"stt": STT_MODEL is not None, "stt_model": STT_NAME, "stt_status": STT_STATUS, "tts": TTS_ENGINE})


    @app.route('/api/stt', methods=['POST'])
    def api_stt():
        """Body: raw 16 kHz mono 16-bit PCM of one utterance → {"text": ...}."""
        if STT_MODEL is None:
            return jsonify({"ok": False, "error": "no offline speech model loaded"}), 503
        data = flask_request.get_data()
        if len(data) < 3200:        # < 0.1 s of audio
            return jsonify({"ok": True, "text": ""})
        try:
            return jsonify({"ok": True, "text": transcribe_pcm(data)})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500


    @app.route('/api/tts', methods=['POST'])
    def api_tts():
        """Body: {"text": ...} → audio/wav rendered by the server's local engine."""
        data = flask_request.get_json(silent=True) or {}
        wav = synth_wav(data.get('text', ''))
        if not wav:
            return jsonify({"ok": False, "error": "no offline speech output available"}), 503
        return Response(wav, mimetype='audio/wav')


    @app.route('/')
    def index():
        return UI

    @app.route('/manifest.json')
    def web_manifest():
        return jsonify({
            "name": "EMLII", "short_name": "EMLII", "start_url": "/", "scope": "/",
            "display": "standalone", "background_color": "#0b0a10", "theme_color": "#0b0a10",
            "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
                      {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"}],
        })

    @app.route('/icon-<int:size>.png')
    def web_icon(size):
        for folder in (Path(__file__).resolve().parent, Path.cwd()):
            for name in (f"icon-{size}.png", "emlii.png"):
                f = folder / name
                if f.is_file():
                    return Response(f.read_bytes(), mimetype="image/png",
                                    headers={"Cache-Control": "public, max-age=86400"})
        return Response(status=404)


    UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0b0a10">
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon-192.png">
<title>EMLII v3.5</title>
<style>
:root {
  --bg:        #0b0a10;
  --surface:   #151220;
  --border:    #2a2240;
  --purple:    #b84bff;
  --purple-lo: #b84bff1c;
  --purple-hi: #d9a3ff;
  --glow:      0 0 10px #b84bff55, 0 0 2px #b84bff;
  --green:     #00e676;
  --green-lo:  #00e67614;
  --cyan:      #00c8b4;
  --yellow:    #f0c040;
  --yellow-lo: #f0c04018;
  --red:       #e05555;
  --pink:      #ff4fd8;
  --text:      #d4cdea;
  --muted:     #5a5280;
  --mono: 'JetBrains Mono','Fira Mono','Courier New',monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: var(--mono); font-size: 13px; overflow: hidden; }

.app { display: flex; flex-direction: column; height: 100vh; }

/* ── Top bar ── */
.topbar {
  flex-shrink: 0; background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 10px 16px;
  display: flex; align-items: center; justify-content: space-between;
}
.logo { font-size: 15px; font-weight: 700; color: var(--purple-hi); letter-spacing: .12em; text-shadow: var(--glow); }
.logo-ver { color: var(--muted); font-size: 11px; margin-top: 2px; }
.top-right { display: flex; align-items: center; gap: 10px; }
.session-tag { font-size: 11px; color: var(--muted); }
.session-tag b { color: #d9a3ff; font-weight: 400; }
.plugin-tag {
  font-size: 11px; color: var(--muted); border: 1px solid var(--border);
  border-radius: 5px; padding: 4px 10px; cursor: pointer; transition: all .15s;
}
.plugin-tag:hover { border-color: var(--purple); color: var(--purple-hi); background: var(--purple-lo); }
.plugin-tag b { color: var(--purple-hi); font-weight: 400; }
.ico-btn {
  background: none; border: 1px solid var(--border); border-radius: 5px;
  color: var(--muted); font-family: var(--mono); font-size: 11px;
  padding: 4px 10px; cursor: pointer; transition: all .15s;
}
.ico-btn:hover { border-color: var(--purple); color: var(--purple-hi); background: var(--purple-lo); }

/* ── Chat area ── */
.chat { flex: 1; overflow-y: auto; padding: 16px 16px 8px;
  display: flex; flex-direction: column; gap: 12px; }

/* ── Bubbles ── */
.msg { display: flex; flex-direction: column; gap: 3px; }
.msg.user { align-items: flex-end; }
.msg.bot  { align-items: flex-start; }
.msg.sys  { align-items: center; }
.who { font-size: 10px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.msg.bot .who { color: var(--purple-hi); }
.ts { font-size: 9px; color: var(--muted); text-transform: none; letter-spacing: 0; opacity: .7; }
/* predicted badge shown next to EMLII label */
.pred-badge {
  display: inline-block; font-size: 9px; padding: 1px 6px;
  border-radius: 3px; text-transform: lowercase;
  letter-spacing: .05em; vertical-align: middle;
}
.pred-badge.similarity { background: #1a1f0a; color: #d298e6; border: 1px solid #d298e6; }
.pred-badge.pattern    { background: #1a1f0a; color: #d298e6;     border: 1px solid #d298e6; }
.pred-badge.api        { background: #2a0f24; color: var(--pink); border: 1px solid var(--pink); }
.pred-badge.plugin     { background: #1f0f33; color: var(--purple-hi); border: 1px solid var(--purple); box-shadow: 0 0 6px #b84bff33; }

.bubble {
  max-width: min(88%, 620px);
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 9px 13px;
  line-height: 1.65; white-space: pre-wrap; word-break: break-word;
  position: relative;
}
.msg.user .bubble { background: #1c1430; border-color: #3a2a5c; }
.msg.bot  .bubble { color: var(--purple-hi); border-color: #2a2240; box-shadow: inset 0 0 18px #b84bff0a; }
/* Predicted bubbles get a subtle yellow tint to distinguish them */
.msg.bot .bubble.predicted { border-color: #ff4fd8; background: #2a0f24; color: #d9a3ff; }
.msg.sys .bubble { background: transparent; border-style: dashed; color: var(--muted); font-size: 11px; text-align: center; }
.copy-btn {
  display: none; position: absolute; top: 6px; right: 8px;
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  color: var(--muted); font-family: var(--mono); font-size: 9px;
  padding: 1px 6px; cursor: pointer;
}
.bubble:hover .copy-btn { display: inline-block; }
.copy-btn:hover { border-color: var(--purple); color: var(--purple-hi); }

/* ── Prediction action row (under predicted bubbles) ── */
.pred-actions {
  display: flex; align-items: center; gap: 8px;
  margin-top: 8px; padding-top: 8px;
  border-top: 1px solid #2a2510;
}
.pred-hint { font-size: 10px; color: var(--muted); flex: 1; }
.pred-btn {
  font-family: var(--mono); font-size: 11px; border-radius: 4px;
  padding: 3px 10px; cursor: pointer; border: 1px solid; transition: all .15s;
}
.pred-btn.ok   { background: transparent; border-color: var(--green); color: var(--green); }
.pred-btn.ok:hover { background: var(--green-lo); }
.pred-btn.edit { background: transparent; border-color: var(--muted); color: var(--muted); }
.pred-btn.edit:hover { border-color: var(--yellow); color: var(--yellow); }

/* ── Rich content ── */
.sub-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px; }
.hash-tbl { border-collapse: collapse; width: 100%; }
.hash-tbl td { padding: 4px 6px; border-bottom: 1px solid var(--border); }
.hash-tbl td:first-child { color: #ff4fd8; font-size: 11px; white-space: nowrap; padding-right: 14px; }
.hash-tbl td:last-child  { color: var(--cyan); font-size: 11px; word-break: break-all; }
.cmd-tbl { border-collapse: collapse; width: 100%; }
.cmd-tbl tr { border-bottom: 1px solid var(--border); }
.cmd-tbl td { padding: 5px 6px; }
.cmd-tbl td:first-child { color: #d9a3ff; white-space: nowrap; padding-right: 16px; }
.cmd-tbl td:last-child  { color: var(--muted); }
.crypt-box { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; margin-top: 6px; }
.crypt-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-bottom: 3px; }
.crypt-val { color: var(--yellow); word-break: break-all; }
.server-link { color: var(--cyan); text-decoration: none; }
.server-link:hover { text-decoration: underline; }

/* ── Teach / correct bar ── */
.teach-bar {
  flex-shrink: 0; display: none;
  background: #2a0f24; border-top: 2px solid #ff4fd8;
  padding: 10px 14px; flex-direction: column; gap: 8px;
}
.teach-bar.open { display: flex; }
.teach-prompt { font-size: 11px; color: #d9a3ff; line-height: 1.5; }
.teach-row { display: flex; gap: 8px; }

/* ── Input bar ── */
.input-bar {
  flex-shrink: 0; background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 11px 14px; display: flex; align-items: center; gap: 10px;
}
.prompt-sym { color: var(--purple-hi); white-space: nowrap; text-shadow: var(--glow); }
.chat-in {
  flex: 1; background: transparent; border: none; outline: none;
  color: var(--text); font-family: var(--mono); font-size: 13px;
  caret-color: var(--purple-hi);
}
.chat-in::placeholder { color: var(--muted); }
.chat-in:disabled { opacity: .3; }
.send-btn {
  background: none; border: 1px solid var(--border); border-radius: 5px;
  color: var(--muted); font-family: var(--mono); font-size: 11px;
  padding: 5px 10px; cursor: pointer; transition: all .15s;
}
.send-btn:hover { border-color: var(--purple); color: var(--purple-hi); box-shadow: var(--glow); }

/* ── Voice mode ── */
.voice-ov {
  position: fixed; inset: 0; z-index: 50;
  background: radial-gradient(ellipse at 50% 42%, #1a0b2e 0%, #07040d 52%, #000 100%);
  display: none; flex-direction: column; align-items: center; justify-content: space-between;
  padding: 18px 16px 28px;
}
.voice-ov.open { display: flex; }
.voice-top { width: 100%; display: flex; justify-content: space-between; align-items: center; }
.voice-title { color: var(--purple-hi); letter-spacing: .14em; font-weight: 700; text-shadow: var(--glow); }
.voice-stage { flex: 1; width: 100%; min-height: 0; position: relative; cursor: pointer; }
#voiceCanvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.voice-status { color: var(--purple-hi); text-shadow: var(--glow); font-size: 12px; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 8px; }
.voice-text {
  color: var(--text); font-size: 14px; text-align: center; max-width: 560px;
  min-height: 42px; line-height: 1.5; margin-bottom: 16px; padding: 0 8px;
}
.voice-ctl { display: flex; gap: 10px; }
.voice-ctl .ico-btn.on { border-color: var(--red); color: var(--red); }
.send-btn.mic-on { border-color: var(--purple); color: var(--purple-hi); }

/* ── Command autocomplete ── */
.autocomplete {
  position: absolute; bottom: 100%; left: 14px; right: 14px; margin-bottom: 4px;
  background: var(--surface); border: 1px solid var(--purple); border-radius: 6px;
  overflow: hidden; display: none; max-height: 180px; overflow-y: auto;
  box-shadow: var(--glow);
}
.autocomplete.open { display: block; }
.ac-item { padding: 6px 12px; font-size: 12px; cursor: pointer; display: flex; justify-content: space-between; gap: 10px; }
.ac-item .ac-cmd { color: var(--purple-hi); white-space: nowrap; }
.ac-item .ac-desc { color: var(--muted); font-size: 10px; text-align: right; overflow: hidden; text-overflow: ellipsis; }
.ac-item.sel, .ac-item:hover { background: var(--purple-lo); }

/* ── Shared controls ── */
.t-in { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 5px; padding: 7px 11px; color: var(--text); font-family: var(--mono); font-size: 13px; outline: none; }
.t-in:focus { border-color: var(--purple); box-shadow: var(--glow); }
.btn { border: none; border-radius: 5px; padding: 7px 14px; font-family: var(--mono); font-size: 12px; font-weight: 600; cursor: pointer; transition: opacity .15s, box-shadow .15s; }
.btn:active { opacity: .7; }
.btn-g { background: var(--purple); color: #0b0a10; }
.btn-g:hover { box-shadow: var(--glow); }
.btn-m { background: var(--border); color: var(--muted); }

/* ── Setup screen ── */
.setup {
  position: fixed; inset: 0; background: var(--bg); z-index: 100;
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 24px; padding: 24px;
}
.setup-art { color: var(--purple-hi); font-size: clamp(9px,2.2vw,13px); white-space: pre; line-height: 1.55; text-align: center; letter-spacing: .05em; text-shadow: var(--glow); }
.setup-sub { color: var(--muted); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; }
.setup-card { background: var(--surface); border: 1px solid #53127f; border-radius: 10px; padding: 22px; width: 100%; max-width: 360px; }
.setup-title { color: var(--muted); font-size: 11px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: .12em; }
.setup-row { display: flex; gap: 8px; }
.setup-foot { color: var(--muted); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; }

/* ── Help / plugin panels ── */
.help-panel { position: fixed; top: 0; right: -300px; width: 290px; height: 100%; background: var(--surface); border-left: 1px solid var(--border); z-index: 50; transition: right .22s ease; overflow-y: auto; padding: 18px; box-shadow: -8px 0 24px #00000066; }
.help-panel.open { right: 0; }
.hp-title { color: var(--purple-hi); font-size: 12px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; text-shadow: var(--glow); }
.hp-close { background: none; border: none; color: var(--muted); font-size: 18px; cursor: pointer; }
.hp-close:hover { color: var(--text); }
.hp-note { margin-top: 14px; color: var(--muted); font-size: 11px; line-height: 1.7; }
.hp-sub { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .1em; margin: 16px 0 8px; border-top: 1px solid var(--border); padding-top: 12px; }
.plugin-row { display: flex; justify-content: space-between; gap: 8px; padding: 5px 0; font-size: 11px; }
.plugin-row .pname { color: var(--purple-hi); white-space: nowrap; }
.plugin-row .pdesc { color: var(--muted); text-align: right; }
.plugin-empty { color: var(--muted); font-size: 11px; font-style: italic; }

/* ── Loading dots ── */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
.dots { display: inline-flex; gap: 4px; vertical-align: middle; }
.dots span { width: 5px; height: 5px; border-radius: 50%; background: var(--purple-hi); box-shadow: var(--glow); animation: blink 1s infinite; }
.dots span:nth-child(2) { animation-delay: .1s; }
.dots span:nth-child(3) { animation-delay: .3s; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: var(--purple); border-radius: 2px; opacity: .5; }
</style>
</head>
<body>

<div class="setup" id="setup">
  <pre class="setup-art">
 ######  ##     ##  #       ###  ###
 #       # #   # #  #        #    #
 ######  #  # #  #  #        #    #
 #       #   #   #  #        #    #
 ######  #       #  ######  ###  ###</pre>
  <div class="setup-sub">[ EVOLVING-MACHINE-LEARNING-INTELLIGENCE ]</div>
  <div class="setup-card">
    <div class="setup-title">// identify yourself to begin</div>
    <div class="setup-row">
      <input class="t-in" id="unameIn" placeholder="username" maxlength="20" autofocus>
      <button class="btn btn-g" onclick="startSession()">→</button>
    </div>
  </div>
  <div class="setup-foot">v3.5</div>
</div>

<div class="app" id="mainApp" style="display:none">
  <div class="topbar">
    <div>
      <div class="logo">[EMLII]</div>
      <div class="logo-ver">v3.5</div>
    </div>
    <div class="top-right">
      <div class="session-tag">session: <b id="unameShow">—</b></div>
      <div class="plugin-tag" id="pluginTag" onclick="toggleHelp('plugins')" title="Loaded tool plugins">
        <b id="pluginCount">0</b> plugins
      </div>
      <button class="ico-btn" onclick="clearChat()">clear</button>
      <button class="ico-btn" onclick="toggleHelp()">!help</button>
    </div>
  </div>

  <div class="chat" id="chat"></div>

  <!-- Teach / correct bar — reused for both new teaches and corrections -->
  <div class="teach-bar" id="teachBar">
    <div class="teach-prompt" id="teachPrompt">EMLII: I don't know how to respond. Teach me:</div>
    <div class="teach-row">
      <input class="t-in" id="teachIn" placeholder="Enter the response EMLII should give…">
      <button class="btn btn-g" onclick="submitTeach()">Save</button>
      <button class="btn btn-m" onclick="cancelTeach()">Skip</button>
    </div>
  </div>

  <div class="input-bar" style="position:relative">
    <div class="autocomplete" id="autocomplete"></div>
    <span class="prompt-sym" id="promptSym">▶</span>
    <input class="chat-in" id="chatIn"
      placeholder="type a message or !command…"
      autocomplete="off" autocorrect="off" spellcheck="false">
    <button class="send-btn" id="voiceBtn" onclick="openVoice()" title="Talk to EMLII">🎙 voice</button>
    <button class="send-btn" onclick="send()">send</button>
  </div>
</div>

<div class="voice-ov" id="voiceOv">
  <div class="voice-top">
    <div class="voice-title">[EMLII] VOICE <span id="voiceMode" style="color:var(--muted);font-size:10px;font-weight:400;letter-spacing:.05em"></span></div>
    <button class="ico-btn" onclick="closeVoice()">✕ close</button>
  </div>
  <div class="voice-stage" id="voiceStage" onclick="voiceTap()"><canvas id="voiceCanvas"></canvas></div>
  <div class="voice-status" id="voiceStatus">idle</div>
  <div class="voice-text" id="voiceText"></div>
  <div class="voice-ctl">
    <button class="ico-btn" id="voiceMute" onclick="voiceToggleMute()">mute mic</button>
    <button class="ico-btn" id="voiceSpk" onclick="voiceToggleSpeech()">voice replies: on</button>
  </div>
</div>

<div class="help-panel" id="helpPanel">
  <div class="hp-title">
    <span id="hpTitleText">Commands</span>
    <button class="hp-close" onclick="toggleHelp()">✕</button>
  </div>
  <table class="cmd-tbl">
    <tr><td>!help</td><td>All commands</td></tr>
    <tr><td>!version</td><td>Version info</td></tr>
    <tr><td>!date</td><td>Date & time</td></tr>
    <tr><td>!calculate &lt;expr&gt;</td><td>Math eval</td></tr>
    <tr><td>!hash &lt;text&gt;</td><td>MD5/SHA1/SHA256/SHA512</td></tr>
    <tr><td>!crack &lt;hash&gt;</td><td>Crack a hash</td></tr>
    <tr><td>!encrypt &lt;text&gt;</td><td>ROT-13 encrypt</td></tr>
    <tr><td>!decrypt &lt;text&gt;</td><td>ROT-13 decrypt</td></tr>
    <tr><td>!dirserver [path]</td><td>LAN file server</td></tr>
    <tr><td>!stopserver</td><td>Stop file server</td></tr>
    <tr><td>!memory</td><td>Learning stats</td></tr>
    <tr><td>!exit</td><td>End session</td></tr>
  </table>
  <div class="hp-sub">AI model (layer 4)</div>
  <div id="modelStatus" class="plugin-empty">loading…</div>
  <div class="hp-sub" id="pluginsHeading">Loaded plugins</div>
  <div id="pluginsList"><div class="plugin-empty">loading…</div></div>
  <div class="hp-note">
    EMLII predicts responses using<br>
    similarity search → pattern rules → AI.<br><br>
    Tap ✓ to accept a prediction.<br>
    Tap ✎ to correct it.
  </div>
</div>

<script>
let username   = '';
let pending    = null;   // { hash, original_text, mode: 'teach'|'correct' }
let msgId      = 0;
let pluginCmds = [];      // [['!cmd', 'help text'], ...] — loaded from /api/plugins
let acIndex    = -1;      // selected index in the open autocomplete list

const BUILTIN_CMDS = [
  ['!help',              'Show all commands'],
  ['!version',           'EMLII version info'],
  ['!date',               'Current date & time'],
  ['!calculate <expr>',  'Evaluate a math expression'],
  ['!hash <text>',       'MD5 / SHA1 / SHA256 / SHA512'],
  ['!crack <hash>',      'Dictionary + brute-force crack'],
  ['!encrypt <text>',    'ROT-13 Caesar cipher encrypt'],
  ['!decrypt <text>',    'ROT-13 Caesar cipher decrypt'],
  ['!dirserver [path]',  'Start LAN file server on port 8080'],
  ['!stopserver',        'Stop the file server'],
  ['!memory',            'Show how much EMLII has learned'],
  ['!exit',              'End session'],
];
function allCommands() { return BUILTIN_CMDS.concat(pluginCmds); }
function cmdWord(c) { return c.replace(/^!/, '').split(/[ <\[]/)[0].toLowerCase(); }


// ── Plugin system ─────────────────────────────────────────────────────────────
// Pulls the live list of tools/*.py plugins EMLII loaded at startup, so the
// UI always reflects what's actually registered — no hardcoding plugin names.
async function loadPlugins() {
  try {
    const r = await fetch('/api/plugins');
    const d = await r.json();
    pluginCmds = (d.plugins || []).map(([c, h]) => ['!' + c, h || '(no description)']);
    document.getElementById('pluginCount').textContent = pluginCmds.length;
    const list = document.getElementById('pluginsList');
    list.innerHTML = pluginCmds.length
      ? pluginCmds.map(([c, h]) =>
          `<div class="plugin-row"><span class="pname">${esc(c)}</span><span class="pdesc">${esc(h)}</span></div>`
        ).join('')
      : '<div class="plugin-empty">No plugins loaded in tools/</div>';

    const m = d.model || {};
    const modelEl = document.getElementById('modelStatus');
    modelEl.innerHTML = m.name
      ? `<div class="plugin-row"><span class="pname">${esc(m.name)}</span><span class="pdesc">${m.local ? 'local · models/' : 'pulled'}</span></div>`
      : '<div class="plugin-empty">Not configured</div>';
  } catch (e) {
    document.getElementById('pluginsList').innerHTML = '<div class="plugin-empty">Could not reach /api/plugins.</div>';
    document.getElementById('modelStatus').innerHTML = '<div class="plugin-empty">Could not reach /api/plugins.</div>';
  }
}
loadPlugins();


// ── Session setup ────────────────────────────────────────────────────────────
function startSession() {
  const val = document.getElementById('unameIn').value.trim();
  if (!val) return;
  username = val;
  document.getElementById('setup').style.display   = 'none';
  document.getElementById('mainApp').style.display = 'flex';
  document.getElementById('unameShow').textContent  = username;
  document.getElementById('promptSym').textContent  = username + ' ▶';
  document.getElementById('chatIn').focus();
  botMsg(`Hi!, ${username}, I'm EMLII.\nYou're personal chat assistant.`);
}
document.getElementById('unameIn').addEventListener('keydown', e => { if(e.key==='Enter') startSession(); });


// ── Help / plugins panel ───────────────────────────────────────────────────────
function toggleHelp(focus) {
  const panel = document.getElementById('helpPanel');
  panel.classList.toggle('open');
  if (panel.classList.contains('open') && focus === 'plugins') {
    document.getElementById('pluginsHeading').scrollIntoView({ block: 'start' });
  }
}

// ── Clear chat ─────────────────────────────────────────────────────────────────
function clearChat() {
  document.getElementById('chat').innerHTML = '';
  msgId = 0;
  sysMsg('Chat cleared — the on-disk history log is untouched.');
}


// ── Command autocomplete ────────────────────────────────────────────────────────
function updateAutocomplete() {
  const inp = document.getElementById('chatIn');
  const ac  = document.getElementById('autocomplete');
  const val = inp.value;
  if (!val.startsWith('!') || /\s/.test(val)) { closeAutocomplete(); return; }
  const q = val.slice(1).toLowerCase();
  const matches = allCommands().filter(([c]) => cmdWord(c).startsWith(q));
  if (!matches.length) { closeAutocomplete(); return; }
  acIndex = 0;
  ac.innerHTML = matches.map(([c, d], i) =>
    `<div class="ac-item${i === 0 ? ' sel' : ''}" data-cmd="${esc(cmdWord(c))}">
       <span class="ac-cmd">${esc(c)}</span><span class="ac-desc">${esc(d)}</span>
     </div>`).join('');
  ac.querySelectorAll('.ac-item').forEach(item =>
    item.addEventListener('click', () => selectAutocomplete(item.dataset.cmd)));
  ac.classList.add('open');
}
function closeAutocomplete() {
  acIndex = -1;
  document.getElementById('autocomplete').classList.remove('open');
}
function selectAutocomplete(cmd) {
  const inp = document.getElementById('chatIn');
  inp.value = '!' + cmd + ' ';
  closeAutocomplete();
  inp.focus();
}
function handleChatKeydown(e) {
  const ac   = document.getElementById('autocomplete');
  const open = ac.classList.contains('open');
  if (open && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
    e.preventDefault();
    const items = ac.querySelectorAll('.ac-item');
    if (!items.length) return;
    items[acIndex]?.classList.remove('sel');
    acIndex = e.key === 'ArrowDown' ? (acIndex + 1) % items.length : (acIndex - 1 + items.length) % items.length;
    items[acIndex].classList.add('sel');
    items[acIndex].scrollIntoView({ block: 'nearest' });
    return;
  }
  if (open && (e.key === 'Tab' || (e.key === 'Enter' && acIndex >= 0))) {
    e.preventDefault();
    const items = ac.querySelectorAll('.ac-item');
    if (items[acIndex]) selectAutocomplete(items[acIndex].dataset.cmd);
    return;
  }
  if (e.key === 'Escape') { closeAutocomplete(); return; }
  if (e.key === 'Enter') send();
}


// ── Send ──────────────────────────────────────────────────────────────────────
document.getElementById('chatIn').addEventListener('keydown', handleChatKeydown);
document.getElementById('chatIn').addEventListener('input', updateAutocomplete);
document.getElementById('teachIn').addEventListener('keydown', e => { if(e.key==='Enter') submitTeach(); });

async function send() {
  const inp = document.getElementById('chatIn');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  closeAutocomplete();
  userMsg(msg);

  const lower = msg.replace(/^!/, '').toLowerCase();
  if (['exit','leave','quit','q'].includes(lower)) {
    botMsg(`Goodbye, ${username}. Session ended.`);
    inp.disabled = true;
    document.querySelector('.send-btn').disabled = true;
    return;
  }

  // Did this message invoke a loaded tools/ plugin? Used to badge the reply.
  const isPlugin = msg.startsWith('!') &&
    pluginCmds.some(([c]) => cmdWord(c) === lower.split(/\s+/)[0]);

  const lid = loadingMsg();
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, username }),
    });
    removeMsg(lid);
    handle(await r.json(), isPlugin);
  } catch(e) {
    removeMsg(lid);
    sysMsg('Connection error.');
  }
}


// ── Response dispatcher ───────────────────────────────────────────────────────
// Routes each result type to the appropriate render function.
// `isPlugin` (set by send()) tags replies that came from a tools/ plugin so
// botMsg can show the purple "plugin" badge instead of a plain reply.
function handle(d, isPlugin) {
  if (!d || !d.type) return;
  switch(d.type) {
    case 'reply':       botMsg(d.text, isPlugin ? {plugin:true} : null); break;
    case 'learned':     botMsg(d.text);    break;
    case 'kill':
      botMsg(`Goodbye, ${username}. Session ended.`);
      document.getElementById('chatIn').disabled = true; break;
    case 'needs_teaching':
      // Store pending info and open teach bar
      pending = { hash: d.hash, original_text: d.prompt, mode: 'teach' };
      document.getElementById('teachPrompt').textContent =
        `EMLII: I don't know how to respond to "${d.prompt}". Teach me:`;
      document.getElementById('teachIn').value = '';
      document.getElementById('teachBar').classList.add('open');
      document.getElementById('teachIn').focus();
      break;
    case 'predicted':   predictedMsg(d);   break;
    case 'help':        helpMsg(d.commands); break;
    case 'hash_result': hashMsg(d);        break;
    case 'crack_start': doCrack(d.hash);   break;
    case 'crypt_result':cryptMsg(d);       break;
    case 'dirserver':
      botMsg(`File server started!\nDirectory: ${d.directory}\n\n${d.url}`, {link: d.url});
      break;
    default: if(d.text) botMsg(d.text);
  }
}


// ── Teach / correct bar ───────────────────────────────────────────────────────

async function submitTeach() {
  // submitTeach handles BOTH new teaches and corrections.
  // `pending.mode` tells us which API endpoint to call.
  const val = document.getElementById('teachIn').value.trim();
  if (!val || !pending) return;
  document.getElementById('teachIn').value = '';
  document.getElementById('teachBar').classList.remove('open');

  const endpoint = pending.mode === 'correct' ? '/api/correct' : '/api/teach';
  const r = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      hash:          pending.hash,
      response:      val,
      original_text: pending.original_text,
    }),
  });
  const d = await r.json();
  pending = null;
  if (d.ok) {
    sysMsg(pending === null
      ? (endpoint === '/api/correct' ? '✓ Correction saved' : '✓ Response saved')
      : '✓ Saved');
    botMsg(val);
  }
}

function cancelTeach() {
  document.getElementById('teachBar').classList.remove('open');
  pending = null;
  sysMsg('Skipped');
}


// ── Trigger correction from a predicted bubble ─────────────────────────────
// Called by the "✎ Correct" button rendered inside a predicted message.
// `msgHash` and `msgOriginal` are embedded in the button's onclick attribute.
function correctPrediction(msgHash, msgOriginal, currentText) {
  pending = { hash: msgHash, original_text: msgOriginal, mode: 'correct' };
  document.getElementById('teachPrompt').textContent = 'Correct EMLII\'s response:';
  // Pre-fill with the current (wrong) response so the user can edit rather than retype
  document.getElementById('teachIn').value = currentText;
  document.getElementById('teachBar').classList.add('open');
  document.getElementById('teachIn').focus();
}

// Called by the "✓ Yes" button — the prediction was already saved, so just confirm.
function acceptPrediction(msgId) {
  sysMsg('✓ Response accepted');
  // Remove the action row so the bubble looks like a normal reply
  const actRow = document.querySelector(`#${msgId} .pred-actions`);
  if (actRow) actRow.remove();
}


// ── Crack ──────────────────────────────────────────────────────────────────────
async function doCrack(hash) {
  const lid = loadingMsg('Cracking…');
  try {
    const r = await fetch('/api/crack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hash }),
    });
    removeMsg(lid);
    const d = await r.json();
    botMsg(d.found
      ? `Hash cracked!\n\nAlgorithm : ${d.algo.toUpperCase()}\nPlaintext : ${d.text}`
      : 'Could not crack hash.\n(Dictionary + 3-char brute force tried)');
  } catch(e) {
    removeMsg(lid); sysMsg('Crack failed: connection error');
  }
}


// ── DOM helpers ───────────────────────────────────────────────────────────────

function mkMsg(cls) {
  const id   = 'm' + (++msgId);
  const chat = document.getElementById('chat');
  const el   = document.createElement('div');
  el.className = 'msg ' + cls;
  el.id        = id;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return { id, el };
}
function scrollChat() {
  const c = document.getElementById('chat'); c.scrollTop = c.scrollHeight;
}
function nowTs() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function copyText(text, btn) {
  const done = ok => { const old = btn.textContent; btn.textContent = ok ? '✓' : '✗'; setTimeout(() => btn.textContent = old, 1000); };
  if (!navigator.clipboard) { done(false); return; }
  navigator.clipboard.writeText(text).then(() => done(true)).catch(() => done(false));
}

function userMsg(text) {
  const { el } = mkMsg('user');
  el.innerHTML = `<div class="who">${esc(username)} <span class="ts">${nowTs()}</span></div><div class="bubble">${esc(text)}</div>`;
  scrollChat();
}

function botMsg(text, extra) {
  const { el } = mkMsg('bot');
  const badge = extra && extra.plugin ? ` <span class="pred-badge plugin">plugin</span>` : '';
  let inner = `<div class="bubble">${esc(text)}`;
  if (extra && extra.link)
    inner += `\n<a class="server-link" href="${esc(extra.link)}" target="_blank">${esc(extra.link)}</a>`;
  inner += `<button class="copy-btn" title="Copy to clipboard">copy</button></div>`;
  el.innerHTML = `<div class="who">EMLII${badge} <span class="ts">${nowTs()}</span></div>${inner}`;
  el.querySelector('.copy-btn').addEventListener('click', e => copyText(text, e.currentTarget));
  scrollChat();
}

function sysMsg(text) {
  const { el } = mkMsg('sys');
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  scrollChat();
}

function loadingMsg(label='') {
  const { id, el } = mkMsg('bot');
  el.innerHTML = `<div class="who">EMLII</div>
    <div class="bubble">${label ? esc(label)+' ' : ''}<span class="dots"><span></span><span></span><span></span></span></div>`;
  scrollChat();
  return id;
}

function removeMsg(id) {
  const el = document.getElementById(id); if(el) el.remove();
}


// ── Predicted message (distinct style + action row) ───────────────────────────
// Predicted messages get:
//   - A coloured badge showing HOW EMLII predicted (similarity / pattern)
//   - A tinted bubble so they're visually distinct from exact-match replies
//   - "✓ Yes / ✎ Correct" buttons below the text
function predictedMsg(d) {
  const { id, el } = mkMsg('bot');

  // Badge labels and classes keyed by prediction source
  const badges = {
    similarity: ['~similar',    'similarity'],
    pattern:    ['~pattern',    'pattern'],
    ai:         ['~ai (SmolLM2)', 'api'],
  };
  const [label, cls] = badges[d.source] || ['~predicted', 'similarity'];
  const conf = Math.round((d.confidence || 0) * 100);

  // Embed hash and original_text in the button onclick so correctPrediction()
  // can retrieve them without a separate data store.
  // btoa() base64-encodes the text to avoid quote-escaping issues in the attribute.
  const hashSafe = esc(d.hash || '');
  const origSafe = esc(d.matched || '');
  const textB64  = btoa(unescape(encodeURIComponent(d.text)));

  el.innerHTML = `
    <div class="who">EMLII <span class="pred-badge ${cls}">${label} · ${conf}%</span> <span class="ts">${nowTs()}</span></div>
    <div class="bubble predicted" id="${id}-bubble">
      ${esc(d.text)}
      <div class="pred-actions">
        <span class="pred-hint">Was this right?</span>
        <button class="pred-btn ok"   onclick="acceptPrediction('${id}')">✓ Yes</button>
        <button class="pred-btn edit" onclick="correctPrediction('${hashSafe}','${origSafe}',decodeURIComponent(escape(atob('${textB64}'))))">✎ Correct</button>
      </div>
    </div>`;
  scrollChat();
}


// ── Specialised render functions ───────────────────────────────────────────────

function helpMsg(commands) {
  const { el } = mkMsg('bot');
  const rows = commands.map(([c,d]) =>
    `<tr><td>${esc(c)}</td><td>${esc(d)}</td></tr>`).join('');
  el.innerHTML = `<div class="who">EMLII</div>
    <div class="bubble"><div class="sub-label">Commands</div>
    <table class="cmd-tbl">${rows}</table></div>`;
  scrollChat();
}

function hashMsg(d) {
  const { el } = mkMsg('bot');
  const rows = Object.entries(d.results).map(([a,h]) =>
    `<tr><td>${a.toUpperCase()}</td><td>${h}</td></tr>`).join('');
  el.innerHTML = `<div class="who">EMLII</div>
    <div class="bubble"><div class="sub-label">Input: "${esc(d.input)}"</div>
    <table class="hash-tbl">${rows}</table></div>`;
  scrollChat();
}

function cryptMsg(d) {
  const { el } = mkMsg('bot');
  el.innerHTML = `<div class="who">EMLII</div>
    <div class="bubble">
      <div class="sub-label">${esc(d.mode)} · ${esc(d.method)}</div>
      <div class="crypt-box"><div class="crypt-label">Input</div><div>${esc(d.original)}</div></div>
      <div class="crypt-box" style="margin-top:7px"><div class="crypt-label">Output</div>
      <div class="crypt-val">${esc(d.output)}</div></div>
    </div>`;
  scrollChat();
}


// ── Voice mode ────────────────────────────────────────────────────────────────
// Browser-side only: SpeechRecognition (mic → text), speechSynthesis (text → voice).
// Transcripts go through the same /api/chat pipeline as typed messages.
// Needs Chrome / Edge / Safari, and a secure origin (localhost or https) for the mic.
const SR_CLASS = window.SpeechRecognition || window.webkitSpeechRecognition;
const VO = {
  open: false, state: 'idle', muted: false, speak: true,
  rec: null, teach: null, level: 0, target: 0, t: 0, raf: 0,
  actx: null, analyser: null, stream: null, buf: null, pulse: 0,
  stt: 'browser', caps: null, proc: null, cap: null, sr: 48000, audio: null, warned: false,
};
const VO_THRESH = 0.018;   // mic RMS that counts as speech for the offline recorder

function voiceSetState(st, label) {
  VO.state = st;
  document.getElementById('voiceStatus').textContent = label || st;
}
function voiceSetText(t) { document.getElementById('voiceText').textContent = t || ''; }

async function openVoice() {
  if (!username) return;
  // Prime speech synthesis inside the click gesture so browsers allow audio later.
  if (window.speechSynthesis) speechSynthesis.speak(new SpeechSynthesisUtterance(''));
  VO.open = true;
  document.getElementById('voiceOv').classList.add('open');
  document.getElementById('voiceBtn').classList.add('mic-on');
  voiceSetText('');
  voiceStartLoop();
  await voiceInitMic();
  try { VO.caps = await (await fetch('/api/voice_caps')).json(); } catch (e) { VO.caps = { stt: false, tts: null }; }
  VO.stt = (VO.caps.stt && VO.analyser) ? 'local' : 'browser';
  voiceShowMode();
  if (VO.stt !== 'local' && !VO.warned) {
    VO.warned = true;
    sysMsg('Offline voice input unavailable: ' + (VO.caps.stt_status || 'unknown reason') + '. Using the browser online speech service instead. Check the EMLII console for details.');
  }
  if (VO.stt !== 'local' && !SR_CLASS) {
    voiceSetState('idle', 'speech input not supported in this browser');
    voiceSetText('Try Chrome, Edge or Safari. Replies can still be spoken.');
    return;
  }
  // Greet out loud (confirms audio works), then start listening when it finishes.
  voiceSay(`Hi ${username}, I'm listening.`);
}

function closeVoice() {
  VO.open = false;
  VO.teach = null;
  voiceStopSpeech();
  try { VO.rec && VO.rec.abort(); } catch(e) {}
  VO.rec = null;
  if (VO.stream) { VO.stream.getTracks().forEach(t => t.stop()); VO.stream = null; }
  VO.cap = null;
  if (VO.proc) { VO.proc.onaudioprocess = null; VO.proc = null; }
  if (VO.actx) { VO.actx.close().catch(()=>{}); VO.actx = null; VO.analyser = null; }
  cancelAnimationFrame(VO.raf);
  document.getElementById('voiceOv').classList.remove('open');
  document.getElementById('voiceBtn').classList.remove('mic-on');
  voiceSetState('idle');
  document.getElementById('chatIn').focus();
}

// Mic level meter — drives the orb while you talk.
async function voiceInitMic() {
  if (VO.analyser || !navigator.mediaDevices) return;
  try {
    VO.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    VO.actx = new (window.AudioContext || window.webkitAudioContext)();
    const src = VO.actx.createMediaStreamSource(VO.stream);
    VO.analyser = VO.actx.createAnalyser();
    VO.analyser.fftSize = 512;
    src.connect(VO.analyser);
    VO.buf = new Uint8Array(VO.analyser.fftSize);
    // Recorder tap for offline speech input (ScriptProcessor: simple, works in every browser).
    VO.proc = VO.actx.createScriptProcessor(4096, 1, 1);
    src.connect(VO.proc);
    VO.proc.connect(VO.actx.destination);   // output stays silent; needed so the node runs
    VO.proc.onaudioprocess = voiceOnAudio;
  } catch (e) {
    // No level meter — the orb falls back to a gentle idle motion while listening.
    VO.analyser = null;
  }
}
function voiceMicLevel() {
  if (!VO.analyser) return 0;
  VO.analyser.getByteTimeDomainData(VO.buf);
  let sum = 0;
  for (let i = 0; i < VO.buf.length; i++) { const v = (VO.buf[i] - 128) / 128; sum += v * v; }
  return Math.min(1, Math.sqrt(sum / VO.buf.length) * 5);
}

// ── Listening ──
function voiceListen() {
  if (!VO.open) return;
  if (VO.muted) { voiceSetState('idle', 'mic muted'); return; }
  if (VO.stt === 'local') voiceListenLocal(); else voiceListenBrowser();
}

// Offline path: record one utterance (energy-based start/stop), send it to /api/stt (Vosk).
function voiceListenLocal() {
  voiceStopSpeech();
  voiceSetState('listening', VO.teach ? 'listening — tell me the response' : 'listening · offline');
  VO.cap = { pre: [], chunks: [], talking: false, silentMs: 0, ms: 0 };
}

function voiceOnAudio(e) {
  const c = VO.cap;
  if (!c || VO.state !== 'listening') return;
  const data = new Float32Array(e.inputBuffer.getChannelData(0));   // copy — the buffer is reused
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
  const rms = Math.sqrt(sum / data.length);
  const ms = data.length / e.inputBuffer.sampleRate * 1000;
  VO.sr = e.inputBuffer.sampleRate;

  if (!c.talking) {                                   // waiting for speech; keep ~0.4 s pre-roll
    c.pre.push(data);
    if (c.pre.length > 5) c.pre.shift();
    if (rms > VO_THRESH) { c.talking = true; c.chunks = c.pre; c.pre = []; voiceSetText('…'); }
    return;
  }
  c.chunks.push(data);
  c.ms += ms;
  c.silentMs = rms > VO_THRESH ? 0 : c.silentMs + ms;
  if (c.silentMs > 900 || c.ms > 15000) voiceFinishLocal(c);
}

function voicePCM16k(chunks, sr) {
  let n = 0; chunks.forEach(c => n += c.length);
  const all = new Float32Array(n); let o = 0;
  chunks.forEach(c => { all.set(c, o); o += c.length; });
  const ratio = sr / 16000, out = new Int16Array(Math.floor(n / ratio));
  for (let i = 0; i < out.length; i++) {              // average each window = cheap low-pass + downsample
    const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
    let acc = 0; for (let j = a; j < b; j++) acc += all[j];
    out[i] = Math.max(-1, Math.min(1, acc / (b - a))) * 32767;
  }
  return out;
}

async function voiceFinishLocal(c) {
  VO.cap = null;                                      // stop collecting before any await
  if (c.ms - c.silentMs < 250) { voiceListenLocal(); return; }   // a click or cough, not speech
  voiceSetState('thinking', 'transcribing');
  try {
    const r = await fetch('/api/stt', {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
      body: voicePCM16k(c.chunks, VO.sr).buffer,
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'request failed');
    const text = (d.text || '').trim();
    if (text) { voiceHandle(text); return; }
  } catch (e) {
    voiceSetText('Speech recognition failed: ' + e.message);
  }
  if (VO.open) setTimeout(voiceListen, 400);
}

function voiceListenBrowser() {
  if (!VO.open || !SR_CLASS || VO.muted) {
    if (VO.muted) voiceSetState('idle', 'mic muted');
    return;
  }
  voiceStopSpeech();
  const rec = new SR_CLASS();
  VO.rec = rec;
  rec.lang = navigator.language || 'en-US';
  rec.interimResults = true;
  rec.continuous = false;
  let finalText = '';
  voiceSetState('listening', VO.teach ? 'listening — tell me the response' : 'listening');

  rec.onresult = e => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const tr = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += tr; else interim += tr;
    }
    voiceSetText(finalText + interim);
  };
  rec.onerror = e => {
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      VO.muted = true;
      voiceSetState('idle', 'mic blocked');
      voiceSetText('Allow microphone access (needs localhost or https), then tap the orb.');
    } else if (e.error === 'audio-capture') {
      VO.muted = true;
      voiceSetState('idle', 'no microphone found');
      voiceSetText('Plug in or enable a microphone, then tap the orb.');
    } else if (e.error === 'network') {
      voiceSetState('idle', 'speech service unreachable');
      voiceSetText('Chrome sends speech to an online service. Check your internet connection.');
    }
  };
  rec.onend = () => {
    if (VO.rec !== rec) return;
    const text = finalText.trim();
    if (text) { VO.rec = null; voiceHandle(text); }
    else if (VO.open && VO.state === 'listening') setTimeout(voiceListen, 200);  // silence timeout → restart
  };
  try { rec.start(); } catch (e) {}
}

// ── Handling a transcript ──
async function voiceHandle(text) {
  voiceSetState('thinking');
  voiceSetText(text);
  userMsg(text);

  if (/^(stop|goodbye|close|exit voice|end voice)\.?$/i.test(text)) { closeVoice(); return; }

  // Previous turn was "I don't know that" → this utterance is the answer to learn.
  if (VO.teach) {
    const t = VO.teach; VO.teach = null;
    try {
      const r = await fetch('/api/teach', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hash: t.hash, response: text, original_text: t.original_text }),
      });
      const d = await r.json();
      if (d.ok) { sysMsg('✓ Response saved'); botMsg(text); voiceSay('Got it. I will remember that.'); return; }
    } catch (e) {}
    voiceSay('Sorry, I could not save that.');
    return;
  }

  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, username }),
    });
    const d = await r.json();
    voiceRespond(d);
  } catch (e) {
    sysMsg('Connection error.');
    voiceSay('I lost the connection.');
  }
}

function voiceRespond(d) {
  if (!d || !d.type) { voiceSay('I did not get a response.'); return; }
  switch (d.type) {
    case 'needs_teaching':
      VO.teach = { hash: d.hash, original_text: d.prompt };
      sysMsg(`Voice teach: waiting for a response to "${d.prompt}"`);
      voiceSay('I do not know how to respond to that yet. What should I say?');
      return;
    case 'kill':
      handle(d); closeVoice(); return;
    case 'reply': case 'learned': case 'predicted':
      handle(d); voiceSay(d.text); return;
    case 'crack_start':
      handle(d); voiceSay('Cracking that hash. I will put the result in the chat.'); return;
    case 'help': case 'hash_result': case 'crypt_result': case 'dirserver':
      handle(d); voiceSay('Done. The result is in the chat.'); return;
    default:
      handle(d); voiceSay(d.text || 'Done.');
  }
}

// ── Speaking ──
// SPEAK_TOKEN invalidates callbacks from cancelled utterances, so a cancel()
// can never trigger a stale "finished speaking" that restarts the mic mid-reply.
let SPEAK_TOKEN = 0, VOICE_PICK = null, SPEAK_KEEP = null, SPEAK_DOG = null;

function voicePick() {
  const vs = window.speechSynthesis ? speechSynthesis.getVoices() : [];
  if (!vs.length) return null;
  const local = vs.filter(v => v.localService);       // on-device voices work offline
  const pool  = local.length ? local : vs;
  const base = (navigator.language || 'en-US').toLowerCase().split('-')[0];
  const same = pool.filter(v => v.lang.toLowerCase().startsWith(base));
  const nice = /natural|samantha|aria|jenny|zira|daniel|karen|google/i;
  return same.find(v => nice.test(v.name)) || same.find(v => v.default) || same[0] || pool[0];
}
// 'local' = on-device browser voice, 'server' = espeak/pyttsx3 via /api/tts, 'browser' = may be online.
function voiceTtsMode() {
  const vs = window.speechSynthesis ? speechSynthesis.getVoices() : [];
  if (vs.some(v => v.localService)) return 'local';
  if (VO.caps && VO.caps.tts) return 'server';
  return 'browser';
}
function voiceShowMode() {
  const tts = voiceTtsMode();
  document.getElementById('voiceMode').textContent =
    ' · in: ' + (VO.stt === 'local' ? 'offline' : 'online') + ' · out: ' + (tts === 'browser' ? 'online' : 'offline');
}
if (window.speechSynthesis) {
  VOICE_PICK = voicePick();
  speechSynthesis.onvoiceschanged = () => { VOICE_PICK = voicePick(); };
}

function voiceStopSpeech() {
  SPEAK_TOKEN++;
  clearInterval(SPEAK_KEEP); clearTimeout(SPEAK_DOG);
  if (VO.audio) { VO.audio.onended = VO.audio.onerror = null; VO.audio.pause(); VO.audio = null; }
  if (window.speechSynthesis) speechSynthesis.cancel();
}

function voiceSay(text) {
  voiceSetText(text);
  if (!VO.open) return;
  if (!VO.speak || (!window.speechSynthesis && !(VO.caps && VO.caps.tts))) { voiceAfterSpeech(); return; }
  voiceStopSpeech();
  const token = SPEAK_TOKEN;
  voiceSetState('speaking');

  const clean = String(text).replace(/https?:\/\/\S+/g, 'link').replace(/[`*_#>|]/g, ' ').replace(/\s+/g, ' ').trim();
  const parts = (clean.match(/[^.!?\n]+[.!?]*/g) || [clean]).map(x => x.trim()).filter(Boolean);
  if (!parts.length) { voiceAfterSpeech(); return; }

  const done = () => {
    if (token !== SPEAK_TOKEN) return;          // cancelled or superseded
    clearInterval(SPEAK_KEEP); clearTimeout(SPEAK_DOG);
    voiceAfterSpeech();
  };
  if (voiceTtsMode() === 'server') { voiceSayServer(parts.join(' '), token, done); return; }
  parts.forEach((part, i) => {
    const u = new SpeechSynthesisUtterance(part);
    const v = VOICE_PICK || voicePick();
    if (v) { u.voice = v; u.lang = v.lang; }
    u.rate = 1.02;
    u.onstart    = () => { if (i === 0) clearTimeout(SPEAK_DOG); };
    u.onboundary = () => { VO.pulse = 1; };
    if (i === parts.length - 1) { u.onend = done; u.onerror = done; }
    speechSynthesis.speak(u);
  });

  // Chrome silently stops long speech after ~15s; a pause/resume nudge keeps it going.
  SPEAK_KEEP = setInterval(() => {
    if (speechSynthesis.speaking) { speechSynthesis.pause(); speechSynthesis.resume(); }
  }, 10000);
  // If audio never starts (blocked / no voices), don't hang — go back to listening.
  SPEAK_DOG = setTimeout(() => {
    if (token === SPEAK_TOKEN && !speechSynthesis.speaking) {
      voiceStopSpeech();
      voiceSetState('idle', 'could not play audio — check volume / browser sound permission');
      setTimeout(() => { if (VO.open) voiceListen(); }, 1500);
    }
  }, 4000);
}
// Server-rendered speech (espeak-ng / pyttsx3) — used when the browser has no on-device voice.
async function voiceSayServer(text, token, done) {
  try {
    const r = await fetch('/api/tts', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) throw new Error('tts unavailable');
    const blob = await r.blob();
    if (token !== SPEAK_TOKEN) return;                 // interrupted while rendering
    const url = URL.createObjectURL(blob);
    const a = new Audio(url);
    VO.audio = a;
    a.onended = a.onerror = () => { URL.revokeObjectURL(url); VO.audio = null; done(); };
    await a.play();
  } catch (e) {
    if (token === SPEAK_TOKEN) done();
  }
}
function voiceAfterSpeech() {
  if (VO.open) voiceListen();
}

// ── Controls ──
function voiceTap() {
  if (VO.state === 'speaking') { voiceStopSpeech(); voiceListen(); return; }  // interrupt
  if (VO.muted) voiceToggleMute();
  else if (VO.state === 'idle') voiceListen();
}
function voiceToggleMute() {
  VO.muted = !VO.muted;
  document.getElementById('voiceMute').classList.toggle('on', VO.muted);
  document.getElementById('voiceMute').textContent = VO.muted ? 'unmute mic' : 'mute mic';
  if (VO.muted) { VO.cap = null; try { VO.rec && VO.rec.abort(); } catch(e) {} VO.rec = null; voiceSetState('idle', 'mic muted'); }
  else voiceListen();
}
function voiceToggleSpeech() {
  VO.speak = !VO.speak;
  if (!VO.speak && VO.state === 'speaking') { voiceStopSpeech(); voiceListen(); }
  document.getElementById('voiceSpk').textContent = 'voice replies: ' + (VO.speak ? 'on' : 'off');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape' && VO.open) closeVoice(); });

// ── Orb: purple plasma core — dot-mesh sphere, orbital rings, radial level ring ──
const ORB_N = 2600;
const ORB_PTS = (() => {
  const pts = new Float32Array(ORB_N * 3), g = Math.PI * (3 - Math.sqrt(5));  // Fibonacci sphere
  for (let i = 0; i < ORB_N; i++) {
    const y = 1 - (i / (ORB_N - 1)) * 2, r = Math.sqrt(1 - y * y), th = g * i;
    pts[i*3] = Math.cos(th) * r; pts[i*3+1] = y; pts[i*3+2] = Math.sin(th) * r;
  }
  return pts;
})();

// Palette — the EMLII accent (#b84bff) with a deep violet for the far side and lavender / hot pink for peaks.
const ORB_C = { deep: [58, 10, 107], core: [184, 75, 255], hi: [217, 163, 255], hot: [255, 110, 235] };
const ORB_DB = 8, ORB_HB = 4, ORB_NB = ORB_DB * ORB_HB;          // depth × heat colour buckets
const ORB_BX = Array.from({ length: ORB_NB }, () => new Float32Array(ORB_N * 3));
const ORB_BC = new Int32Array(ORB_NB);
const ORB_REDUCE = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
let ORB_HEAT = 0, ORB_SPIN = 1, ORB_ANG = 0, ORB_SWEEP = 0;
const ORB_RINGS = [                                              // radius, tilt, yaw speed, yaw offset, satellite speed
  { r: 1.62, tilt: 1.15, ys:  0.55, y0: 0.0, sat:  1.6 },
  { r: 1.42, tilt: 0.35, ys: -0.40, y0: 1.7, sat: -1.1 },
  { r: 1.82, tilt: -0.85, ys: 0.28, y0: 3.1, sat:  0.8 },
];

function orbMix(a, b, k) { return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k]; }
function orbRGBA(c, a) { return `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a.toFixed(3)})`; }

function voiceStartLoop() {
  const cv = document.getElementById('voiceCanvas');
  const ctx = cv.getContext('2d');
  const frame = () => {
    if (!VO.open) return;
    VO.raf = requestAnimationFrame(frame);
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, w, h);

    // ── state → motion ──
    const calm = ORB_REDUCE ? 0.25 : 1;
    const st = VO.state;
    VO.t += 0.016 * calm;
    VO.pulse *= 0.9;
    const t = VO.t;
    if (st === 'listening')      VO.target = Math.max(0.06, voiceMicLevel());
    else if (st === 'speaking')  VO.target = 0.22 + 0.18 * Math.sin(t * 7) * Math.sin(t * 2.3) + 0.45 * VO.pulse;
    else if (st === 'thinking')  VO.target = 0.14 + 0.08 * Math.sin(t * 4);
    else                         VO.target = 0.04;
    VO.level += (VO.target - VO.level) * 0.14;
    const L = VO.level;
    const heatT = st === 'speaking' ? 1 : st === 'thinking' ? 0.55 : st === 'listening' ? 0.2 + L * 0.6 : 0;
    ORB_HEAT  += (heatT - ORB_HEAT) * 0.06;
    ORB_SPIN  += ((st === 'thinking' ? 3.2 : 1) - ORB_SPIN) * 0.05;
    ORB_SWEEP += ((st === 'thinking' ? 1 : 0) - ORB_SWEEP) * 0.08;
    ORB_ANG   += 0.016 * calm * ORB_SPIN;

    const R = Math.min(w, h) * 0.25, cx = w / 2, cy = h / 2;
    const tint = orbMix(ORB_C.core, ORB_C.hot, ORB_HEAT * 0.45);

    // ── halo + core glow ──
    let g = ctx.createRadialGradient(cx, cy, R * 0.2, cx, cy, R * 2.7);
    g.addColorStop(0,   orbRGBA(tint, 0.20 + 0.30 * L));
    g.addColorStop(0.4, orbRGBA(ORB_C.deep, 0.16 + 0.10 * L));
    g.addColorStop(1,   orbRGBA(ORB_C.deep, 0));
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = 'lighter';
    g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.05);
    g.addColorStop(0, orbRGBA(ORB_C.hi, 0.10 + 0.32 * L + 0.08 * ORB_HEAT));
    g.addColorStop(1, orbRGBA(ORB_C.core, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R * 1.05, 0, 6.2832); ctx.fill();

    // ── radial level ring: bars around the orb that follow the voice ──
    const BARS = 96, rr = R * 1.34;
    ctx.lineCap = 'round'; ctx.lineWidth = 2;
    ctx.strokeStyle = orbRGBA(tint, 0.55 + 0.3 * ORB_HEAT);
    ctx.beginPath();
    for (let i = 0; i < BARS; i++) {
      const a = (i / BARS) * 6.2832 + ORB_ANG * 0.08;
      const wob = 0.5 + 0.5 * Math.sin(i * 0.9 + t * 3.1) * Math.sin(i * 0.37 - t * 2.1);
      const len = 2 + (L * 1.5 + VO.pulse * 0.35) * wob * R * 0.42;
      const c = Math.cos(a), s = Math.sin(a);
      ctx.moveTo(cx + c * rr, cy + s * rr);
      ctx.lineTo(cx + c * (rr + len), cy + s * (rr + len));
    }
    ctx.stroke();

    // ── thinking sweep: a comet arc circling the level ring ──
    if (ORB_SWEEP > 0.02) {
      const base = ORB_ANG * 2.2, r2 = R * 1.22;
      ctx.lineWidth = 2.5;
      for (let k = 0; k < 14; k++) {
        ctx.strokeStyle = orbRGBA(ORB_C.hi, ORB_SWEEP * (1 - k / 14) * 0.9);
        ctx.beginPath(); ctx.arc(cx, cy, r2, base - (k + 1) * 0.07, base - k * 0.07); ctx.stroke();
      }
    }

    // ── orbital rings with a satellite each ──
    ctx.lineWidth = 1;
    for (const rg of ORB_RINGS) {
      const yaw = rg.y0 + ORB_ANG * rg.ys;
      const ct = Math.cos(rg.tilt), st2 = Math.sin(rg.tilt), cy2 = Math.cos(yaw), sy2 = Math.sin(yaw);
      const pt = th => {
        const x = Math.cos(th) * rg.r, y = Math.sin(th) * rg.r;
        const y1 = y * ct, z1 = y * st2;
        const x2 = x * cy2 + z1 * sy2, z2 = -x * sy2 + z1 * cy2;
        const p = 1 / (1 - z2 * 0.13);
        return [cx + x2 * R * p, cy + y1 * R * p, z2, p];
      };
      ctx.strokeStyle = orbRGBA(ORB_C.core, 0.22 + 0.18 * ORB_HEAT);
      ctx.beginPath();
      for (let i = 0; i <= 96; i++) { const q = pt((i / 96) * 6.2832); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); }
      ctx.stroke();
      const q = pt(t * rg.sat * 1.3 + rg.y0);
      const depth = Math.max(0.35, Math.min(1, 0.7 + q[2] * 0.25));
      ctx.fillStyle = orbRGBA(ORB_C.hi, 0.9 * depth);
      ctx.beginPath(); ctx.arc(q[0], q[1], 2.2 * q[3], 0, 6.2832); ctx.fill();
      ctx.fillStyle = orbRGBA(ORB_C.core, 0.25 * depth);
      ctx.beginPath(); ctx.arc(q[0], q[1], 7 * q[3], 0, 6.2832); ctx.fill();
    }

    // ── the dot-mesh sphere: colour by depth (violet → purple) and by displacement (→ lavender / pink) ──
    const ay = ORB_ANG * 0.35, ax = 0.45 + Math.sin(t * 0.2) * 0.15;
    const cyw = Math.cos(ay), syw = Math.sin(ay), cxw = Math.cos(ax), sxw = Math.sin(ax);
    ORB_BC.fill(0);
    for (let i = 0; i < ORB_N; i++) {
      const x = ORB_PTS[i*3], y = ORB_PTS[i*3+1], z = ORB_PTS[i*3+2];
      const d = L * (0.55 * Math.sin(x * 3.1 + t * 1.9) * Math.cos(y * 2.7 - t * 1.4)
                   + 0.40 * Math.sin(z * 4.2 + t * 2.4)
                   + 0.30 * Math.sin((x + y) * 5 - t * 3));
      const r = 1 + d + 0.015 * Math.sin(t * 1.5);
      const px = x * r, py = y * r, pz = z * r;
      const x1 = px * cyw + pz * syw, z1 = -px * syw + pz * cyw;       // spin
      const y1 = py * cxw - z1 * sxw, z2 = py * sxw + z1 * cxw;        // tilt
      const persp = 1 / (1 - z2 * 0.22);
      const tz = Math.max(0, Math.min(0.999, (z2 + 1.2) / 2.4));
      const hz = Math.max(0, Math.min(0.999, Math.abs(d) * 3 + ORB_HEAT * 0.7));
      const b = ((tz * ORB_DB) | 0) * ORB_HB + ((hz * ORB_HB) | 0);
      const arr = ORB_BX[b], n = ORB_BC[b]++ * 3;
      arr[n] = cx + x1 * R * persp; arr[n+1] = cy + y1 * R * persp; arr[n+2] = (0.8 + 0.9 * tz) * persp;
    }
    for (let db = 0; db < ORB_DB; db++) {
      const tt = (db + 0.5) / ORB_DB;
      for (let hb = 0; hb < ORB_HB; hb++) {
        const b = db * ORB_HB + hb, cnt = ORB_BC[b];
        if (!cnt) continue;
        const hh = (hb + 0.5) / ORB_HB;
        let c = orbMix(ORB_C.deep, ORB_C.core, Math.min(1, tt * 1.15));
        c = orbMix(c, ORB_C.hi, hh * 0.75);
        c = orbMix(c, ORB_C.hot, ORB_HEAT * 0.35 * hh);
        ctx.fillStyle = orbRGBA(c, 0.10 + 0.80 * tt);
        const arr = ORB_BX[b];
        for (let j = 0; j < cnt; j++) { const o = j * 3; ctx.fillRect(arr[o], arr[o+1], arr[o+2], arr[o+2]); }
      }
    }
    ctx.globalCompositeOperation = 'source-over';
  };
  cancelAnimationFrame(VO.raf);
  frame();
}

// ── XSS protection ────────────────────────────────────────────────────────────
// All user-provided text goes through esc() before being placed in innerHTML.
// & must be replaced first to avoid double-escaping (&amp; → &amp;amp;).
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


# ── run_web() ─────────────────────────────────────────────────────────────────
def run_web(port=5000):
    if not FLASK_AVAILABLE:
        print(f"\n{RED} Flask not installed.{RESET}")
        print(f" Run: {CYAN}pip install flask --break-system-packages{RESET}\n")
        sys.exit(1)
    host = os.environ.get("EMLII_WEB_HOST") or "0.0.0.0"     # 127.0.0.1 = only this device can connect
    lan = get_lan_ip()
    net = (f"{MUTED} Network →{RESET} {CYAN}http://{lan}:{port}{RESET}" if host == "0.0.0.0"
           else f"{MUTED} Network →{RESET} off (EMLII_WEB_HOST={host})")
    print(f"""
{GREEN}{BOLD} EMLII v{EMLII.VERSION} — Web Mode{RESET}
 {"─"*40}
{MUTED} Local   →{RESET} {CYAN}http://127.0.0.1:{port}{RESET}
{net}
 {"─"*40}
""")
    app.run(host=host, port=port, debug=False)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    load_tools()
    load_models()
    load_voice()
    if '--web' in sys.argv:
        try: port = int(sys.argv[sys.argv.index('--web') + 1])
        except (IndexError, ValueError): port = 5000
        run_web(port)
    elif '--terminal' in sys.argv or '--term' in sys.argv:
        run_terminal()
    else:
        print(f"""
{GREEN}{BOLD} EMLII v{EMLII.VERSION}{RESET}
 {"─"*32}
  {CYAN}1.{RESET} Web mode    {MUTED}(browser, shareable on LAN){RESET}
  {CYAN}2.{RESET} Terminal    {MUTED}(coloured terminal){RESET}
 {"─"*32}""")
        choice = input(f"  {GREEN}Choose [1/2]: {RESET}").strip()
        if choice == '1':
            p = input(f"  {MUTED}Port [default 5000]: {RESET}").strip()
            run_web(int(p) if p.isdigit() else 5000)
        else:
            run_terminal()