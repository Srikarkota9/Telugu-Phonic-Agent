"""
Telugu Phonic Agent — Streamlit Interface
Run with: streamlit run app.py
API key is stored in .streamlit/secrets.toml
"""

import os
import json
import tempfile
import difflib
import re
import random
import streamlit as st
from elevenlabs.client import ElevenLabs
import whisper

WORD_BANK_PATH = "telugu_word_bank.json"
OUTPUT_DIR = "generated_audio"
MODEL_ID = "eleven_multilingual_v2"

# Default ElevenLabs voice for users who haven't cloned yet
DEFAULT_VOICE_ID = "PSGmTr1P7xsA5H9W7obv"  # Balachander — South Indian accent
DEFAULT_VOICE_LABEL = "Default Voice (Balachander)"


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_word_bank():
    with open(WORD_BANK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_client():
    api_key = st.secrets.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return None
    return ElevenLabs(api_key=api_key)


def get_active_voice_id():
    """Return the user's cloned voice_id if available, otherwise the default."""
    return st.session_state.get("voice_id", DEFAULT_VOICE_ID)


def get_user_audio_dir():
    """
    Return the audio output directory for the current user.
    - Cloned users: generated_audio/<voice_id>/
    - Default users: generated_audio/default/
    """
    voice_id = st.session_state.get("voice_id", None)
    if voice_id:
        return os.path.join(OUTPUT_DIR, voice_id)
    else:
        return os.path.join(OUTPUT_DIR, "default")


def clone_voice(client, audio_bytes):
    """Clone voice from uploaded/recorded audio bytes. Returns voice_id."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    try:
        with open(tmp.name, "rb") as f:
            voice = client.voices.ivc.create(
                name="Telugu Teacher",
                files=[f],
                description="Cloned voice for Telugu pronunciation teaching",
            )
        return voice.voice_id
    finally:
        os.remove(tmp.name)


def generate_single_audio(client, voice_id, text, pronunciation_hint=""):
    """Generate audio for a single text and return bytes.
    If pronunciation_hint is provided, use it instead of Telugu script
    since romanized text produces better pronunciation.
    """
    from elevenlabs import VoiceSettings

    # Use romanized text if available — it pronounces better
    tts_text = pronunciation_hint if pronunciation_hint else text

    audio_iter = client.text_to_speech.convert(
        voice_id=voice_id,
        text=tts_text,
        model_id=MODEL_ID,
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.85,           # High = consistent, clear pronunciation
            similarity_boost=0.90,    # High = stays close to voice
            style=0.0,                # Zero = cleanest, most neutral delivery
            use_speaker_boost=True,   # Enhances voice clarity
        ),
    )
    audio_bytes = b"".join(audio_iter)
    return audio_bytes


def save_audio(audio_bytes, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)


def get_word_filepath(user_dir, cat_key, romanized):
    """Get the filepath for a word's audio file."""
    safe_name = romanized.replace(" ", "_").replace("/", "_").lower()
    return os.path.join(user_dir, cat_key, f"{safe_name}.mp3")


def get_rhyme_dir(user_dir, rhyme_title):
    """Get the directory for a rhyme's audio files."""
    safe_title = rhyme_title.replace(" ", "_").lower()
    return os.path.join(user_dir, "nursery_rhymes", safe_title)


@st.cache_resource
def load_whisper_model():
    """Load Whisper small model (much better for Telugu than base)."""
    model_path = os.path.expanduser("~/.cache/whisper/small.pt")
    if os.path.exists(model_path):
        return whisper.load_model(model_path)
    return whisper.load_model("small")


def transcribe_audio(audio_bytes, expected_telugu="", expected_roman=""):
    """
    Transcribe user's recorded audio using Whisper.
    Uses expected text as a prompt hint to improve accuracy.
    Returns both Telugu script and romanized transcription.
    """
    model = load_whisper_model()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    try:
        # First pass: transcribe as Telugu script with prompt hint
        prompt_hint = expected_telugu if expected_telugu else ""
        result_telugu = model.transcribe(
            tmp.name,
            language="te",
            initial_prompt=prompt_hint,
            temperature=0.0,
            no_speech_threshold=0.3,
            condition_on_previous_text=False,
        )
        telugu_text = result_telugu["text"].strip()

        # Second pass: transcribe as romanized (English mode picks up phonetics)
        roman_hint = expected_roman if expected_roman else ""
        result_roman = model.transcribe(
            tmp.name,
            language="en",
            initial_prompt=f"The speaker is saying the Telugu word: {roman_hint}",
            temperature=0.0,
            no_speech_threshold=0.3,
            condition_on_previous_text=False,
        )
        roman_text = result_roman["text"].strip()

        return telugu_text, roman_text
    finally:
        os.remove(tmp.name)


def transliterate_telugu_to_roman(text):
    """Convert Telugu script to a rough romanized form for comparison."""
    telugu_map = {
        "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ee", "ఉ": "u", "ఊ": "oo",
        "ఎ": "e", "ఏ": "e", "ఐ": "ai", "ఒ": "o", "ఓ": "o", "ఔ": "au",
        "క": "ka", "ఖ": "kha", "గ": "ga", "ఘ": "gha", "ఙ": "nga",
        "చ": "cha", "ఛ": "chha", "జ": "ja", "ఝ": "jha", "ఞ": "nya",
        "ట": "ta", "ఠ": "tha", "డ": "da", "ఢ": "dha", "ణ": "na",
        "త": "ta", "థ": "tha", "ద": "da", "ధ": "dha", "న": "na",
        "ప": "pa", "ఫ": "pha", "బ": "ba", "భ": "bha", "మ": "ma",
        "య": "ya", "ర": "ra", "ల": "la", "వ": "va", "శ": "sha",
        "ష": "sha", "స": "sa", "హ": "ha", "ళ": "la", "క్ష": "ksha",
        "ఱ": "rra",
        # Vowel marks (matras)
        "ా": "aa", "ి": "i", "ీ": "ee", "ు": "u", "ూ": "oo",
        "ె": "e", "ే": "e", "ై": "ai", "ొ": "o", "ో": "o", "ౌ": "au",
        "ం": "m", "ః": "h",
        "్": "",  # virama — suppresses inherent vowel
        "ృ": "ru", "ౄ": "roo",
    }
    result = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i:i+2] in telugu_map:
            result.append(telugu_map[text[i:i+2]])
            i += 2
        elif text[i] in telugu_map:
            result.append(telugu_map[text[i]])
            i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def normalize(text):
    """Normalize text for comparison: lowercase, remove punctuation/spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z āēīōūṁṃḍṇṭḷṛśṣ]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def compare_pronunciation(expected_roman, actual_transcription):
    """
    Compare expected romanized word with Whisper's transcription.
    Returns a list of (char, status) tuples and an overall score.
    """
    expected = normalize(expected_roman)
    actual = normalize(actual_transcription)

    matcher = difflib.SequenceMatcher(None, expected, actual)
    opcodes = matcher.get_opcodes()

    result = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for ch in expected[i1:i2]:
                result.append((ch, "correct"))
        elif tag == "replace":
            exp_chunk = expected[i1:i2]
            act_chunk = actual[j1:j2]
            for k, ch in enumerate(exp_chunk):
                wrong_ch = act_chunk[k] if k < len(act_chunk) else "?"
                result.append((ch, "wrong", wrong_ch))
            for k in range(len(exp_chunk), len(act_chunk)):
                result.append(("", "extra", act_chunk[k]))
        elif tag == "delete":
            for ch in expected[i1:i2]:
                result.append((ch, "missing"))
        elif tag == "insert":
            for ch in actual[j1:j2]:
                result.append(("", "extra", ch))

    correct = sum(1 for r in result if r[1] == "correct")
    total = sum(1 for r in result if r[1] != "extra")
    score = int((correct / max(total, 1)) * 100)

    return result, score


def render_feedback_html(comparison_result, expected_roman):
    """Build colored HTML showing which letters are correct/wrong/missing."""
    html_parts = ['<div style="font-size: 28px; font-family: monospace; line-height: 2;">']
    html_parts.append('<p style="margin-bottom: 5px;"><b>Your pronunciation:</b></p>')

    for item in comparison_result:
        if item[1] == "correct":
            html_parts.append(
                f'<span style="color: #00cc66; font-weight: bold; '
                f'background: #003311; padding: 2px 4px; border-radius: 4px; margin: 1px;">'
                f'{item[0]}</span>'
            )
        elif item[1] == "wrong":
            wrong_ch = item[2] if len(item) > 2 else "?"
            html_parts.append(
                f'<span title="You said: {wrong_ch}" '
                f'style="color: #ff4444; font-weight: bold; text-decoration: underline wavy red; '
                f'background: #330000; padding: 2px 4px; border-radius: 4px; margin: 1px; '
                f'cursor: help;">'
                f'{item[0]}</span>'
            )
        elif item[1] == "missing":
            html_parts.append(
                f'<span title="You missed this sound" '
                f'style="color: #ffaa00; font-weight: bold; text-decoration: line-through; '
                f'background: #332200; padding: 2px 4px; border-radius: 4px; margin: 1px; '
                f'cursor: help;">'
                f'{item[0]}</span>'
            )
        elif item[1] == "extra":
            extra_ch = item[2] if len(item) > 2 else "?"
            html_parts.append(
                f'<span title="Extra sound added" '
                f'style="color: #aa88ff; font-weight: bold; '
                f'background: #220033; padding: 2px 4px; border-radius: 4px; margin: 1px; '
                f'cursor: help;">'
                f'+{extra_ch}</span>'
            )

    html_parts.append('</div>')
    html_parts.append(
        '<div style="font-size: 14px; margin-top: 15px; color: #888;">'
        '<span style="color: #00cc66;">&#9632;</span> Correct &nbsp; '
        '<span style="color: #ff4444;">&#9632;</span> Wrong &nbsp; '
        '<span style="color: #ffaa00;">&#9632;</span> Missing &nbsp; '
        '<span style="color: #aa88ff;">&#9632;</span> Extra'
        '</div>'
    )

    return "".join(html_parts)


def play_or_generate_word(client, voice_id, user_dir, cat_key, word):
    """Play existing audio or generate on demand. Returns filepath."""
    filepath = get_word_filepath(user_dir, cat_key, word["romanized"])
    if not os.path.exists(filepath):
        audio = generate_single_audio(client, voice_id, word["telugu"], pronunciation_hint=word["romanized"])
        save_audio(audio, filepath)
    return filepath


# ── Progress Tracking ──────────────────────────────────────────────────────

def init_progress():
    """Initialize progress tracking in session state."""
    if "progress" not in st.session_state:
        st.session_state["progress"] = {}
    if "quiz_stats" not in st.session_state:
        st.session_state["quiz_stats"] = {"correct": 0, "total": 0, "streak": 0, "best_streak": 0}


def record_practice_score(cat_key, romanized, score):
    """Record a pronunciation practice score for a word."""
    init_progress()
    key = f"{cat_key}::{romanized}"
    if key not in st.session_state["progress"]:
        st.session_state["progress"][key] = {"attempts": 0, "best_score": 0, "mastered": False}
    entry = st.session_state["progress"][key]
    entry["attempts"] += 1
    entry["best_score"] = max(entry["best_score"], score)
    entry["mastered"] = entry["best_score"] >= 90


def record_quiz_result(correct):
    """Record a quiz answer result."""
    init_progress()
    stats = st.session_state["quiz_stats"]
    stats["total"] += 1
    if correct:
        stats["correct"] += 1
        stats["streak"] += 1
        stats["best_streak"] = max(stats["best_streak"], stats["streak"])
    else:
        stats["streak"] = 0


def get_all_words(word_categories):
    """Get a flat list of all words across categories."""
    all_words = []
    for cat_key, cat in word_categories.items():
        for word in cat["words"]:
            all_words.append({"cat_key": cat_key, **word})
    return all_words


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Telugu Phonic Agent",
    page_icon="🗣️",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ─── Global ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Noto+Sans+Telugu:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Animations ─── */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255,107,53,0.4); }
        50% { box-shadow: 0 0 12px 4px rgba(255,107,53,0.15); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* ─── Header Banner ─── */
    .hero-banner {
        background: linear-gradient(135deg, #FF6B35, #F7931E, #E8530E, #FFB347, #FF6B35);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        border-radius: 20px;
        padding: 48px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(255,107,53,0.25), 0 0 60px rgba(255,107,53,0.08);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 450px;
        height: 450px;
        background: rgba(255,255,255,0.07);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }
    .hero-banner h1 {
        font-size: 2.8em;
        font-weight: 800;
        color: white;
        margin: 0 0 10px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-banner p {
        font-size: 1.2em;
        color: rgba(255,255,255,0.92);
        margin: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .hero-telugu {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.2em;
        color: rgba(255,255,255,0.6);
        margin-top: 6px;
        position: relative;
        z-index: 1;
    }

    /* ─── Glassmorphism Base ─── */
    .glass {
        background: rgba(30, 32, 40, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ─── Word Cards ─── */
    .word-card {
        background: rgba(30, 32, 40, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 107, 53, 0.1);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 14px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .word-card:hover {
        border-color: rgba(255, 107, 53, 0.45);
        box-shadow: 0 4px 24px rgba(255, 107, 53, 0.12), 0 0 40px rgba(255,107,53,0.06);
        transform: translateY(-2px);
        background: rgba(35, 37, 48, 0.75);
    }
    .word-telugu {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 2em;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 6px;
        text-shadow: 0 0 20px rgba(255,107,53,0.2);
    }
    .word-roman {
        font-size: 1.08em;
        color: #B8C0D0;
        font-style: italic;
        margin-bottom: 3px;
    }
    .word-english {
        font-size: 0.95em;
        color: #6B7585;
    }

    /* ─── Audio Player Wrapper ─── */
    .audio-wrapper {
        background: rgba(25, 27, 35, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,107,53,0.1);
        border-radius: 12px;
        padding: 12px;
        margin-top: 8px;
        transition: all 0.3s ease;
    }
    .audio-wrapper:hover {
        border-color: rgba(255,107,53,0.25);
        box-shadow: 0 0 16px rgba(255,107,53,0.08);
    }

    /* ─── Rhyme Card ─── */
    .rhyme-card {
        background: rgba(26, 29, 36, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 179, 71, 0.12);
        border-radius: 16px;
        padding: 28px;
        margin: 14px 0;
        transition: all 0.3s ease;
    }
    .rhyme-card:hover {
        border-color: rgba(255, 179, 71, 0.3);
        box-shadow: 0 4px 24px rgba(255,179,71,0.08);
    }
    .rhyme-title {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.6em;
        font-weight: 700;
        color: #FFB347;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(255,179,71,0.15);
    }
    .rhyme-verse {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.12em;
        color: #D4D8E0;
        line-height: 1.9;
        white-space: pre-wrap;
    }
    .rhyme-romanized {
        font-size: 0.95em;
        color: #8B95A5;
        font-style: italic;
        line-height: 1.7;
        white-space: pre-wrap;
        margin-top: 10px;
    }

    /* ─── Practice Section ─── */
    .practice-word {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 3.5em;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        padding: 24px;
        text-shadow: 0 0 30px rgba(255,107,53,0.25);
    }
    .practice-info {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.15em;
    }

    /* ─── Score Badge ─── */
    .score-badge {
        display: inline-block;
        font-size: 2.2em;
        font-weight: 700;
        padding: 12px 28px;
        border-radius: 14px;
        margin: 10px 0;
    }
    .score-great {
        background: rgba(0,204,102,0.12);
        color: #00cc66;
        border: 1px solid rgba(0,204,102,0.3);
        box-shadow: 0 0 20px rgba(0,204,102,0.1);
    }
    .score-good {
        background: rgba(255,170,0,0.12);
        color: #ffaa00;
        border: 1px solid rgba(255,170,0,0.3);
        box-shadow: 0 0 20px rgba(255,170,0,0.1);
    }
    .score-poor {
        background: rgba(255,68,68,0.12);
        color: #ff4444;
        border: 1px solid rgba(255,68,68,0.3);
        box-shadow: 0 0 20px rgba(255,68,68,0.1);
    }

    /* ─── Sidebar Styling ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1117, #151820, #1A1D24);
        border-right: 1px solid rgba(255,107,53,0.08);
    }
    .sidebar-status {
        background: rgba(30, 58, 47, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,204,102,0.25);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 18px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,204,102,0.06);
    }
    .sidebar-status-default {
        background: rgba(42, 34, 53, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(170,136,255,0.25);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 18px;
        text-align: center;
        box-shadow: 0 0 20px rgba(170,136,255,0.06);
    }

    /* ─── Tab Styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(20, 22, 28, 0.5);
        border-radius: 14px 14px 0 0;
        padding: 6px 6px 0 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0;
        padding: 12px 22px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,107,53,0.08);
    }
    .stTabs [aria-selected="true"] {
        box-shadow: 0 -2px 8px rgba(255,107,53,0.15);
    }

    /* ─── Category Pill ─── */
    .category-label {
        display: inline-block;
        background: rgba(255,107,53,0.1);
        color: #FF6B35;
        padding: 8px 20px;
        border-radius: 24px;
        font-weight: 600;
        font-size: 0.95em;
        margin-bottom: 18px;
        border: 1px solid rgba(255,107,53,0.15);
        box-shadow: 0 0 12px rgba(255,107,53,0.06);
    }

    /* ─── Buttons ─── */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        box-shadow: 0 4px 16px rgba(255,107,53,0.2);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        box-shadow: 0 2px 8px rgba(255,107,53,0.15);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(255,107,53,0.3);
    }

    /* ─── Misc ─── */
    .divider-accent {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,107,53,0.35), transparent);
        margin: 28px 0;
    }

    /* ─── How It Works Steps ─── */
    .how-it-works {
        background: rgba(26, 29, 36, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,107,53,0.12);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 32px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .how-it-works h3 {
        color: #FF6B35;
        margin: 0 0 20px 0;
        font-size: 1.2em;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .step-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .step-row:last-child {
        margin-bottom: 0;
    }
    .step-number {
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        color: white;
        font-weight: 700;
        font-size: 0.85em;
        width: 32px;
        height: 32px;
        min-width: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 16px;
        margin-top: 2px;
        animation: pulse 3s ease-in-out infinite;
        box-shadow: 0 2px 8px rgba(255,107,53,0.25);
    }
    .step-text {
        color: #C8CDD5;
        font-size: 0.97em;
        line-height: 1.55;
    }
    .step-text b {
        color: #FFFFFF;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,107,53,0.2);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,107,53,0.4);
    }

    /* ─── Progress Dashboard ─── */
    .progress-card {
        background: rgba(30, 32, 40, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,107,53,0.12);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .progress-card:hover {
        border-color: rgba(255,107,53,0.3);
        box-shadow: 0 4px 20px rgba(255,107,53,0.1);
    }
    .progress-number {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B35, #FFB347);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .progress-label {
        font-size: 0.9em;
        color: #8B95A5;
        margin-top: 4px;
    }
    .progress-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 8px 0;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #FF6B35, #FFB347);
        transition: width 0.5s ease;
    }
    .progress-bar-fill-green {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00cc66, #00ff88);
        transition: width 0.5s ease;
    }

    /* ─── Quiz Cards ─── */
    .quiz-option {
        background: rgba(30, 32, 40, 0.6);
        backdrop-filter: blur(8px);
        border: 2px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-size: 1.1em;
        font-weight: 500;
        color: #C8CDD5;
    }
    .quiz-option:hover {
        border-color: rgba(255,107,53,0.5);
        box-shadow: 0 0 20px rgba(255,107,53,0.1);
        transform: translateY(-2px);
        color: white;
    }
    .quiz-correct {
        background: rgba(0,204,102,0.15) !important;
        border-color: rgba(0,204,102,0.5) !important;
        color: #00cc66 !important;
        box-shadow: 0 0 20px rgba(0,204,102,0.15);
    }
    .quiz-wrong {
        background: rgba(255,68,68,0.15) !important;
        border-color: rgba(255,68,68,0.5) !important;
        color: #ff4444 !important;
        box-shadow: 0 0 20px rgba(255,68,68,0.15);
    }
    .quiz-streak {
        background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,179,71,0.1));
        border: 1px solid rgba(255,107,53,0.25);
        border-radius: 12px;
        padding: 12px 20px;
        display: inline-block;
        font-weight: 600;
        color: #FFB347;
    }
</style>
""", unsafe_allow_html=True)

# ── Header Banner ────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <h1>Telugu Phonic Agent</h1>
    <p>Learn Telugu pronunciation in your own voice</p>
    <div class="hero-telugu">తెలుగు ఫోనిక్ ఏజెంట్</div>
</div>
""", unsafe_allow_html=True)

# ── How It Works ────────────────────────────────────────────────────────────

st.markdown("""
<div class="how-it-works">
    <h3>How It Works</h3>
    <div class="step-row">
        <div class="step-number">1</div>
        <div class="step-text"><b>Clone your voice</b> — Record a 60-second voice sample in the sidebar on the left. Read the sample script clearly, then click "Clone My Voice".</div>
    </div>
    <div class="step-row">
        <div class="step-number">2</div>
        <div class="step-text"><b>Generate audio</b> — Go to the <b>Generate All</b> tab and click "Generate Everything". This creates all Telugu word and rhyme audio in your own voice.</div>
    </div>
    <div class="step-row">
        <div class="step-number">3</div>
        <div class="step-text"><b>Listen & learn</b> — Go to the <b>Words</b> tab to browse categories and hear each Telugu word pronounced in your voice. Check the <b>Nursery Rhymes</b> tab for fun rhymes.</div>
    </div>
    <div class="step-row">
        <div class="step-number">4</div>
        <div class="step-text"><b>Practice pronunciation</b> — Go to the <b>Pronunciation Practice</b> tab, pick a word, listen to it, then record yourself saying it. The app tells you exactly which sounds you got right or wrong.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Voice Setup ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎙️ Voice Setup")

    # Show current voice status
    if "voice_id" in st.session_state:
        st.markdown(
            '<div class="sidebar-status">'
            '✅ <b>Your voice is active</b><br>'
            '<span style="font-size:0.85em; color:#8B95A5;">All audio uses your cloned voice</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sidebar-status-default">'
            '🔮 <b>Using default voice</b><br>'
            '<span style="font-size:0.85em; color:#8B95A5;">Clone your voice below to personalize!</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

    st.markdown(
        "Record or upload a **~60 second** voice sample. "
        "Speak **clearly and slowly** in your natural voice."
    )

    with st.expander("📝 Sample script to read", expanded=False):
        st.markdown(
            "Read this script **slowly and clearly:**\n\n"
            '*"Hello, my name is ___. I am recording my voice so that '
            "it can be used to help me learn Telugu pronunciation. "
            "Telugu is a beautiful language spoken by millions of people. "
            "I am excited to learn how to speak it.\n\n"
            "The sun is shining brightly today. "
            "My mother and father are at home. "
            "I like to drink water and eat good food. "
            "The tall tree has a beautiful flower. "
            "I go to school to read my book every day.\n\n"
            "I will now count slowly. "
            'One, two, three, four, five, six, seven, eight, nine, ten."*'
        )

    audio_input = st.audio_input("🎤 Record your voice")
    uploaded_file = st.file_uploader(
        "Or upload a WAV/MP3 file", type=["wav", "mp3"]
    )

    voice_sample = None
    if audio_input is not None:
        voice_sample = audio_input.getvalue()
    elif uploaded_file is not None:
        voice_sample = uploaded_file.getvalue()

    if voice_sample:
        st.audio(voice_sample, format="audio/wav")

    if st.button("🎙️ Clone My Voice", type="primary", disabled=voice_sample is None, use_container_width=True):
        client = get_client()
        if client is None:
            st.error("ElevenLabs API key not configured.")
        else:
            with st.spinner("Cloning your voice..."):
                try:
                    voice_id = clone_voice(client, voice_sample)
                    st.session_state["voice_id"] = voice_id
                    st.success("Voice cloned! All audio will now use your voice.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Cloning failed: {e}")

# ── Main Content ─────────────────────────────────────────────────────────────

word_bank = load_word_bank()
word_categories = {k: v for k, v in word_bank["categories"].items() if "words" in v}
has_rhymes = "nursery_rhymes" in word_bank["categories"]

# Current user's audio directory and voice
user_dir = get_user_audio_dir()
active_voice = get_active_voice_id()

init_progress()

tab_words, tab_rhymes, tab_practice, tab_quiz, tab_progress, tab_generate_all = st.tabs(
    ["📚 Words", "🎶 Nursery Rhymes", "🎯 Practice", "🧠 Quiz", "📊 Progress", "⚡ Generate All"]
)

# ── Tab 1: Words ─────────────────────────────────────────────────────────────

with tab_words:
    category_labels = {k: v["label"] for k, v in word_categories.items()}
    selected_cat = st.selectbox(
        "Choose a category", options=list(category_labels.keys()),
        format_func=lambda k: category_labels[k],
    )

    if selected_cat:
        category = word_categories[selected_cat]
        st.markdown(f'<span class="category-label">{category["label"]}</span>',
                    unsafe_allow_html=True)

        for word in category["words"]:
            col_info, col_audio = st.columns([3, 1])

            with col_info:
                st.markdown(
                    f'<div class="word-card">'
                    f'<div class="word-telugu">{word["telugu"]}</div>'
                    f'<div class="word-roman">{word["romanized"]}</div>'
                    f'<div class="word-english">{word["english"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col_audio:
                filepath = get_word_filepath(user_dir, selected_cat, word["romanized"])

                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                else:
                    safe_name = word["romanized"].replace(" ", "_").replace("/", "_").lower()
                    btn_key = f"gen_{selected_cat}_{safe_name}"
                    if st.button("🔊 Generate", key=btn_key):
                        client = get_client()
                        if client is None:
                            st.error("API key not configured.")
                        else:
                            with st.spinner("Generating..."):
                                audio = generate_single_audio(client, active_voice, word["telugu"], pronunciation_hint=word["romanized"])
                                save_audio(audio, filepath)
                                st.rerun()

# ── Tab 2: Nursery Rhymes ───────────────────────────────────────────────────

with tab_rhymes:
    if not has_rhymes:
        st.info("No nursery rhymes in the word bank yet.")
    else:
        rhymes = word_bank["categories"]["nursery_rhymes"]["rhymes"]
        rhyme_titles = {r["title"]: r for r in rhymes}

        selected_rhyme_title = st.selectbox(
            "Choose a rhyme",
            options=list(rhyme_titles.keys()),
            format_func=lambda t: f"{rhyme_titles[t]['title_telugu']}  ({t})",
        )

        rhyme = rhyme_titles[selected_rhyme_title]
        rhyme_dir = get_rhyme_dir(user_dir, rhyme["title"])

        # Rhyme header card
        st.markdown(
            f'<div class="rhyme-card">'
            f'<div class="rhyme-title">{rhyme["title_telugu"]}</div>'
            f'<div style="color:#9CA3AF; font-size:1.05em;">{rhyme["title"]}</div>'
            f'<div style="color:#6B7280; font-size:0.9em; margin-top:6px;">{rhyme["english_meaning"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Full rhyme playback
        full_path = os.path.join(rhyme_dir, "full_rhyme.mp3")
        col_full_label, col_full_audio = st.columns([1, 2])
        with col_full_label:
            st.markdown("**▶️ Full Rhyme**")
        with col_full_audio:
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            else:
                if st.button("🔊 Generate Full Rhyme", key="gen_full_rhyme"):
                    client = get_client()
                    if client:
                        full_text = "\n\n".join(v["telugu"] for v in rhyme["verses"])
                        with st.spinner("Generating full rhyme..."):
                            audio = generate_single_audio(client, active_voice, full_text)
                            save_audio(audio, full_path)
                            st.rerun()

        st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

        # Individual verses
        for verse in rhyme["verses"]:
            vnum = verse["verse_number"]
            safe_title = rhyme["title"].replace(" ", "_").lower()
            with st.expander(f"🎵 Verse {vnum}", expanded=True):
                st.markdown(
                    f'<div class="rhyme-card">'
                    f'<div class="rhyme-verse">{verse["telugu"]}</div>'
                    f'<div class="rhyme-romanized">{verse["romanized"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                verse_path = os.path.join(rhyme_dir, f"verse_{vnum}.mp3")
                if os.path.exists(verse_path):
                    with open(verse_path, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                else:
                    if st.button(f"🔊 Generate Verse {vnum}", key=f"gen_v{vnum}_{safe_title}"):
                        client = get_client()
                        if client:
                            with st.spinner(f"Generating verse {vnum}..."):
                                audio = generate_single_audio(
                                    client, active_voice, verse["telugu"],
                                )
                                save_audio(audio, verse_path)
                                st.rerun()

# ── Tab 3: Pronunciation Practice ────────────────────────────────────────────

with tab_practice:
    st.markdown(
        '<p style="color: #9CA3AF; font-size: 1.05em;">'
        'Pick a word, listen to the correct pronunciation, then record yourself. '
        'The app highlights exactly which sounds you nailed and which need work.'
        '</p>',
        unsafe_allow_html=True,
    )

    prac_cat_labels = {k: v["label"] for k, v in word_categories.items()}
    prac_cat = st.selectbox(
        "Category", options=list(prac_cat_labels.keys()),
        format_func=lambda k: prac_cat_labels[k],
        key="prac_cat",
    )

    if prac_cat:
        prac_words = word_categories[prac_cat]["words"]
        prac_word_labels = {
            i: f"{w['telugu']}  ({w['romanized']}) — {w['english']}"
            for i, w in enumerate(prac_words)
        }
        prac_word_idx = st.selectbox(
            "Word to practice",
            options=list(prac_word_labels.keys()),
            format_func=lambda i: prac_word_labels[i],
            key="prac_word",
        )

        word = prac_words[prac_word_idx]

        st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

        # Step 1: Show the word in a styled card
        st.markdown(
            f'<div class="word-card" style="text-align: center; padding: 30px;">'
            f'<div class="practice-word">{word["telugu"]}</div>'
            f'<div class="practice-info">{word["romanized"]} — {word["english"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Play correct audio
        filepath = get_word_filepath(user_dir, prac_cat, word["romanized"])
        col_listen_label, col_listen_audio = st.columns([1, 2])
        with col_listen_label:
            st.markdown("**🔊 Listen first:**")
        with col_listen_audio:
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
            else:
                client = get_client()
                if client:
                    with st.spinner("Generating..."):
                        audio = generate_single_audio(client, active_voice, word["telugu"], pronunciation_hint=word["romanized"])
                        save_audio(audio, filepath)
                    with open(filepath, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")
                else:
                    st.info("Generate this word's audio first.")

        st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

        # Step 2: Record user's attempt
        st.markdown("### 🎤 Now try saying it:")
        user_audio = st.audio_input("Record your pronunciation", key="prac_record")

        if user_audio is not None:
            user_bytes = user_audio.getvalue()

            if st.button("Check My Pronunciation", type="primary"):
                with st.spinner("Listening to your pronunciation (this may take a moment)..."):
                    telugu_heard, roman_heard = transcribe_audio(
                        user_bytes,
                        expected_telugu=word["telugu"],
                        expected_roman=word["romanized"],
                    )

                st.divider()
                st.markdown("### Results")

                expected_roman = word["romanized"]

                if any("\u0C00" <= ch <= "\u0C7F" for ch in telugu_heard):
                    telugu_as_roman = transliterate_telugu_to_roman(telugu_heard)
                else:
                    telugu_as_roman = telugu_heard

                from difflib import SequenceMatcher
                score_telugu = SequenceMatcher(None, normalize(expected_roman), normalize(telugu_as_roman)).ratio()
                score_roman = SequenceMatcher(None, normalize(expected_roman), normalize(roman_heard)).ratio()

                if score_telugu >= score_roman:
                    actual_roman = telugu_as_roman
                else:
                    actual_roman = roman_heard

                col_exp, col_act = st.columns(2)
                with col_exp:
                    st.markdown(f"**Expected:** `{expected_roman}`")
                with col_act:
                    st.markdown(f"**You said:** `{actual_roman}`")
                    st.caption(f"Telugu heard: {telugu_heard} | Roman heard: {roman_heard}")

                comparison, score = compare_pronunciation(expected_roman, actual_roman)

                # Record progress
                record_practice_score(prac_cat, word["romanized"], score)

                if score >= 90:
                    badge_class = "score-great"
                    badge_msg = "Excellent! 🎉"
                elif score >= 70:
                    badge_class = "score-good"
                    badge_msg = "Good, some sounds need work"
                elif score >= 50:
                    badge_class = "score-good"
                    badge_msg = "Keep practicing! Focus on red letters"
                else:
                    badge_class = "score-poor"
                    badge_msg = "Let's work on this one"

                st.markdown(
                    f'<div class="score-badge {badge_class}">{score}%</div>'
                    f'<p style="color: #9CA3AF; margin-top: 4px;">{badge_msg}</p>',
                    unsafe_allow_html=True,
                )

                feedback_html = render_feedback_html(comparison, expected_roman)
                st.markdown(feedback_html, unsafe_allow_html=True)

                wrong_letters = [item for item in comparison if item[1] in ("wrong", "missing")]
                if wrong_letters:
                    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Focus on these sounds:")
                    for item in wrong_letters:
                        if item[1] == "wrong" and len(item) > 2:
                            st.markdown(
                                f"- You said **{item[2]}** instead of **{item[0]}** — "
                                f"try emphasizing the **{item[0]}** sound"
                            )
                        elif item[1] == "missing":
                            st.markdown(
                                f"- You missed the **{item[0]}** sound — "
                                f"make sure to pronounce it clearly"
                            )

                if os.path.exists(filepath):
                    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)
                    st.markdown("### 🔁 Listen again and retry:")
                    with open(filepath, "rb") as f:
                        st.audio(f.read(), format="audio/mp3")

# ── Tab 4: Quiz Mode ───────────────────────────────────────────────────────

with tab_quiz:
    st.markdown(
        '<p style="color: #9CA3AF; font-size: 1.05em;">'
        'Listen to a Telugu word and pick the correct English meaning. '
        'Build your streak and test your knowledge!'
        '</p>',
        unsafe_allow_html=True,
    )

    all_words = get_all_words(word_categories)
    quiz_stats = st.session_state["quiz_stats"]

    # Show current streak and stats
    col_streak, col_score, col_best = st.columns(3)
    with col_streak:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_stats["streak"]}</div>'
            f'<div class="progress-label">Current Streak</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_score:
        pct = int((quiz_stats["correct"] / max(quiz_stats["total"], 1)) * 100)
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_stats["correct"]}/{quiz_stats["total"]}</div>'
            f'<div class="progress-label">Correct ({pct}%)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_best:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_stats["best_streak"]}</div>'
            f'<div class="progress-label">Best Streak</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

    # Generate a new question
    if "quiz_word" not in st.session_state or st.button("Next Question", type="primary"):
        if len(all_words) < 4:
            st.warning("Need at least 4 words in the word bank for quiz mode.")
        else:
            correct_word = random.choice(all_words)
            wrong_words = [w for w in all_words if w["english"] != correct_word["english"]]
            wrong_options = random.sample(wrong_words, min(3, len(wrong_words)))
            options = [correct_word] + wrong_options
            random.shuffle(options)

            st.session_state["quiz_word"] = correct_word
            st.session_state["quiz_options"] = options
            st.session_state["quiz_answered"] = False
            st.session_state["quiz_selected"] = None
            st.rerun()

    if "quiz_word" in st.session_state:
        qword = st.session_state["quiz_word"]
        qoptions = st.session_state["quiz_options"]

        # Show the Telugu word
        st.markdown(
            f'<div class="word-card" style="text-align: center; padding: 30px;">'
            f'<div class="practice-word">{qword["telugu"]}</div>'
            f'<div style="color: #6B7585; font-size: 0.95em;">What does this word mean?</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Play audio if available
        filepath = get_word_filepath(user_dir, qword["cat_key"], qword["romanized"])
        if os.path.exists(filepath):
            col_a1, col_a2, col_a3 = st.columns([1, 2, 1])
            with col_a2:
                with open(filepath, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")

        st.markdown("")

        # Show options as buttons
        answered = st.session_state.get("quiz_answered", False)

        for i, opt in enumerate(qoptions):
            is_correct = opt["english"] == qword["english"]
            was_selected = st.session_state.get("quiz_selected") == i

            if answered:
                if is_correct:
                    st.success(f"✅  {opt['english']}")
                elif was_selected:
                    st.error(f"❌  {opt['english']}")
                else:
                    st.button(opt["english"], key=f"quiz_opt_{i}", disabled=True)
            else:
                if st.button(opt["english"], key=f"quiz_opt_{i}", use_container_width=True):
                    st.session_state["quiz_answered"] = True
                    st.session_state["quiz_selected"] = i
                    record_quiz_result(is_correct)
                    st.rerun()

        if answered:
            selected_idx = st.session_state["quiz_selected"]
            selected_opt = qoptions[selected_idx]
            if selected_opt["english"] == qword["english"]:
                st.markdown(
                    f'<div class="quiz-streak">🔥 Streak: {st.session_state["quiz_stats"]["streak"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'The correct answer was **{qword["english"]}** ({qword["telugu"]} — {qword["romanized"]})',
                )
                st.session_state["quiz_stats"]["streak"] = 0

# ── Tab 5: Progress Dashboard ─────────────────────────────────────────────

with tab_progress:
    st.markdown(
        '<p style="color: #9CA3AF; font-size: 1.05em;">'
        'Track your learning journey across all categories.'
        '</p>',
        unsafe_allow_html=True,
    )

    progress_data = st.session_state.get("progress", {})
    quiz_stats = st.session_state["quiz_stats"]

    # Overall stats
    total_words = sum(len(c["words"]) for c in word_categories.values())
    practiced_words = len(progress_data)
    mastered_words = sum(1 for v in progress_data.values() if v.get("mastered", False))
    total_attempts = sum(v.get("attempts", 0) for v in progress_data.values())

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{practiced_words}</div>'
            f'<div class="progress-label">Words Practiced</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{mastered_words}</div>'
            f'<div class="progress-label">Words Mastered (90%+)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{total_attempts}</div>'
            f'<div class="progress-label">Total Attempts</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_s4:
        quiz_pct = int((quiz_stats["correct"] / max(quiz_stats["total"], 1)) * 100)
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_pct}%</div>'
            f'<div class="progress-label">Quiz Accuracy</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Overall progress bar
    overall_pct = int((mastered_words / max(total_words, 1)) * 100)
    st.markdown(f"### Overall Mastery: {mastered_words}/{total_words} words")
    st.markdown(
        f'<div class="progress-bar-bg">'
        f'<div class="progress-bar-fill-green" style="width: {overall_pct}%;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

    # Per-category breakdown
    st.markdown("### Category Breakdown")

    for cat_key, category in word_categories.items():
        words_in_cat = category["words"]
        cat_total = len(words_in_cat)

        cat_practiced = 0
        cat_mastered = 0
        cat_details = []

        for word in words_in_cat:
            key = f"{cat_key}::{word['romanized']}"
            entry = progress_data.get(key, None)
            if entry:
                cat_practiced += 1
                if entry.get("mastered"):
                    cat_mastered += 1
                cat_details.append({
                    "word": word,
                    "attempts": entry["attempts"],
                    "best_score": entry["best_score"],
                    "mastered": entry["mastered"],
                })
            else:
                cat_details.append({
                    "word": word,
                    "attempts": 0,
                    "best_score": 0,
                    "mastered": False,
                })

        cat_pct = int((cat_mastered / max(cat_total, 1)) * 100)

        with st.expander(f"{category['label']}  —  {cat_mastered}/{cat_total} mastered", expanded=False):
            st.markdown(
                f'<div class="progress-bar-bg">'
                f'<div class="progress-bar-fill" style="width: {cat_pct}%;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            for detail in cat_details:
                w = detail["word"]
                if detail["mastered"]:
                    status = "✅"
                    color = "#00cc66"
                elif detail["attempts"] > 0:
                    status = "🔶"
                    color = "#ffaa00"
                else:
                    status = "⬜"
                    color = "#6B7585"

                col_w, col_s = st.columns([3, 1])
                with col_w:
                    st.markdown(
                        f'<span style="color:{color};">{status}</span> '
                        f'**{w["telugu"]}** ({w["romanized"]}) — {w["english"]}',
                        unsafe_allow_html=True,
                    )
                with col_s:
                    if detail["attempts"] > 0:
                        st.markdown(
                            f'Best: **{detail["best_score"]}%** · {detail["attempts"]} tries'
                        )
                    else:
                        st.markdown('*Not practiced yet*')

    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)

    # Quiz stats section
    st.markdown("### Quiz Performance")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_stats["total"]}</div>'
            f'<div class="progress-label">Questions Answered</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_q2:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_stats["correct"]}</div>'
            f'<div class="progress-label">Correct Answers</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_q3:
        st.markdown(
            f'<div class="progress-card">'
            f'<div class="progress-number">{quiz_stats["best_streak"]}</div>'
            f'<div class="progress-label">Best Streak</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Reset progress button
    st.markdown('<hr class="divider-accent">', unsafe_allow_html=True)
    if st.button("🗑️ Reset All Progress", help="Clear all practice scores and quiz stats"):
        st.session_state["progress"] = {}
        st.session_state["quiz_stats"] = {"correct": 0, "total": 0, "streak": 0, "best_streak": 0}
        st.success("Progress reset!")
        st.rerun()

# ── Tab 6: Generate All ─────────────────────────────────────────────────────

with tab_generate_all:
    if "voice_id" in st.session_state:
        st.markdown(
            '<div class="word-card" style="text-align:center; padding:24px;">'
            '<div style="font-size:1.2em; color:#00cc66; font-weight:600;">🎙️ Using Your Cloned Voice</div>'
            '<div style="color:#8B95A5; margin-top:6px;">Generate audio for every word and nursery rhyme in your voice.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="word-card" style="text-align:center; padding:24px;">'
            '<div style="font-size:1.2em; color:#aa88ff; font-weight:600;">🔮 Using Default Voice</div>'
            '<div style="color:#8B95A5; margin-top:6px;">Clone your voice first (sidebar) to hear everything in YOUR voice!</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    regenerate = st.checkbox(
        "Regenerate all (overwrite existing files)",
        value=True,
        help="Check this if you cloned a new voice and want to replace all audio.",
    )

    if st.button("Generate Everything", type="primary"):
        client = get_client()
        if client is None:
            st.error("API key not configured.")
        else:
            voice_id = active_voice

            total_words = sum(len(c["words"]) for c in word_categories.values())
            total_verses = 0
            if has_rhymes:
                for r in word_bank["categories"]["nursery_rhymes"]["rhymes"]:
                    total_verses += len(r["verses"]) + 1
            total = total_words + total_verses

            progress = st.progress(0, text="Starting...")
            done = 0

            # Generate words
            for cat_key, category in word_categories.items():
                for word in category["words"]:
                    filepath = get_word_filepath(user_dir, cat_key, word["romanized"])
                    if regenerate or not os.path.exists(filepath):
                        audio = generate_single_audio(client, voice_id, word["telugu"], pronunciation_hint=word["romanized"])
                        save_audio(audio, filepath)
                    done += 1
                    progress.progress(
                        done / total,
                        text=f"[{done}/{total}] {word['telugu']} ({word['romanized']})",
                    )

            # Generate nursery rhymes
            if has_rhymes:
                for rhyme in word_bank["categories"]["nursery_rhymes"]["rhymes"]:
                    r_dir = get_rhyme_dir(user_dir, rhyme["title"])

                    for verse in rhyme["verses"]:
                        vpath = os.path.join(r_dir, f"verse_{verse['verse_number']}.mp3")
                        if regenerate or not os.path.exists(vpath):
                            audio = generate_single_audio(client, voice_id, verse["telugu"])
                            save_audio(audio, vpath)
                        done += 1
                        progress.progress(
                            done / total,
                            text=f"[{done}/{total}] {rhyme['title']} — Verse {verse['verse_number']}",
                        )

                    full_p = os.path.join(r_dir, "full_rhyme.mp3")
                    if regenerate or not os.path.exists(full_p):
                        full_text = "\n\n".join(v["telugu"] for v in rhyme["verses"])
                        audio = generate_single_audio(client, voice_id, full_text)
                        save_audio(audio, full_p)
                    done += 1
                    progress.progress(
                        done / total,
                        text=f"[{done}/{total}] {rhyme['title']} — Full rhyme",
                    )

            progress.progress(1.0, text="All done!")
            st.success(f"Generated {total} audio files!")
            st.balloons()
