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
import streamlit as st
from elevenlabs.client import ElevenLabs
import whisper

WORD_BANK_PATH = "telugu_word_bank.json"
OUTPUT_DIR = "generated_audio"
MODEL_ID = "eleven_multilingual_v2"

# Default ElevenLabs voice for users who haven't cloned yet
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # Sarah — clear, professional
DEFAULT_VOICE_LABEL = "Default Voice (Sarah)"


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


def generate_single_audio(client, voice_id, text):
    """Generate audio for a single text and return bytes."""
    audio_iter = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=MODEL_ID,
        output_format="mp3_44100_128",
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
        audio = generate_single_audio(client, voice_id, word["telugu"])
        save_audio(audio, filepath)
    return filepath


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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Telugu:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ─── Header Banner ─── */
    .hero-banner {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFB347 100%);
        border-radius: 16px;
        padding: 40px 35px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .hero-banner h1 {
        font-size: 2.5em;
        font-weight: 700;
        color: white;
        margin: 0 0 8px 0;
    }
    .hero-banner p {
        font-size: 1.15em;
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-weight: 300;
    }
    .hero-telugu {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.1em;
        color: rgba(255,255,255,0.7);
        margin-top: 4px;
    }

    /* ─── Word Cards ─── */
    .word-card {
        background: linear-gradient(145deg, #1E2028, #252830);
        border: 1px solid rgba(255, 107, 53, 0.15);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .word-card:hover {
        border-color: rgba(255, 107, 53, 0.4);
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.1);
        transform: translateY(-1px);
    }
    .word-telugu {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.8em;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 4px;
    }
    .word-roman {
        font-size: 1.05em;
        color: #B0B8C8;
        font-style: italic;
        margin-bottom: 2px;
    }
    .word-english {
        font-size: 0.95em;
        color: #6B7280;
    }

    /* ─── Rhyme Card ─── */
    .rhyme-card {
        background: linear-gradient(145deg, #1A1D24, #22252D);
        border: 1px solid rgba(255, 179, 71, 0.15);
        border-radius: 14px;
        padding: 24px;
        margin: 12px 0;
    }
    .rhyme-title {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.5em;
        font-weight: 700;
        color: #FFB347;
        margin-bottom: 8px;
    }
    .rhyme-verse {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 1.1em;
        color: #D1D5DB;
        line-height: 1.8;
        white-space: pre-wrap;
    }
    .rhyme-romanized {
        font-size: 0.95em;
        color: #8B95A5;
        font-style: italic;
        line-height: 1.7;
        white-space: pre-wrap;
        margin-top: 8px;
    }

    /* ─── Practice Section ─── */
    .practice-word {
        font-family: 'Noto Sans Telugu', sans-serif;
        font-size: 3em;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        padding: 20px;
    }
    .practice-info {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1em;
    }

    /* ─── Score Badge ─── */
    .score-badge {
        display: inline-block;
        font-size: 2em;
        font-weight: 700;
        padding: 10px 24px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .score-great { background: rgba(0,204,102,0.15); color: #00cc66; border: 1px solid rgba(0,204,102,0.3); }
    .score-good { background: rgba(255,170,0,0.15); color: #ffaa00; border: 1px solid rgba(255,170,0,0.3); }
    .score-poor { background: rgba(255,68,68,0.15); color: #ff4444; border: 1px solid rgba(255,68,68,0.3); }

    /* ─── Sidebar Styling ─── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13151A, #1A1D24);
        border-right: 1px solid rgba(255,107,53,0.1);
    }
    .sidebar-status {
        background: linear-gradient(135deg, #1E3A2F, #1A2E25);
        border: 1px solid rgba(0,204,102,0.3);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 16px;
        text-align: center;
    }
    .sidebar-status-default {
        background: linear-gradient(135deg, #2A2235, #1F1A2E);
        border: 1px solid rgba(170,136,255,0.3);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 16px;
        text-align: center;
    }

    /* ─── Tab Styling ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* ─── Category Pill ─── */
    .category-label {
        display: inline-block;
        background: rgba(255,107,53,0.12);
        color: #FF6B35;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95em;
        margin-bottom: 16px;
    }

    /* ─── Misc ─── */
    .divider-accent {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,107,53,0.3), transparent);
        margin: 24px 0;
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
        "Record or upload a **~30 second** voice sample. "
        "Speak clearly in your natural voice."
    )

    with st.expander("📝 Sample script to read", expanded=False):
        st.markdown(
            '*"Hello, my name is ___. I am recording my voice so that '
            "it can be used to help people learn Telugu pronunciation. "
            "Telugu is a beautiful language spoken by millions of people. "
            "I will now count from one to ten slowly and clearly. "
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

tab_words, tab_rhymes, tab_practice, tab_generate_all = st.tabs(
    ["📚 Words", "🎶 Nursery Rhymes", "🎯 Pronunciation Practice", "⚡ Generate All"]
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
                                audio = generate_single_audio(client, active_voice, word["telugu"])
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
                        audio = generate_single_audio(client, active_voice, word["telugu"])
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

# ── Tab 4: Generate All ─────────────────────────────────────────────────────

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
                        audio = generate_single_audio(client, voice_id, word["telugu"])
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
