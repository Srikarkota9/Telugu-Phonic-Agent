"""
Usage:
    python voice_generator.py

Requires ELEVENLABS_API_KEY environment variable.
"""

import os
import json
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from elevenlabs.client import ElevenLabs

SAMPLE_RATE = 44100
RECORD_DURATION = 30  # seconds — ElevenLabs needs ~30s for good cloning
WORD_BANK_PATH = "telugu_word_bank.json"
OUTPUT_DIR = "generated_audio"
MODEL_ID = "eleven_multilingual_v2"


def record_voice_sample(duration: int = RECORD_DURATION) -> str:
    """Record a voice sample from the microphone and save as WAV."""
    print(f"\n--- Voice Recording ---")
    print(f"You will be recorded for {duration} seconds.")
    print("Please read the following passage clearly in your natural voice:\n")
    print('  "Hello, my name is [your name]. I am recording my voice so that')
    print('   it can be used to help people learn Telugu pronunciation.')
    print('   Telugu is a beautiful language spoken by millions of people.')
    print('   I will now count from one to ten slowly and clearly.')
    print('   One, two, three, four, five, six, seven, eight, nine, ten."')
    print()
    input("Press Enter when you are ready to start recording...")

    print(f"Recording... Speak now! ({duration} seconds)")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Recording complete!\n")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio_data)
    return tmp.name


def clone_voice(client: ElevenLabs, audio_path: str, voice_name: str = "Telugu Teacher") -> str:
    """Clone a voice using ElevenLabs Instant Voice Cloning. Returns voice_id."""
    print(f"Cloning voice from recording...")
    with open(audio_path, "rb") as f:
        voice = client.voices.ivc.create(
            name=voice_name,
            files=[f],
            description="Cloned voice for Telugu pronunciation teaching",
        )
    print(f"Voice cloned successfully! Voice ID: {voice.voice_id}\n")
    return voice.voice_id


def load_word_bank(path: str = WORD_BANK_PATH) -> dict:
    """Load the Telugu word bank from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_word_audio(client: ElevenLabs, voice_id: str, word_bank: dict, output_dir: str = OUTPUT_DIR):
    """Generate audio for each Telugu word in the word bank using the cloned voice."""
    os.makedirs(output_dir, exist_ok=True)

    # Count words (excluding nursery_rhymes which have a different structure)
    word_categories = {k: v for k, v in word_bank["categories"].items() if "words" in v}
    total_words = sum(len(cat["words"]) for cat in word_categories.values())
    generated = 0

    print(f"Generating audio for {total_words} Telugu words...\n")

    for cat_key, category in word_categories.items():
        cat_dir = os.path.join(output_dir, cat_key)
        os.makedirs(cat_dir, exist_ok=True)

        print(f"  Category: {category['label']}")

        for word_entry in category["words"]:
            telugu = word_entry["telugu"]
            romanized = word_entry["romanized"]
            english = word_entry["english"]

            # Use Telugu script as text input for TTS
            audio_iter = client.text_to_speech.convert(
                voice_id=voice_id,
                text=telugu,
                model_id=MODEL_ID,
                output_format="mp3_44100_128",
            )

            # Sanitize filename from romanized form
            safe_name = romanized.replace(" ", "_").replace("/", "_").lower()
            filename = f"{safe_name}.mp3"
            filepath = os.path.join(cat_dir, filename)

            with open(filepath, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)

            generated += 1
            print(f"    [{generated}/{total_words}] {telugu} ({romanized}) - {english}")

    print(f"\nDone! Generated {generated} audio files in '{output_dir}/'")

    # Generate nursery rhyme audio
    if "nursery_rhymes" in word_bank["categories"]:
        generate_nursery_rhyme_audio(client, voice_id, word_bank, output_dir)

    return generated


def generate_nursery_rhyme_audio(client: ElevenLabs, voice_id: str, word_bank: dict, output_dir: str = OUTPUT_DIR):
    """Generate audio for each nursery rhyme verse and a full version using the cloned voice."""
    rhymes_category = word_bank["categories"]["nursery_rhymes"]
    rhymes_dir = os.path.join(output_dir, "nursery_rhymes")
    os.makedirs(rhymes_dir, exist_ok=True)

    print(f"\n--- Generating Nursery Rhymes in Your Voice ---\n")

    for rhyme in rhymes_category["rhymes"]:
        title = rhyme["title"]
        safe_title = title.replace(" ", "_").lower()
        rhyme_dir = os.path.join(rhymes_dir, safe_title)
        os.makedirs(rhyme_dir, exist_ok=True)

        print(f"  Rhyme: {rhyme['title_telugu']} ({title})")
        print(f"  ({rhyme['english_meaning']})\n")

        # Generate audio for each verse
        for verse in rhyme["verses"]:
            verse_num = verse["verse_number"]
            telugu_text = verse["telugu"]

            print(f"    Generating verse {verse_num}...")
            audio_iter = client.text_to_speech.convert(
                voice_id=voice_id,
                text=telugu_text,
                model_id=MODEL_ID,
                output_format="mp3_44100_128",
            )

            filepath = os.path.join(rhyme_dir, f"verse_{verse_num}.mp3")
            with open(filepath, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)

            print(f"    Verse {verse_num} saved: {filepath}")

        # Generate full rhyme as a single audio file
        full_telugu = "\n\n".join(v["telugu"] for v in rhyme["verses"])
        print(f"\n    Generating full rhyme...")
        audio_iter = client.text_to_speech.convert(
            voice_id=voice_id,
            text=full_telugu,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
        )

        full_path = os.path.join(rhyme_dir, "full_rhyme.mp3")
        with open(full_path, "wb") as f:
            for chunk in audio_iter:
                f.write(chunk)

        print(f"    Full rhyme saved: {full_path}\n")

    print(f"Nursery rhymes generated in '{rhymes_dir}/'!")


def print_summary(word_bank: dict, output_dir: str = OUTPUT_DIR):
    """Print a summary of all generated audio files."""
    print("\n" + "=" * 60)
    print("GENERATED TELUGU AUDIO FILES")
    print("=" * 60)

    for cat_key, category in word_bank["categories"].items():
        if cat_key == "nursery_rhymes":
            continue
        print(f"\n  {category['label']}:")
        cat_dir = os.path.join(output_dir, cat_key)
        for word_entry in category["words"]:
            telugu = word_entry["telugu"]
            romanized = word_entry["romanized"]
            english = word_entry["english"]
            safe_name = romanized.replace(" ", "_").replace("/", "_").lower()
            filepath = os.path.join(cat_dir, f"{safe_name}.mp3")
            exists = "OK" if os.path.exists(filepath) else "MISSING"
            print(f"    [{exists}] {telugu}  ({romanized}) — {english}")
            print(f"           File: {filepath}")

    # Nursery rhymes summary
    if "nursery_rhymes" in word_bank["categories"]:
        print(f"\n  Nursery Rhymes:")
        for rhyme in word_bank["categories"]["nursery_rhymes"]["rhymes"]:
            title = rhyme["title"]
            safe_title = title.replace(" ", "_").lower()
            rhyme_dir = os.path.join(output_dir, "nursery_rhymes", safe_title)

            full_path = os.path.join(rhyme_dir, "full_rhyme.mp3")
            exists = "OK" if os.path.exists(full_path) else "MISSING"
            print(f"    [{exists}] {rhyme['title_telugu']}  ({title})")
            print(f"           Full: {full_path}")

            for verse in rhyme["verses"]:
                v_path = os.path.join(rhyme_dir, f"verse_{verse['verse_number']}.mp3")
                v_exists = "OK" if os.path.exists(v_path) else "MISSING"
                print(f"    [{v_exists}]   Verse {verse['verse_number']}: {v_path}")

    print("\n" + "=" * 60)


def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: Set ELEVENLABS_API_KEY environment variable.")
        print("  export ELEVENLABS_API_KEY='your-api-key-here'")
        return

    client = ElevenLabs(api_key=api_key)

    # Step 1: Record voice
    audio_path = record_voice_sample()

    try:
        # Step 2: Clone voice
        voice_id = clone_voice(client, audio_path)

        # Step 3: Load word bank
        word_bank = load_word_bank()

        # Step 4: Generate audio for all words
        generate_word_audio(client, voice_id, word_bank)

        # Step 5: Print summary
        print_summary(word_bank)

    finally:
        # Clean up temp recording
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    main()
