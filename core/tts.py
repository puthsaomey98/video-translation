import os
import re
import asyncio
import subprocess
from typing import Optional
import edge_tts

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = "ffmpeg"


def create_silent_audio_fallback(output_path: str, duration: float = 1.0) -> bool:
    """Creates a brief silent audio file so the FFmpeg rendering pipeline never crashes."""
    try:
        cmd = [
            FFMPEG_PATH, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(max(0.5, duration)),
            "-c:a", "libmp3lame",
            "-b:a", "64k",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f"⚠️ Failed to create silent fallback audio: {e}")
        return False


def clean_khmer_tts_text(text: str) -> str:
    """
    Cleans invisible Unicode characters, ellipses, and formatting symbols
    that cause Microsoft Edge-TTS km-KH neural models to return 'No audio was received'.
    """
    if not text:
        return ""

    # Strip zero-width spaces, non-joiners, byte-order marks, and non-breaking spaces
    cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', ' ', text)

    # Strip subtitle annotations, music tags, and brackets
    cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned)
    cleaned = re.sub(r'<.*?>', '', cleaned)
    cleaned = re.sub(r'[♪♫\-\_#@*^~]+', ' ', cleaned)

    # Replace multiple dots/ellipses with a standard Khmer pause sign (។)
    cleaned = re.sub(r'\.{2,}|…', ' ។ ', cleaned)

    # Remove problematic quotation marks
    cleaned = re.sub(r'["\'“”«»`]+', '', cleaned)

    # Collapse multiple whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


async def generate_segment_audio(
    text: str,
    output_path: str,
    gender: str = "male",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    retries: int = 3
) -> bool:
    """
    Generates natural Khmer neural voiceover audio using Edge TTS.
    Strictly forces Male (Piseth) or Female (Sreymom).
    """
    is_male = str(gender).strip().lower() in ["male", "force_male", "m"]
    primary_voice = "km-KH-PisethNeural" if is_male else "km-KH-SreymomNeural"
    fallback_voice = "km-KH-SreymomNeural" if is_male else "km-KH-PisethNeural"

    clean_text = clean_khmer_tts_text(text)

    # Verify string contains readable Khmer unicode or alphanumeric characters
    if not clean_text or not any(char.isalnum() or ('\u1780' <= char <= '\u17FF') for char in clean_text):
        return create_silent_audio_fallback(output_path, duration=0.8)

    for attempt in range(1, retries + 1):
        voice_to_use = primary_voice if attempt < retries else fallback_voice

        try:
            if attempt > 1:
                await asyncio.sleep(1.2 * attempt)

            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=voice_to_use,
                rate=rate,
                volume=volume,
                pitch=pitch
            )
            await communicate.save(output_path)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 300:
                return True

        except Exception as e:
            print(f"⚠️ TTS attempt {attempt}/{retries} failed ({voice_to_use}): {e}")

    # Fallback to silent segment to keep the pipeline moving if network times out
    print(f"⚠️ Warning: TTS unavailable for '{clean_text[:25]}...'. Generating silent placeholder audio.")
    return create_silent_audio_fallback(output_path, duration=1.2)


def format_time(seconds: float) -> str:
    """Converts seconds into SRT timestamp format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds >= 1000:
        secs += 1
        milliseconds = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"