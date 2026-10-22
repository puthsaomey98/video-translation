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
    """
    Creates a brief silent audio file using FFmpeg.
    Ensures MoviePy/FFmpeg never crashes with 'No audio tracks selected for mixing'
    if the online TTS service temporarily blocks requests.
    """
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


async def generate_segment_audio(
    text: str,
    output_path: str,
    gender: str = "female",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    retries: int = 3
) -> bool:
    """
    Generates natural Khmer neural voiceover audio using Edge TTS.
    Features exponential backoff, voice fallback, and emergency silent audio generation.
    """
    is_male = str(gender).strip().lower() == "male"
    primary_voice = "km-KH-PisethNeural" if is_male else "km-KH-SreymomNeural"
    fallback_voice = "km-KH-SreymomNeural" if is_male else "km-KH-PisethNeural"

    # Clean punctuation, bracketed annotations, and music symbols
    clean_text = re.sub(r'[♪♫\[\]\(\)\-\_]+', '', text).strip()
    
    # Verify string contains Khmer unicode or alphanumeric characters
    if not clean_text or not any(char.isalnum() or ('\u1780' <= char <= '\u17FF') for char in clean_text):
        return create_silent_audio_fallback(output_path, duration=0.8)

    for attempt in range(1, retries + 1):
        # Alternate to fallback voice on subsequent attempts
        voice_to_use = primary_voice if attempt == 1 else fallback_voice
        
        try:
            # Back off to clear Microsoft WebSocket rate limits
            if attempt > 1:
                await asyncio.sleep(1.0 * attempt)

            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=voice_to_use,
                rate=rate,
                volume=volume,
                pitch=pitch
            )
            await communicate.save(output_path)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 200:
                return True

        except Exception as e:
            print(f"⚠️ TTS attempt {attempt}/{retries} failed ({voice_to_use}): {e}")

    # Emergency safety net: create silence so the video rendering pipeline continues uninterrupted
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