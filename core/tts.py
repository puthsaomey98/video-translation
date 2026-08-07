import os
import re
import edge_tts

async def generate_segment_audio(text: str, output_path: str, gender: str = "female") -> bool:
    """
    Generates natural Khmer neural voiceover audio.
    Dynamically switches between female (Sreymom) and male (Piseth) voices.
    """
    # Select voice model based on gender
    if str(gender).lower() == "male":
        voice = "km-KH-PisethNeural"
    else:
        voice = "km-KH-SreymomNeural"
    
    clean_text = re.sub(r'[♪♫\[\]\(\)\-\_]+', '', text).strip()
    if not clean_text or not any(char.isalnum() or ('\u1780' <= char <= '\u17FF') for char in clean_text):
        return False

    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"❌ Error generating TTS ({gender}/{voice}) for text '{clean_text}': {e}")
        return False

def format_time(seconds: float) -> str:
    """Converts seconds into SRT timestamp format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"