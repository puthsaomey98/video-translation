import os
import subprocess
from pathlib import Path

# Version-safe MoviePy import
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

def extract_and_separate_audio(video_path: str, output_dir: str = "./temp_separated") -> dict:
    """
    Extracts audio from video and uses Meta's Demucs to split it into 
    background music ('no_vocals.wav') and isolated speech ('vocals.wav').
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "extracted_source_audio.mp3")
    
    print("1. Extracting raw audio track from video...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='mp3', logger=None)
    clip.close()
    
    print("2. Separating background music/effects from speech using Meta Demucs...")
    cmd = [
        "python", "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", output_dir,
        audio_path
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("❌ Demucs background/vocal separation failed.")
        
    filename_stem = Path(audio_path).stem
    stem_dir = Path(output_dir) / "htdemucs" / filename_stem
    
    vocals_path = str(stem_dir / "vocals.wav")
    no_vocals_path = str(stem_dir / "no_vocals.wav")
    
    if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
        raise FileNotFoundError("❌ Demucs failed to generate the audio tracks.")
        
    print("✅ Successfully separated background music and speech stems!")
    return {
        "vocals": vocals_path,
        "no_vocals": no_vocals_path
    }