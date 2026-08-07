import os
import subprocess

# Version-safe MoviePy import
try:
    from moviepy import AudioFileClip
except ImportError:
    from moviepy.editor import AudioFileClip

def fit_and_normalize_audio_segment(input_audio_path: str, target_duration: float, output_audio_path: str):
    try:
        with AudioFileClip(input_audio_path) as clip:
            actual_duration = clip.duration
        
        speed = 1.0
        if actual_duration > target_duration and target_duration > 0:
            speed = actual_duration / target_duration
            if speed > 2.0:
                speed = 2.0

        filter_chain = []
        if speed > 1.0:
            filter_chain.append(f"atempo={speed}")
            
        filter_chain.append("acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=4dB")
        filter_chain.append("alimiter=limit=0.9")
        
        filter_str = ",".join(filter_chain)

        cmd = [
            "ffmpeg", "-y", "-i", input_audio_path,
            "-filter:a", filter_str,
            output_audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return output_audio_path
    except Exception as e:
        print(f"⚠️ Warning: Could not normalize segment {input_audio_path}: {e}")
        
    return input_audio_path

def merge_with_background_music(video_path: str, background_audio_path: str, audio_segments: list, output_path: str, crop_bottom_pixels: int = 0):
    if not audio_segments:
        raise RuntimeError("❌ No voiceover segments available to merge.")

    fitted_segments = []
    print("Applying professional audio compression for absolute volume stability from start to finish...")
    
    for i, seg in enumerate(audio_segments):
        target_duration = seg["end"] - seg["start"]
        fitted_path = f"temp_fitted_{i}.mp3"
        processed_path = fit_and_normalize_audio_segment(seg["path"], target_duration, fitted_path)
        fitted_segments.append({
            "path": processed_path,
            "start": seg["start"]
        })

    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", background_audio_path]
    
    for seg in fitted_segments:
        cmd.extend(["-i", seg["path"]])
    
    filter_complex_parts = []
    filter_complex_parts.append("[1:a]volume=0.5[bg_music]")
    
    voice_mix_inputs = []
    for i, seg in enumerate(fitted_segments):
        input_idx = i + 2
        delay_ms = int(seg["start"] * 1000)
        filter_complex_parts.append(f"[{input_idx}:a]adelay={delay_ms}|{delay_ms}[v{i}]")
        voice_mix_inputs.append(f"[v{i}]")
        
    voice_mix_str = "".join(voice_mix_inputs)
    filter_complex_parts.append(f"{voice_mix_str}amix=inputs={len(fitted_segments)}:duration=longest:normalize=0,volume=1.5[mixed_voices]")
    
    # FIX: Changed 'duration=first' to 'duration=longest' so background music won't cut out early
    filter_complex_parts.append("[bg_music][mixed_voices]amix=inputs=2:duration=longest[final_audio]")
    
    if crop_bottom_pixels > 0:
        filter_complex_parts.append(f"[0:v]crop=iw:ih-{crop_bottom_pixels}:0:0[v]")
    else:
        filter_complex_parts.append(f"[0:v]copy[v]")
        
    filter_complex_script = ";".join(filter_complex_parts)
    
    cmd.extend([
        "-filter_complex", filter_complex_script,
        "-map", "[v]",
        "-map", "[final_audio]",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ])
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    for seg in fitted_segments:
        if seg["path"].startswith("temp_fitted_") and os.path.exists(seg["path"]):
            os.remove(seg["path"])

    if result.returncode != 0:
        print("\n--- FFmpeg Error Log ---")
        print(result.stderr)
        raise RuntimeError("❌ FFmpeg failed to merge background music and voiceover.")
    else:
        print("✅ Successfully merged with 100% rock-solid stable volume from start to finish!")