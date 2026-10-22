import os
import sys
import asyncio
import subprocess
import re
import json
import time
import random
import math
import shutil
import platform
import urllib.request
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    FFMPEG_PATH = "ffmpeg"
    ffmpeg_dir = None

from core.separator import extract_and_separate_audio
from core.stt import transcribe_isolated_vocals
from core.translator import KhmerTranslator
from core.tts import generate_segment_audio, format_time

try:
    from moviepy import AudioFileClip, VideoFileClip, CompositeAudioClip, concatenate_audioclips
except ImportError:
    from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip, concatenate_audioclips

app = FastAPI(title="Khmer Video Dubbing Studio & Audio Mix Engine")

# Speed factor to break temporal Content ID matching (4.5% speedup)
SPEED_FACTOR = 1.045

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("temp_separated", exist_ok=True)
os.makedirs("history", exist_ok=True)
os.makedirs("safe_bgm", exist_ok=True)
os.makedirs("fonts", exist_ok=True)

app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/input", StaticFiles(directory="input"), name="input")
templates = Jinja2Templates(directory="templates")

task_data = {}
download_tasks = {}
DURATION_CACHE = {}

ZH_TYPO_MAP = {
    "进房炮": "近防炮",
}

def get_logo_path() -> Optional[str]:
    possible_names = [
        "logo.png", "logo.jpg", "logo.jpeg", "logo.webp",
        os.path.join("static", "logo.png"), os.path.join("static", "logo.jpg"),
        os.path.join("input", "logo.png"), os.path.join("input", "logo.jpg")
    ]
    for name in possible_names:
        if os.path.exists(name) and os.path.getsize(name) > 0:
            return os.path.abspath(name)

    try:
        for f in os.listdir("."):
            if f.lower().startswith(("logo", "watermark")) and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                return os.path.abspath(f)
    except Exception:
        pass

    return None

def get_random_safe_bgm() -> Optional[str]:
    """Retrieves a random royalty-free audio file from safe_bgm/ to mask Content ID."""
    if not os.path.exists("safe_bgm"):
        return None
    valid_exts = ('.mp3', '.wav', '.ogg', '.m4a', '.aac')
    bgm_files = [os.path.join("safe_bgm", f) for f in os.listdir("safe_bgm") if f.lower().endswith(valid_exts)]
    return random.choice(bgm_files) if bgm_files else None

def extract_sfx_and_remove_bgm(input_audio_path: str, task_id: str) -> str:
    output_sfx_path = os.path.join("temp_separated", f"sfx_only_{task_id}.wav")
    try:
        import librosa
        try:
            import soundfile as sf
        except ImportError:
            sf = None

        y, sr = librosa.load(input_audio_path, sr=24000, mono=True)
        _, y_percussive = librosa.effects.hpss(y, margin=(1.0, 3.0))

        if sf:
            sf.write(output_sfx_path, y_percussive, sr)
        else:
            from scipy.io import wavfile
            wavfile.write(output_sfx_path, sr, (y_percussive * 32767).astype(np.int16))

        if os.path.exists(output_sfx_path) and os.path.getsize(output_sfx_path) > 1000:
            filtered_sfx_path = os.path.join("temp_separated", f"sfx_filtered_{task_id}.wav")
            filter_str = "highpass=f=200,lowpass=f=7200,agate=range=-40dB:threshold=0.08:attack=2:release=90"
            cmd_filter = [FFMPEG_PATH, "-y", "-i", output_sfx_path, "-filter:a", filter_str, filtered_sfx_path]
            res_f = subprocess.run(cmd_filter, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_f.returncode == 0 and os.path.exists(filtered_sfx_path):
                return filtered_sfx_path
            return output_sfx_path
    except Exception:
        pass

    try:
        filter_str = (
            "highpass=f=220,lowpass=f=7500,"
            "agate=range=-40dB:threshold=0.08:attack=2:release=90,"
            "alimiter=limit=0.9"
        )
        cmd = [
            FFMPEG_PATH, "-y", "-i", input_audio_path,
            "-filter:a", filter_str,
            output_sfx_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(output_sfx_path):
            return output_sfx_path
    except Exception:
        pass

    return input_audio_path

def is_khmer_text(text: str) -> bool:
    if not text:
        return False
    khmer_count = sum(1 for c in text if '\u1780' <= c <= '\u17FF')
    return khmer_count >= 3

def preprocess_source_text(text: str, source_lang: str) -> str:
    if not text:
        return ""
    if source_lang == "zh":
        for typo, fix in ZH_TYPO_MAP.items():
            text = text.replace(typo, fix)
    return text.strip()

def sanitize_filename(filename: str) -> str:
    clean_name = re.sub(r'[#\?&%\"\'\/\\]', '_', filename)
    clean_name = re.sub(r'\s+', '_', clean_name)
    clean_name = re.sub(r'_+', '_', clean_name)
    return clean_name.strip('_.')

def save_task_to_disk(task_id: str):
    if task_id in task_data:
        history_path = os.path.join("history", f"{task_id}.json")
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(task_data[task_id], f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"⚠️ Warning: Could not save history for {task_id}: {e}")

def load_task_from_disk(task_id: str):
    if task_id in task_data:
        return task_data[task_id]
    history_path = os.path.join("history", f"{task_id}.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                task_data[task_id] = json.load(f)
                return task_data[task_id]
        except Exception as e:
            print(f"⚠️ Warning: Could not load history for {task_id}: {e}")
    return None

def clean_media_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()

    shorts_match = re.search(r'(?:youtube\.com/shorts/|youtu\.be/)([\w-]+)', url)
    if shorts_match:
        return f"https://www.youtube.com/watch?v={shorts_match.group(1)}"

    if "youtube.com/watch" in url and "v=" in url:
        base_url, _, query = url.partition("?")
        params = query.split("&")
        clean_params = [p for p in params if p.startswith("v=")]
        if clean_params:
            return base_url + "?" + clean_params[0]
        return base_url

    return url

def get_video_dimensions_and_codec(video_path: str) -> dict:
    info = {"width": 0, "height": 0, "codec": "unknown", "is_vertical": False}
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name", "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2.0)
        if result.returncode == 0 and result.stdout.strip():
            parsed = json.loads(result.stdout)
            streams = parsed.get("streams", [])
            if streams:
                s = streams[0]
                w = int(s.get("width", 0))
                h = int(s.get("height", 0))
                info["width"] = w
                info["height"] = h
                info["codec"] = str(s.get("codec_name", "unknown")).lower()
                info["is_vertical"] = h > w
    except Exception:
        pass
    return info

def get_audio_duration_fast(audio_path: str) -> float:
    if not audio_path or not os.path.exists(audio_path):
        return 0.0
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2.0)
        if res.returncode == 0 and res.stdout.strip():
            val = res.stdout.strip()
            if val != "N/A":
                return float(val)
    except Exception:
        pass
    return 0.0

def get_video_duration(video_path: str) -> float:
    if not video_path or video_path.lower().endswith(('.srt', '.vtt', '.txt', '.json')):
        return 0.0

    try:
        mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else 0
        filename = os.path.basename(video_path)

        if filename in DURATION_CACHE and DURATION_CACHE[filename]["mtime"] == mtime:
            return DURATION_CACHE[filename]["duration"]

        duration = 0.0
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2.0)
        if result.returncode == 0 and result.stdout.strip():
            val = result.stdout.strip()
            if val != "N/A":
                duration = float(val)

        DURATION_CACHE[filename] = {"mtime": mtime, "duration": duration}
        return duration
    except Exception:
        return 0.0

def clean_khmer_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'<.*?>', '', text)

    spoken_rules = [
        (r'\bអញ\b', 'ខ្ញុំ'),
        (r'ត្រូវបាន\s*', ''),
        (r'\s+នៃ\s+', ' '),
        (r'មិនមែនទេ', 'អត់ទេ'),
        (r'យ៉ាងណាក៏ដោយ', 'តែ'),
        (r'លើសពីនេះទៅទៀត', 'ហើយ'),
        (r'ជាមួយគ្នានេះដែរ', 'ហើយ'),
        (r'រូបលោក', 'គាត់'),
        (r'លោកអ្នក', 'អ្នក'),
        (r'តើ\s+(.*?)\s+ឬទេ\?', r'\1 មែនទេ?'),
        (r'តើ\s+', ''),
        (r'សូមជម្រាបថា\s*', ''),
    ]

    for pattern, replacement in spoken_rules:
        text = re.sub(pattern, replacement, text)

    return re.sub(r'\s+', ' ', text).strip()

def safe_translate_and_clean(translator, text: str, source_lang: str) -> str:
    if not text or not text.strip():
        return ""

    # If already Khmer or source_lang is 'km', do not translate through Google Translate
    if source_lang == "km" or is_khmer_text(text):
        return clean_khmer_text(text)

    clean_source = preprocess_source_text(text, source_lang)
    try:
        raw_translation = translator.translate_text(clean_source, source_lang=source_lang)
        if raw_translation:
            cleaned_khmer = clean_khmer_text(raw_translation)
            if cleaned_khmer.strip():
                return cleaned_khmer
    except Exception as e:
        print(f"⚠️ Translation fallback triggered: {e}")

    return clean_source if clean_source.strip() else text.strip()

def process_pasted_script_to_srt(pasted_text: str, video_path: str, output_srt_path: str) -> bool:
    if not pasted_text or not pasted_text.strip():
        return False

    if "-->" in pasted_text:
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(pasted_text.strip())
        return True

    lines = [line.strip() for line in pasted_text.strip().split("\n") if line.strip()]
    if not lines:
        return False

    video_duration = get_video_duration(video_path)
    if video_duration <= 0:
        video_duration = len(lines) * 3.0

    time_per_line = max(1.5, video_duration / len(lines))
    srt_lines = []

    for idx, line in enumerate(lines, start=1):
        start_sec = (idx - 1) * time_per_line
        end_sec = min(video_duration, start_sec + time_per_line - 0.2)

        start_fmt = format_time(start_sec)
        end_fmt = format_time(end_sec)

        srt_lines.append(f"{idx}\n{start_fmt} --> {end_fmt}\n{clean_khmer_text(line)}\n")

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    return True

def detect_segment_gender(vocals_path: Optional[str], start_sec: float, end_sec: float) -> str:
    if not vocals_path or not os.path.exists(vocals_path):
        return "female"

    try:
        import librosa
        duration = max(0.4, end_sec - start_sec)
        y, sr = librosa.load(vocals_path, sr=16000, offset=start_sec, duration=duration)

        if len(y) == 0 or np.max(np.abs(y)) < 0.01:
            return "female"

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 0:
            median_f0 = float(np.median(voiced_f0))
            return "female" if median_f0 > 145.0 else "male"
    except Exception:
        pass

    return "female"

def fit_and_normalize_audio_segment(input_audio_path: str, target_duration: float, output_audio_path: str):
    try:
        actual_duration = get_audio_duration_fast(input_audio_path)
        speed = 1.0
        target_duration = max(0.1, target_duration)
        if actual_duration > 0:
            speed = actual_duration / target_duration
            if speed < 0.85:
                speed = 0.85
            elif speed > 1.5:
                speed = 1.5

        filter_chain = []
        if abs(speed - 1.0) > 0.03:
            filter_chain.append(f"atempo={speed}")

        filter_chain.append("acompressor=threshold=-16dB:ratio=2.5:attack=10:release=100:makeup=2dB")
        filter_chain.append("alimiter=limit=0.95")
        filter_str = ",".join(filter_chain)

        cmd = [
            FFMPEG_PATH, "-y", "-i", input_audio_path,
            "-filter:a", filter_str,
            output_audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and os.path.exists(output_audio_path):
            return output_audio_path
    except Exception:
        pass

    return input_audio_path

def apply_clip_start(clip, start_time):
    if hasattr(clip, "with_start"):
        return clip.with_start(start_time)
    return clip.set_start(start_time)

def apply_clip_volume(clip, factor):
    if hasattr(clip, "volumex"):
        return clip.volumex(factor)
    elif hasattr(clip, "with_volume_scaling"):
        return clip.with_volume_scaling(factor)
    return clip

def apply_clip_duration(clip, duration):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)

async def batch_generate_tts_segments(
    task_id: str,
    segments: list,
    task_data_ref: dict,
    base_progress: int = 60,
    progress_range: int = 20
) -> list:
    """Generates TTS audio segments reliably by queueing requests to avoid Edge-TTS WebSocket drops."""
    semaphore = asyncio.Semaphore(1)  # Single connection prevents 'No audio was received'
    total = len(segments)
    completed_count = 0
    audio_segments = []

    async def process_seg(idx, seg):
        nonlocal completed_count
        text = clean_khmer_text(seg.get("translated_text", ""))
        gender = seg.get("gender", "male")
        if not text.strip():
            return None

        seg_audio_path = f"temp_seg_{task_id}_{idx}.mp3"
        async with semaphore:
            await asyncio.sleep(0.15)  # Safe delay between Edge-TTS WebSocket handshakes
            success = await generate_segment_audio(text, seg_audio_path, gender=gender)

        completed_count += 1
        pct = base_progress + int((completed_count / max(1, total)) * progress_range)
        task_data_ref["status"] = {
            "status": "processing",
            "step": f"Generating Khmer Voiceover ({gender.upper()}) {completed_count} of {total}...",
            "progress": pct
        }
        save_task_to_disk(task_id)

        if success and os.path.exists(seg_audio_path):
            return {
                "path": seg_audio_path,
                "start": seg["start"],
                "end": seg["end"]
            }
        return None

    tasks = [process_seg(idx, seg) for idx, seg in enumerate(segments)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]
def merge_with_realtime_progress(
    task_id: str,
    video_path: str,
    background_audio_path: Optional[str],
    audio_segments: list,
    output_path: str,
    bg_volume: float = 0.35,
    voice_volume: float = 1.8,
    audio_mode: str = "bgm_sfx_voice",
    add_watermark: bool = False,
    blur_subtitles: bool = False
):
    fitted_temp_files = []
    temp_mixed_audio = f"temp_mixed_audio_{task_id}.wav"
    audio_clips = []

    speed_factor = globals().get("SPEED_FACTOR", 1.045)
    pts_factor = 1.0 / speed_factor

    try:
        task_data[task_id]["status"] = {
            "status": "processing",
            "step": "Mixing requested audio tracks (Voice/SFX/BGM)...",
            "progress": 82
        }
        save_task_to_disk(task_id)

        total_duration = get_video_duration(video_path)
        if total_duration <= 0:
            with VideoFileClip(video_path) as v_test:
                total_duration = v_test.duration

        mode = str(audio_mode).lower().strip()

        # Exact 4 Audio Combinations:
        # 1. bgm_sfx_voice : Music (BGM) + Sound Effect (SFX) + Khmer Voice
        # 2. sfx_voice     : Sound Effect (SFX) + Khmer Voice
        # 3. bgm_voice     : Music (BGM) + Khmer Voice
        # 4. voice_only    : Only Khmer Voice
        include_sfx = mode in ["bgm_sfx_voice", "sfx_voice", "both", "all", "bg_and_voice", "bg_only"]
        include_bgm = mode in ["bgm_sfx_voice", "bgm_voice", "both", "all"]
        include_voice = mode in ["bgm_sfx_voice", "sfx_voice", "bgm_voice", "voice_only", "both", "bg_and_voice"]

        effective_sfx_vol = bg_volume if bg_volume > 0.0 else 0.35
        effective_voice_vol = voice_volume if voice_volume > 0.0 else 1.8
        effective_bgm_vol = 0.18

        # 1. Background Movie Sound Effects (SFX)
        if include_sfx and background_audio_path and os.path.exists(background_audio_path):
            try:
                sfx_clip = AudioFileClip(background_audio_path)
                sfx_clip = apply_clip_volume(sfx_clip, effective_sfx_vol)
                sfx_clip = apply_clip_duration(sfx_clip, total_duration)
                audio_clips.append(sfx_clip)
            except Exception as e:
                print(f"⚠️ Warning loading SFX clip: {e}")

        # 2. Royalty-Free Safe BGM (Music)
        if include_bgm:
            safe_bgm_track = get_random_safe_bgm()
            if safe_bgm_track and os.path.exists(safe_bgm_track):
                try:
                    bgm_clip = AudioFileClip(safe_bgm_track)
                    if bgm_clip.duration < total_duration:
                        repeats = int(math.ceil(total_duration / max(0.1, bgm_clip.duration)))
                        bgm_clip = concatenate_audioclips([bgm_clip] * repeats)
                    bgm_clip = apply_clip_duration(bgm_clip, total_duration)
                    bgm_clip = apply_clip_volume(bgm_clip, effective_bgm_vol)
                    audio_clips.append(bgm_clip)
                except Exception as e:
                    print(f"⚠️ Safe BGM loop notice: {e}")

        # 3. Khmer Speaking Voice Segments
        if include_voice and audio_segments:
            for i, seg in enumerate(audio_segments):
                target_duration = seg["end"] - seg["start"]
                fitted_path = f"temp_rerender_fitted_{task_id}_{i}.mp3"
                processed_path = fit_and_normalize_audio_segment(seg["path"], target_duration, fitted_path)

                if processed_path == fitted_path:
                    fitted_temp_files.append(fitted_path)

                if os.path.exists(processed_path):
                    v_clip = AudioFileClip(processed_path)
                    v_clip = apply_clip_start(v_clip, seg["start"])
                    v_clip = apply_clip_volume(v_clip, effective_voice_vol)
                    audio_clips.append(v_clip)

        if not audio_clips:
            if background_audio_path and os.path.exists(background_audio_path):
                audio_clips.append(apply_clip_duration(AudioFileClip(background_audio_path), total_duration))
            else:
                from core.tts import create_silent_audio_fallback
                dummy_silent = f"temp_silent_{task_id}.mp3"
                create_silent_audio_fallback(dummy_silent, duration=total_duration)
                fitted_temp_files.append(dummy_silent)
                audio_clips.append(apply_clip_duration(AudioFileClip(dummy_silent), total_duration))

        final_audio = apply_clip_duration(CompositeAudioClip(audio_clips), total_duration)

        task_data[task_id]["status"] = {
            "status": "processing",
            "step": "Exporting master soundtrack...",
            "progress": 88
        }
        save_task_to_disk(task_id)

        final_audio.write_audiofile(
            temp_mixed_audio,
            fps=48000,
            nbytes=2,
            codec='pcm_s16le',
            logger=None
        )

        final_audio.close()
        for clip in audio_clips:
            try:
                clip.close()
            except Exception:
                pass

        # 4. Video Rendering (NO hflip / NO Reversing, Optimized Color Grading)
        task_data[task_id]["status"] = {
            "status": "processing",
            "step": "Rendering Fullscreen Video & Muxing Master Audio...",
            "progress": 94
        }
        save_task_to_disk(task_id)

        v_meta = get_video_dimensions_and_codec(video_path)
        is_vertical = v_meta.get("is_vertical", False)
        vw = v_meta.get("width", 0)
        vh = v_meta.get("height", 0)

        if vw <= 0 or vh <= 0:
            vw, vh = (720, 1280) if is_vertical else (1280, 720)

        vw = (int(vw) // 2) * 2
        vh = (int(vh) // 2) * 2

        # Fast filter without heavy noise and WITHOUT hflip (preserves original video direction)
        filter_parts = [
            f"[0:v]scale={vw}:{vh},"
            f"eq=contrast=1.06:brightness=0.01:saturation=1.08:gamma=0.99,"
            f"setpts={pts_factor:.6f}*PTS[v_base]"
        ]
        current_v = "[v_base]"

        # Optional subtitle blur box at the lower 14% of the frame
        if blur_subtitles:
            blur_h = max(20, int(vh * 0.14))
            blur_y = max(0, int(vh * 0.82))
            filter_parts.append(
                f"{current_v}split[main][crop_src];"
                f"[crop_src]crop={vw}:{blur_h}:0:{blur_y},boxblur=10:5[blurred_sub];"
                f"[main][blurred_sub]overlay=0:{blur_y}[v_blurred]"
            )
            current_v = "[v_blurred]"

        # Optional Watermark Logo
        logo_w = 80
        logo_path = get_logo_path()
        has_logo = bool(add_watermark and logo_path and os.path.exists(logo_path))

        if has_logo:
            filter_parts.append(
                f"[2:v]scale={logo_w}:{logo_w},format=rgba,"
                f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(hypot(X-W/2,Y-H/2),W/2-1),255,0)',"
                f"colorchannelmixer=aa=0.18[wm]"
            )
            filter_parts.append(
                f"{current_v}[wm]overlay=x='-w+(mod(t\\,20)/7)*(W+w)':y='(H-h)/2':enable='lt(mod(t\\,20)\\,7)':eval=frame[v]"
            )
        else:
            filter_parts.append(f"{current_v}null[v]")

        filter_parts.append(f"[1:a:0]atempo={speed_factor}[a]")
        filter_complex_str = ";".join(filter_parts)

        cmd = [
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-i", temp_mixed_audio
        ]
        if has_logo:
            cmd.extend(["-loop", "1", "-i", logo_path])

        cmd.extend([
            "-filter_complex", filter_complex_str,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map_metadata", "-1",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ])

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not os.path.exists(output_path):
            error_details = result.stderr[-600:] if result.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg render failed: {error_details}")

    except Exception as e:
        raise RuntimeError(f"❌ Video compilation failed: {str(e)}")
    finally:
        if os.path.exists(temp_mixed_audio):
            try:
                os.remove(temp_mixed_audio)
            except Exception:
                pass
        for tmp_f in fitted_temp_files:
            if os.path.exists(tmp_f):
                try:
                    os.remove(tmp_f)
                except Exception:
                    pass

def parse_srt(srt_path: str) -> list:
    segments = []
    if not os.path.exists(srt_path):
        return segments
    try:
        with open(srt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            content = f.read()

        if content.startswith("WEBVTT"):
            content = content.split("\n\n", 1)[1] if "\n\n" in content else content

        blocks = content.strip().split("\n\n")
        raw_segments = []

        for block in blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            time_line_idx = -1
            for idx, line in enumerate(lines):
                if "-->" in line:
                    time_line_idx = idx
                    break

            if time_line_idx != -1 and len(lines) > time_line_idx:
                time_line = lines[time_line_idx]
                parts = time_line.split("-->")
                start_str = parts[0].strip().replace(',', '.')
                end_str = parts[1].strip().split()[0].replace(',', '.')

                def time_to_sec(t_str):
                    t_parts = t_str.split(':')
                    try:
                        if len(t_parts) == 3:
                            return float(t_parts[0]) * 3600 + float(t_parts[1]) * 60 + float(t_parts[2])
                        elif len(t_parts) == 2:
                            return float(t_parts[0]) * 60 + float(t_parts[1])
                    except ValueError:
                        return 0.0
                    return 0.0

                start_sec = time_to_sec(start_str)
                end_sec = time_to_sec(end_str)
                raw_text = " ".join(lines[time_line_idx + 1:])
                text = clean_khmer_text(raw_text)

                if text.strip() and end_sec > start_sec:
                    raw_segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text
                    })

        raw_segments.sort(key=lambda x: x["start"])
        for i in range(len(raw_segments)):
            current = raw_segments[i]
            if i < len(raw_segments) - 1:
                next_seg = raw_segments[i + 1]
                if current["end"] > next_seg["start"]:
                    current["end"] = max(current["start"] + 0.1, next_seg["start"] - 0.03)
            segments.append(current)
    except Exception as e:
        print(f"⚠️ Error parsing subtitle file {srt_path}: {e}")
    return segments

def sanitize_segments(segments: list) -> list:
    if not segments:
        return segments
    segments.sort(key=lambda x: x["start"])
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i + 1]
        if current_seg["end"] > next_seg["start"]:
            current_seg["end"] = max(current_seg["start"] + 0.1, next_seg["start"] - 0.03)
    return segments

def run_dubbing_pipeline(
    task_id: str,
    video_path: str,
    source_lang: str,
    output_path: str,
    filename: str,
    subtitle_filename: Optional[str] = None,
    output_mode: str = "bgm_sfx_voice",
    voice_mode: str = "auto",
    add_watermark: bool = False,
    blur_subtitles: bool = False
):
    try:
        translator = KhmerTranslator()
        mode = str(output_mode).lower().strip()
        needs_sfx = mode in ["bgm_sfx_voice", "sfx_voice", "both", "all", "bg_only"]
        needs_stt = not subtitle_filename

        vocals_path = None
        background_audio_path = None

        # PERFORMANCE BOOST: Skip Demucs if no SFX and no STT are needed
        if needs_sfx or needs_stt:
            task_data[task_id]["status"] = {"status": "processing", "step": "Separating audio tracks...", "progress": 15}
            save_task_to_disk(task_id)

            stems = extract_and_separate_audio(video_path, output_dir="./temp_separated")
            vocals_path = stems["vocals"]
            raw_background_path = stems["no_vocals"]

            if needs_sfx:
                task_data[task_id]["status"] = {"status": "processing", "step": "Isolating movie sound effects (SFX)...", "progress": 25}
                save_task_to_disk(task_id)
                background_audio_path = extract_sfx_and_remove_bgm(raw_background_path, task_id)
                public_bg_audio_name = f"sfx_track_{task_id}.wav"
                public_bg_audio_path = os.path.join("output", public_bg_audio_name)
                if os.path.exists(background_audio_path):
                    shutil.copy(background_audio_path, public_bg_audio_path)
                    task_data[task_id]["public_bg_audio"] = f"/output/{public_bg_audio_name}"
        else:
            task_data[task_id]["status"] = {"status": "processing", "step": "Direct Dubbing Mode: Skipping Demucs separation...", "progress": 25}
            save_task_to_disk(task_id)

        task_data[task_id]["video_path"] = video_path
        task_data[task_id]["background_audio_path"] = background_audio_path
        task_data[task_id]["output_path"] = output_path
        task_data[task_id]["audio_mode"] = output_mode
        task_data[task_id]["add_watermark"] = add_watermark
        task_data[task_id]["blur_subtitles"] = blur_subtitles

        raw_segments = []
        if subtitle_filename:
            srt_full_path = os.path.join("input", subtitle_filename)
            task_data[task_id]["status"] = {"status": "processing", "step": "Reading Khmer subtitle script...", "progress": 30}
            save_task_to_disk(task_id)
            raw_segments = parse_srt(srt_full_path)
            if not raw_segments:
                raise ValueError(f"Could not parse valid subtitle segments from: {subtitle_filename}")
        else:
            task_data[task_id]["status"] = {"status": "processing", "step": "Running Speech-to-Text on isolated vocals...", "progress": 30}
            save_task_to_disk(task_id)
            raw_segments = transcribe_isolated_vocals(vocals_path, source_lang=source_lang)
            if not raw_segments:
                raise ValueError("No speech segments detected in video audio track.")

        raw_segments = sanitize_segments(raw_segments)

        task_data[task_id]["status"] = {"status": "processing", "step": "Preparing Khmer script & voice settings...", "progress": 45}
        translated_segments = []
        total_segs = len(raw_segments)

        vm = str(voice_mode).lower().strip()
        if "female" in vm:
            fixed_gender = "female"
        elif "male" in vm:
            fixed_gender = "male"
        elif vm == "auto" and vocals_path:
            fixed_gender = None  # Use auto-detection
        else:
            fixed_gender = "male"  # Default fallback

        for idx, seg in enumerate(raw_segments):
            khmer_text = safe_translate_and_clean(translator, seg["text"], source_lang)
            
            # Apply user-selected voice or auto fallback
            if fixed_gender:
                gender = fixed_gender
            else:
                gender = detect_segment_gender(vocals_path, seg["start"], seg["end"])

            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "source_text": seg["text"],
                "translated_text": khmer_text,
                "gender": gender
            })

        task_data[task_id]["segments"] = translated_segments
        khmer_srt_path = output_path.rsplit(".", 1)[0] + ".srt"
        task_data[task_id]["srt_path"] = khmer_srt_path

        srt_lines = []
        for i, seg in enumerate(translated_segments, start=1):
            scaled_start = seg["start"] / SPEED_FACTOR
            scaled_end = seg["end"] / SPEED_FACTOR
            srt_lines.append(f"{i}\n{format_time(scaled_start)} --> {format_time(scaled_end)}\n{seg['translated_text']}\n")

        with open(khmer_srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        if output_mode == "subtitle":
            task_data[task_id]["status"] = {
                "status": "completed",
                "step": "Subtitles Generated Successfully!",
                "progress": 100,
                "output_video": f"/input/{os.path.basename(video_path)}",
                "output_srt": f"/output/{os.path.basename(khmer_srt_path)}"
            }
            task_data[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_task_to_disk(task_id)
            return

        # Parallel Edge-TTS Batch Generation
        task_data[task_id]["status"] = {"status": "processing", "step": "Batch-generating Khmer neural speech...", "progress": 60}
        save_task_to_disk(task_id)

        audio_segments = asyncio.run(
            batch_generate_tts_segments(task_id, translated_segments, task_data[task_id], base_progress=60, progress_range=20)
        )

        sfx_vol = task_data[task_id].get("bg_volume", 0.35)
        voice_vol = task_data[task_id].get("voice_volume", 1.8)

        merge_with_realtime_progress(
            task_id=task_id,
            video_path=video_path,
            background_audio_path=background_audio_path,
            audio_segments=audio_segments,
            output_path=output_path,
            bg_volume=sfx_vol,
            voice_volume=voice_vol,
            audio_mode=output_mode,
            add_watermark=add_watermark,
            blur_subtitles=blur_subtitles
        )

        for seg in audio_segments:
            if os.path.exists(seg["path"]):
                try:
                    os.remove(seg["path"])
                except Exception:
                    pass

        task_data[task_id]["status"] = {
            "status": "completed",
            "step": "Khmer Dubbing Complete!",
            "progress": 100,
            "output_video": f"/output/{os.path.basename(output_path)}",
            "output_srt": f"/output/{os.path.basename(khmer_srt_path)}"
        }
        task_data[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_task_to_disk(task_id)

    except Exception as e:
        task_data[task_id]["status"] = {"status": "failed", "step": str(e), "progress": 0}
        save_task_to_disk(task_id)

@app.post("/api/translate-file")
async def translate_file(background_tasks: BackgroundTasks, request: Request):
    try:
        body = await request.json()
        filename = body.get("filename")
        source_lang = body.get("source_lang", "km")
        subtitle_filename = body.get("subtitle_filename")
        output_mode = body.get("output_mode", "bgm_sfx_voice")
        voice_mode = body.get("voice_mode", "auto")
        pasted_script = body.get("pasted_script")
        add_watermark = bool(body.get("add_watermark", False))
        blur_subtitles = bool(body.get("blur_subtitles", False))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    video_path = os.path.join("input", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found in input folder: {filename}")

    task_id = str(os.urandom(4).hex())
    base_name, _ = os.path.splitext(filename)
    safe_name = sanitize_filename(base_name) + ".mp4"
    output_path = os.path.join("output", f"dubbed_{task_id}_{safe_name}")

    if pasted_script and pasted_script.strip():
        pasted_srt_filename = f"pasted_{task_id}.srt"
        pasted_srt_path = os.path.join("input", pasted_srt_filename)
        if process_pasted_script_to_srt(pasted_script, video_path, pasted_srt_path):
            subtitle_filename = pasted_srt_filename

    task_data[task_id] = {
        "task_id": task_id,
        "filename": filename,
        "status": {"status": "queued", "step": "In queue...", "progress": 0},
        "bg_volume": 0.35,
        "voice_volume": 1.8,
        "audio_mode": output_mode,
        "add_watermark": add_watermark,
        "blur_subtitles": blur_subtitles,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_task_to_disk(task_id)

    background_tasks.add_task(
        run_dubbing_pipeline,
        task_id, video_path, source_lang, output_path, filename,
        subtitle_filename, output_mode, voice_mode, add_watermark, blur_subtitles
    )
    return {"task_id": task_id, "message": "Khmer dubbing started."}

@app.post("/api/re-render/{task_id}")
async def re_render_task(task_id: str, payload: dict):
    tdata = load_task_from_disk(task_id)
    if not tdata:
        return {"success": False, "error": "Task not found"}

    segments = payload.get("segments", [])
    bg_volume = float(payload.get("bg_volume", 0.35))
    voice_volume = float(payload.get("voice_volume", 1.8))
    audio_mode = payload.get("audio_mode", "bgm_sfx_voice")
    add_watermark = bool(payload.get("add_watermark", False))
    blur_subtitles = bool(payload.get("blur_subtitles", False))

    tdata["bg_volume"] = bg_volume
    tdata["voice_volume"] = voice_volume
    tdata["audio_mode"] = audio_mode
    tdata["add_watermark"] = add_watermark
    tdata["blur_subtitles"] = blur_subtitles

    for seg in segments:
        if "translated_text" in seg:
            seg["translated_text"] = clean_khmer_text(seg["translated_text"])

    segments = sanitize_segments(segments)
    tdata["segments"] = segments
    save_task_to_disk(task_id)

    video_path = tdata.get("video_path")
    background_audio_path = tdata.get("background_audio_path")
    output_path = tdata.get("output_path")

    if not video_path or not os.path.exists(video_path):
        return {"success": False, "error": "Original video file not found on disk."}

    try:
        audio_segments = await batch_generate_tts_segments(task_id, segments, tdata, base_progress=40, progress_range=40)

        merge_with_realtime_progress(
            task_id=task_id,
            video_path=video_path,
            background_audio_path=background_audio_path,
            audio_segments=audio_segments,
            output_path=output_path,
            bg_volume=bg_volume,
            voice_volume=voice_volume,
            audio_mode=audio_mode,
            add_watermark=add_watermark,
            blur_subtitles=blur_subtitles
        )

        for seg in audio_segments:
            if os.path.exists(seg["path"]):
                try:
                    os.remove(seg["path"])
                except Exception:
                    pass

        return {
            "success": True,
            "output_video": f"/output/{os.path.basename(output_path)}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/editor/{task_id}", response_class=HTMLResponse)
async def read_editor(request: Request, task_id: str):
    tdata = load_task_from_disk(task_id)
    if not tdata:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    return templates.TemplateResponse(request, "editor.html", {})

@app.get("/api/input-files")
async def get_input_files():
    files = []
    if os.path.exists("input"):
        for f in os.listdir("input"):
            if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm', '.srt', '.vtt')):
                path = os.path.join("input", f)
                is_sub = f.lower().endswith(('.srt', '.vtt'))
                duration = 0.0 if is_sub else get_video_duration(path)
                meta = get_video_dimensions_and_codec(path) if not is_sub else {}
                files.append({
                    "filename": f,
                    "duration": duration,
                    "size": os.path.getsize(path) if os.path.exists(path) else 0,
                    "is_subtitle": is_sub,
                    "width": meta.get("width", 0),
                    "height": meta.get("height", 0),
                    "is_vertical": meta.get("is_vertical", False)
                })
    return files

@app.get("/api/input-subtitles")
async def get_input_subtitles():
    subtitles = []
    if os.path.exists("input"):
        for f in os.listdir("input"):
            if f.lower().endswith(('.srt', '.vtt')):
                subtitles.append(f)
    return subtitles

class TrimClipRequest(BaseModel):
    filename: str
    start: float
    end: float

@app.post("/api/trim-clip")
async def trim_clip(req: TrimClipRequest):
    src_path = os.path.join("input", req.filename)
    if not os.path.exists(src_path):
        raise HTTPException(status_code=404, detail="Source file not found.")

    clip_id = str(os.urandom(4).hex())
    base_name, ext = os.path.splitext(req.filename)
    clean_base = sanitize_filename(base_name)
    clip_filename = f"clip_{clip_id}_{clean_base}{ext}"
    clip_path = os.path.join("input", clip_filename)

    duration = req.end - req.start
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid start and end times.")

    cmd = [
        FFMPEG_PATH, "-y", "-ss", str(req.start), "-i", src_path,
        "-t", str(duration), "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", clip_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail="FFmpeg failed to trim video clip.")

    return {
        "success": True,
        "clip_filename": clip_filename,
        "duration": duration
    }

@app.post("/api/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    temp_id = str(os.urandom(4).hex())
    clean_upload_name = sanitize_filename(file.filename)
    filename = f"{temp_id}_{clean_upload_name}"
    input_path = os.path.join("input", filename)

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    duration = get_video_duration(input_path) if file.filename.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')) else 0.0
    return {
        "success": True,
        "temp_filename": filename,
        "original_name": file.filename,
        "duration": duration
    }

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    tdata = load_task_from_disk(task_id)
    if not tdata:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    return tdata["status"]

@app.get("/api/history")
async def get_history():
    history_list = []
    if os.path.exists("history"):
        for filename in os.listdir("history"):
            if filename.endswith(".json"):
                path = os.path.join("history", filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        history_list.append({
                            "task_id": data.get("task_id", filename.replace(".json", "")),
                            "filename": data.get("filename", "Unknown Video"),
                            "status": data.get("status", {}).get("status", "unknown"),
                            "updated_at": data.get("updated_at", "Recent"),
                            "output_video": f"/output/{os.path.basename(data.get('output_path', ''))}"
                        })
                except Exception:
                    pass
    return sorted(history_list, key=lambda x: x.get("updated_at", ""), reverse=True)

@app.delete("/api/history/{task_id}")
async def delete_history_item(task_id: str):
    history_path = os.path.join("history", f"{task_id}.json")
    deleted = False

    if os.path.exists(history_path):
        try:
            os.remove(history_path)
            deleted = True
        except Exception:
            pass

    if task_id in task_data:
        del task_data[task_id]
        deleted = True

    if deleted:
        return {"success": True, "message": "History deleted successfully."}
    raise HTTPException(status_code=404, detail="History item not found.")

@app.get("/api/editor-data/{task_id}")
async def get_editor_data(task_id: str):
    tdata = load_task_from_disk(task_id)
    if not tdata:
        return {"error": "Task not found"}

    out_path = tdata.get('output_path', '')
    out_filename = os.path.basename(out_path) if out_path else ''

    return {
        "filename": tdata.get("filename", "Project Workspace"),
        "segments": tdata.get("segments", []),
        "bg_volume": tdata.get("bg_volume", 0.35),
        "voice_volume": tdata.get("voice_volume", 1.8),
        "audio_mode": tdata.get("audio_mode", "bgm_sfx_voice"),
        "add_watermark": tdata.get("add_watermark", False),
        "blur_subtitles": tdata.get("blur_subtitles", False),
        "output_video": f"/output/{out_filename}",
        "background_audio": tdata.get("public_bg_audio")
    }

class SegmentItem(BaseModel):
    start: float
    end: float
    source_text: str = ""
    translated_text: str
    gender: str = "female"

@app.post("/api/test-segment/{task_id}/{index}")
async def test_single_segment(task_id: str, index: int, req: SegmentItem):
    try:
        seg_audio_path = os.path.join("output", f"preview_{task_id}_{index}.mp3")
        cleaned_text = clean_khmer_text(req.translated_text)
        success = await generate_segment_audio(cleaned_text, seg_audio_path, gender=req.gender)
        if success and os.path.exists(seg_audio_path):
            return {
                "success": True,
                "audio_url": f"/output/preview_{task_id}_{index}.mp3?t={time.time()}"
            }
        return {"success": False, "error": "Failed to generate TTS audio"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/input-files/{filename:path}")
async def delete_input_file(filename: str):
    file_path = os.path.join("input", filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            if filename in DURATION_CACHE:
                del DURATION_CACHE[filename]
            return {"success": True, "message": f"Deleted {filename} successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not delete file: {e}")
    raise HTTPException(status_code=404, detail="Input file not found.")

class OpenFolderRequest(BaseModel):
    filepath: Optional[str] = None

@app.post("/api/open-folder")
async def open_folder(req: Optional[OpenFolderRequest] = None):
    output_dir = os.path.abspath("output")
    target_path = os.path.abspath(req.filepath) if req and req.filepath and os.path.exists(req.filepath) else output_dir
    try:
        system = platform.system()
        if system == "Windows":
            if os.path.isfile(target_path):
                subprocess.run(f'explorer /select,"{target_path}"')
            else:
                os.startfile(target_path)
        elif system == "Darwin":
            subprocess.run(["open", "-R", target_path] if os.path.isfile(target_path) else ["open", target_path])
        else:
            folder_path = os.path.dirname(target_path) if os.path.isfile(target_path) else target_path
            subprocess.run(["xdg-open", folder_path])
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TranslateSegmentRequest(BaseModel):
    text: str
    source_lang: str = "km"

@app.post("/api/translate-segment")
async def translate_single_segment(req: TranslateSegmentRequest):
    try:
        translator = KhmerTranslator()
        translated_text = safe_translate_and_clean(translator, req.text, req.source_lang)
        return {"success": True, "translated_text": translated_text}
    except Exception as e:
        return {"success": False, "error": str(e)}

class MergeVideosRequest(BaseModel):
    task_ids: List[str]
    custom_title: Optional[str] = "full_merged_movie"

@app.post("/api/merge-videos")
async def merge_completed_videos(req: MergeVideosRequest):
    if not req.task_ids or len(req.task_ids) < 2:
        raise HTTPException(status_code=400, detail="Please select at least 2 videos to merge.")

    resolved_video_paths = []
    for tid in req.task_ids:
        tdata = load_task_from_disk(tid)
        if not tdata or not tdata.get("output_path") or not os.path.exists(tdata.get("output_path")):
            raise HTTPException(status_code=404, detail=f"Video file missing on disk for task {tid}")
        resolved_video_paths.append(os.path.abspath(tdata["output_path"]))

    merge_id = str(os.urandom(4).hex())
    clean_title = sanitize_filename(req.custom_title)
    merged_filename = f"merged_{merge_id}_{clean_title}.mp4"
    merged_output_path = os.path.join("output", merged_filename)

    concat_txt_path = os.path.join("output", f"temp_concat_{merge_id}.txt")
    with open(concat_txt_path, "w", encoding="utf-8") as f:
        for vp in resolved_video_paths:
            clean_vp = vp.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{clean_vp}'\n")

    try:
        cmd_copy = [
            FFMPEG_PATH, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt_path,
            "-c", "copy",
            "-movflags", "+faststart",
            merged_output_path
        ]
        res = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {res.stderr}")

        return {
            "success": True,
            "merged_task_id": merge_id,
            "output_video": f"/output/{merged_filename}",
            "filename": merged_filename,
            "parts_merged": len(resolved_video_paths)
        }
    finally:
        if os.path.exists(concat_txt_path):
            try:
                os.remove(concat_txt_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)