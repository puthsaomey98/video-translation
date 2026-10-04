import os
import sys
import asyncio
import subprocess
import re
import json
import time
import random
import math
import base64
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
from PIL import Image, ImageDraw, ImageFont

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

app = FastAPI(title="Khmer Video Dubbing Studio & Anti-Copyright Engine")

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

def find_headless_browser() -> Optional[str]:
    """Finds Microsoft Edge or Google Chrome to render HarfBuzz Khmer shaping."""
    win_paths = [
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for p in win_paths:
        if os.path.exists(p):
            return p
    for name in ["msedge", "chrome", "google-chrome", "chromium"]:
        w = shutil.which(name)
        if w:
            return w
    return None

def get_font_file_path() -> Optional[str]:
    """Finds the path to KhmerFont.ttf or standard Windows Khmer fonts."""
    candidate_paths = [
        os.path.join("fonts", "KhmerFont.ttf"),
        "KhmerFont.ttf",
        os.path.join(os.getcwd(), "fonts", "KhmerFont.ttf"),
        os.path.join(os.getcwd(), "KhmerFont.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "LeelawUI.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "LeelaUIb.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "KhmerUI.ttf")
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.getsize(p) > 2000:
            return os.path.abspath(p)
    return None

def create_recap_banner(vw: int, vh: int, banner_text: str, output_path: str) -> Optional[str]:
    """
    Renders 100% grammatically correct Khmer script (stacked subscripts & vowels)
    at the top banner on a single line.
    """
    top_bar_h = int(vh * 0.13)
    # Scale font size based on width and height to strictly fit within one line
    font_size = max(18, min(int(top_bar_h * 0.32), int(vw * 0.038)))

    browser_path = find_headless_browser()
    font_path = get_font_file_path()

    font_face_css = ""
    if font_path and os.path.exists(font_path):
        try:
            with open(font_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            font_face_css = f"""
            @font-face {{
                font-family: 'LocalKhmer';
                src: url('data:font/truetype;charset=utf-8;base64,{b64}') format('truetype');
                font-weight: bold;
            }}
            """
        except Exception as e:
            print(f"⚠️ Font embedding warning: {e}")

    # 1. Primary Engine: Headless Browser for Perfect Khmer Typography
    if browser_path:
        temp_html_path = output_path.replace(".png", ".html")
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Kantumruy+Pro:wght@700&family=Noto+Sans+Khmer:wght@700&display=swap');
{font_face_css}
* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}
body {{
    width: {vw}px;
    height: {vh}px;
    background-color: transparent;
    overflow: hidden;
    font-family: 'LocalKhmer', 'Kantumruy Pro', 'Noto Sans Khmer', 'Leelawadee UI', 'Khmer OS Siemreap', 'Khmer OS', sans-serif;
}}
.top-bar {{
    position: absolute;
    top: 0;
    left: 0;
    width: {vw}px;
    height: {top_bar_h}px;
    background-color: #0c0c10;
    border-bottom: 3.5px solid #ffd700;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 0 16px;
}}
.top-title {{
    color: #ffffff;
    font-size: {font_size}px;
    font-weight: bold;
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.95);
    line-height: 1.3;
    white-space: nowrap;
}}
</style>
</head>
<body>
    <div class="top-bar">
        <div class="top-title">{banner_text}</div>
    </div>
</body>
</html>"""
        try:
            with open(temp_html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            file_url = f"file:///{os.path.abspath(temp_html_path).replace(os.sep, '/')}"
            cmd = [
                browser_path,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--window-size={vw},{vh}",
                "--force-device-scale-factor=1",
                "--default-background-color=00000000",
                "--hide-scrollbars",
                f"--screenshot={os.path.abspath(output_path)}",
                file_url
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                with Image.open(output_path) as im:
                    if im.size != (vw, vh):
                        im = im.resize((vw, vh), Image.Resampling.LANCZOS)
                        im.save(output_path, "PNG")
                return output_path
        except Exception as e:
            print(f"⚠️ Headless browser render failed: {e}")
        finally:
            if os.path.exists(temp_html_path):
                try:
                    os.remove(temp_html_path)
                except Exception:
                    pass

    # 2. Secondary Engine: FFmpeg ASS Subtitle (Native HarfBuzz)
    try:
        temp_ass = output_path.replace(".png", ".ass")
        top_font_size = max(18, min(int(vh * 0.025), int(vw * 0.038)))
        top_margin = int(vh * 0.045)

        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {vw}
PlayResY: {vh}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TopStyle,KhmerFont,{top_font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,8,15,15,{top_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,TopStyle,,0,0,0,,{banner_text}
"""
        with open(temp_ass, "w", encoding="utf-8") as f:
            f.write(ass_content)

        clean_ass = os.path.abspath(temp_ass).replace("\\", "/").replace(":", "\\:")
        filter_str = (
            f"drawbox=x=0:y=0:w={vw}:h={top_bar_h}:color=0x0c0c10@1.0:t=fill,"
            f"drawbox=x=0:y={top_bar_h - 3}:w={vw}:h=3:color=0xffd700@1.0:t=fill,"
            f"subtitles={clean_ass}:fontsdir=fonts"
        )
        cmd = [
            FFMPEG_PATH, "-y",
            "-f", "lavfi", "-i", f"color=c=black@0.0:s={vw}x{vh}:d=1",
            "-vf", filter_str,
            "-frames:v", "1",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.exists(temp_ass):
            try:
                os.remove(temp_ass)
            except Exception:
                pass
        if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
    except Exception as e:
        print(f"⚠️ FFmpeg ASS banner fallback error: {e}")

    # 3. Tertiary Fallback: Pillow basic banner box
    try:
        img = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, vw, top_bar_h], fill=(12, 12, 16, 255))
        draw.line([0, top_bar_h - 3, vw, top_bar_h - 3], fill=(255, 215, 0, 240), width=3)
        img.save(output_path, "PNG")
        return output_path
    except Exception:
        return None

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

def get_common_ydl_opts(download_id: Optional[str] = None) -> dict:
    opts = {
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': ffmpeg_dir if ffmpeg_dir else FFMPEG_PATH,
    }

    cookie_paths = [
        os.path.join(os.getcwd(), "cookies.txt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cookies.txt")
    ]
    for cp in cookie_paths:
        if os.path.exists(cp):
            opts['cookiefile'] = cp
            break

    if download_id:
        opts['progress_hooks'] = [lambda d: ydl_progress_hook(d, download_id)]
    return opts

def safe_extract_info(url: str, is_download: bool = False, custom_opts: dict = None, download_id: Optional[str] = None):
    base_opts = get_common_ydl_opts(download_id)
    if custom_opts:
        base_opts.update(custom_opts)

    if not is_download:
        base_opts.pop('format', None)
        base_opts['ignore_no_formats_error'] = True

    is_youtube = "youtube.com" in url or "youtu.be" in url

    if is_youtube:
        strategies = [
            {'extractor_args': {'youtube': {'player_client': ['ios', 'android']}}},
            {'extractor_args': {'youtube': {'player_client': ['web', 'android']}}},
            {'extractor_args': {'youtube': {'player_client': ['mweb', 'ios']}}},
            {'extractor_args': {'youtube': {'player_client': ['tv', 'android_vr']}}},
            {}
        ]
    else:
        strategies = [{}]

    last_err = None
    for strat in strategies:
        try:
            run_opts = dict(base_opts)
            run_opts.update(strat)
            run_opts.pop('ignoreerrors', None)

            with yt_dlp.YoutubeDL(run_opts) as ydl:
                info = ydl.extract_info(url, download=is_download)
                if info is not None:
                    return info, ydl
        except Exception as e:
            last_err = e
            continue

    clean_err = re.sub(r'\x1b\[[0-9;]*m', '', str(last_err)) if last_err else 'Stream unavailable'
    raise RuntimeError(f"Error fetching stream info: {clean_err}")

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
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.5,
            encoding="utf-8", errors="ignore"
        )
        if result.returncode == 0 and result.stdout.strip():
            val = result.stdout.strip()
            if val != "N/A":
                duration = float(val)

        if duration == 0.0 and video_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')):
            try:
                with VideoFileClip(video_path) as clip:
                    duration = clip.duration
            except Exception:
                duration = 0.0

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

    if source_lang == "km":
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

def detect_segment_gender(vocals_path: str, start_sec: float, end_sec: float) -> str:
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
        with AudioFileClip(input_audio_path) as clip:
            actual_duration = clip.duration

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
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="ignore"
        )
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

def merge_with_realtime_progress(
    task_id: str,
    video_path: str,
    background_audio_path: str,
    audio_segments: list,
    output_path: str,
    bg_volume: float = 0.0,
    voice_volume: float = 1.8,
    audio_mode: str = "both",
    add_watermark: bool = False,
    blur_subtitles: bool = False,
    banner_bottom_text: str = "ចុច Follow ដើម្បីទស្សនាភាគបន្ត"
):
    fitted_temp_files = []
    temp_mixed_audio = f"temp_mixed_audio_{task_id}.wav"
    temp_banner_path = f"temp_banner_{task_id}.png"
    audio_clips = []

    speed_factor = globals().get("SPEED_FACTOR", 1.045)
    pts_factor = 1.0 / speed_factor

    try:
        task_data[task_id]["status"] = {
            "status": "processing",
            "step": "Mixing Khmer voiceover, reduced SFX, and Safe BGM...",
            "progress": 82
        }
        save_task_to_disk(task_id)

        total_duration = get_video_duration(video_path)
        if total_duration <= 0:
            with VideoFileClip(video_path) as v_test:
                total_duration = v_test.duration

        # 1. Background Movie SFX
        include_sfx = (audio_mode in ["both", "bg_and_voice", "bg_only"]) and (bg_volume > 0.0)
        if include_sfx and background_audio_path and os.path.exists(background_audio_path):
            try:
                sfx_clip = AudioFileClip(background_audio_path)
                sfx_clip = apply_clip_volume(sfx_clip, bg_volume)
                sfx_clip = apply_clip_duration(sfx_clip, total_duration)
                audio_clips.append(sfx_clip)
            except Exception as e:
                print(f"⚠️ Warning loading SFX clip: {e}")

        # 2. Safe Royalty-Free BGM Layering
        safe_bgm_track = get_random_safe_bgm()
        if safe_bgm_track and os.path.exists(safe_bgm_track):
            try:
                bgm_clip = AudioFileClip(safe_bgm_track)
                if bgm_clip.duration < total_duration:
                    repeats = int(math.ceil(total_duration / max(0.1, bgm_clip.duration)))
                    bgm_clip = concatenate_audioclips([bgm_clip] * repeats)
                bgm_clip = apply_clip_duration(bgm_clip, total_duration)
                bgm_clip = apply_clip_volume(bgm_clip, 0.18)
                audio_clips.append(bgm_clip)
            except Exception as e:
                print(f"⚠️ Safe BGM loop notice: {e}")

        # 3. Khmer Narration Segments
        include_voice = (audio_mode in ["both", "bg_and_voice", "voice_only"]) and (voice_volume > 0.0)
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
                    v_clip = apply_clip_volume(v_clip, voice_volume)
                    audio_clips.append(v_clip)

        if not audio_clips:
            raise RuntimeError("❌ No audio tracks selected for mixing.")

        final_audio = apply_clip_duration(CompositeAudioClip(audio_clips), total_duration)

        task_data[task_id]["status"] = {
            "status": "processing",
            "step": "Exporting master anti-copyright audio track...",
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

        # 4. Anti-Copyright Visual Canvas & Recap Banner Filters
        task_data[task_id]["status"] = {
            "status": "processing",
            "step": "Rendering Title Banner & Anti-Fingerprint Canvas...",
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

        # Generate recap banner with correct single-line Khmer typography
        banner_file = create_recap_banner(vw, vh, banner_bottom_text, temp_banner_path)

        # Subtitle blur bounding box
        if is_vertical:
            sub_w = (int(vw * 0.80) // 2) * 2
            sub_h = (int(vh * 0.068) // 2) * 2
            sub_x = (int((vw - sub_w) / 2) // 2) * 2
            sub_y = (int(vh * 0.70) // 2) * 2
        else:
            sub_w = (int(vw * 0.75) // 2) * 2
            sub_h = (int(vh * 0.075) // 2) * 2
            sub_x = (int((vw - sub_w) / 2) // 2) * 2
            sub_y = (int(vh * 0.81) // 2) * 2

        # Scale centered drama clip to 72% height to leave clear margins for banner
        scale_ratio = 0.72 if is_vertical else 0.75
        fg_w = (int(vw * scale_ratio) // 2) * 2
        fg_h = (int(vh * scale_ratio) // 2) * 2

        fg_crop_w = (int(fg_w * 0.96) // 2) * 2
        fg_crop_h = (int(fg_h * 0.96) // 2) * 2
        fg_crop_x = (int((fg_w - fg_crop_w) / 2) // 2) * 2
        fg_crop_y = (int((fg_h - fg_crop_h) / 2) // 2) * 2

        filter_parts = []

        if blur_subtitles:
            filter_parts.append(f"[0:v]scale={vw}:{vh},split=2[v_base1][v_crop]")
            filter_parts.append(f"[v_crop]crop={sub_w}:{sub_h}:{sub_x}:{sub_y},boxblur=16:2[v_blur]")
            filter_parts.append(f"[v_base1][v_blur]overlay={sub_x}:{sub_y}[v_source]")
        else:
            filter_parts.append(f"[0:v]scale={vw}:{vh}[v_source]")

        # Recap Canvas Generation (Blurred backdrop + Scaled centered drama clip)
        filter_parts.extend([
            f"[v_source]split=2[bg_in][fg_in]",
            f"[bg_in]boxblur=26:5,eq=brightness=-0.25:contrast=1.05,setpts={pts_factor:.6f}*PTS[bg]",
            f"[fg_in]scale={fg_w}:{fg_h},crop={fg_crop_w}:{fg_crop_h}:{fg_crop_x}:{fg_crop_y},"
            f"hflip,eq=contrast=1.12:brightness=0.02:saturation=1.15:gamma=0.98,noise=alls=6:allf=t+u,"
            f"setpts={pts_factor:.6f}*PTS[fg]",
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v_canvas]"
        ])

        input_index_tracker = 2
        if banner_file and os.path.exists(banner_file):
            filter_parts.append(f"[v_canvas][{input_index_tracker}:v]overlay=0:0[v_bannered]")
            current_v = "[v_bannered]"
            input_index_tracker += 1
        else:
            current_v = "[v_canvas]"

        # Logo / Watermark
        logo_w = 80
        logo_path = get_logo_path()
        has_logo = bool(add_watermark and logo_path and os.path.exists(logo_path))

        if has_logo:
            filter_parts.append(
                f"[{input_index_tracker}:v]scale={logo_w}:{logo_w},format=rgba,"
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

        if banner_file and os.path.exists(banner_file):
            cmd.extend(["-i", banner_file])

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
            print(f"⚠️ Primary encode failed, retrying with fallback flags:\n{error_details}")

            fb_cmd = [
                FFMPEG_PATH, "-y",
                "-i", video_path,
                "-i", temp_mixed_audio
            ]
            if banner_file and os.path.exists(banner_file):
                fb_cmd.extend(["-i", banner_file])
            if has_logo:
                fb_cmd.extend(["-loop", "1", "-i", logo_path])

            fb_cmd.extend([
                "-filter_complex", filter_complex_str,
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-map_metadata", "-1",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ])
            fb_res = subprocess.run(fb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if fb_res.returncode != 0:
                fb_error = fb_res.stderr[-600:] if fb_res.stderr else "Unknown fallback error"
                raise RuntimeError(f"FFmpeg muxing failed: {fb_error}")

    except Exception as e:
        raise RuntimeError(f"❌ Video compilation failed: {str(e)}")
    finally:
        if os.path.exists(temp_mixed_audio):
            try:
                os.remove(temp_mixed_audio)
            except Exception:
                pass
        if os.path.exists(temp_banner_path):
            try:
                os.remove(temp_banner_path)
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
                            h, m, s = t_parts
                            return float(h) * 3600 + float(m) * 60 + float(s)
                        elif len(t_parts) == 2:
                            m, s = t_parts
                            return float(m) * 60 + float(s)
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

def split_srt_for_clip(source_srt, output_srt, start_time, end_time):
    if not os.path.exists(source_srt):
        return
    segments = parse_srt(source_srt)
    new_segments = []
    for seg in segments:
        if seg["start"] >= start_time and seg["end"] <= end_time:
            new_segments.append({
                "start": seg["start"] - start_time,
                "end": seg["end"] - start_time,
                "text": seg["text"]
            })
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(new_segments, 1):
            f.write(f"{i}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text']}\n\n")

def run_dubbing_pipeline(
    task_id: str,
    video_path: str,
    source_lang: str,
    output_path: str,
    filename: str,
    subtitle_filename: Optional[str] = None,
    output_mode: str = "both",
    voice_mode: str = "auto",
    add_watermark: bool = False,
    blur_subtitles: bool = False,
    banner_bottom_text: str = "ចុច Follow ដើម្បីទស្សនាភាគបន្ត"
):
    try:
        translator = KhmerTranslator()

        task_data[task_id]["status"] = {"status": "processing", "step": "Separating vocals from background audio...", "progress": 15}
        save_task_to_disk(task_id)

        stems = extract_and_separate_audio(video_path, output_dir="./temp_separated")
        vocals_path = stems["vocals"]
        raw_background_path = stems["no_vocals"]

        task_data[task_id]["status"] = {"status": "processing", "step": "Filtering out background music & isolating SFX...", "progress": 25}
        save_task_to_disk(task_id)

        sfx_isolated_path = extract_sfx_and_remove_bgm(raw_background_path, task_id)
        background_audio_path = sfx_isolated_path

        public_bg_audio_name = f"sfx_track_{task_id}.wav"
        public_bg_audio_path = os.path.join("output", public_bg_audio_name)
        if os.path.exists(background_audio_path):
            import shutil
            shutil.copy(background_audio_path, public_bg_audio_path)
            task_data[task_id]["public_bg_audio"] = f"/output/{public_bg_audio_name}"

        task_data[task_id]["video_path"] = video_path
        task_data[task_id]["background_audio_path"] = background_audio_path
        task_data[task_id]["output_path"] = output_path
        task_data[task_id]["audio_mode"] = output_mode
        task_data[task_id]["add_watermark"] = add_watermark
        task_data[task_id]["blur_subtitles"] = blur_subtitles
        task_data[task_id]["banner_bottom_text"] = banner_bottom_text

        if output_mode == "bg_only":
            task_data[task_id]["status"] = {"status": "processing", "step": "Exporting SFX Only video...", "progress": 75}
            save_task_to_disk(task_id)

            merge_with_realtime_progress(
                task_id=task_id,
                video_path=video_path,
                background_audio_path=background_audio_path,
                audio_segments=[],
                output_path=output_path,
                bg_volume=0.25,
                voice_volume=0.0,
                audio_mode="bg_only",
                add_watermark=add_watermark,
                blur_subtitles=blur_subtitles,
                banner_bottom_text=banner_bottom_text
            )

            task_data[task_id]["status"] = {
                "status": "completed",
                "step": "SFX Track Export Complete!",
                "progress": 100,
                "output_video": f"/output/{os.path.basename(output_path)}",
                "output_srt": ""
            }
            task_data[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_task_to_disk(task_id)
            return

        raw_segments = []
        if subtitle_filename:
            srt_full_path = os.path.join("input", subtitle_filename)
            task_data[task_id]["status"] = {"status": "processing", "step": "Reading provided subtitle file...", "progress": 30}
            save_task_to_disk(task_id)

            raw_segments = parse_srt(srt_full_path)
            if not raw_segments:
                raise ValueError(f"Could not parse valid subtitle segments from: {subtitle_filename}")
        else:
            task_data[task_id]["status"] = {"status": "processing", "step": "Running Speech-to-Text on isolated vocals...", "progress": 30}
            save_task_to_disk(task_id)

            raw_segments = transcribe_isolated_vocals(vocals_path, source_lang=source_lang)
            if not raw_segments:
                raise ValueError("No speech segments detected in the video audio track.")

        raw_segments = sanitize_segments(raw_segments)

        task_data[task_id]["status"] = {"status": "processing", "step": "Preparing Khmer script & voice modes...", "progress": 45}
        translated_segments = []
        total_segs = len(raw_segments)

        for idx, seg in enumerate(raw_segments):
            khmer_text = safe_translate_and_clean(translator, seg["text"], source_lang)

            if voice_mode == "force_male":
                gender = "male"
            elif voice_mode == "force_female":
                gender = "female"
            else:
                gender = detect_segment_gender(vocals_path, seg["start"], seg["end"])

            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "source_text": seg["text"],
                "translated_text": khmer_text,
                "gender": gender
            })

            current_progress = 45 + int(((idx + 1) / max(1, total_segs)) * 15)
            task_data[task_id]["status"] = {
                "status": "processing",
                "step": f"Processed Khmer segment {idx + 1} of {total_segs} ({gender.upper()})...",
                "progress": current_progress
            }
            save_task_to_disk(task_id)

        task_data[task_id]["segments"] = translated_segments

        khmer_srt_path = output_path.rsplit(".", 1)[0] + ".srt"
        task_data[task_id]["srt_path"] = khmer_srt_path

        srt_lines = []
        for i, seg in enumerate(translated_segments, start=1):
            scaled_start = seg["start"] / SPEED_FACTOR
            scaled_end = seg["end"] / SPEED_FACTOR
            start_time = format_time(scaled_start)
            end_time = format_time(scaled_end)
            srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{seg['translated_text']}\n")

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

        audio_segments = []
        for idx, seg in enumerate(translated_segments):
            if seg["translated_text"].strip():
                seg_audio_path = f"temp_seg_{task_id}_{idx}.mp3"
                detected_gender = seg.get("gender", "female")

                success = asyncio.run(generate_segment_audio(
                    seg["translated_text"],
                    seg_audio_path,
                    gender=detected_gender
                ))

                if success and os.path.exists(seg_audio_path):
                    audio_segments.append({
                        "path": seg_audio_path,
                        "start": seg["start"],
                        "end": seg["end"]
                    })

            current_progress = 60 + int(((idx + 1) / max(1, total_segs)) * 20)
            task_data[task_id]["status"] = {
                "status": "processing",
                "step": f"Generating Khmer TTS Voiceover {idx + 1} of {total_segs}...",
                "progress": current_progress
            }
            save_task_to_disk(task_id)

        sfx_vol = 0.0 if output_mode == "voice_only" else task_data[task_id].get("bg_volume", 0.0)
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
            blur_subtitles=blur_subtitles,
            banner_bottom_text=banner_bottom_text
        )

        for seg in audio_segments:
            if os.path.exists(seg["path"]):
                try:
                    os.remove(seg["path"])
                except Exception:
                    pass

        task_data[task_id]["status"] = {
            "status": "completed",
            "step": "Anti-Copyright Dubbing Complete!",
            "progress": 100,
            "output_video": f"/output/{os.path.basename(output_path)}",
            "output_srt": f"/output/{os.path.basename(khmer_srt_path)}"
        }
        task_data[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_task_to_disk(task_id)

    except Exception as e:
        task_data[task_id]["status"] = {"status": "failed", "step": str(e), "progress": 0}
        save_task_to_disk(task_id)

class MergeVideosRequest(BaseModel):
    task_ids: List[str]
    custom_title: Optional[str] = "full_merged_movie"

@app.post("/api/merge-videos")
async def merge_completed_videos(req: MergeVideosRequest):
    if not req.task_ids or len(req.task_ids) < 2:
        raise HTTPException(status_code=400, detail="Please select at least 2 completed video parts to merge.")

    resolved_video_paths = []
    for tid in req.task_ids:
        tdata = load_task_from_disk(tid)
        if not tdata:
            raise HTTPException(status_code=404, detail=f"Task record not found for ID: {tid}")

        out_path = tdata.get("output_path")
        if not out_path or not os.path.exists(out_path):
            candidate = os.path.join("output", f"dubbed_{tid}_{sanitize_filename(tdata.get('filename', ''))}.mp4")
            if os.path.exists(candidate):
                out_path = candidate
            else:
                raise HTTPException(status_code=404, detail=f"Video file missing on disk for task {tid}")
        resolved_video_paths.append(os.path.abspath(out_path))

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

        if res.returncode != 0 or not os.path.exists(merged_output_path) or os.path.getsize(merged_output_path) < 1000:
            input_args = []
            filter_inputs = []
            for i, vp in enumerate(resolved_video_paths):
                input_args.extend(["-i", vp])
                filter_inputs.append(f"[{i}:v:0][{i}:a:0]")

            filter_complex_str = f"{''.join(filter_inputs)}concat=n={len(resolved_video_paths)}:v=1:a=1[outv][outa]"
            cmd_reencode = [
                FFMPEG_PATH, "-y",
                *input_args,
                "-filter_complex", filter_complex_str,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                merged_output_path
            ]
            res_fb = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res_fb.returncode != 0:
                fb_error = res_fb.stderr[-500:] if res_fb.stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg video merge failed: {fb_error}")

        task_data[merge_id] = {
            "task_id": merge_id,
            "filename": f"Full Merged: {clean_title}",
            "status": {
                "status": "completed",
                "step": f"Successfully merged {len(resolved_video_paths)} parts!",
                "progress": 100,
                "output_video": f"/output/{merged_filename}",
                "output_srt": ""
            },
            "output_path": merged_output_path,
            "segments": [],
            "bg_volume": 0.0,
            "voice_volume": 1.8,
            "audio_mode": "both",
            "add_watermark": False,
            "blur_subtitles": False,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_task_to_disk(merge_id)

        return {
            "success": True,
            "merged_task_id": merge_id,
            "output_video": f"/output/{merged_filename}",
            "filename": merged_filename,
            "parts_merged": len(resolved_video_paths)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge error: {str(e)}")
    finally:
        if os.path.exists(concat_txt_path):
            try:
                os.remove(concat_txt_path)
            except Exception:
                pass

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
    if req.filename.lower().endswith(('.srt', '.vtt')):
        raise HTTPException(status_code=400, detail="Cannot perform video trimming on subtitle files.")
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
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="ignore"
    )
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail="FFmpeg failed to trim video clip.")

    source_srt = None
    for f in os.listdir("input"):
        if f.lower().endswith(('.srt', '.vtt')) and clean_base in f:
            source_srt = os.path.join("input", f)
            break

    if source_srt and os.path.exists(source_srt):
        target_srt_name = f"clip_{clip_id}_{clean_base}.srt"
        split_srt_for_clip(source_srt, os.path.join("input", target_srt_name), req.start, req.end)

    return {
        "success": True,
        "clip_filename": clip_filename,
        "duration": duration
    }

@app.post("/api/translate-file")
async def translate_file(background_tasks: BackgroundTasks, request: Request):
    try:
        body = await request.json()
        filename = body.get("filename")
        source_lang = body.get("source_lang", "km")
        subtitle_filename = body.get("subtitle_filename")
        output_mode = body.get("output_mode", "both")
        voice_mode = body.get("voice_mode", "auto")
        pasted_script = body.get("pasted_script")
        add_watermark = bool(body.get("add_watermark", False))
        blur_subtitles = bool(body.get("blur_subtitles", True))
        banner_bottom_text = body.get("banner_bottom_text", "ចុច Follow ដើម្បីទស្សនាភាគបន្ត")
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
        "bg_volume": 0.0,
        "voice_volume": 1.8 if output_mode != "bg_only" else 0.0,
        "audio_mode": output_mode,
        "add_watermark": add_watermark,
        "blur_subtitles": blur_subtitles,
        "banner_bottom_text": banner_bottom_text,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_task_to_disk(task_id)

    background_tasks.add_task(
        run_dubbing_pipeline,
        task_id, video_path, source_lang, output_path, filename,
        subtitle_filename, output_mode, voice_mode, add_watermark, blur_subtitles,
        banner_bottom_text
    )
    return {"task_id": task_id, "message": "Anti-copyright Khmer dubbing started."}

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

def ydl_progress_hook(d, download_id):
    if d['status'] == 'downloading':
        raw_percent = d.get('_percent_str', '0%')
        clean_percent = re.sub(r'\x1b\[[0-9;]*m', '', raw_percent)
        percent_str = clean_percent.strip().replace('%', '')
        try:
            percent = float(percent_str)
        except Exception:
            percent = 0.0

        download_tasks[download_id] = {
            "status": "downloading",
            "progress": int(percent),
            "step": f"Downloading source quality stream... ({clean_percent.strip()})"
        }
    elif d['status'] == 'finished':
        download_tasks[download_id] = {
            "status": "processing",
            "progress": 95,
            "step": "Finalizing uncompressed video..."
        }

class URLInfoRequest(BaseModel):
    url: str

@app.post("/api/fetch-url-info")
async def fetch_url_info(req: URLInfoRequest):
    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed.")
    try:
        cleaned_url = clean_media_url(req.url)

        if any(ext in cleaned_url.lower() for ext in [".f4v", ".m3u8", ".mpd", "71edge.com", "googlevideo.com"]):
            return {
                "success": True,
                "title": "Direct Media Stream (Auto Source)",
                "languages": ["zh", "en", "km"],
                "qualities": [{"id": "best", "label": "🌟 Source Stream Resolution"}]
            }

        def get_info():
            info, _ = safe_extract_info(cleaned_url, is_download=False, custom_opts={'skip_download': True})
            return info

        info = await asyncio.to_thread(get_info)
        title = info.get('title', 'Unknown Video')

        subs = list(info.get('subtitles', {}).keys()) if info.get('subtitles') else []
        auto_subs = list(info.get('automatic_captions', {}).keys()) if info.get('automatic_captions') else []
        all_langs = sorted(list(set(subs + auto_subs)))

        if not all_langs:
            all_langs = ['en', 'km', 'zh', 'ja', 'ko']

        available_res_set = set()
        formats = info.get('formats', []) or []
        for f in formats:
            if f.get('vcodec') and f.get('vcodec') != 'none':
                w = f.get('width')
                h = f.get('height')
                if w and h:
                    min_dim = min(w, h)
                    if min_dim >= 144:
                        available_res_set.add(min_dim)

        quality_options = [{"id": "best", "label": "🌟 Best Source Quality (Auto Max 1080p/4K)"}]

        known_labels = {
            2160: "4K (2160x3840 / 3840x2160)",
            1440: "2K (1440x2560 / 2560x1440)",
            1080: "1080p (1080x1920 / Full HD)",
            720: "720p (720x1280 / HD)",
            480: "480p (480x854 / SD)",
            360: "360p",
            240: "240p",
            144: "144p"
        }

        added_resolutions = set()
        for res in sorted(list(available_res_set), reverse=True):
            target_bucket = res
            for std_res in [2160, 1440, 1080, 720, 480, 360, 240, 144]:
                if abs(res - std_res) <= 40:
                    target_bucket = std_res
                    break

            if target_bucket not in added_resolutions:
                added_resolutions.add(target_bucket)
                label = known_labels.get(target_bucket, f"{target_bucket}p")
                quality_options.append({"id": str(target_bucket), "label": f"📱 {label}"})

        return {
            "success": True,
            "title": title,
            "languages": all_langs,
            "qualities": quality_options
        }
    except Exception as e:
        print(f"❌ fetch_url_info error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

class URLDownloadRequest(BaseModel):
    url: str
    sublang: Optional[str] = None
    quality: Optional[str] = "best"

async def background_download_video(download_id: str, url: str, sublang: Optional[str] = None, quality: Optional[str] = "best"):
    try:
        download_tasks[download_id] = {"status": "downloading", "progress": 0, "step": "Connecting to video stream..."}
        cleaned_url = clean_media_url(url)
        temp_id = download_id

        if any(ext in cleaned_url.lower() for ext in [".f4v", ".m3u8", ".mpd", "71edge.com", "googlevideo.com"]):
            output_filename = f"{temp_id}_stream_video.mp4"
            output_path = os.path.join("input", output_filename)
            temp_raw_file = os.path.join("input", f"{temp_id}_raw.f4v")

            referer = "https://www.iqiyi.com/" if "71edge.com" in cleaned_url or "iqiyi" in cleaned_url else "https://www.youtube.com/"
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

            header_str = f"Referer: {referer}\r\nUser-Agent: {user_agent}\r\n"

            cmd = [
                FFMPEG_PATH, "-y",
                "-headers", header_str,
                "-i", cleaned_url,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path
            ]
            res = await asyncio.to_thread(subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if res.returncode != 0 or not os.path.exists(output_path):
                print(f"⚠️ Notice: FFmpeg direct stream error: {res.stderr[:160]}... Trying chunked Python fetcher...")
                download_tasks[download_id]["step"] = "Streaming chunked segments via Python buffer..."

                req = urllib.request.Request(
                    cleaned_url,
                    headers={
                        "User-Agent": user_agent,
                        "Referer": referer,
                        "Accept": "*/*"
                    }
                )

                def run_stream_download():
                    with urllib.request.urlopen(req, timeout=30) as response, open(temp_raw_file, 'wb') as out_f:
                        while True:
                            chunk = response.read(1024 * 512)
                            if not chunk:
                                break
                            out_f.write(chunk)

                await asyncio.to_thread(run_stream_download)

                if os.path.exists(temp_raw_file) and os.path.getsize(temp_raw_file) > 1000:
                    conv_cmd = [
                        FFMPEG_PATH, "-y",
                        "-i", temp_raw_file,
                        "-c", "copy",
                        "-movflags", "+faststart",
                        output_path
                    ]
                    subprocess.run(conv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if os.path.exists(temp_raw_file):
                        try:
                            os.remove(temp_raw_file)
                        except Exception:
                            pass

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise RuntimeError("Failed to capture video data. The CDN link may have expired or is blocked. Please re-capture a fresh link.")

            duration = get_video_duration(output_path)
            download_tasks[download_id] = {
                "status": "completed",
                "progress": 100,
                "step": "Download complete!",
                "temp_filename": output_filename,
                "duration": duration
            }
            return

        output_template = os.path.join("input", f"{temp_id}_%(title).100B.%(ext)s")
        languages_to_fetch = [sublang] if sublang and sublang.strip() else ['en', 'en-US', 'zh', 'ja', 'ko', 'all']

        if quality and quality != "best" and quality.isdigit():
            target_res = int(quality)
            format_sort_rules = [f'res:{target_res}', 'fps', 'codec:vp9:av01:h264', 'size', 'br']
        else:
            format_sort_rules = ['res', 'fps', 'codec:vp9:av01:h264', 'size', 'br']

        custom_opts = {
            'format': 'bv*+ba/b',
            'format_sort': format_sort_rules,
            'merge_output_format': 'mp4',
            'outtmpl': output_template,
            'noplaylist': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': languages_to_fetch,
            'subtitlesformat': 'srt/vtt/best',
            'extractor_retries': 3,
            'sleep_interval_subtitles': 2,
            'postprocessors': [{
                'key': 'FFmpegSubtitlesConvertor',
                'format': 'srt',
            }],
        }

        def run_ydl():
            info, ydl_instance = safe_extract_info(
                cleaned_url,
                is_download=True,
                custom_opts=custom_opts,
                download_id=download_id
            )

            width = info.get('width') or 0
            height = info.get('height') or 0
            fps = info.get('fps') or 0
            vcodec = info.get('vcodec') or 'unknown'
            format_id = info.get('format_id') or 'unknown'
            print(f"🌟 [High-Quality Stream] Downloaded: {width}x{height} @ {fps}fps | Format ID: {format_id} | Codec: {vcodec}")

            filename = ydl_instance.prepare_filename(info)
            base, _ = os.path.splitext(filename)
            expected_mp4 = base + ".mp4"
            if os.path.exists(expected_mp4):
                return expected_mp4
            elif os.path.exists(filename):
                return filename
            return filename

        filename = await asyncio.to_thread(run_ydl)
        real_filename = os.path.basename(filename)
        duration = get_video_duration(filename)

        download_tasks[download_id] = {
            "status": "completed",
            "progress": 100,
            "step": "Download complete!",
            "temp_filename": real_filename,
            "duration": duration
        }
    except Exception as e:
        print(f"❌ Download error: {e}")
        download_tasks[download_id] = {
            "status": "failed",
            "progress": 0,
            "step": str(e)
        }

@app.post("/api/download-url-preview")
async def download_url_preview(background_tasks: BackgroundTasks, req: URLDownloadRequest):
    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed.")

    download_id = str(os.urandom(4).hex())
    download_tasks[download_id] = {"status": "queued", "progress": 0, "step": "In queue..."}
    background_tasks.add_task(background_download_video, download_id, req.url, req.sublang, req.quality)
    return {"download_id": download_id}

@app.get("/api/download-status/{download_id}")
async def get_download_status(download_id: str):
    if download_id not in download_tasks:
        raise HTTPException(status_code=404, detail="Download ID not found.")
    return download_tasks[download_id]

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
        except Exception as e:
            print(f"⚠️ Warning: Could not delete history file {history_path}: {e}")

    if task_id in task_data:
        del task_data[task_id]
        deleted = True

    if deleted:
        return {"success": True, "message": "History and data deleted successfully."}
    else:
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
        "bg_volume": tdata.get("bg_volume", 0.0),
        "voice_volume": tdata.get("voice_volume", 1.8),
        "audio_mode": tdata.get("audio_mode", "both"),
        "add_watermark": tdata.get("add_watermark", False),
        "blur_subtitles": tdata.get("blur_subtitles", True),
        "banner_bottom_text": tdata.get("banner_bottom_text", "ចុច Follow ដើម្បីទស្សនាភាគបន្ត"),
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
    tdata = load_task_from_disk(task_id)
    if not tdata:
        return {"success": False, "error": "Task not found"}
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

@app.post("/api/re-render/{task_id}")
async def re_render_task(task_id: str, payload: dict):
    tdata = load_task_from_disk(task_id)
    if not tdata:
        return {"success": False, "error": "Task not found"}

    segments = payload.get("segments", [])
    bg_volume = float(payload.get("bg_volume", 0.0))
    voice_volume = float(payload.get("voice_volume", 1.8))
    audio_mode = payload.get("audio_mode", "both")
    add_watermark = bool(payload.get("add_watermark", False))
    blur_subtitles = bool(payload.get("blur_subtitles", True))
    banner_bottom_text = payload.get("banner_bottom_text", "ចុច Follow ដើម្បីទស្សនាភាគបន្ត")

    tdata["bg_volume"] = bg_volume
    tdata["voice_volume"] = voice_volume
    tdata["audio_mode"] = audio_mode
    tdata["add_watermark"] = add_watermark
    tdata["blur_subtitles"] = blur_subtitles
    tdata["banner_bottom_text"] = banner_bottom_text

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
        audio_segments = []
        if audio_mode != "bg_only" and voice_volume > 0:
            for idx, seg in enumerate(segments):
                text_to_speak = clean_khmer_text(seg.get("translated_text", ""))
                gender = seg.get("gender", "female")
                if text_to_speak:
                    seg_audio_path = f"temp_seg_{task_id}_{idx}.mp3"
                    success = await generate_segment_audio(text_to_speak, seg_audio_path, gender=gender)
                    if success and os.path.exists(seg_audio_path):
                        audio_segments.append({
                            "path": seg_audio_path,
                            "start": seg["start"],
                            "end": seg["end"]
                        })

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
            blur_subtitles=blur_subtitles,
            banner_bottom_text=banner_bottom_text
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filepath = req.filepath if req and req.filepath else None
    target_path = os.path.abspath(filepath) if filepath and os.path.exists(filepath) else output_dir

    try:
        system = platform.system()
        if system == "Windows":
            if os.path.isfile(target_path):
                subprocess.run(f'explorer /select,"{target_path}"')
            else:
                os.startfile(target_path)
        elif system == "Darwin":
            if os.path.isfile(target_path):
                subprocess.run(["open", "-R", target_path])
            else:
                subprocess.run(["open", target_path])
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
        return {
            "success": True,
            "translated_text": translated_text
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

class AutoSplitRequest(BaseModel):
    filename: str
    chunk_minutes: float = 5.0

@app.post("/api/auto-split-video")
async def auto_split_video(req: AutoSplitRequest):
    src_path = os.path.join("input", req.filename)
    if not os.path.exists(src_path):
        raise HTTPException(status_code=404, detail="Source file not found.")

    total_dur = get_video_duration(src_path)
    chunk_duration_sec = req.chunk_minutes * 60.0
    overlap = 3.0

    parts_created = 0
    start_time = 0.0
    base_name, ext = os.path.splitext(req.filename)
    clean_base = sanitize_filename(base_name)

    matching_srt_path = None
    for f in os.listdir("input"):
        if f.lower().endswith(('.srt', '.vtt')) and (clean_base in f or clean_base[:10] in f):
            matching_srt_path = os.path.join("input", f)
            break

    srt_blocks = []
    if matching_srt_path and os.path.exists(matching_srt_path):
        try:
            with open(matching_srt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                srt_content = f.read()
            if srt_content.startswith("WEBVTT"):
                srt_content = srt_content.split("\n\n", 1)[1] if "\n\n" in srt_content else srt_content
            blocks = srt_content.strip().split("\n\n")
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
                    def time_to_sec(t_str):
                        t_parts = t_str.strip().replace(',', '.').split(':')
                        try:
                            if len(t_parts) == 3:
                                return float(t_parts[0]) * 3600 + float(t_parts[1]) * 60 + float(t_parts[2])
                            elif len(t_parts) == 2:
                                return float(t_parts[0]) * 60 + float(t_parts[1])
                        except ValueError:
                            return 0.0
                        return 0.0
                    s_sec = time_to_sec(parts[0])
                    e_sec = time_to_sec(parts[1].split()[0])
                    text_lines = lines[time_line_idx + 1:]
                    srt_blocks.append({
                        "start": s_sec,
                        "end": e_sec,
                        "lines": text_lines
                    })
        except Exception as e:
            print(f"⚠️ Warning: Could not parse matching srt for splitting: {e}")

    try:
        while start_time < total_dur:
            end_time = min(start_time + chunk_duration_sec, total_dur)
            actual_duration = end_time - start_time

            clip_id = str(os.urandom(3).hex())
            clip_filename = f"part_{parts_created + 1}_{clip_id}_{clean_base}{ext}"
            clip_path = os.path.join("input", clip_filename)

            cmd = [
                FFMPEG_PATH, "-y", "-ss", str(start_time), "-i", src_path,
                "-t", str(actual_duration), "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", clip_path
            ]
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="ignore"
            )

            if res.returncode == 0:
                parts_created += 1

                if srt_blocks:
                    part_srt_filename = f"part_{parts_created}_{clip_id}_{clean_base}.srt"
                    part_srt_path = os.path.join("input", part_srt_filename)
                    part_lines = []
                    counter = 1
                    for block in srt_blocks:
                        if block["end"] >= start_time and block["start"] <= end_time:
                            rel_start = max(0.0, block["start"] - start_time)
                            rel_end = min(actual_duration, block["end"] - start_time)
                            if rel_end > rel_start:
                                start_fmt = format_time(rel_start)
                                end_fmt = format_time(rel_end)
                                part_lines.append(f"{counter}\n{start_fmt} --> {end_fmt}\n" + "\n".join(block["lines"]) + "\n")
                                counter += 1
                    if part_lines:
                        with open(part_srt_path, "w", encoding="utf-8") as sf:
                            sf.write("\n".join(part_lines))

            if end_time >= total_dur:
                break

            start_time = end_time - overlap

        return {
            "success": True,
            "total_parts": parts_created
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)