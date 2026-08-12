import os
import asyncio
import subprocess
import re
import json
import time
import platform
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

from core.separator import extract_and_separate_audio
from core.stt import transcribe_isolated_vocals
from core.translator import KhmerTranslator
from core.tts import generate_segment_audio, format_time

try:
    from moviepy import AudioFileClip, VideoFileClip, CompositeAudioClip
except ImportError:
    from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip

app = FastAPI(title="Khmer Video Dubbing Studio & Timeline Editor")

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("temp_separated", exist_ok=True)
os.makedirs("history", exist_ok=True)

app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/input", StaticFiles(directory="input"), name="input")
templates = Jinja2Templates(directory="templates")

task_data = {}
download_tasks = {}

# In-memory duration cache for instant folder scanning
DURATION_CACHE = {}

ZH_TYPO_MAP = {
    "进房炮": "近防炮",  # Fixes "entering room cannon" -> Close-In Weapon System (CIWS)
}

def preprocess_source_text(text: str, source_lang: str) -> str:
    """Pre-cleans source text typos (e.g. Chinese homophones) before translation."""
    if not text:
        return ""
    if source_lang == "zh":
        for typo, fix in ZH_TYPO_MAP.items():
            text = text.replace(typo, fix)
    return text.strip()

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

def clean_youtube_url(url: str) -> str:
    """Strips playlist parameters (&list=, &index=) to ensure single-video extraction."""
    if "youtube.com/watch" in url and "v=" in url:
        base_url, _, query = url.partition("?")
        params = query.split("&")
        clean_params = [p for p in params if not p.startswith("list=") and not p.startswith("index=")]
        if clean_params:
            return base_url + "?" + "&".join(clean_params)
        return base_url
    return url

def get_video_duration(video_path: str) -> float:
    """Fast, cached duration scanner to prevent slow folder listing."""
    if not video_path or video_path.lower().endswith(('.srt', '.vtt', '.txt', '.json')):
        return 0.0

    try:
        mtime = os.path.getmtime(video_path) if os.path.exists(video_path) else 0
        filename = os.path.basename(video_path)

        # ⚡ Instant Return from Memory Cache
        if filename in DURATION_CACHE and DURATION_CACHE[filename]["mtime"] == mtime:
            return DURATION_CACHE[filename]["duration"]

        duration = 0.0

        # Fast FFprobe Probe
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.2,
            encoding="utf-8", errors="ignore"
        )
        if result.returncode == 0 and result.stdout.strip():
            val = result.stdout.strip()
            if val != "N/A":
                duration = float(val)

        # Fallback to MoviePy ONLY if ffprobe fails
        if duration == 0.0 and video_path.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')):
            try:
                with VideoFileClip(video_path) as clip:
                    duration = clip.duration
            except Exception:
                duration = 0.0

        # Cache Result
        DURATION_CACHE[filename] = {"mtime": mtime, "duration": duration}
        return duration

    except Exception:
        return 0.0

def clean_khmer_text(text: str) -> str:
    """Sanitizes and naturalizes Khmer text into spoken drama style (ភាសានិយាយ)."""
    if not text:
        return ""
    
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    spoken_rules = [
        (r'\bអញ\b', 'ខ្ញុំ'),
        (r'ត្រូវបាន\s*', ''),
        (r'\s+នៃ\s+', ' '),
        (r'មិនមែនទេ', 'អត់ទេ'),
        (r'យ៉ាងណាក៏ដោយ', 'ប៉ុន្តែ'),
        (r'លើសពីនេះទៅទៀត', 'ហើយ'),
        (r'ជាមួយគ្នានេះដែរ', 'ហើយ'),
        (r'រូបលោក', 'គាត់'),
        (r'លោកអ្នក', 'អ្នក'),
        (r'តើ\s+(.*?)\s+ឬទេ\?', r'\1 មែនទេ?'),
        (r'តើ\s+', ''),
    ]
    
    for pattern, replacement in spoken_rules:
        text = re.sub(pattern, replacement, text)
        
    return re.sub(r'\s+', ' ', text).strip()

def safe_translate_and_clean(translator, text: str, source_lang: str) -> str:
    """Translates source text or cleans Khmer directly with full fallback safety."""
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
    """
    Saves pasted text as an SRT file.
    If plain text, calculates distributed timing across video duration.
    """
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
            if speed < 0.75:
                speed = 0.75
            elif speed > 1.9:
                speed = 1.9

        filter_chain = []
        if abs(speed - 1.0) > 0.05:
            filter_chain.append(f"atempo={speed}")
            
        filter_chain.append("acompressor=threshold=-14dB:ratio=3.5:attack=5:release=80:makeup=3dB")
        filter_chain.append("alimiter=limit=0.92")
        
        filter_str = ",".join(filter_chain)

        cmd = [
            "ffmpeg", "-y", "-i", input_audio_path,
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

# ➕ ADD THESE TWO COMPATIBILITY HELPERS:
def apply_clip_duration(clip, duration):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)

def apply_clip_audio(video_clip, audio_clip):
    if hasattr(video_clip, "with_audio"):
        return video_clip.with_audio(audio_clip)
    return video_clip.set_audio(audio_clip)

def merge_with_realtime_progress(task_id: str, video_path: str, background_audio_path: str, audio_segments: list, output_path: str, bg_volume: float = 0.7, voice_volume: float = 1.5):
    if not audio_segments:
        raise RuntimeError("❌ No voiceover segments available to merge.")

    fitted_temp_files = []
    try:
        task_data[task_id]["status"] = {"status": "processing", "step": "Mixing clean audio tracks and BGM...", "progress": 82}
        save_task_to_disk(task_id)

        video_clip = VideoFileClip(video_path)
        total_duration = video_clip.duration

        audio_clips = []
        if background_audio_path and os.path.exists(background_audio_path):
            bg_clip = AudioFileClip(background_audio_path)
            bg_clip = apply_clip_volume(bg_clip, bg_volume)
            audio_clips.append(bg_clip)

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

        final_audio = apply_clip_duration(CompositeAudioClip(audio_clips), total_duration)
        final_video = apply_clip_audio(video_clip, final_audio)

        task_data[task_id]["status"] = {"status": "processing", "step": "Rendering final dubbed video...", "progress": 90}
        save_task_to_disk(task_id)

        # Fixed MoviePy parameter: pix_fmt="yuv420p"
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            fps=video_clip.fps if video_clip.fps else 30,
            preset="medium",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"], # ✅ Universal FFmpeg parameter
            logger=None
        )

        video_clip.close()
        final_audio.close()
        for clip in audio_clips:
            clip.close()

    except Exception as e:
        raise RuntimeError(f"❌ Audio mixing failed: {str(e)}")
    finally:
        for tmp_f in fitted_temp_files:
            if os.path.exists(tmp_f):
                try:
                    os.remove(tmp_f)
                except Exception:
                    pass
def sanitize_filename(filename: str) -> str:
    """Removes emojis, special characters, and brackets to create clean, browser-safe filenames."""
    # Keep alphanumeric characters, dots, underscores, and dashes
    clean_name = re.sub(r'[^\w\.-]', '_', filename)
    # Collapse multiple underscores into one
    return re.sub(r'_+', '_', clean_name).strip('_')


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

def run_dubbing_pipeline(task_id: str, video_path: str, source_lang: str, output_path: str, filename: str, subtitle_filename: Optional[str] = None, output_mode: str = "both", voice_mode: str = "auto"):
    try:
        translator = KhmerTranslator()

        # Step 1: Extract background audio (BGM)
        task_data[task_id]["status"] = {"status": "processing", "step": "Isolating background music and audio tracks...", "progress": 15}
        save_task_to_disk(task_id)

        stems = extract_and_separate_audio(video_path, output_dir="./temp_separated")
        vocals_path = stems["vocals"]
        background_music_path = stems["no_vocals"]
        
        public_bg_audio_name = f"bg_track_{task_id}.wav"
        public_bg_audio_path = os.path.join("output", public_bg_audio_name)
        if os.path.exists(background_music_path):
            import shutil
            shutil.copy(background_music_path, public_bg_audio_path)
            task_data[task_id]["public_bg_audio"] = f"/output/{public_bg_audio_name}"

        task_data[task_id]["video_path"] = video_path
        task_data[task_id]["background_audio_path"] = background_music_path
        task_data[task_id]["output_path"] = output_path
        
        # Step 2: Extract or load subtitle segments
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

        # Step 3: Process/Translate script
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

        # Save output SRT
        khmer_srt_path = output_path.rsplit(".", 1)[0] + ".srt"
        task_data[task_id]["srt_path"] = khmer_srt_path
        
        srt_lines = []
        for i, seg in enumerate(translated_segments, start=1):
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])
            srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{seg['translated_text']}\n")
        
        with open(khmer_srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        # Step 4: Generate Khmer TTS Audio
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

        # Step 5: Merge BGM + Speech
        bg_vol = task_data[task_id].get("bg_volume", 0.7)
        voice_vol = task_data[task_id].get("voice_volume", 1.5)

        merge_with_realtime_progress(
            task_id=task_id,
            video_path=video_path,
            background_audio_path=background_music_path,
            audio_segments=audio_segments,
            output_path=output_path,
            bg_volume=bg_vol,
            voice_volume=voice_vol
        )

        for seg in audio_segments:
            if os.path.exists(seg["path"]):
                try:
                    os.remove(seg["path"])
                except Exception:
                    pass

        task_data[task_id]["status"] = {
            "status": "completed", 
            "step": "Dubbing Complete!", 
            "progress": 100,
            "output_video": f"/output/{os.path.basename(output_path)}",
            "output_srt": f"/output/{os.path.basename(khmer_srt_path)}"
        }
        task_data[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_task_to_disk(task_id)

    except Exception as e:
        task_data[task_id]["status"] = {"status": "failed", "step": str(e), "progress": 0}
        save_task_to_disk(task_id)

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
                files.append({
                    "filename": f,
                    "duration": duration,
                    "size": os.path.getsize(path) if os.path.exists(path) else 0,
                    "is_subtitle": is_sub
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
    clip_filename = f"clip_{clip_id}_{base_name}{ext}"
    clip_path = os.path.join("input", clip_filename)
    
    duration = req.end - req.start
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid start and end times.")
    
    cmd = [
        "ffmpeg", "-y", "-ss", str(req.start), "-i", src_path,
        "-t", str(duration), "-c:v", "copy", "-c:a", "copy", clip_path
    ]
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="ignore"
    )
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail="FFmpeg failed to trim video clip.")

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
        
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
        
    video_path = os.path.join("input", filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found in input folder: {filename}")
    
    task_id = str(os.urandom(4).hex())

    safe_name = sanitize_filename(filename)
    output_path = os.path.join("output", f"dubbed_{task_id}_{safe_name}")

    # Process Pasted Script if provided
    if pasted_script and pasted_script.strip():
        pasted_srt_filename = f"pasted_{task_id}.srt"
        pasted_srt_path = os.path.join("input", pasted_srt_filename)
        if process_pasted_script_to_srt(pasted_script, video_path, pasted_srt_path):
            subtitle_filename = pasted_srt_filename

    task_data[task_id] = {
        "task_id": task_id,
        "filename": filename,
        "status": {"status": "queued", "step": "In queue...", "progress": 0},
        "bg_volume": 0.7,
        "voice_volume": 1.5,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_task_to_disk(task_id)

    background_tasks.add_task(
        run_dubbing_pipeline, 
        task_id, video_path, source_lang, output_path, filename, 
        subtitle_filename, output_mode, voice_mode
    )
    return {"task_id": task_id, "message": "Khmer video dubbing pipeline started."}

@app.post("/api/upload-preview")
async def upload_preview(file: UploadFile = File(...)):
    temp_id = str(os.urandom(4).hex())
    filename = f"{temp_id}_{file.filename}"
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
            "step": f"Downloading from URL... ({clean_percent.strip()})"
        }
    elif d['status'] == 'finished':
        download_tasks[download_id] = {
            "status": "processing",
            "progress": 95,
            "step": "Finalizing downloaded file..."
        }

class URLInfoRequest(BaseModel):
    url: str

@app.post("/api/fetch-url-info")
async def fetch_url_info(req: URLInfoRequest):
    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed.")
    try:
        cleaned_url = clean_youtube_url(req.url)
        
        def get_info():
            ydl_opts = {
                'skip_download': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'allsubtitles': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(cleaned_url, download=False)
        
        info = await asyncio.to_thread(get_info)
        title = info.get('title', 'Unknown Video')
        
        subs = list(info.get('subtitles', {}).keys())
        auto_subs = list(info.get('automatic_captions', {}).keys())
        all_langs = sorted(list(set(subs + auto_subs)))
        
        if not all_langs:
            all_langs = ['en']
        
        return {
            "success": True,
            "title": title,
            "languages": all_langs
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

async def background_download_video(download_id: str, url: str, sublang: Optional[str] = None):
    try:
        download_tasks[download_id] = {"status": "downloading", "progress": 0, "step": "Connecting to URL..."}
        cleaned_url = clean_youtube_url(url)
        temp_id = download_id
        output_template = os.path.join("input", f"{temp_id}_%(title)s.%(ext)s")
        
        languages_to_fetch = [sublang] if sublang and sublang.strip() else ['en', 'en-US', 'zh', 'ja', 'ko']
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_template,
            'noplaylist': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': languages_to_fetch,
            'subtitlesformat': 'srt/vtt/best',
            'postprocessors': [{
                'key': 'FFmpegSubtitlesConvertor',
                'format': 'srt',
            }],
            'progress_hooks': [lambda d: ydl_progress_hook(d, download_id)]
        }

        def run_ydl():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(cleaned_url, download=True)
                filename = ydl.prepare_filename(info)
                if not filename.endswith('.mp4'):
                    base, _ = os.path.splitext(filename)
                    new_filename = base + ".mp4"
                    if os.path.exists(filename):
                        os.rename(filename, new_filename)
                    filename = new_filename
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
        download_tasks[download_id] = {
            "status": "failed",
            "progress": 0,
            "step": str(e)
        }

class URLDownloadRequest(BaseModel):
    url: str
    sublang: Optional[str] = None

@app.post("/api/download-url-preview")
async def download_url_preview(background_tasks: BackgroundTasks, req: URLDownloadRequest):
    if not yt_dlp:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed.")
    
    download_id = str(os.urandom(4).hex())
    download_tasks[download_id] = {"status": "queued", "progress": 0, "step": "In queue..."}
    background_tasks.add_task(background_download_video, download_id, req.url, req.sublang)
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
                            "output_video": f"/output/dubbed_{data.get('task_id', '')}_{data.get('filename', '')}" if "dubbed_" not in data.get('output_path', '') else f"/output/{os.path.basename(data.get('output_path', ''))}"
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
    
    return {
        "filename": tdata.get("filename", "Project Workspace"),
        "segments": tdata.get("segments", []),
        "bg_volume": tdata.get("bg_volume", 0.7),
        "voice_volume": tdata.get("voice_volume", 1.5),
        "output_video": f"/output/{os.path.basename(tdata.get('output_path', ''))}",
        "background_audio": tdata.get("public_bg_audio")
    }

class SegmentItem(BaseModel):
    start: float
    end: float
    source_text: str = ""
    translated_text: str
    gender: str = "female"

class ReRenderRequest(BaseModel):
    segments: List[SegmentItem]
    bg_volume: float = 0.7
    voice_volume: float = 1.5

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
    bg_volume = payload.get("bg_volume", 0.7)
    voice_volume = payload.get("voice_volume", 1.5)
    
    tdata["bg_volume"] = bg_volume
    tdata["voice_volume"] = voice_volume
    
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
            voice_volume=voice_volume
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
    
    matching_srt_path = None
    for f in os.listdir("input"):
        if f.lower().endswith(('.srt', '.vtt')) and (base_name in f or f.startswith(base_name[:10])):
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
            clip_filename = f"part_{parts_created + 1}_{clip_id}_{base_name}{ext}"
            clip_path = os.path.join("input", clip_filename)
            
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-i", src_path,
                "-t", str(actual_duration), "-c:v", "copy", "-c:a", "copy", clip_path
            ]
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="ignore"
            )
            
            if res.returncode == 0:
                parts_created += 1
                
                if srt_blocks:
                    part_srt_filename = f"part_{parts_created}_{clip_id}_{base_name}.srt"
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