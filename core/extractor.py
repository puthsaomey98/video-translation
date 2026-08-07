import os
from videocr import save_subtitles_to_file

def extract_hardcoded_subtitles(video_path: str, source_lang: str = "en", output_srt: str = "extracted_subs.srt") -> list:
    """Extracts hardcoded subtitles from video frames using OCR and parses them."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Map languages for Tesseract OCR engine
    tesseract_lang_map = {
        "en": "eng",
        "zh": "chi_sim"
    }
    lang_code = tesseract_lang_map.get(source_lang, "eng")
    
    print(f"Extracting hardcoded subtitles via OCR (Language: {lang_code})...")
    
    # Extract and save subtitles directly to an SRT file
    save_subtitles_to_file(
        video_path=video_path,
        file_path=output_srt,
        lang=lang_code,
        conf_threshold=65,
        sim_threshold=90
    )
    
    return parse_srt_file(output_srt)

def parse_srt_file(srt_path: str) -> list:
    """Parses an SRT file into segment dictionaries with timestamps and text."""
    if not os.path.exists(srt_path):
        return []
        
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    blocks = content.strip().split("\n\n")
    segments = []
    
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            time_line = lines[1]
            text_line = " ".join(lines[2:])
            
            parts = time_line.split(" --> ")
            if len(parts) == 2:
                start_sec = time_to_seconds(parts[0])
                end_sec = time_to_seconds(parts[1])
                segments.append({
                    "start": start_sec,
                    "end": end_sec,
                    "text": text_line.strip()
                })
                
    return segments

def time_to_seconds(time_str: str) -> float:
    """Converts SRT timestamp format into total seconds."""
    time_str = time_str.replace(',', '.')
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)