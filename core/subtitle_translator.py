import os
import re
from core.translator import KhmerTranslator

def parse_subtitle_file(file_path: str) -> list:
    """Parses SRT/VTT subtitle files line-by-line to guarantee exact 1-to-1 millisecond block isolation."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Subtitle file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    segments = []
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i].strip()
        time_line = None
        time_line_idx = -1
        
        # Identify the timestamp line containing '-->'
        if "-->" in line:
            time_line = line
            time_line_idx = i
        elif i + 1 < n and "-->" in lines[i + 1].strip():
            time_line = lines[i + 1].strip()
            time_line_idx = i + 1
            
        if time_line and time_line_idx != -1:
            parts = time_line.split("-->")
            if len(parts) == 2:
                # Extract precise start and end time strings
                start_str = parts[0].strip().split()[-1]
                end_str = parts[1].strip().split()[0]
                
                # Collect all text lines belonging exclusively to this timestamp block
                text_lines = []
                j = time_line_idx + 1
                while j < n:
                    next_line = lines[j].strip()
                    if not next_line:  # Blank line signals end of block
                        j += 1
                        break
                    if "-->" in next_line:  # Safety check against malformed blocks
                        break
                    # Check if it's a standalone number index for the next subtitle block
                    if next_line.isdigit() and j + 1 < n and "-->" in lines[j + 1]:
                        break
                        
                    text_lines.append(next_line)
                    j += 1
                
                text_content = " ".join(text_lines).strip()
                text_content = re.sub(r'<[^>]+>', '', text_content) # Strip HTML tags if any
                
                if text_content:
                    start_sec = time_to_seconds(start_str)
                    end_sec = time_to_seconds(end_str)
                    segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text_content
                    })
                i = j
                continue
        i += 1
        
    return segments

def time_to_seconds(time_str: str) -> float:
    """Converts millisecond timestamp string (with comma or dot) into precise float seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return 0.0

def translate_subtitles_to_khmer(input_subtitle_path: str, output_srt_path: str, source_lang: str = "en") -> list:
    """Reads millisecond subtitles, translates each segment independently, and saves a perfectly synchronized SRT file."""
    print(f"Parsing millisecond subtitle file: {input_subtitle_path}")
    raw_segments = parse_subtitle_file(input_subtitle_path)
    
    if not raw_segments:
        raise ValueError("❌ No valid subtitle segments found in the file. Please check the SRT format.")
        
    translator = KhmerTranslator()
    translated_segments = []
    print(f"Translating {len(raw_segments)} subtitle segments into Khmer with exact millisecond preservation...")
    
    for seg in raw_segments:
        translated_text = translator.translate_text(seg["text"], source_lang=source_lang)
        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "translated_text": translated_text
        })
        
    srt_lines = []
    for i, seg in enumerate(translated_segments, start=1):
        from core.tts import format_time
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{seg['translated_text']}\n")
        
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
        
    print(f"✅ Khmer millisecond subtitle file successfully saved to: {output_srt_path}")
    return translated_segments