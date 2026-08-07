import os
from faster_whisper import WhisperModel

def transcribe_isolated_vocals(vocals_audio_path: str, source_lang: str = "en") -> list:
    """Transcribes the isolated vocal audio file with high precision and VAD filtering."""
    if not os.path.exists(vocals_audio_path):
        raise FileNotFoundError(f"Vocals audio file not found: {vocals_audio_path}")
    
    print("Running Faster-Whisper on isolated vocal track for precise millisecond timestamps...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    segments, _ = model.transcribe(
        vocals_audio_path, 
        language=source_lang, 
        beam_size=5, 
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    transcript_segments = []
    for segment in segments:
        start_time = float(segment.start)
        end_time = float(segment.end)
        text = segment.text.strip()
        
        if text:
            transcript_segments.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
        
    print(f"✅ Successfully extracted {len(transcript_segments)} high-precision speech segments.")
    return transcript_segments