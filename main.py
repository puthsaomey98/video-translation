import asyncio
import os
from core.subtitle_translator import translate_subtitles_to_khmer
from core.tts import generate_segment_audio
from core.merger import merge_synchronized_audio_and_subtitles

async def process_video_with_subtitle_file(video_path: str, subtitle_path: str, source_lang: str, output_path: str):
    print(f"1. Processing video: {video_path}")
    print(f"   Using subtitle file: {subtitle_path}")
    
    khmer_srt_path = output_path.rsplit(".", 1)[0] + ".srt"
    
    try:
        translated_segments = translate_subtitles_to_khmer(subtitle_path, khmer_srt_path, source_lang=source_lang)
    except Exception as e:
        print(f"❌ Error translating subtitles: {e}")
        return

    print("2. Generating individual Khmer Voiceover segments (TTS)...")
    audio_segments = []
    
    for i, seg in enumerate(translated_segments):
        if seg["translated_text"].strip():
            seg_audio_path = f"temp_seg_{i}.mp3"
            await generate_segment_audio(seg["translated_text"], seg_audio_path)
            if os.path.exists(seg_audio_path):
                audio_segments.append({
                    "path": seg_audio_path,
                    "start": seg["start"]
                })

    print("3. Compiling and syncing final video with FFmpeg...")
    merge_synchronized_audio_and_subtitles(video_path, audio_segments, khmer_srt_path, output_path)

    print("4. Cleaning up temporary workspace files...")
    for seg in audio_segments:
        if os.path.exists(seg["path"]):
            os.remove(seg["path"])

    print(f"✅ Video successfully dubbed in Khmer and saved to: {output_path}")
    print(f"✅ Khmer subtitle file ready at: {khmer_srt_path}")

if __name__ == "__main__":
    input_video = "Me at the zoo.webm"
    input_subtitle = "Me at the zoo.en.vtt" # (Or your Spanish subtitle file)
    output_video = "output_khmer_dubbed_video.mp4"
    
    # Set source language to Spanish ("es") since the input text is in Spanish
    source_language = "es"  
    
    if os.path.exists(input_video) and os.path.exists(input_subtitle):
        print(f"Auto-detected video: {input_video}")
        print(f"Auto-detected subtitles: {input_subtitle}")
        asyncio.run(process_video_with_subtitle_file(input_video, input_subtitle, source_language, output_video))
    else:
        print(f"❌ Error: Could not find '{input_video}' or '{input_subtitle}' in the directory.")