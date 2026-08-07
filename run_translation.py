import asyncio
import os
from core.separator import extract_and_separate_audio
from core.stt import transcribe_isolated_vocals
from core.translator import KhmerTranslator
from core.tts import generate_segment_audio, format_time
from core.merger import merge_with_background_music

async def process_video_with_background_retention(video_path: str, source_lang: str, output_path: str):
    print(f"==================================================")
    print(f"🎬 Starting Audio Separation & Dubbing Pipeline")
    print(f"📁 Target Video: {video_path}")
    print(f"==================================================")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    khmer_srt_path = output_path.rsplit(".", 1)[0] + ".srt"
    
    try:
        # Step 1: Separate video into background music and isolated vocals using Demucs
        print("\n[Step 1/5] Separating background music and speech stems...")
        stems = extract_and_separate_audio(video_path)
        vocals_path = stems["vocals"]
        background_music_path = stems["no_vocals"]
    except Exception as e:
        print(f"❌ Error during audio separation: {e}")
        return

    try:
        # Step 2: Transcribe clean isolated vocals to get high-precision millisecond timestamps
        print("\n[Step 2/5] Analyzing speech timing from isolated vocals...")
        raw_segments = transcribe_isolated_vocals(vocals_path, source_lang=source_lang)
    except Exception as e:
        print(f"❌ Error during speech-to-text transcription: {e}")
        return

    if not raw_segments:
        print("❌ No speech segments detected in the vocal track.")
        return

    # Step 3: Translate script into Khmer
    print("\n[Step 3/5] Translating script into Khmer...")
    translator = KhmerTranslator()
    
    translated_segments = []
    for seg in raw_segments:
        translated_text = translator.translate_text(seg["text"], source_lang=source_lang)
        translated_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "translated_text": translated_text
        })

    # Save reference Khmer SRT file
    srt_lines = []
    for i, seg in enumerate(translated_segments, start=1):
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{seg['translated_text']}\n")
    
    with open(khmer_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))
    print(f"✅ Generated synced Khmer SRT reference file: {khmer_srt_path}")

    # Step 4: Generate Khmer voiceover segments (TTS)
    print("\n[Step 4/5] Generating individual Khmer Voiceover segments (TTS)...")
    audio_segments = []
    
    for i, seg in enumerate(translated_segments):
        if seg["translated_text"].strip():
            seg_audio_path = f"temp_seg_{i}.mp3"
            success = await generate_segment_audio(seg["translated_text"], seg_audio_path)
            if success and os.path.exists(seg_audio_path):
                audio_segments.append({
                    "path": seg_audio_path,
                    "start": seg["start"],
                    "end": seg["end"]
                })

    # Step 5: Merge original background music, new Khmer voiceover, and video cleanly
    print("\n[Step 5/5] Mixing background music, Khmer voiceover, and video...")
    merge_with_background_music(
        video_path=video_path,
        background_audio_path=background_music_path,
        audio_segments=audio_segments,
        output_path=output_path,
        crop_bottom_pixels=0  # Set to e.g. 80 if you need to crop hardcoded text
    )

    # Clean up temporary individual TTS and separated workspace audio files
    print("🧹 Cleaning up temporary workspace files...")
    for seg in audio_segments:
        if os.path.exists(seg["path"]):
            os.remove(seg["path"])

    print(f"\n==================================================")
    print(f"🎉 SUCCESS! Dubbed video with original background music saved at: {output_path}")
    print(f"==================================================")

if __name__ == "__main__":
    input_video = "input/part_002.mp4"
    output_video = "output/put_part_002.mp4"
    source_language = "en"  # Spoken language of the original video ("en", "zh", etc.)
    
    if os.path.exists(input_video):
        asyncio.run(process_video_with_background_retention(input_video, source_language, output_video))
    else:
        print(f"❌ Error: Make sure '{input_video}' exists inside your 'input' folder.")