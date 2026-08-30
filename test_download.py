import os
import sys

try:
    import yt_dlp
except ImportError:
    print("❌ Error: yt-dlp is not installed. Run: pip install -U yt-dlp")
    sys.exit(1)

# Auto-locate FFmpeg via imageio-ffmpeg if available
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ffmpeg_exe = "ffmpeg"

VIDEO_URL = "https://www.youtube.com/watch?v=QUPPzfqb8YY"
COOKIE_FILE = "cookies.txt"

if not os.path.exists(COOKIE_FILE):
    print(f"⚠️ Warning: '{COOKIE_FILE}' not found in current directory: {os.getcwd()}")
else:
    print(f"✅ Found cookie file: {COOKIE_FILE} ({os.path.getsize(COOKIE_FILE)} bytes)")

ydl_opts = {
    # Authentication & Session
    "cookiefile": COOKIE_FILE if os.path.exists(COOKIE_FILE) else None,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    },
    # Client strategy to bypass reload challenges
    "extractor_args": {
        "youtube": {
            "player_client": ["web", "ios"]
        }
    },
    # Quality: Best video stream + best audio stream merged
    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
    "format_sort": ["res", "fps", "codec:h264:vp9", "size", "br"],
    "merge_output_format": "mp4",
    "outtmpl": "test_output_%(id)s.%(ext)s",
    "ffmpeg_location": ffmpeg_exe,
    "noplaylist": True,
    "quiet": False,
    "no_warnings": False,
}

print(f"\n🚀 Attempting test download for: {VIDEO_URL}\n" + "=" * 50)

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(VIDEO_URL, download=True)
        
        print("\n" + "=" * 50)
        print("🎉 Download successful!")
        print(f"📌 Title:      {info.get('title')}")
        print(f"📐 Resolution: {info.get('width')}x{info.get('height')} @ {info.get('fps')}fps")
        print(f"🎞️ Format ID:  {info.get('format_id')}")
        print(f"💾 File:       test_output_{info.get('id')}.mp4")

except Exception as e:
    print("\n" + "=" * 50)
    print(f"❌ Test download failed with error:\n{e}")