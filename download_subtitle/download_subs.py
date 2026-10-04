import os
import re
import urllib.parse
import urllib.request

# Base configuration
# SAMPLE_URL = "https://subtitle.stardust-tv.com/final_cut/21362/5/%E8%B4%A8%E6%A3%80_%E9%87%8D%E7%94%9F%E5%A9%9A%E5%AE%B4%EF%BC%9A%E6%88%91%E8%BD%AC%E8%BA%AB%E5%AB%81%E7%BB%99%E8%B4%A2%E9%98%80%E4%BF%9D%E9%95%96_001.srt"
SAMPLE_URL = "https://subtitle.stardust-tv.com/final_cut/19461/5/%E8%B4%A8%E6%A3%80_%E8%A2%AB%E6%89%AB%E5%9C%B0%E5%87%BA%E9%97%A8%E5%90%8E%EF%BC%8C%E6%88%91%E9%9D%A0%E5%85%BB%E8%9B%99%E6%88%90%E4%BA%86%E5%85%A8%E6%9D%91%E9%A6%96%E5%AF%8C_001.srt"
START_EP = 1
END_EP = 96
PADDING = 3
OUTPUT_DIR = "subtitles"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for ep in range(START_EP, END_EP + 1):
    padded_ep = str(ep).zfill(PADDING)
    # Replace the trailing episode number
    url = re.sub(r'([_-])\d+(\.[a-zA-Z0-9]+)?$', rf'\g<1>{padded_ep}\g<2>', SAMPLE_URL)
    
    # Extract clean file name
    filename = urllib.parse.unquote(url.split('/')[-1])
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[OK] Downloaded Ep {ep:03d} -> {filename}")
    except urllib.error.HTTPError as e:
        print(f"[404 / Error] Ep {ep:03d}: HTTP {e.code}")
        if e.code == 404:
            print("Reached end of available episodes. Stopping.")
            break
    except Exception as e:
        print(f"[Failed] Ep {ep:03d}: {e}")

print("Done!")