import subprocess
import json

queries = [
    '"ZED" driving test',
    '"ZED 2" road test',
    '"OAK-D" road test',
    '"OAK-D" driving',
    '"stereo vision" road surface condition',
    '"stereo camera" road roughness',
    'dashcam "potholes" SBS',
    'stereo dashcam pothole',
    'ZED camera vehicle pothole',
    'stereo dashcam "India"',
    'stereo camera car "bad road"',
    'ZED 2i "road"'
]

seen = set()
for q in queries:
    cmd = ['uv', 'tool', 'run', 'yt-dlp', f'ytsearch5:{q}', '--dump-json', '--flat-playlist']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        for line in p.stdout.strip().split('\n'):
            if not line: continue
            try:
                d = json.loads(line)
                vid = d.get('id')
                if vid and vid not in seen:
                    seen.add(vid)
                    title = d.get('title','').encode('ascii', 'replace').decode('ascii')
                    dur = d.get('duration', 0)
                    print(f"{vid} | {dur}s | {title} | ({q})")
            except Exception: pass
    except Exception as e:
        print(f"Error {q}: {e}")
