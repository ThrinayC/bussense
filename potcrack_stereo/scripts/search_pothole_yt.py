import subprocess
import json

queries = [
    'pothole "stereo camera"',
    '"pothole detection" stereo',
    'pothole "ZED"',
    'pothole "ZED 2"',
    '"potholes" "3D SBS"',
    'pothole 3D camera driving',
    'pothole detection "stereo vision"',
    'pothole dataset stereo',
    'autonomous driving potholes 3D'
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
