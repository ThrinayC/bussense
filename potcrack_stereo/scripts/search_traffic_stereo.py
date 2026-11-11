import subprocess
import json

queries = [
    '"3D SBS" driving traffic',
    '"3D SBS" dashcam road',
    '"3D 180" driving traffic road',
    '"ZED 2" car driving traffic',
    '"stereo camera" driving in traffic',
    '"stereo vision" autonomous driving traffic',
    '"OAK-D" driving test traffic'
]

seen = set()
for q in queries:
    cmd = ['uv', 'tool', 'run', 'yt-dlp', f'ytsearch5:{q}', '--dump-json', '--flat-playlist']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
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
