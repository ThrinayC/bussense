import subprocess
import json

vids = {
    "gopro_offroad": "IBtbb3Zz4Yg",
    "f150_table_mesa": "mQ11F8k-al0",
    "pothole_area_est": "cLvNj1Gbg58",
    "cambodia_rural": "cNmA6Ababps"
}

for name, vid in vids.items():
    cmd = ['uv', 'tool', 'run', 'yt-dlp', f'https://www.youtube.com/watch?v={vid}', '--dump-json']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    try:
        d = json.loads(p.stdout)
        title = d.get('title', '').encode('ascii', 'replace').decode('ascii')
        desc = d.get('description', '')[:200].replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
        print(f"=== {name} ({vid}, {d.get('duration')}s) ===")
        print(f"Title: {title}")
        print(f"Desc: {desc}")
    except Exception as e:
        print(f"Error {name}: {e}")
