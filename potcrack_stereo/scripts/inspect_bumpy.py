import subprocess
import json

vids = {
    "casares_spain": "6gtp3ScwMNQ",
    "kandao_driving": "t8cAlr-O2l0",
    "car_ride_sbs": "scqHl0Ljpak",
    "3d_offroad_107s": "5XLN15MXi98",
    "3d_offroad_48s": "7MaQ1bfkcbg"
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
