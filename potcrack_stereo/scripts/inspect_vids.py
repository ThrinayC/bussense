import subprocess
import json

vids = ['TeYCVHm3u_g', 'dqTsCVU4xc0', 'kzIswymBWtE', '0pDbV6IDwLE', 'RaoWS_yiF4g', 'lf8nYOKtw4E', 'X6yBA9e5r2U', 'Acgt9Ar4NTg']

for vid in vids:
    cmd = ['uv', 'tool', 'run', 'yt-dlp', f'https://www.youtube.com/watch?v={vid}', '--dump-json']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    try:
        d = json.loads(p.stdout)
        title = d.get('title', '').encode('ascii', 'replace').decode('ascii')
        desc = d.get('description', '')[:200].replace('\n', ' ').encode('ascii', 'replace').decode('ascii')
        print(f"=== {vid} ({d.get('duration')}s) ===")
        print(f"Title: {title}")
        print(f"Desc: {desc}")
    except Exception as e:
        print(f"Error {vid}: {e}")
