import urllib.request

base = "https://sean-shiyuez.github.io/extreme-driving-dataset-web/"
demos = [
    "video_demo/demo_show/Complex_Traffic_Driving.mp4",
    "video_demo/demo_show/Critical_Driving.mp4",
    "video_demo/demo_show/Critical_Driving1.mp4",
    "video_demo/demo_show/Critical_Driving2.mp4",
]

for d in demos:
    url = base + d
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        print(f"[OK] {d} -> Content-Length: {res.headers.get('Content-Length')}")
    except Exception as e:
        print(f"[ERR] {d} -> {e}")
