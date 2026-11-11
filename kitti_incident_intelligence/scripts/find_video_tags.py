import re

with open('extreme_driving.html', 'r', encoding='utf-8') as f:
    html = f.read()

videos = re.findall(r'<video[^>]*>.*?</video>', html, re.DOTALL)
sources = re.findall(r'src=["\']([^"\']+\.mp4)["\']', html)
print(f"Found {len(videos)} video tags, {len(sources)} mp4 sources:")
for s in sources:
    print(" ", s)
