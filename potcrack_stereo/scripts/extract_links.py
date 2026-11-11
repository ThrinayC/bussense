import re
import os

p = r"C:\Users\Integra\.gemini\antigravity\brain\bdb57915-d8a4-47a7-b023-aa7ea297fbb4\.system_generated\steps\1202\content.md"
with open(p, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

urls = set()
for m in re.finditer(r'https?://[^\s<>"\'\)]+', text):
    u = m.group(0)
    low = u.lower()
    if any(k in low for k in ["drive", "github", "dropbox", "kaggle", "pothole", "data"]):
        urls.add(u)

for u in sorted(urls):
    print(u)
