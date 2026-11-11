import urllib.request
import re

url = "https://nail-hnu.github.io/EvTTC/calibration/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')

# Print text inside <article> or headings
print("Calibration page text:")
clean = re.sub(r'<[^>]+>', ' ', html)
lines = [l.strip() for l in clean.split('\n') if l.strip()]
for line in lines[20:70]:
    print(line)

links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', html)
print("\nLinks on calibration page:")
for u, t in links:
    print(f"  {t}: {u}")
