import urllib.request
import re
import json

def fetch_evttc():
    url = "https://nail-hnu.github.io/EvTTC/download/download/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
    with open('evttc_download.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved evttc_download.html ({len(html)} bytes)")

    # Find sequences
    # Look for <strong>(CCR... | CPL... | CPN...)</strong>
    matches = re.findall(r'<strong>(CCRs-[^<]+|CCRm-[^<]+|CPLA-[^<]+|CPNA-[^<]+|CPNAO-[^<]+)</strong>', html)
    print(f"Found {len(matches)} EvTTC sequence mentions:")
    for m in matches[:15]:
        print("  -", m)

def fetch_extreme_driving():
    url = "https://sean-shiyuez.github.io/extreme-driving-dataset-web/"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
        with open('extreme_driving.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Saved extreme_driving.html ({len(html)} bytes)")
    except Exception as e:
        print(f"Failed to fetch Extreme Driving page: {e}")

if __name__ == "__main__":
    fetch_evttc()
    fetch_extreme_driving()
