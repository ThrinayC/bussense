import re

def parse_evttc():
    with open('evttc_download.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Find sequence blocks and their download links
    print("=== EvTTC Sequences ===")
    pattern = r'<li>\s*<strong>([^<]+)</strong>:\s*(.*?)</li>'
    items = re.findall(pattern, html, re.DOTALL)
    print(f"Found {len(items)} sequence entries.")
    for name, links_html in items[:10]:
        links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', links_html)
        link_str = ", ".join([f"{text}: {url}" for url, text in links])
        print(f"\nSequence: {name.strip()}")
        print(f"  Links: {link_str}")

def parse_extreme_driving():
    with open('extreme_driving.html', 'r', encoding='utf-8') as f:
        html = f.read()
    print("\n=== Extreme Driving Dataset ===")
    # Extract links, github links, download links, google drive or huggingface
    links = re.findall(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', html)
    print(f"Found {len(links)} links:")
    for url, text in links:
        if any(w in url.lower() or w in text.lower() for w in ['download', 'drive', 'huggingface', 'github', 'zenodo', 'data', 'box']):
            print(f"  {text.strip()}: {url}")

if __name__ == "__main__":
    parse_evttc()
    parse_extreme_driving()
