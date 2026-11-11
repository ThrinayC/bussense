import re

with open('extreme_driving.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Look for any url links in html
urls = re.findall(r'href=["\']([^"\']+)["\']', html)
print("All URLs in extreme_driving.html:")
for u in set(urls):
    print(" ", u)

# Check text for mentions of download or drive or baidu
for line in html.split('\n'):
    if any(k in line.lower() for k in ['download', 'drive', 'baidu', 'pan', 'onedrive', 'box', 'data']):
        print(" MATCH:", re.sub(r'<[^>]+>', ' ', line).strip())
