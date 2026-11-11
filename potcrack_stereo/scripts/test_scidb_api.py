import urllib.request
import json

urls_to_try = [
    "https://www.scidb.cn/api/detail?dataSetId=3f222c92e3af47539988affb6e987985",
    "https://www.scidb.cn/api/files?dataSetId=3f222c92e3af47539988affb6e987985",
    "https://www.scidb.cn/api/dataset/files?dataSetId=3f222c92e3af47539988affb6e987985",
    "https://www.scidb.cn/api/v1/dataset/3f222c92e3af47539988affb6e987985/files"
]

for url in urls_to_try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
            print(f"Success {url}: {len(data)} bytes")
            try:
                js = json.loads(data)
                print(str(js)[:500])
            except:
                pass
    except Exception as e:
        print(f"Failed {url}: {e}")
