import re

with open(r'C:\Users\Thrinay\.gemini\antigravity\brain\03d7fdef-cb64-4066-b840-f0b324ee084e\.system_generated\steps\2321\content.md', 'r', encoding='utf-8') as f:
    text = f.read()

urls = re.findall(r'https?://[^\s"\'<>]+', text)
print(f"Total URLs found: {len(urls)}")
for u in sorted(set(urls)):
    if any(k in u.lower() for k in ['zip', 'tar', 'download', 'file', 'rsrd', 'scidb', 'dataset']):
        print(u)
