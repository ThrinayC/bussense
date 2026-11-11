import json

log_path = r'C:\Users\Thrinay\.gemini\antigravity\brain\03d7fdef-cb64-4066-b840-f0b324ee084e\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in [336, 337, 338, 339, 340]:
            obj = json.loads(line)
            content = obj.get('content', '')
            if content:
                print(f"Line {i}: {content[:1000]}")
