import json

log_path = r'C:\Users\Thrinay\.gemini\antigravity\brain\03d7fdef-cb64-4066-b840-f0b324ee084e\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in [325, 326, 327, 335, 336, 337]:
            obj = json.loads(line)
            print(f"--- Line {i} Step {obj.get('step_index')} ---")
            print("Content:", str(obj.get('content'))[:300])
            for tc in obj.get('tool_calls', []):
                args = tc.get('args') or tc.get('parameters') or {}
                print("CMD:", args.get('CommandLine'))
