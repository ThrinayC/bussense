import json

log_path = r'C:\Users\Thrinay\.gemini\antigravity\brain\03d7fdef-cb64-4066-b840-f0b324ee084e\.system_generated\logs\transcript.jsonl'
with open(log_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'long_debug_5' in line or 'check_frame_95' in line:
            obj = json.loads(line)
            print(f"Line {i}: step={obj.get('step_index')} type={obj.get('type')}")
            for tc in obj.get('tool_calls', []):
                args = tc.get('args') or tc.get('parameters') or {}
                cmd = args.get('CommandLine', '')
                if cmd:
                    print(f"  CMD: {cmd}")
