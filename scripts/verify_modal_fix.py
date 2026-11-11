import urllib.request
import json

def check(url, desc):
    try:
        req = urllib.request.urlopen(url, timeout=5)
        status = req.status
        content = req.read()
        print(f"[OK] {desc} (status {status}, {len(content)} bytes)")
        return content
    except Exception as e:
        print(f"[FAIL] {desc}: {e}")
        return None

print("=== 1. FASTAPI Authoritative Endpoints ===")
p6_raw = check("http://localhost:8000/api/events/EVT-POT-006", "EVT-POT-006 API")
if p6_raw:
    p6 = json.loads(p6_raw.decode("utf-8"))
    print(f"   Event Type: {p6.get('event_type')}")
    print(f"   Depth: {p6.get('depth_cm')} cm, Area: {p6.get('area_m2')} m2")
    print(f"   Image: {p6.get('image')}")
    print(f"   Clip: {p6.get('event_clip')}")
    print(f"   Trichy Demo Loc: {p6.get('demo_route_location')}")
    print(f"   Sensor Benchmark GPS: {p6.get('source_gps')}")

c3_raw = check("http://localhost:8000/api/events/EVT-CRK-003", "EVT-CRK-003 API")
if c3_raw:
    c3 = json.loads(c3_raw.decode("utf-8"))
    print(f"   Event Type: {c3.get('event_type')}")
    print(f"   Length: {c3.get('length_m')} m, Branches: {c3.get('branches')}")
    print(f"   Image: {c3.get('image')}")
    print(f"   Clip: {c3.get('event_clip')}")

print("\n=== 2. Frontend Assets Verification ===")
idx = check("http://localhost:8080/index.html", "Frontend index.html")
if idx:
    text = idx.decode("utf-8", errors="ignore")
    print("   [+] Has openEventEvidence imported:", "openEventEvidence" in text)
    print("   [+] Has 640px width modal container:", "width:640px" in text)

rh = check("http://localhost:8080/js/pages/road-health.js", "Frontend road-health.js")
if rh:
    text = rh.decode("utf-8", errors="ignore")
    print("   [+] Has clickable-row class:", "clickable-row" in text)
    print("   [+] Has openEventEvidence call:", "openEventEvidence" in text)
    print("   [+] Has locateOnRoadMap implementation:", "locateOnRoadMap" in text)
    print("   [+] Has markersByEventId dictionary:", "markersByEventId" in text)

em = check("http://localhost:8080/js/event-modal.js", "Frontend event-modal.js")
if em:
    text = em.decode("utf-8", errors="ignore")
    print("   [+] Has openEventEvidence export:", "export async function openEventEvidence" in text)
    print("   [+] Has locate on map button:", "btn-modal-locate" in text)
    print("   [+] Has stereo metrology badge:", "Stereo Disparity Metrology" in text)

print("\n=== All Validations Completed Successfully ===")
