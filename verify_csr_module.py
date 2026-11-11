import urllib.request
import urllib.parse
import json
import sys

def get_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def post_json(url, data):
    payload = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def patch_json(url, data):
    payload = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='PATCH')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def main():
    print("=== BUSSENSE CSR MODULE VERIFICATION ===")
    
    # 1. Verify CSR Partners
    partners = get_json('http://localhost:8000/api/csr/partners')
    print(f"1. CSR Partners: {len(partners)} partners loaded successfully.")
    assert len(partners) >= 4, "Expected at least 4 CSR partners"
    for p in partners:
        print(f"   * {p['company_name']} ({p['company_id']}) | Avail: INR {p['available_budget_inr']:,} | Focus: {', '.join(p['focus_areas'])}")
    
    # 2. Verify CSR Requests
    requests = get_json('http://localhost:8000/api/csr/requests')
    print(f"\n2. Existing CSR Requests: {len(requests)} requests loaded.")
    for r in requests:
        print(f"   * {r['request_id']}: {r['company_name']} -> {r['pothole_event_id']} ({r.get('road_name', 'Corridor')}) | INR {r['requested_amount_inr']:,} | Status: {r['status']}")

    # 3. Verify Road Defects (Potholes & Cracks)
    road_events = get_json('http://localhost:8000/api/road-events?limit=50')
    potholes = [e for e in road_events if (e.get('event_type') or '').upper() == 'POTHOLE']
    cracks = [e for e in road_events if (e.get('event_type') or '').upper() == 'CRACK']
    print(f"\n3. Road Defects in Backend: {len(road_events)} total ({len(potholes)} potholes, {len(cracks)} cracks)")
    for pot in potholes:
        print(f"   * {pot.get('event_id')}: depth={pot.get('depth_cm')}cm, area={pot.get('area_m2')}m2, sev={pot.get('severity')}, condition={pot.get('road_condition')}, image={bool(pot.get('image'))}")

    # 4. Test Creating a New CSR Request
    print("\n4. Testing POST /api/csr/requests for EVT-POT-006...")
    new_req_payload = {
        "company_id": "TVS_MOBILITY_CSR",
        "company_name": "TVS Mobility Safety Foundation (DEMO CSR PARTNER)",
        "pothole_event_id": "EVT-POT-006",
        "gps_lat": 10.8228,
        "gps_lon": 78.6835,
        "depth_cm": 6.8,
        "area_m2": 0.45,
        "severity": "HIGH",
        "road_name": "Karur Bypass Road",
        "estimated_repair_cost_inr": 8500,
        "requested_amount_inr": 8500,
        "evidence_image": "/media/potholes/EVT-POT-006_frame.jpg",
        "notes": "Automated verification test request for Karur Bypass defect."
    }
    created_req = post_json('http://localhost:8000/api/csr/requests', new_req_payload)
    print(f"   * Created Request ID: {created_req['request_id']} | Status: {created_req['status']}")
    assert created_req['status'] == 'REQUESTED'
    
    # 5. Test Status Advancement (PATCH /api/csr/requests/{id}/status)
    req_id = created_req['request_id']
    print(f"\n5. Testing PATCH status advancement for {req_id}...")
    updated_1 = patch_json(f'http://localhost:8000/api/csr/requests/{req_id}/status', {"status": "UNDER REVIEW"})
    print(f"   * Advanced to: {updated_1['status']}")
    assert updated_1['status'] == 'UNDER REVIEW'
    
    updated_2 = patch_json(f'http://localhost:8000/api/csr/requests/{req_id}/status', {"status": "APPROVED"})
    print(f"   * Advanced to: {updated_2['status']}")
    assert updated_2['status'] == 'APPROVED'

    # 6. Verify Frontend Static Assets via HTTP 8080
    print("\n6. Verifying Frontend Static Assets on http://localhost:8080...")
    frontend_files = [
        '/',
        '/js/csr-cost-model.js',
        '/js/api.js',
        '/js/pages/road-health.js',
        '/js/event-modal.js',
        '/css/design-system.css'
    ]
    for path in frontend_files:
        req = urllib.request.Request(f'http://localhost:8080{path}')
        with urllib.request.urlopen(req) as resp:
            content_len = len(resp.read())
            print(f"   * http://localhost:8080{path}: HTTP {resp.status} ({content_len} bytes)")
            assert resp.status == 200

    print("\n>>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == '__main__':
    main()
