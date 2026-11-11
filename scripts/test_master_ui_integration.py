"""
BusSense — Master UI & Evidence Integration Verification Suite
==============================================================
Validates:
1. Canonical event ID resolution across /api/events/{id} and /events/{id}.
2. Fallback resolution for legacy prefixed IDs (e.g., INFRA-EVT-DIV-002).
3. Local disk existence and HTTP 200 for all image frames and video clips.
4. Clean, evidence-first UI pages without giant decorative containers.
5. Elimination of internal debug jargon ('HONEST') in user-facing templates.
6. Verified presence of sidebar count elements.
"""

import sys
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"
REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_EVENTS = [
    "EVT-POT-001",
    "EVT-CRK-001",
    "EVT-WTR-001",
    "EVT-MRK-001",
    "EVT-ZBR-001",
    "EVT-SGN-001",
    "EVT-MSN-001",
    "EVT-DIV-001",
    "EVT-DIV-002",
    "EVT-HZD-001",
    "EVT-VEH-001",
    "EVT-VRU-001",
    "EVT-ANP-001",
    "EVT-MBU-001",
]

LEGACY_PREFIX_TESTS = [
    "INFRA-EVT-DIV-002",
    "INFRA-EVT-POT-001",
    "INFRA-EVT-WTR-001",
    "INFRA-EVT-SGN-001",
]

VERIFIED_VIDEOS = [
    "/outputs/videos/bussense_waterlogging_verified.mp4",
    "/outputs/videos/bussense_road_marking_health_verified_v2.mp4",
    "/outputs/videos/bussense_signs_verified.mp4",
    "/outputs/videos/bussense_missing_infrastructure_verified_v2.mp4",
    "/outputs/videos/bussense_divider_verified.mp4",
    "/outputs/videos/bussense_hazards_verified.mp4",
]


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "BusSenseTest/2.4"})
    with urllib.request.urlopen(req) as resp:
        return resp.getcode(), resp.read()


def test_server_health():
    print("[1/6] Testing Server Health...")
    code, data = http_get(f"{BASE_URL}/api/health")
    assert code == 200, f"Expected 200, got {code}"
    health = json.loads(data.decode("utf-8"))
    assert health.get("status") == "healthy", f"Health status was {health.get('status')}"
    print(f"  ✓ Server healthy. Total events: {health['storage']['total_events']}")


def test_canonical_event_ids_and_evidence():
    print("[2/6] Testing Canonical Event IDs & Local Evidence Assets...")
    passed = 0
    for ev_id in CANONICAL_EVENTS:
        # Test /api/events/{id}
        url = f"{BASE_URL}/api/events/{ev_id}"
        code, data = http_get(url)
        assert code == 200, f"Failed GET {url}: status {code}"
        ev = json.loads(data.decode("utf-8"))
        assert ev.get("event_id") == ev_id or ev_id in ev.get("event_id", ""), f"Mismatch ID: {ev.get('event_id')}"

        # Test /events/{id} route alias
        url_alias = f"{BASE_URL}/events/{ev_id}"
        code_alias, _ = http_get(url_alias)
        assert code_alias == 200, f"Failed GET {url_alias}: status {code_alias}"

        # Verify evidence image
        img_path = ev.get("image") or ev.get("best_frame") or ev.get("evidence_image_url")
        if img_path:
            clean_rel = img_path.lstrip("/").replace("/", os.sep)
            local_file = REPO_ROOT / clean_rel
            assert local_file.exists(), f"Image file not found on disk: {local_file}"
            img_code, _ = http_get(f"{BASE_URL}{img_path}")
            assert img_code == 200, f"HTTP fetch failed for image: {img_path}"

        # Verify evidence clip
        clip_path = ev.get("event_clip")
        if clip_path:
            clean_rel = clip_path.lstrip("/").replace("/", os.sep)
            local_file = REPO_ROOT / clean_rel
            assert local_file.exists(), f"Video clip file not found on disk: {local_file}"
            clip_code, _ = http_get(f"{BASE_URL}{clip_path}")
            assert clip_code == 200, f"HTTP fetch failed for clip: {clip_path}"

        passed += 1

    print(f"  ✓ All {passed}/{len(CANONICAL_EVENTS)} canonical events and evidence files verified 100%.")


def test_legacy_prefix_fallback():
    print("[3/6] Testing Legacy INFRA- Prefix Resolution...")
    for leg_id in LEGACY_PREFIX_TESTS:
        url = f"{BASE_URL}/api/events/{leg_id}"
        code, data = http_get(url)
        assert code == 200, f"Failed legacy prefix resolution for {leg_id}"
        ev = json.loads(data.decode("utf-8"))
        assert "event_id" in ev, f"Missing event_id in response for {leg_id}"
        print(f"  ✓ Legacy {leg_id} resolved -> {ev['event_id']}")


def test_verified_video_assets():
    print("[4/6] Testing Verified Domain Stream Videos...")
    for vid in VERIFIED_VIDEOS:
        clean_rel = vid.lstrip("/").replace("/", os.sep)
        local_file = REPO_ROOT / clean_rel
        assert local_file.exists(), f"Domain video missing on disk: {local_file}"
        code, _ = http_get(f"{BASE_URL}{vid}")
        assert code == 200, f"Failed HTTP 200 for domain video: {vid}"
        print(f"  ✓ Verified domain video exists and serves 200: {vid}")


def test_ui_cleanliness_and_debug_invariants():
    print("[5/6] Testing UI Cleanliness & Elimination of Debug Jargon...")
    domain_pages = [
        "bussense_frontend/js/pages/waterlogging.js",
        "bussense_frontend/js/pages/road-markings.js",
        "bussense_frontend/js/pages/traffic-signs.js",
        "bussense_frontend/js/pages/missing-infrastructure.js",
        "bussense_frontend/js/pages/road-dividers.js",
        "bussense_frontend/js/pages/road-hazards.js",
    ]

    for p in domain_pages:
        full_p = REPO_ROOT / p
        assert full_p.exists(), f"Page file does not exist: {p}"
        content = full_p.read_text(encoding="utf-8")

        # Invariant 1: No giant 380px black box showcase with debug lists
        assert 'style="grid-template-columns:1.8fr 1fr;background:#000;"' not in content, f"Giant dark box found in {p}"
        assert 'style="display:grid;grid-template-columns:1.8fr 1fr;background:#000;"' not in content, f"Giant dark box found in {p}"

        # Invariant 2: No 'HONEST' in HTML markup
        assert '>HONEST<' not in content, f"Debug word HONEST found in markup of {p}"

        # Invariant 3: Contains 'Evidence: Verified' or 'Verified'
        assert 'Evidence: Verified' in content or 'Verified' in content, f"Missing Verified status badge in {p}"

        print(f"  ✓ Page {p} passed evidence-first design check.")


def test_frontend_entrypoint_and_sidebar():
    print("[6/6] Testing Frontend Entrypoint & Sidebar Badges...")
    code, data = http_get(f"{BASE_URL}/")
    assert code == 200, f"Frontend root returned {code}"
    html = data.decode("utf-8")
    assert "BusSense Command Center" in html

    required_badges = [
        "sidebar-count-potholes",
        "sidebar-count-waterlogging",
        "sidebar-count-markings",
        "sidebar-count-vehicles",
        "sidebar-count-vru",
        "sidebar-count-traffic",
        "sidebar-count-incidents",
        "sidebar-count-signs",
        "sidebar-count-missing",
        "sidebar-count-dividers",
        "sidebar-count-hazards",
        "sidebar-count-multibus",
    ]

    for b in required_badges:
        assert f'id="{b}"' in html, f"Missing sidebar badge id '{b}' in index.html"

    # Also check window.openEventEvidence exists in script
    assert "window.openEventEvidence" in html, "window.openEventEvidence missing in index.html"
    print("  ✓ Frontend entrypoint and all 12 sidebar badges confirmed.")


if __name__ == "__main__":
    print("=" * 70)
    print("BUSSENSE MASTER UI & EVIDENCE INTEGRATION TEST SUITE")
    print("=" * 70)
    try:
        test_server_health()
        test_canonical_event_ids_and_evidence()
        test_legacy_prefix_fallback()
        test_verified_video_assets()
        test_ui_cleanliness_and_debug_invariants()
        test_frontend_entrypoint_and_sidebar()
        print("=" * 70)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! ZERO DEFECTS.")
        print("=" * 70)
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)
