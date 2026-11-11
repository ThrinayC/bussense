"""
Regression and Acceptance Tests for:
1. License Plate ANPR HUD presentation rules (Strict silence on unreadable plates, persistence, isolation)
2. VRU Cyclist vs. Pedestrian Semantic Disambiguation (Rider association, false alert suppression)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from scripts.calibration import KittiCalibration
from scripts.detector import Detection
from scripts.vehicle_tracker import TrackedObject, SceneTracker
from scripts.vru_engine import VRUEngine
from scripts.anpr_engine import ANPREngine


class DummyCalib:
    fx = 721.5377
    fy = 721.5377
    cx = 609.5593
    cy = 172.8540
    fB = 384.3815
    image_width = 1242
    image_height = 375


@pytest.fixture
def calib():
    return DummyCalib()


def test_readable_plate_appears(calib):
    """Test 1: When a plate is verified, HUD label displays 'PLATE: <REGISTRATION>'."""
    det = Detection(bbox=np.array([100, 100, 300, 250]), conf=0.9, cls_name="car", category="vehicle")
    trk = TrackedObject(track_id=1, detection=det, init_X=0.0, init_Z=5.0, frame_id=1, timestamp=0.1)

    # Initially unverified
    assert ANPREngine.get_hud_plate_label(trk) is None

    # Supply plate observations
    trk.update_plate_observation("KA01AB1234", conf=0.88, timestamp=0.1)
    assert trk.plate_verified is True
    assert trk.plate_text == "KA01AB1234"

    hud_label = ANPREngine.get_hud_plate_label(trk)
    assert hud_label == "PLATE: KA01AB1234"


def test_unreadable_plate_silent(calib):
    """Test 2: When plate is unreadable, HUD displays NOTHING related to plate.
    Strictly forbids 'UNREADABLE', 'UNKNOWN', 'NO PLATE', 'N/A'."""
    det = Detection(bbox=np.array([100, 100, 300, 250]), conf=0.9, cls_name="car", category="vehicle")
    trk = TrackedObject(track_id=2, detection=det, init_X=0.0, init_Z=18.0, frame_id=1, timestamp=0.1)

    # Distant vehicle, no plate reading
    hud_label = ANPREngine.get_hud_plate_label(trk)
    assert hud_label is None

    # Ensure forbidden strings are never returned
    forbidden = ["UNREADABLE", "UNKNOWN", "NO PLATE", "N/A", "OCR FAILED", "NULL"]
    if hud_label is not None:
        for f in forbidden:
            assert f not in hud_label.upper()


def test_side_by_side_isolation(calib):
    """Test 3: Two side-by-side vehicles: plate on car A does not leak or contaminate car B."""
    det_a = Detection(bbox=np.array([100, 100, 250, 250]), conf=0.9, cls_name="car", category="vehicle")
    det_b = Detection(bbox=np.array([300, 100, 450, 250]), conf=0.9, cls_name="car", category="vehicle")

    trkA = TrackedObject(track_id=10, detection=det_a, init_X=-1.5, init_Z=6.0, frame_id=1, timestamp=0.1)
    trkB = TrackedObject(track_id=20, detection=det_b, init_X=1.5, init_Z=6.0, frame_id=1, timestamp=0.1)

    trkA.update_plate_observation("DL01XY9999", conf=0.90, timestamp=0.1)

    assert ANPREngine.get_hud_plate_label(trkA) == "PLATE: DL01XY9999"
    assert ANPREngine.get_hud_plate_label(trkB) is None
    assert trkB.plate_text is None
    assert trkB.plate_verified is False


def test_cyclist_rider_composite(calib):
    """Test 4: When a person is riding a bicycle, system suppresses false PEDESTRIAN_IN_PATH
    and generates CYCLIST_IN_PATH. False alert count must be 0."""
    vru_engine = VRUEngine(calib=calib, corridor_max_range=25.0)

    # Bicycle in corridor
    det_bike = Detection(bbox=np.array([500, 160, 560, 240]), conf=0.85, cls_name="bicycle", category="vru")
    trk_bike = TrackedObject(track_id=1, detection=det_bike, init_X=0.2, init_Z=8.0, frame_id=1, timestamp=0.1)
    trk_bike.hits = 5

    # Rider detected as 'person', overlapping bike spatially
    det_rider = Detection(bbox=np.array([510, 130, 550, 200]), conf=0.80, cls_name="person", category="vru")
    trk_rider = TrackedObject(track_id=2, detection=det_rider, init_X=0.2, init_Z=8.1, frame_id=1, timestamp=0.1)
    trk_rider.hits = 5

    tracks = [trk_bike, trk_rider]
    events = vru_engine.evaluate_vru_tracks(tracks, ego_speed_mps=5.0)

    event_types = [ev["event_type"] for obs in events for ev in obs["events"]]
    ped_alerts = [e for e in event_types if e == "PEDESTRIAN_IN_PATH"]
    cyclist_alerts = [e for e in event_types if e == "CYCLIST_IN_PATH"]

    # Suppression assertion: 0 false pedestrian alerts for rider
    assert len(ped_alerts) == 0, f"False PEDESTRIAN_IN_PATH alert triggered for mounted rider: {ped_alerts}"
    assert len(cyclist_alerts) >= 1, "Expected CYCLIST_IN_PATH alert for cyclist composite entity"
    assert trk_rider.is_associated_rider is True
    assert trk_rider.associated_bike_id == trk_bike.track_id


def test_pedestrian_standalone(calib):
    """Test 5: Standalone pedestrian in corridor triggers PEDESTRIAN_IN_PATH."""
    vru_engine = VRUEngine(calib=calib, corridor_max_range=25.0)

    det_ped = Detection(bbox=np.array([520, 130, 560, 240]), conf=0.88, cls_name="person", category="vru")
    trk_ped = TrackedObject(track_id=5, detection=det_ped, init_X=0.1, init_Z=9.0, frame_id=1, timestamp=0.1)
    trk_ped.hits = 5

    events = vru_engine.evaluate_vru_tracks([trk_ped], ego_speed_mps=5.0)
    event_types = [ev["event_type"] for obs in events for ev in obs["events"]]

    assert "PEDESTRIAN_IN_PATH" in event_types
    assert "CYCLIST_IN_PATH" not in event_types


def test_cyclist_and_pedestrian_simultaneous(calib):
    """Test 6: Cyclist and distinct pedestrian simultaneously in scene: both alerted accurately."""
    vru_engine = VRUEngine(calib=calib, corridor_max_range=25.0)

    # Cyclist at center
    det_bike = Detection(bbox=np.array([500, 160, 560, 240]), conf=0.85, cls_name="bicycle", category="vru")
    trk_bike = TrackedObject(track_id=1, detection=det_bike, init_X=0.0, init_Z=7.0, frame_id=1, timestamp=0.1)
    trk_bike.hits = 5

    # Separate pedestrian to the side
    det_ped = Detection(bbox=np.array([200, 140, 240, 230]), conf=0.85, cls_name="person", category="vru")
    trk_ped = TrackedObject(track_id=2, detection=det_ped, init_X=-1.2, init_Z=10.0, frame_id=1, timestamp=0.1)
    trk_ped.hits = 5

    events = vru_engine.evaluate_vru_tracks([trk_bike, trk_ped], ego_speed_mps=5.0)
    obs_by_id = {obs["track_id"]: obs for obs in events}

    bike_evs = [e["event_type"] for e in obs_by_id[1]["events"]]
    ped_evs = [e["event_type"] for e in obs_by_id[2]["events"]]

    assert "CYCLIST_IN_PATH" in bike_evs
    assert "PEDESTRIAN_IN_PATH" in ped_evs


def test_occlusion_persistence(calib):
    """Test 7: Temporary occlusion/blur does not reset or flicker verified plate text."""
    det = Detection(bbox=np.array([100, 100, 300, 250]), conf=0.9, cls_name="car", category="vehicle")
    trk = TrackedObject(track_id=8, detection=det, init_X=0.0, init_Z=5.0, frame_id=1, timestamp=0.1)

    trk.update_plate_observation("MH12AB1234", conf=0.85, timestamp=0.1)
    assert ANPREngine.get_hud_plate_label(trk) == "PLATE: MH12AB1234"

    # Simulate 3 frames of prediction (occlusion / missed detection)
    trk.predict()
    trk.predict()
    trk.predict()

    # Plate remains verified and persistent during occlusion
    assert trk.plate_verified is True
    assert trk.plate_text == "MH12AB1234"
    assert ANPREngine.get_hud_plate_label(trk) == "PLATE: MH12AB1234"
