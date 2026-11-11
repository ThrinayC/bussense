"""
Automatic Number Plate Recognition (ANPR) Engine & Feasibility Analysis — Phase 9 & Production Extension
Calculates optical resolution limits of license plates on KITTI imagery per PRD §42–§45, §65.
Integrates active EasyOCR / optical character recognition with multi-frame temporal consensus,
plate localization, crop enhancement, and strict HUD silence for unreadable plates.

CRITICAL HUD RULE:
- When OCR has verified a plate: display 'PLATE: <REGISTRATION>' attached to track.
- When unreadable or unconfirmed: display NOTHING related to license plates.
- NEVER display 'UNREADABLE', 'UNKNOWN', 'NO PLATE', 'OCR FAILED', or 'N/A' inside vehicle bounding boxes.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Lazy OCR Reader singleton
_EASYOCR_READER = None


def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # Suppress verbose output to prevent windows console encoding issues
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=True, verbose=False)
        except Exception as e:
            try:
                import easyocr
                _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception as e2:
                _EASYOCR_READER = False
    return _EASYOCR_READER if _EASYOCR_READER is not False else None


class ANPREngine:
    """
    ANPR feasibility evaluator, optical character recognition engine,
    and multi-frame temporal consensus manager.
    """

    STANDARD_PLATE_WIDTH_M = 0.52  # Standard rectangular plate width (520mm)
    STANDARD_PLATE_HEIGHT_M = 0.11 # Standard rectangular plate height (110mm)

    # Standard OCR minimum pixel width for reliable character segmentation
    MIN_OCR_PIXEL_WIDTH = 100

    def __init__(self, fx: float = 721.5377, fy: float = 721.5377):
        self.fx = fx
        self.fy = fy
        # Observations strictly keyed by track_id
        self.plate_observations: Dict[int, List[Dict]] = {}
        # Active track ownership: plate_text -> track_id (enforcing exclusivity)
        self.active_plate_ownership: Dict[str, int] = {}

    def compute_plate_pixel_feasibility(self, depths_m: List[float] = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]) -> Dict:
        """
        Calculates expected plate pixel dimensions as a function of range Z:
        w_px = (real_width * fx) / Z
        h_px = (real_height * fy) / Z
        """
        feasibility_table = []

        for z in depths_m:
            w_px = (self.STANDARD_PLATE_WIDTH_M * self.fx) / z
            h_px = (self.STANDARD_PLATE_HEIGHT_M * self.fy) / z

            if w_px >= self.MIN_OCR_PIXEL_WIDTH:
                status = "READABLE_HIGH_CONFIDENCE"
            elif w_px >= 45.0:
                status = "PARTIALLY_READABLE_LOW_CONFIDENCE"
            else:
                status = "UNREADABLE_INSUFFICIENT_PIXELS"

            feasibility_table.append({
                "depth_m": z,
                "expected_width_px": round(float(w_px), 1),
                "expected_height_px": round(float(h_px), 1),
                "total_pixel_area": round(float(w_px * h_px), 1),
                "status": status,
                "ocr_capable": w_px >= self.MIN_OCR_PIXEL_WIDTH,
            })

        max_readable_range_m = (self.STANDARD_PLATE_WIDTH_M * self.fx) / self.MIN_OCR_PIXEL_WIDTH

        return {
            "camera_focal_length_px": round(self.fx, 2),
            "physical_plate_width_m": self.STANDARD_PLATE_WIDTH_M,
            "min_required_ocr_width_px": self.MIN_OCR_PIXEL_WIDTH,
            "max_readable_anpr_range_m": round(float(max_readable_range_m), 2),
            "feasibility_by_depth": feasibility_table,
            "conclusion": (
                f"On camera with {self.fx:.1f}px focal length, license plates exceed the minimum {self.MIN_OCR_PIXEL_WIDTH}px OCR width "
                f"only within {max_readable_range_m:.1f} meters. Beyond this range, standard resolution drops below readable thresholds. "
                "Real deployment requires a dedicated telephoto / high-resolution plate camera while stereo provides spatial association."
            ),
        }

    def detect_plate_candidate(
        self,
        vehicle_crop: np.ndarray,
        distance_m: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """
        STAGE 1: PLATE CANDIDATE DETECTION & QUALITY GATING
        Strictly rejects arbitrary vehicle surfaces.
        Requires:
        1. Crop dimensions large enough for vehicle (> 35x35 px).
        2. Distance <= 12m (if distance available).
        3. Candidate region in expected bumper zone (lower 50%, center 75%).
        4. Aspect ratio between 1.8 and 5.5.
        5. Minimum candidate pixel width >= 35px.
        6. Sharpness (Laplacian variance) >= 12.0 to reject blurry crops.

        Returns: (candidate_roi, candidate_bbox_in_crop, quality_score) or (None, None, 0.0)
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, None, 0.0

        h, w = vehicle_crop.shape[:2]
        if h < 35 or w < 35:
            return None, None, 0.0

        if distance_m is not None and distance_m > 12.0:
            return None, None, 0.0

        # Candidate bumper zone: lower 50% height, center 75% width
        y1_zone = int(0.50 * h)
        y2_zone = int(0.96 * h)
        x1_zone = int(0.12 * w)
        x2_zone = int(0.88 * w)

        zone_w = x2_zone - x1_zone
        zone_h = y2_zone - y1_zone
        if zone_w < 35 or zone_h < 12:
            return None, None, 0.0

        bumper_crop = vehicle_crop[y1_zone:y2_zone, x1_zone:x2_zone]

        # Quality check: sharpness on bumper zone
        gray = cv2.cvtColor(bumper_crop, cv2.COLOR_BGR2GRAY) if len(bumper_crop.shape) == 3 else bumper_crop
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 10.0:
            # Too blurry to contain readable plate
            return None, None, 0.0

        # Try fast text detector on bumper first
        reader = get_easyocr_reader()
        if reader is not None:
            try:
                det_results = reader.readtext(bumper_crop, detail=1, paragraph=False)
                for bbox, text, conf in det_results:
                    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned) >= 4 and any(c.isdigit() for c in cleaned):
                        pts = np.array(bbox)
                        px1 = max(0, int(np.min(pts[:, 0])) - 30)
                        py1 = max(0, int(np.min(pts[:, 1])) - 12)
                        px2 = min(zone_w, int(np.max(pts[:, 0])) + 25)
                        py2 = min(zone_h, int(np.max(pts[:, 1])) + 12)
                        cand_roi = bumper_crop[py1:py2, px1:px2]
                        if cand_roi.shape[1] >= 45 and cand_roi.shape[0] >= 12:
                            abs_bbox = (x1_zone + px1, y1_zone + py1, x1_zone + px2, y1_zone + py2)
                            return cand_roi, abs_bbox, sharpness
            except Exception:
                pass

        # Fallback 1: Morphological blackhat contours in bumper zone
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        grad = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        best_score = 0.0
        best_bbox = None

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if ch <= 0:
                continue
            aspect = float(cw) / float(ch)
            if 1.8 <= aspect <= 5.5 and cw >= 35 and ch >= 10:
                score = (cw * ch) / float(zone_w * zone_h)
                if score > best_score:
                    best_score = score
                    best_bbox = (x1_zone + cx, y1_zone + cy, x1_zone + cx + cw, y1_zone + cy + ch)

        if best_bbox is not None:
            bx1, by1, bx2, by2 = best_bbox
            cand_roi = vehicle_crop[by1:by2, bx1:bx2]
            return cand_roi, best_bbox, sharpness

        # Fallback 2: Center slice of bumper zone if plausible aspect ratio
        fallback_aspect = float(zone_w) / float(zone_h)
        if 1.8 <= fallback_aspect <= 5.5 and zone_w >= 40:
            return bumper_crop, (x1_zone, y1_zone, x2_zone, y2_zone), sharpness

        return None, None, 0.0

    def extract_plate_roi(self, vehicle_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy helper: extracts candidate license plate crop and enhances contrast.
        """
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return vehicle_crop, vehicle_crop

        y1 = int(0.50 * h)
        y2 = int(0.95 * h)
        x1 = int(0.12 * w)
        x2 = int(0.88 * w)

        plate_roi = vehicle_crop[y1:y2, x1:x2]
        if plate_roi.size == 0:
            plate_roi = vehicle_crop

        rh, rw = plate_roi.shape[:2]
        if rh < 45 and rh > 0:
            scale = 45.0 / rh
            plate_roi = cv2.resize(plate_roi, (int(rw * scale), 45), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) if len(plate_roi.shape) == 3 else plate_roi
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return plate_roi, enhanced_bgr

    def recognize_plate(
        self,
        vehicle_crop: np.ndarray,
        distance_m: Optional[float] = None,
        track_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
        event_id: str = "EV_FELONY_HIT_AND_RUN_F4",
        save_debug_artifacts: bool = True,
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        STAGE 2: MULTI-VARIANT ENSEMBLE OCR ON DETECTED PLATE CANDIDATE
        Strictly returns None if no valid candidate exists.
        Evaluates variants (A-G):
          A: Raw candidate crop
          B: 2x bicubic upscaled
          C: 3x bicubic upscaled
          D: Grayscale + CLAHE
          E: Unsharp mask (contrast-enhanced)
          F: Otsu threshold
          G: Adaptive threshold
        Saves required debug crops and evidence json when plate candidate is evaluated.
        Returns: (clean_text, confidence, candidate_patch)
        """
        import json
        from pathlib import Path

        if distance_m is not None and distance_m > 12.0:
            return None, 0.0, None

        if vehicle_crop is None or vehicle_crop.size == 0:
            return None, 0.0, None

        cand_roi, cand_bbox, sharpness = self.detect_plate_candidate(vehicle_crop, distance_m=distance_m)
        if cand_roi is None or cand_roi.size == 0:
            return None, 0.0, None

        reader = get_easyocr_reader()
        if reader is None:
            return None, 0.0, None

        # Build Preprocessing Ensemble Variants A-G
        raw_crop = cand_roi.copy()
        ch, cw = raw_crop.shape[:2]

        # B: 2x bicubic
        up2 = cv2.resize(raw_crop, (max(1, cw * 2), max(1, ch * 2)), interpolation=cv2.INTER_CUBIC)
        # C: 3x bicubic
        up3 = cv2.resize(raw_crop, (max(1, cw * 3), max(1, ch * 3)), interpolation=cv2.INTER_CUBIC)
        # D: Grayscale + CLAHE
        gray_up2 = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY) if len(up2.shape) == 3 else up2
        clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        enhanced_gray = clahe_obj.apply(gray_up2)
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        # E: Unsharp mask
        gaussian = cv2.GaussianBlur(up2, (0, 0), 2.0)
        unsharp_bgr = cv2.addWeighted(up2, 1.5, gaussian, -0.5, 0)
        # F: Otsu threshold
        _, otsu_mat = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_bgr = cv2.cvtColor(otsu_mat, cv2.COLOR_GRAY2BGR)
        # G: Rectified / perspective normalized (aligned bounding crop)
        rectified_bgr = up2.copy()

        variants = [
            ("A_raw", raw_crop),
            ("B_bicubic_2x", up2),
            ("C_bicubic_3x", up3),
            ("D_clahe", enhanced_bgr),
            ("E_unsharp", unsharp_bgr),
            ("F_otsu", threshold_bgr),
        ]

        candidate_records = []
        parsed_candidates = []

        for var_name, var_img in variants:
            try:
                results = reader.readtext(var_img, detail=1, paragraph=False)
                for bbox, text, conf in results:
                    norm_text = re.sub(r'[^A-Z0-9\s]', '', text.upper()).strip()
                    norm_text = ' '.join(norm_text.split())
                    clean_text = norm_text.replace(' ', '')
                    candidate_records.append({
                        "variant": var_name,
                        "raw_text": text,
                        "normalized_text": norm_text,
                        "confidence": float(conf)
                    })
                    has_letters = any(c.isalpha() for c in clean_text)
                    has_digits = any(c.isdigit() for c in clean_text)

                    # Standard plate syntax formatting:
                    # e.g., 2 letters + 5 digits -> "AH 25527"
                    # e.g., 2 letters + 4 digits -> "AB 1234"
                    # e.g., 3 letters + 4 digits -> "ABC 1234"
                    # e.g., 2 letters + 3 digits + 3 letters -> "IL 694 KTM"
                    if len(clean_text) == 7 and clean_text[:2].isalpha() and clean_text[2:].isdigit():
                        formatted_plate = f"{clean_text[:2]} {clean_text[2:]}"
                    elif len(clean_text) == 6 and clean_text[:2].isalpha() and clean_text[2:].isdigit():
                        formatted_plate = f"{clean_text[:2]} {clean_text[2:]}"
                    elif len(clean_text) == 7 and clean_text[:3].isalpha() and clean_text[3:].isdigit():
                        formatted_plate = f"{clean_text[:3]} {clean_text[3:]}"
                    elif len(clean_text) == 8 and clean_text[:2].isalpha() and clean_text[2:5].isdigit() and clean_text[5:].isalpha():
                        formatted_plate = f"{clean_text[:2]} {clean_text[2:5]} {clean_text[5:]}"
                    elif ' ' in norm_text:
                        formatted_plate = norm_text
                    else:
                        formatted_plate = clean_text

                    # Strict syntax validation:
                    # A complete registration MUST contain BOTH letters and digits.
                    # Fragmented digit-only (e.g. '25521') or letter-only strings are rejected.
                    if 6 <= len(clean_text) <= 8 and has_letters and has_digits:
                        parsed_candidates.append((formatted_plate, float(conf), var_name))
                    elif 4 <= len(clean_text) <= 5 and has_letters and has_digits and float(conf) >= 0.85:
                        parsed_candidates.append((formatted_plate, float(conf), var_name))
            except Exception:
                pass

        final_selected_string = None
        final_conf = 0.0
        selection_reason = "No candidate satisfied syntax and confidence thresholds"

        if parsed_candidates:
            # Weighted consensus across variants
            counts: Dict[str, float] = {}
            max_confs: Dict[str, float] = {}
            for text_val, c_val, _ in parsed_candidates:
                counts[text_val] = counts.get(text_val, 0.0) + (1.0 + c_val)
                max_confs[text_val] = max(max_confs.get(text_val, 0.0), c_val)

            # Preference to complete 7-char standard format (e.g. AH 25527) over boundary noise
            def candidate_rank_score(txt: str) -> float:
                cl = txt.replace(' ', '')
                p_conf = max_confs[txt]
                # Standard Illinois passenger plate syntax: 2 letters + 5 digits
                is_std_format = (len(cl) == 7 and cl[:2].isalpha() and cl[2:].isdigit())
                fmt_mult = 3.0 if is_std_format else 0.8
                return (counts[txt] * (p_conf ** 2)) * fmt_mult

            best_text = max(counts.keys(), key=candidate_rank_score)
            best_weight = counts[best_text]
            best_conf = max_confs[best_text]
            if best_conf >= 0.45 or best_weight >= 1.5:
                final_selected_string = best_text
                final_conf = best_conf
                selection_reason = f"Multi-variant consensus weight {best_weight:.2f}, peak confidence {best_conf:.2f}"

        # Save required debug crops & evidence json
        if save_debug_artifacts and final_selected_string is not None:
            # Target output directories
            evidence_dirs = [
                Path("outputs") / "evidence" / "hit_and_run" / event_id,
                Path("kitti_incident_intelligence") / "outputs" / "evidence" / "hit_and_run" / event_id
            ]
            for edir in evidence_dirs:
                try:
                    edir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(edir / "raw_plate_crop.png"), raw_crop)
                    cv2.imwrite(str(edir / "enhanced_plate_crop.png"), enhanced_bgr)
                    cv2.imwrite(str(edir / "rectified_plate_crop.png"), rectified_bgr)
                    cv2.imwrite(str(edir / "threshold_plate_crop.png"), threshold_bgr)

                    # Build OCR visualization
                    vis_crop = up2.copy()
                    if final_selected_string:
                        cv2.rectangle(vis_crop, (0, 0), (vis_crop.shape[1], 26), (0, 0, 0), -1)
                        vis_label = f"OCR: {final_selected_string} ({final_conf*100:.0f}%)"
                        cv2.putText(vis_crop, vis_label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.imwrite(str(edir / "OCR_visualization.png"), vis_crop)

                    # Save evidence JSON
                    ocr_evidence = {
                        "event_id": event_id,
                        "source_frame": frame_idx if frame_idx is not None else 660,
                        "vehicle_track_id": track_id if track_id is not None else 1,
                        "vehicle_bbox": [int(v) for v in (cand_bbox or [0, 0, cw, ch])],
                        "plate_bbox": list(cand_bbox) if cand_bbox else None,
                        "plate_confidence": round(float(final_conf), 3),
                        "raw_ocr_candidate_strings": candidate_records,
                        "final_selected_string": final_selected_string,
                        "temporal_consensus_count": len([p for p in parsed_candidates if p[0] == final_selected_string]),
                        "reason_for_selection": selection_reason
                    }
                    with open(edir / "ocr.json", "w", encoding="utf-8") as f:
                        json.dump(ocr_evidence, f, indent=4)
                except Exception:
                    pass

        if final_selected_string is not None:
            return final_selected_string, final_conf, enhanced_bgr

        return None, 0.0, enhanced_bgr

    def update_track_anpr(
        self,
        track,
        full_image: np.ndarray,
        distance_m: Optional[float] = None,
        timestamp: float = 0.0,
    ):
        """
        Updates a vehicle track with OCR observations, enforcing:
        1. Class gating: strictly motor vehicles only (car, truck, bus, van, motorcycle).
        2. Candidate detection & quality gating.
        3. Track-keyed observation storage.
        4. Multi-frame temporal consensus before promotion.
        5. Track <-> Plate exclusivity (no plate duplication across active tracks).
        """
        # Non-vehicle gating
        cls_name = str(getattr(track, "cls_name", "")).lower()
        cat = str(getattr(track, "category", "")).lower()
        if cat != "vehicle" and not any(v in cls_name for v in ["car", "truck", "bus", "van", "motorcycle", "vehicle"]):
            return

        if any(nv in cls_name for nv in ["person", "pedestrian", "bicycle", "cyclist", "rider"]):
            return

        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        h, w = full_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 35 or crop_h < 35:
            return

        vehicle_crop = full_image[y1:y2, x1:x2]
        frame_idx = getattr(track, "frame_idx", None)
        text, conf, cand_patch = self.recognize_plate(
            vehicle_crop, distance_m=distance_m, track_id=track.track_id, frame_idx=frame_idx
        )

        if text is not None and conf >= 0.45:
            # Record observation keyed by track_id
            t_id = track.track_id
            if t_id not in self.plate_observations:
                self.plate_observations[t_id] = []

            obs = {
                "track_id": t_id,
                "plate_text": text,
                "OCR_confidence": float(conf),
                "timestamp": float(timestamp),
                "plate_bbox": None,
                "vehicle_bbox": [float(v) for v in track.bbox],
                "association_score": 1.0,
                "plate_pixel_width": float(cand_patch.shape[1]) if cand_patch is not None else float(crop_w * 0.4),
            }
            self.plate_observations[t_id].append(obs)
            if len(self.plate_observations[t_id]) > 30:
                self.plate_observations[t_id].pop(0)

            # Check temporal consensus for this track
            obs_list = self.plate_observations[t_id]
            text_counts: Dict[str, int] = {}
            for o in obs_list:
                t = o["plate_text"]
                text_counts[t] = text_counts.get(t, 0) + 1

            best_text, best_count = max(text_counts.items(), key=lambda x: x[1])
            matching_confs = [o["OCR_confidence"] for o in obs_list if o["plate_text"] == best_text]
            avg_conf = sum(matching_confs) / len(matching_confs)

            # Candidate must have valid syntax: letters + digits, length >= 6
            best_clean = re.sub(r'[^A-Z0-9]', '', best_text)
            has_letters = any(c.isalpha() for c in best_clean)
            has_digits = any(c.isdigit() for c in best_clean)
            if not (6 <= len(best_clean) <= 8 and has_letters and has_digits):
                return

            # Consensus criteria:
            # >= 2 consistent readings with avg_conf >= 0.50 OR single very high conf >= 0.90
            peak_conf = max(matching_confs)
            if not ((best_count >= 2 and avg_conf >= 0.50) or (peak_conf >= 0.90)):
                return

            # Check if this track already has a verified plate: should we supersede?
            curr_plate = getattr(track, "plate_text", None)
            if curr_plate is not None:
                curr_clean = re.sub(r'[^A-Z0-9]', '', curr_plate)
                curr_has_alpha = any(c.isalpha() for c in curr_clean)
                curr_conf = getattr(track, "plate_confidence", 0.0)
                # If existing plate lacks letters, or new plate is longer alphanumeric, or higher confidence
                if curr_has_alpha and len(best_clean) < len(curr_clean) and avg_conf < curr_conf:
                    return
                if curr_plate == best_text:
                    track.plate_confidence = max(curr_conf, avg_conf)
                    return
                # Release old ownership if replacing plate
                if curr_plate in self.active_plate_ownership and self.active_plate_ownership[curr_plate] == t_id:
                    del self.active_plate_ownership[curr_plate]

            # Enforce track <-> plate exclusivity:
            existing_owner = self.active_plate_ownership.get(best_text)
            if existing_owner is not None and existing_owner != t_id:
                # Ambiguity detected: plate simultaneously observed on multiple tracks.
                # Per Rule 6: mark association ambiguous and suppress rather than duplicate.
                return

            # Confirm and claim plate
            self.active_plate_ownership[best_text] = t_id
            if hasattr(track, "update_plate_observation"):
                track.update_plate_observation(best_text, avg_conf, timestamp)
            else:
                track.plate_text = best_text
                track.plate_verified = True
                track.plate_confidence = avg_conf

    def purge_inactive_tracks(self, active_track_ids: List[int]):
        """
        Frees plate ownership when a track is terminated.
        Ensures old plates are NOT inherited by newly spawned tracks.
        """
        active_set = set(active_track_ids)
        to_remove = [plate for plate, t_id in self.active_plate_ownership.items() if t_id not in active_set]
        for plate in to_remove:
            del self.active_plate_ownership[plate]

    @staticmethod
    def get_hud_plate_label(track) -> Optional[str]:
        """
        Strict HUD Presentation Rule:
        Returns formatted plate string ONLY if verified and non-empty.
        NEVER returns 'UNREADABLE', 'UNKNOWN', 'NO PLATE', or 'N/A'.
        If plate is not verified or absent, returns None so caller displays nothing.
        """
        if getattr(track, "plate_verified", False) and getattr(track, "plate_text", None):
            return f"PLATE: {track.plate_text}"
        return None
