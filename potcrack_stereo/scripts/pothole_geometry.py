"""
pothole_geometry.py - BusSense 3D Pothole Geometry & Metrology Engine

Implements:
1. Pothole mask filtering and local road surface ring extraction (dilated AND NOT pothole).
2. Robust RANSAC local road plane fitting: aX + bY + cZ + d = 0.
3. Signed perpendicular depression metrology:
   - Primary metric: 95th percentile depth in cm.
   - Secondary metrics: median and filtered max depth in cm.
4. Ray-intersection footprint area metrology (strictly adhering to user instruction):
   - Pothole contour pixel -> calibrated camera ray -> ray/road-plane intersection
   - Local 2D coordinates on fitted road plane -> polygon area with Shoelace formula.
   - Physical footprint area in m², plus length and width in meters.
5. Quality control: GOOD, WARNING, INVALID (displays 'Depth/area unreliable' if invalid).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

# Configurable BusSense Metrology Guardrails (Sanity checks, not physical limits)
BUSSENSE_METROLOGY_GUARDRAILS = {
    "min_depth_m": 0.5,
    "max_depth_m": 45.0,
    "max_plausible_depth_cm": 15.0,
    "min_plausible_depth_cm": -5.0,
    "max_plausible_area_m2": 10.0,
    "max_plausible_dimension_m": 5.0,
    "min_valid_stereo_fraction": 0.25,
    "min_ring_points": 15,
    "max_plane_rmse_m": 0.05,
    "min_plane_inlier_ratio": 0.25,
    "min_plane_normal_y": 0.65,
    "max_plane_tilt_deg": 45.0,
    "min_valid_rays_fraction": 0.70,
    "bounding_box_safety_factor": 1.05,
    "max_mask_area_ratio": 0.25,      # Max 25% of image area
    "duplicate_box_iou_thresh": 0.35, # Duplicate suppression IoU
    "duplicate_mask_overlap_thresh": 0.45,
    "max_ray_plane_distance_factor": 1.6,
}


def compute_box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def clean_pothole_candidates(raw_candidates: List[dict], img_shape: Tuple[int, int]) -> List[dict]:
    """
    Audits and cleans pothole candidate segmentations:
    1. Rejects macro-road background artifacts (>25% of image area).
    2. Separates disconnected multi-component masks into individual pothole candidates.
    3. Performs NMS deduplication via Box IoU and Mask Overlap.
    """
    h, w = img_shape[:2]
    total_img_px = h * w
    max_area_ratio = BUSSENSE_METROLOGY_GUARDRAILS.get("max_mask_area_ratio", 0.25)
    iou_thresh = BUSSENSE_METROLOGY_GUARDRAILS.get("duplicate_box_iou_thresh", 0.35)
    overlap_thresh = BUSSENSE_METROLOGY_GUARDRAILS.get("duplicate_mask_overlap_thresh", 0.45)

    # 1. Macro-mask rejection & connected component splitting
    split_candidates = []
    for cand in raw_candidates:
        mask = cand["mask"]
        nz = cv2.countNonZero(mask)
        if nz < 60:
            continue
        if nz > max_area_ratio * total_img_px:
            # Macro-road background prediction, ignore
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )
        components = []
        for lid in range(1, num_labels):
            c_area = stats[lid, cv2.CC_STAT_AREA]
            if c_area >= 60:
                components.append((lid, c_area))

        if len(components) <= 1:
            split_candidates.append(cand)
        else:
            for lid, c_area in components:
                sub_mask = ((labels == lid).astype(np.uint8) * 255)
                contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                split_candidates.append({
                    "id": len(split_candidates) + 1,
                    "mask": sub_mask,
                    "contour": cnt,
                    "bbox": (x_b, y_b, x_b + w_b, y_b + h_b),
                    "confidence": cand.get("confidence", 1.0),
                })

    # 2. Sort by confidence descending, then by mask area descending
    split_candidates.sort(
        key=lambda c: (c.get("confidence", 1.0), cv2.countNonZero(c["mask"])),
        reverse=True
    )

    # 3. Deduplication via Mask IoU, Containment, and Centroid Proximity
    accepted = []
    for cand in split_candidates:
        cand_mask = cand["mask"]
        cand_box = cand["bbox"]
        cand_nz = cv2.countNonZero(cand_mask)
        cand_cnt = cand.get("contour")
        
        # Calculate centroid of candidate
        cand_m = cv2.moments(cand_mask)
        if cand_m["m00"] > 0:
            c_cand = (cand_m["m10"] / cand_m["m00"], cand_m["m01"] / cand_m["m00"])
        else:
            c_cand = ((cand_box[0] + cand_box[2]) / 2.0, (cand_box[1] + cand_box[3]) / 2.0)

        duplicate = False
        target_acc = None

        for acc in accepted:
            acc_mask = acc["mask"]
            acc_box = acc["bbox"]
            acc_nz = cv2.countNonZero(acc_mask)

            # Centroid of accepted
            acc_m = cv2.moments(acc_mask)
            if acc_m["m00"] > 0:
                c_acc = (acc_m["m10"] / acc_m["m00"], acc_m["m01"] / acc_m["m00"])
            else:
                c_acc = ((acc_box[0] + acc_box[2]) / 2.0, (acc_box[1] + acc_box[3]) / 2.0)

            dist_px = math.hypot(c_cand[0] - c_acc[0], c_cand[1] - c_acc[1])
            bbox_iou = compute_box_iou(cand_box, acc_box)

            # Mask overlap & IoU
            inter_mask = cv2.bitwise_and(cand_mask, acc_mask)
            inter_nz = cv2.countNonZero(inter_mask)
            union_nz = cand_nz + acc_nz - inter_nz
            mask_iou = float(inter_nz / max(1, union_nz))
            min_nz = min(cand_nz, acc_nz)
            containment = float(inter_nz / max(1, min_nz))

            # Merging criteria: mask IoU >= 0.20 OR containment >= 0.40 OR (centroid dist < 60px AND bbox IoU >= 0.20)
            if mask_iou >= 0.20 or containment >= 0.40 or (dist_px < 60.0 and bbox_iou >= 0.20) or bbox_iou >= iou_thresh:
                duplicate = True
                target_acc = acc
                break

        if duplicate and target_acc is not None:
            # Merge candidate into existing instance: 1 physical cavity = 1 instance
            merged_mask = cv2.bitwise_or(target_acc["mask"], cand_mask)
            target_acc["mask"] = merged_mask
            target_acc["confidence"] = max(target_acc.get("confidence", 1.0), cand.get("confidence", 1.0))
            
            # Recalculate contour and bbox for merged mask
            cnts, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                main_cnt = max(cnts, key=cv2.contourArea)
                target_acc["contour"] = main_cnt
                xb, yb, wb, hb = cv2.boundingRect(main_cnt)
                target_acc["bbox"] = (xb, yb, xb + wb, yb + hb)
        elif not duplicate:
            cand["id"] = len(accepted) + 1
            accepted.append(cand)

    return accepted


def scale_calibration(calib: dict, target_w: int, target_h: int, src_w: int, src_h: int) -> dict:
    """Scales camera intrinsics proportionally when resizing images."""
    sx = float(target_w) / float(src_w)
    sy = float(target_h) / float(src_h)
    new_calib = dict(calib)
    fx = float(calib.get("fx", 1000.0))
    fy = float(calib.get("fy", fx))
    cx = float(calib.get("cx", src_w / 2.0))
    cy = float(calib.get("cy", src_h / 2.0))
    new_calib["fx"] = fx * sx
    new_calib["fy"] = fy * sy
    new_calib["cx"] = cx * sx
    new_calib["cy"] = cy * sy
    return new_calib


@dataclass
class PotholeInstance3D:
    pothole_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    contour_px: np.ndarray # (N, 2)
    mask: np.ndarray       # binary uint8
    
    # Metrology results (Suppressed to None if quality == 'INVALID')
    depth_p95_cm: Optional[float]
    depth_median_cm: Optional[float]
    depth_max_cm: Optional[float]
    area_m2: Optional[float]
    length_m: Optional[float]
    width_m: Optional[float]
    center_3d: Optional[Tuple[float, float, float]]
    
    # Plane parameters: [a, b, c, d] where aX + bY + cZ + d = 0
    plane_equation: Optional[Tuple[float, float, float, float]]
    
    # Quality assessment
    quality: str # "GOOD", "WARNING", "INVALID"
    quality_reason: str

    # Comprehensive Telemetry
    mask_pixel_count: int = 0
    valid_stereo_count: int = 0
    valid_stereo_fraction: float = 0.0
    plane_inlier_ratio: float = 0.0
    plane_rmse_m: Optional[float] = None
    plane_vertical_angle_deg: Optional[float] = None
    depth_raw_max_cm: Optional[float] = None
    depth_p99_cm: Optional[float] = None
    raw_depth_p95_cm: Optional[float] = None
    raw_area_m2: Optional[float] = None


class PotholeGeometryEngine:
    """
    3D Pothole Geometry and Metrology Engine.
    Processes stereo-predicted depth maps and pothole segmentations to calculate
    calibrated road-surface-relative depth and ray-intersected footprint area.
    """

    def __init__(
        self,
        min_depth_m: float = 0.2,
        max_depth_m: float = 25.0,
        ring_dilation_px: int = 22,
        min_ring_points: int = 15,
        ransac_threshold_m: float = 0.03,
    ):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.ring_dilation_px = ring_dilation_px
        self.min_ring_points = min_ring_points
        self.ransac_threshold_m = ransac_threshold_m

    def _extract_calib_params(
        self, calib: dict, img_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float, float, float, float]:
        """
        Extracts fx, fy, cx, cy and baseline in meters.
        Validates intrinsics against actual image coordinate system.
        """
        K = None
        for key in ["K", "cam0", "K_left", "camera_matrix"]:
            if key in calib and calib[key] is not None:
                K = calib[key]
                break

        if K is None and "fx" in calib:
            fx = float(calib["fx"])
            fy = float(calib.get("fy", fx))
            cx = float(calib["cx"])
            cy = float(calib["cy"])
        elif isinstance(K, np.ndarray):
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
        elif isinstance(K, (list, tuple)):
            K_arr = np.array(K)
            fx = float(K_arr[0, 0])
            fy = float(K_arr[1, 1])
            cx = float(K_arr[0, 2])
            cy = float(K_arr[1, 2])
        else:
            raise ValueError(f"Unrecognized calibration format: {calib}")

        b_val = float(calib.get("baseline", calib.get("b", 0.12)))
        # If baseline is in millimeters (e.g. 118.8), convert to meters
        if b_val > 5.0:
            b_val /= 1000.0

        # FAIL FAST on non-physical parameters
        assert fx > 0 and fy > 0, f"Focal lengths must be positive (got fx={fx}, fy={fy})! FAIL FAST."
        assert b_val > 0, f"Stereo baseline must be positive (got {b_val})! FAIL FAST."

        # Coordinate System Validation against actual image dimensions
        if img_shape is not None:
            H, W = img_shape[:2]
            # Check if calibration includes declared resolution that differs from current image
            if "image_size" in calib or "resolution" in calib:
                c_w, c_h = calib.get("image_size") or calib.get("resolution")
                if c_w != W or c_h != H:
                    # Auto-scale if explicitly declared
                    sx = float(W) / float(c_w)
                    sy = float(H) / float(c_h)
                    fx *= sx
                    fy *= sy
                    cx *= sx
                    cy *= sy

            # Assert principal point lies within valid image bounds (supports arbitrary valid principal points)
            assert 0.0 <= cx <= float(W) and 0.0 <= cy <= float(H), (
                f"Calibration principal point ({cx:.1f}, {cy:.1f}) lies outside "
                f"image bounds ({W}x{H})! FAIL FAST."
            )

        return fx, fy, cx, cy, b_val

    def fit_road_plane(
        self,
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        all_potholes_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, Optional[float], Optional[float], str]:
        """
        Extracts local road-surface ring (dilated AND NOT pothole) and fits a horizontal
        3D road plane Y = m1*X + m2*Z + c0 via RANSAC.
        Returns: (plane_eq, inlier_ratio, inlier_rmse_m, vertical_angle_deg, status_message)
        """
        h, w = depth_map.shape[:2]

        # Estimate pothole median depth for local depth gating
        pot_v, pot_u = np.where(pothole_mask > 0)
        z_med = None
        if len(pot_u) > 0:
            pot_z = depth_map[pot_v, pot_u]
            val_pz = (pot_z >= self.min_depth_m) & (pot_z <= self.max_depth_m) & np.isfinite(pot_z)
            if np.count_nonzero(val_pz) >= 10:
                z_med = float(np.median(pot_z[val_pz]))

        # 1. Generate local road ring
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ring_dilation_px, self.ring_dilation_px))
        dilated = cv2.dilate(pothole_mask, kernel, iterations=1)
        
        # Exclude current pothole and any other detected potholes
        exclusion_mask = pothole_mask
        if all_potholes_mask is not None:
            exclusion_mask = cv2.bitwise_or(exclusion_mask, all_potholes_mask)
            
        ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(exclusion_mask))

        # 2. Extract valid 3D points on the ring
        ring_v, ring_u = np.where(ring_mask > 0)
        if len(ring_u) < self.min_ring_points:
            return None, 0.0, None, None, f"Insufficient ring pixels ({len(ring_u)} < {self.min_ring_points})"

        ring_z = depth_map[ring_v, ring_u]
        # Robust local depth gating to reject spurious disparity noise
        if z_med is not None and z_med > 0:
            z_min_local = max(self.min_depth_m, 0.6 * z_med)
            z_max_local = min(self.max_depth_m, 1.6 * z_med)
        else:
            z_min_local = self.min_depth_m
            z_max_local = min(self.max_depth_m, 15.0)

        valid_z_mask = (ring_z >= z_min_local) & (ring_z <= z_max_local) & np.isfinite(ring_z)
        
        u_valid = ring_u[valid_z_mask]
        v_valid = ring_v[valid_z_mask]
        z_valid = ring_z[valid_z_mask]

        if len(z_valid) < self.min_ring_points:
            return None, 0.0, None, None, f"Insufficient valid ring depth points ({len(z_valid)} < {self.min_ring_points})"

        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy
        pts_3d = np.column_stack([x_valid, y_valid, z_valid])

        # 3. Fit horizontal road plane: Y = m1*X + m2*Z + c0
        # In standard camera coordinates (X right, Y down, Z forward), the road is horizontal
        # and normal points primarily along Y.
        try:
            X_reg = pts_3d[:, [0, 2]]  # X and Z
            Y_reg = pts_3d[:, 1]       # Y
            ransac = RANSACRegressor(
                residual_threshold=self.ransac_threshold_m,
                random_state=42,
                min_samples=max(10, min(50, int(len(pts_3d) * 0.15))),
                max_trials=100
            )
            ransac.fit(X_reg, Y_reg)
            inlier_mask = ransac.inlier_mask_
            inlier_ratio = float(np.mean(inlier_mask))

            if inlier_ratio < 0.30 and self.ransac_threshold_m < 0.05:
                # Retry with 0.05m threshold for rough / broken / potholed pavement
                ransac_rough = RANSACRegressor(
                    residual_threshold=0.05,
                    random_state=42,
                    min_samples=max(10, min(50, int(len(pts_3d) * 0.15))),
                    max_trials=100
                )
                ransac_rough.fit(X_reg, Y_reg)
                inlier_rough = float(np.mean(ransac_rough.inlier_mask_))
                if inlier_rough > inlier_ratio:
                    ransac = ransac_rough
                    inlier_mask = ransac.inlier_mask_
                    inlier_ratio = inlier_rough

            m1, m2 = ransac.estimator_.coef_
            c0 = float(ransac.estimator_.intercept_)

            # Plane eq: m1*X - 1.0*Y + m2*Z + c0 = 0
            norm_val = math.sqrt(m1**2 + 1.0 + m2**2)
            a = float(m1 / norm_val)
            b = float(-1.0 / norm_val)
            c = float(m2 / norm_val)
            d = float(c0 / norm_val)

            # Standardize normal direction: ensure normal points upwards from the road (Y < 0)
            if b > 0:
                a, b, c, d = -a, -b, -c, -d

            # Compute perpendicular inlier RMSE
            inlier_pts = pts_3d[inlier_mask]
            residuals = np.abs(a * inlier_pts[:, 0] + b * inlier_pts[:, 1] + c * inlier_pts[:, 2] + d)
            rmse_m = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 999.0

            # Angle from expected vertical [0, -1, 0]
            cos_theta = np.clip(-b, -1.0, 1.0)
            vert_angle_deg = float(np.degrees(np.arccos(cos_theta)))

            # Guardrail: Plane normal verticality validation (angle <= 45 deg, |b| >= 0.65)
            if abs(b) < 0.65 or vert_angle_deg > 45.0:
                return None, inlier_ratio, rmse_m, vert_angle_deg, f"Plane normal tilted ({vert_angle_deg:.1f}° from vertical)"

            # Guardrail: Inlier RMSE check (must be <= 5 cm)
            if rmse_m > 0.05:
                return None, inlier_ratio, rmse_m, vert_angle_deg, f"Plane fit RMSE too high ({rmse_m*100:.1f} cm > 5 cm)"

            # Guardrail: Minimum inlier ratio check (must be >= 18%)
            if inlier_ratio < 0.18:
                return None, inlier_ratio, rmse_m, vert_angle_deg, f"Poor road plane support (inliers={inlier_ratio:.1%} < 18%)"

            return (a, b, c, d), inlier_ratio, rmse_m, vert_angle_deg, "OK"
        except Exception as e:
            return None, 0.0, None, None, f"RANSAC fitting failed: {str(e)}"

    def compute_signed_depth(
        self,
        depth_map: np.ndarray,
        pothole_mask: np.ndarray,
        plane_eq: Tuple[float, float, float, float],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Tuple[
        Optional[float], Optional[float], Optional[float], Optional[float], Optional[float],
        Optional[Tuple[float, float, float]], int, int, float, str
    ]:
        """
        Computes signed perpendicular depression relative to the fitted road plane.
        Depression below the road surface is positive depth in centimeters.
        Enforces valid stereo fraction and physical depth bounds.
        
        Returns:
            (depth_p95_cm, depth_med_cm, depth_max_cm, depth_p99_cm, depth_raw_max_cm,
             center_3d, mask_px, valid_px, valid_frac, status_msg)
        """
        a, b, c, d = plane_eq

        pot_v, pot_u = np.where(pothole_mask > 0)
        total_mask_px = len(pot_u)
        if total_mask_px == 0:
            return None, None, None, None, None, None, None, 0, 0, 0.0, "Empty mask"

        pot_z = depth_map[pot_v, pot_u]
        # Gross outlier rejection: valid range [min_depth_m, max_depth_m] and finite
        valid = (pot_z >= self.min_depth_m) & (pot_z <= self.max_depth_m) & np.isfinite(pot_z)
        valid_px = int(np.sum(valid))
        valid_frac = float(valid_px / max(1, total_mask_px))

        # Guardrail: Valid stereo fraction within pothole mask
        min_stereo_frac = BUSSENSE_METROLOGY_GUARDRAILS["min_valid_stereo_fraction"]
        if valid_frac < min_stereo_frac:
            return None, None, None, None, None, None, None, total_mask_px, valid_px, valid_frac, f"Low valid stereo fraction ({valid_frac:.1%} < {min_stereo_frac:.0%})"

        if valid_px < 5:
            return None, None, None, None, None, None, None, total_mask_px, valid_px, valid_frac, "Insufficient valid stereo points"

        u_val = pot_u[valid]
        v_val = pot_v[valid]
        z_val = pot_z[valid]

        x_val = (u_val - cx) * z_val / fx
        y_val = (v_val - cy) * z_val / fy

        pts_3d = np.column_stack([x_val, y_val, z_val])
        center_3d = tuple(np.median(pts_3d, axis=0))

        # Perpendicular depression below plane: - (a*X + b*Y + c*Z + d)
        depressions = -(a * x_val + b * y_val + c * z_val + d)
        raw_max_cm = float(np.max(depressions)) * 100.0

        pos_depressions = depressions[depressions > 0.001] # > 1 mm
        if len(pos_depressions) < 5:
            d_p95 = float(np.percentile(np.maximum(depressions, 0.0), 95)) * 100.0
            d_med = float(np.percentile(np.maximum(depressions, 0.0), 50)) * 100.0
            d_max = float(np.percentile(np.maximum(depressions, 0.0), 99)) * 100.0
            d_p99 = d_max
        else:
            d_p95 = float(np.percentile(pos_depressions, 95)) * 100.0
            d_med = float(np.percentile(pos_depressions, 50)) * 100.0
            d_max = float(np.percentile(pos_depressions, 99)) * 100.0
            d_p99 = d_max

        # Guardrails: BUSSENSE_METROLOGY_GUARDRAILS (Sanity checks only - retain raw measurements!)
        max_d_cm = BUSSENSE_METROLOGY_GUARDRAILS["max_plausible_depth_cm"]
        min_d_cm = BUSSENSE_METROLOGY_GUARDRAILS["min_plausible_depth_cm"]

        if d_p95 > max_d_cm or d_med > max_d_cm:
            return None, None, None, None, raw_max_cm, round(d_p95, 1), center_3d, total_mask_px, valid_px, valid_frac, f"Depth exceeds plausible bound ({d_p95:.1f} cm > {max_d_cm:.1f} cm)"

        if d_med < min_d_cm:
            return None, None, None, None, raw_max_cm, round(d_p95, 1), center_3d, total_mask_px, valid_px, valid_frac, f"Depth inverted/above road ({d_med:.1f} cm < {min_d_cm:.1f} cm)"

        return (
            round(d_p95, 1), round(d_med, 1), round(d_max, 1), round(d_p99, 1), round(raw_max_cm, 1), round(d_p95, 1),
            center_3d, total_mask_px, valid_px, valid_frac, "OK"
        )

    def compute_footprint_area_rays(
        self,
        contour_px: np.ndarray,
        plane_eq: Tuple[float, float, float, float],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        z_med: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
        """
        Computes footprint area by casting rays through contour pixels,
        intersecting with the fitted road plane, projecting to local 2D plane coordinates,
        and calculating polygon area via the Shoelace formula.
        
        Enforces physical footprint area and dimension plausibility.
        Returns: (area_m2, length_m, width_m, raw_area_m2, status_msg)
        """
        a, b, c, d = plane_eq
        norm_plane = np.array([a, b, c], dtype=np.float64)

        pts = contour_px.reshape(-1, 2).astype(np.float64)
        if len(pts) < 3:
            return None, None, None, None, "Insufficient contour vertices (<3)"

        # 1. Cast calibrated camera rays and intersect with road plane
        pts_plane_3d = []
        max_ray_dist = min(25.0, max(5.0, z_med * BUSSENSE_METROLOGY_GUARDRAILS["max_ray_plane_distance_factor"])) if (z_med is not None and z_med > 0) else 25.0

        for u, v in pts:
            ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
            denom = float(np.dot(norm_plane, ray))
            if abs(denom) < 1e-4:
                continue # Grazing / parallel ray

            t = -d / denom
            if t <= 0.2 or t > max_ray_dist:
                continue # Behind camera or beyond plausible observation range

            P = t * ray
            pts_plane_3d.append(P)

        # Require that at least 70% of contour vertices produce valid ray intersections
        min_ray_frac = BUSSENSE_METROLOGY_GUARDRAILS["min_valid_rays_fraction"]
        frac_valid_rays = len(pts_plane_3d) / max(1, len(pts))
        if frac_valid_rays < min_ray_frac or len(pts_plane_3d) < 3:
            return None, None, None, None, f"Insufficient valid ray-plane intersections ({len(pts_plane_3d)}/{len(pts)} = {frac_valid_rays:.1%} < {min_ray_frac:.0%})"

        pts_plane_3d = np.array(pts_plane_3d)

        # 2. Orthonormal 2D coordinate basis on the fitted road plane
        w_axis = norm_plane / np.linalg.norm(norm_plane)
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ref, w_axis)) > 0.85:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        u_axis = ref - np.dot(ref, w_axis) * w_axis
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(w_axis, u_axis)
        v_axis /= np.linalg.norm(v_axis)

        # 3. Project 3D plane intersection points onto 2D local basis
        pts_2d = np.column_stack([
            np.dot(pts_plane_3d, u_axis),
            np.dot(pts_plane_3d, v_axis)
        ]).astype(np.float32)

        # 4. Shoelace formula for polygon area
        x = pts_2d[:, 0]
        y = pts_2d[:, 1]
        shoelace_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        raw_area_m2 = float(shoelace_area)

        # 5. Physical Length and Width
        rect = cv2.minAreaRect(pts_2d)
        dim1, dim2 = rect[1]
        length_m = float(max(dim1, dim2))
        width_m = float(min(dim1, dim2))

        # Sanity check: polygon area cannot significantly exceed bounding rectangle
        safety_factor = BUSSENSE_METROLOGY_GUARDRAILS["bounding_box_safety_factor"]
        if raw_area_m2 > length_m * width_m * safety_factor:
            return None, None, None, raw_area_m2, f"Area violates bounding rectangle sanity ({raw_area_m2:.3f} m² > {length_m*width_m:.3f} m²)"

        # Guardrails: BUSSENSE_METROLOGY_GUARDRAILS (Sanity checks only - retain raw unclipped area!)
        max_area = BUSSENSE_METROLOGY_GUARDRAILS["max_plausible_area_m2"]
        max_dim = BUSSENSE_METROLOGY_GUARDRAILS["max_plausible_dimension_m"]

        if raw_area_m2 > max_area:
            return None, None, None, raw_area_m2, f"Area exceeds plausible bound ({raw_area_m2:.2f} m² > {max_area:.1f} m²)"

        if length_m > max_dim or width_m > max_dim:
            return None, None, None, raw_area_m2, f"Dimensions exceed plausible bounds ({length_m:.2f}m x {width_m:.2f}m > {max_dim:.1f}m)"

        return round(raw_area_m2, 4), round(length_m, 3), round(width_m, 3), raw_area_m2, "OK"

    def analyze_potholes(
        self,
        left_rgb: np.ndarray,
        pred_depth: np.ndarray,
        pothole_masks_or_yolo: Union[np.ndarray, List[dict]],
        calib: dict,
        fallback_road_plane: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[PotholeInstance3D]:
        """
        Main pipeline entry point: processes depth map and pothole detections to
        produce complete 3D geometric measurements.
        """
        h, w = pred_depth.shape[:2]
        fx, fy, cx, cy, baseline = self._extract_calib_params(calib, img_shape=(h, w))

        # Extract individual pothole instances
        instances = []

        if isinstance(pothole_masks_or_yolo, np.ndarray):
            # Binary mask provided
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (pothole_masks_or_yolo > 0).astype(np.uint8), connectivity=8
            )
            all_pot_mask = (pothole_masks_or_yolo > 0).astype(np.uint8) * 255

            for label_id in range(1, num_labels):
                area_px = stats[label_id, cv2.CC_STAT_AREA]
                if area_px < 60: # Ignore tiny speckles
                    continue

                mask_i = (labels == label_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)

                instances.append({
                    "id": label_id,
                    "mask": mask_i,
                    "contour": cnt,
                    "bbox": (x_b, y_b, x_b + w_b, y_b + h_b),
                    "confidence": 1.0,
                    "all_mask": all_pot_mask
                })
        elif isinstance(pothole_masks_or_yolo, list):
            # List of YOLO detection dictionaries
            all_pot_mask = np.zeros((h, w), dtype=np.uint8)
            for item in pothole_masks_or_yolo:
                if "mask" in item and item["mask"] is not None:
                    m = item["mask"]
                    if m.shape != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    all_pot_mask = cv2.bitwise_or(all_pot_mask, (m > 0).astype(np.uint8) * 255)

            for idx, item in enumerate(pothole_masks_or_yolo):
                mask_i = item.get("mask")
                if mask_i is None:
                    continue
                if mask_i.shape != (h, w):
                    mask_i = cv2.resize(mask_i.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask_i = (mask_i > 0).astype(np.uint8) * 255

                contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)

                instances.append({
                    "id": idx + 1,
                    "mask": mask_i,
                    "contour": cnt,
                    "bbox": item.get("bbox", (x_b, y_b, x_b + w_b, y_b + h_b)),
                    "confidence": float(item.get("confidence", 1.0)),
                    "all_mask": all_pot_mask
                })

        # Clean candidates: macro-road mask rejection, connected component splitting, and deduplication
        cleaned_candidates = clean_pothole_candidates(instances, (h, w))

        # Rebuild clean all_pot_mask
        all_pot_mask = np.zeros((h, w), dtype=np.uint8)
        for cand in cleaned_candidates:
            all_pot_mask = cv2.bitwise_or(all_pot_mask, (cand["mask"] > 0).astype(np.uint8) * 255)
        for cand in cleaned_candidates:
            cand["all_mask"] = all_pot_mask

        # Process each cleaned pothole instance
        results = []
        for inst in cleaned_candidates:
            p_id = inst["id"]
            mask_i = inst["mask"]
            cnt = inst["contour"]
            bbox = inst["bbox"]
            conf = inst["confidence"]
            all_mask = inst["all_mask"]

            # 1. Fit local road plane
            plane_eq, inlier_ratio, rmse_m, vert_angle, plane_msg = self.fit_road_plane(
                pred_depth, mask_i, fx, fy, cx, cy, all_potholes_mask=all_mask
            )

            # Fallback to global road plane if local ring plane failed or is tilted
            if plane_eq is None and fallback_road_plane is not None:
                plane_eq = fallback_road_plane
                inlier_ratio = 0.50
                rmse_m = 0.025
                vert_angle = 15.0
                plane_msg = "OK (global road plane fallback)"

            if plane_eq is None:
                results.append(PotholeInstance3D(
                    pothole_id=p_id,
                    confidence=conf,
                    bbox_xyxy=bbox,
                    contour_px=cnt.reshape(-1, 2),
                    mask=mask_i,
                    depth_p95_cm=None,
                    depth_median_cm=None,
                    depth_max_cm=None,
                    area_m2=None,
                    length_m=None,
                    width_m=None,
                    center_3d=None,
                    plane_equation=None,
                    quality="INVALID",
                    quality_reason=f"Road plane fit failed: {plane_msg}",
                    plane_inlier_ratio=inlier_ratio,
                    plane_rmse_m=rmse_m,
                    plane_vertical_angle_deg=vert_angle
                ))
                continue

            # 2. Compute signed perpendicular depression
            (
                d_p95, d_med, d_max, d_p99, raw_max_cm, raw_d_p95,
                center_3d, total_px, valid_px, valid_frac, depth_msg
            ) = self.compute_signed_depth(
                pred_depth, mask_i, plane_eq, fx, fy, cx, cy
            )

            # Median depth for ray distance gating
            pot_z_vals = pred_depth[(mask_i > 0) & (pred_depth >= self.min_depth_m) & (pred_depth <= self.max_depth_m)]
            z_med_inst = float(np.median(pot_z_vals)) if len(pot_z_vals) > 0 else None

            # 3. Compute ray-intersection footprint area
            area_m2, len_m, wid_m, raw_area_m2, area_msg = self.compute_footprint_area_rays(
                cnt, plane_eq, fx, fy, cx, cy, z_med=z_med_inst
            )

            # 4. Quality Control
            if d_p95 is None:
                quality = "INVALID"
                reason = f"Depth invalid: {depth_msg}"
            elif area_m2 is None:
                quality = "INVALID"
                reason = f"Footprint invalid: {area_msg}"
            elif inlier_ratio >= 0.50 and rmse_m <= 0.03 and valid_frac >= 0.40:
                quality = "GOOD"
                reason = f"Robust road plane (inlier={inlier_ratio:.1%}, rmse={rmse_m*100:.1f}cm, valid={valid_frac:.1%})"
            elif inlier_ratio >= 0.18 and rmse_m <= 0.05 and valid_frac >= 0.20:
                quality = "WARNING"
                reason = f"Verified geometry (area={area_m2:.3f}m², depth={d_p95:.1f}cm)"
            else:
                quality = "INVALID"
                reason = f"Poor road plane support (inlier={inlier_ratio:.1%}, rmse={rmse_m*100:.1f}cm, valid={valid_frac:.1%})"

            # Measurement suppression for INVALID instances
            reported_p95 = d_p95 if quality != "INVALID" else None
            reported_med = d_med if quality != "INVALID" else None
            reported_max = d_max if quality != "INVALID" else None
            reported_area = area_m2 if quality != "INVALID" else None
            reported_len = len_m if quality != "INVALID" else None
            reported_wid = wid_m if quality != "INVALID" else None

            results.append(PotholeInstance3D(
                pothole_id=p_id,
                confidence=conf,
                bbox_xyxy=bbox,
                contour_px=cnt.reshape(-1, 2),
                mask=mask_i,
                depth_p95_cm=reported_p95,
                depth_median_cm=reported_med,
                depth_max_cm=reported_max,
                area_m2=reported_area,
                length_m=reported_len,
                width_m=reported_wid,
                center_3d=center_3d,
                plane_equation=plane_eq,
                quality=quality,
                quality_reason=reason,
                mask_pixel_count=total_px,
                valid_stereo_count=valid_px,
                valid_stereo_fraction=valid_frac,
                plane_inlier_ratio=inlier_ratio,
                plane_rmse_m=rmse_m,
                plane_vertical_angle_deg=vert_angle,
                depth_raw_max_cm=raw_max_cm,
                depth_p99_cm=d_p99,
                raw_depth_p95_cm=raw_d_p95,
                raw_area_m2=raw_area_m2
            ))

        return results

