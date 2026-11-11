#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BusSense — Phase 2: Stereo Depth Module
Unified swappable interface for stereo disparity and metric depth estimation.

Supported models:
- Fast-FoundationStereo (NVIDIA CVPR foundation model, zero-shot real-time)
- StereoSGBM / Traditional (CPU/GPU fallback)
- IGEV / RAFT pluggable adapter

Standard API:
    estimate_disparity(left, right) -> np.ndarray
    estimate_depth(left, right, calibration) -> Tuple[np.ndarray, np.ndarray]
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Point to foundation stereo model weights if available
FOUNDATION_REPO = Path(r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\models\fast_foundation_stereo_repo")
FOUNDATION_WEIGHTS = Path(r"C:\Users\Integra\.gemini\antigravity\scratch\bussense\models\weights\fast_foundation_stereo\model_best_bp2_serialize.pth")

if FOUNDATION_REPO.exists() and str(FOUNDATION_REPO) not in sys.path:
    sys.path.insert(0, str(FOUNDATION_REPO))


class BaseStereoMatcher:
    """Abstract base class for swappable stereo matchers."""

    def estimate_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def estimate_depth(
        self, left: np.ndarray, right: np.ndarray, calib: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class FastFoundationStereoMatcher(BaseStereoMatcher):
    """
    Wrapper for NVIDIA Fast-FoundationStereo model.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        mixed_precision: bool = True,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.mixed_precision = mixed_precision and (self.device.type == "cuda")
        self.weights_path = weights_path or FOUNDATION_WEIGHTS
        self.model = None
        self.model_name = "Fast-FoundationStereo"
        self._load_model()

    def _load_model(self):
        if self.weights_path.exists() and self.weights_path.stat().st_size > 1000:
            try:
                # Add safe globals for PyTorch 2.6
                try:
                    import core.foundation_stereo
                    torch.serialization.add_safe_globals([core.foundation_stereo.FastFoundationStereo])
                except Exception:
                    pass

                loaded = torch.load(str(self.weights_path), map_location=self.device, weights_only=False)
                if isinstance(loaded, nn.Module):
                    self.model = loaded
                elif isinstance(loaded, dict) and "model" in loaded:
                    self.model = loaded["model"]
                else:
                    self.model = loaded

                self.model.eval()
                # Ensure OmegaConf args has normalize attribute
                if hasattr(self.model, "args"):
                    try:
                        from omegaconf import OmegaConf
                        OmegaConf.set_struct(self.model.args, False)
                    except Exception:
                        pass
                    try:
                        if not hasattr(self.model.args, "normalize") or "normalize" not in self.model.args:
                            self.model.args.normalize = True
                    except Exception:
                        pass
                print(f"[StereoDepth] Loaded {self.model_name} onto {self.device}")
            except Exception as e:
                print(f"[StereoDepth] Warning: Failed to load Fast-FoundationStereo ({e}). Fallback will be used.")
                self.model = None
        else:
            print(f"[StereoDepth] Weights not found at {self.weights_path}. Model will fallback to SGBM.")

    def estimate_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Estimates disparity map from left and right images.
        Both inputs are expected to be RGB or BGR numpy arrays (H x W x 3) or grayscale.
        Returns floating point disparity map (H x W) pixel-aligned to left image.
        """
        if self.model is None:
            # Fallback to SGBM
            return SGBMStereoMatcher().estimate_disparity(left, right)

        # Convert to RGB if BGR
        if len(left.shape) == 3 and left.shape[2] == 3:
            imgL = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            imgR = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        else:
            imgL = left
            imgR = right

        h_orig, w_orig = imgL.shape[:2]

        # Downsample for stereo cost volume if image is high-res (prevents CUDA OOM on 1080p)
        scale = 1.0
        if w_orig > 1000 or h_orig > 600:
            scale = min(960.0 / w_orig, 576.0 / h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            imgL_in = cv2.resize(imgL, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            imgR_in = cv2.resize(imgR, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            imgL_in = imgL
            imgR_in = imgR

        h_in, w_in = imgL_in.shape[:2]

        # Foundation stereo expects (B, 3, H, W) normalized to [0, 1] or standard ImageNet
        tL = torch.from_numpy(imgL_in).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        tR = torch.from_numpy(imgR_in).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad to multiple of 32 for UNet/feature pyramid
        pad_h = (32 - (h_in % 32)) % 32
        pad_w = (32 - (w_in % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            tL = F.pad(tL, (0, pad_w, 0, pad_h), mode="replicate")
            tR = F.pad(tR, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.no_grad():
            try:
                out = self.model(tL, tR, test_mode=True)
            except TypeError:
                out = self.model(tL, tR)

            if isinstance(out, (list, tuple)):
                disp_tensor = out[-1]
            elif isinstance(out, dict):
                disp_tensor = out.get("disp", out.get("disparity", list(out.values())[-1]))
            else:
                disp_tensor = out

            if len(disp_tensor.shape) == 4:
                disp_tensor = disp_tensor.squeeze(1) # (B, H, W)

            # Unpad
            if pad_h > 0 or pad_w > 0:
                disp_tensor = disp_tensor[:, :h_in, :w_in]

            disp = disp_tensor[0].cpu().numpy().astype(np.float32)

        # Upscale disparity back to original resolution if downsampled
        if scale != 1.0:
            disp = cv2.resize(disp, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) * (1.0 / scale)

        # Handle invalid disparity (zero, negative, NaN, inf)
        disp = np.nan_to_num(disp, nan=-1.0, posinf=-1.0, neginf=-1.0)
        disp[disp <= 0] = -1.0
        return disp

    def estimate_depth(
        self, left: np.ndarray, right: np.ndarray, calib: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates metric depth map Z = f * B / disparity.
        Returns:
            depth: float32 (H x W), metric depth in meters. Invalid pixels set to 0.0.
            disparity: float32 (H x W), disparity in pixels.
        """
        disp = self.estimate_disparity(left, right)
        return disparity_to_depth(disp, calib)


class SGBMStereoMatcher(BaseStereoMatcher):
    """OpenCV StereoSGBM baseline matcher."""

    def __init__(self, num_disparities: int = 128, block_size: int = 7):
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def estimate_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if len(left.shape) == 3 else left
        grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if len(right.shape) == 3 else right
        raw_disp = self.matcher.compute(grayL, grayR).astype(np.float32) / 16.0
        raw_disp[raw_disp <= 0] = -1.0
        return raw_disp

    def estimate_depth(
        self, left: np.ndarray, right: np.ndarray, calib: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        disp = self.estimate_disparity(left, right)
        return disparity_to_depth(disp, calib)


def disparity_to_depth(disparity: np.ndarray, calib: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts disparity map to metric depth in meters using calibrated parameters.
    Z = (f * B) / disparity
    where f = fx (focal length in pixels) and B is baseline in meters.
    """
    if "K" in calib:
        K = np.array(calib["K"], dtype=np.float64)
        fx = float(K[0, 0])
    else:
        fx = float(calib.get("fx", 1000.0))

    B_raw = float(calib.get("B", calib.get("baseline", calib.get("b", 0.12))))
    # Calibrated baseline in RSRD is in mm (e.g. ~118.8 mm)
    baseline_m = B_raw / 1000.0 if B_raw > 10.0 else B_raw

    # Mask valid disparities
    valid_mask = (disparity > 0.1) & np.isfinite(disparity)
    depth = np.zeros_like(disparity, dtype=np.float32)

    # Z = f * B / d
    depth[valid_mask] = (fx * baseline_m) / disparity[valid_mask]

    # Filter physical sanity bounds for automotive stereo (0.3m to 80m)
    depth[(depth < 0.3) | (depth > 80.0)] = 0.0
    return depth, disparity


def depth_to_3d_points(depth: np.ndarray, calib: dict) -> np.ndarray:
    """
    Backprojects metric depth map to 3D camera coordinates (X, Y, Z).
    Returns (H, W, 3) float32 array where invalid depth pixels have [0, 0, 0].
    """
    h, w = depth.shape[:2]
    if "K" in calib:
        K = np.array(calib["K"], dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    else:
        fx = float(calib.get("fx", 1000.0))
        fy = float(calib.get("fy", 1000.0))
        cx = float(calib.get("cx", w / 2.0))
        cy = float(calib.get("cy", h / 2.0))

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = (depth > 0.0)
    Z = depth.copy()
    X = np.zeros_like(Z)
    Y = np.zeros_like(Z)

    X[valid] = (u_grid[valid] - cx) * Z[valid] / fx
    Y[valid] = (v_grid[valid] - cy) * Z[valid] / fy

    points_3d = np.stack([X, Y, Z], axis=-1)
    return points_3d


def get_stereo_matcher(model_name: str = "fast_foundation", **kwargs) -> BaseStereoMatcher:
    """Factory function to instantiate swappable stereo backends."""
    model_name = model_name.lower().replace("-", "_")
    if "foundation" in model_name:
        return FastFoundationStereoMatcher(**kwargs)
    elif "sgbm" in model_name:
        return SGBMStereoMatcher(**kwargs)
    else:
        # Fallback to foundation or SGBM
        try:
            return FastFoundationStereoMatcher(**kwargs)
        except Exception:
            return SGBMStereoMatcher(**kwargs)


# Module-level convenience functions matching user API
_GLOBAL_MATCHER: Optional[BaseStereoMatcher] = None

def get_default_matcher() -> BaseStereoMatcher:
    global _GLOBAL_MATCHER
    if _GLOBAL_MATCHER is None:
        _GLOBAL_MATCHER = get_stereo_matcher("fast_foundation")
    return _GLOBAL_MATCHER

def estimate_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return get_default_matcher().estimate_disparity(left, right)

def estimate_depth(left: np.ndarray, right: np.ndarray, calib: dict) -> Tuple[np.ndarray, np.ndarray]:
    return get_default_matcher().estimate_depth(left, right, calib)
