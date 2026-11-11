"""
IGEV-Stereo Engine Implementation
Declared primary offline learned stereo engine using iterative geometry encoding volume.
Loads official KITTI checkpoint from models/igev/kitti15.pth per PRD §6.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from scripts.stereo_engine import StereoEngine

# Add models/igev to sys.path for core module access
IGEV_ROOT = Path(__file__).resolve().parent.parent / "models" / "igev"
if str(IGEV_ROOT) not in sys.path:
    sys.path.insert(0, str(IGEV_ROOT))


class IGEVArgs:
    """Configuration namespace for IGEV-Stereo architecture."""
    hidden_dims = [128, 128, 128]
    corr_levels = 2
    corr_radius = 4
    n_downsample = 2
    n_gru_layers = 3
    max_disp = 192
    mixed_precision = False
    precision_dtype = "float32"


class IGEVStereoEngine(StereoEngine):
    """
    IGEV-Stereo implementation of StereoEngine conforming to PRD §6.
    Loads kitti15.pth pretrained weights on available GPU.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        super().__init__(name="IGEV_Stereo")

        default_ckpt = IGEV_ROOT / "kitti15.pth"
        self.checkpoint_path = str(checkpoint_path or default_ckpt)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_available = False
        self.status = "INITIALIZING"

        self._init_model()

    def _init_model(self):
        if not os.path.exists(self.checkpoint_path):
            self.status = f"BLOCKED: Checkpoint not found at {self.checkpoint_path}"
            self.is_available = False
            return

        try:
            from core.igev_stereo import IGEVStereo

            args = IGEVArgs()
            self.model = IGEVStereo(args)

            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Clean module prefix if saved from DataParallel
            clean_state = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                clean_state[name] = v

            self.model.load_state_dict(clean_state, strict=True)
            self.model.to(self.device)
            self.model.eval()

            self.is_available = True
            self.status = f"OPERATIONAL (weights loaded on {self.device})"
        except Exception as e:
            self.is_available = False
            self.status = f"FAILED_LOAD: {str(e)}"

    def compute_disparity(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Computes dense disparity map using IGEV-Stereo.
        Inputs: left_img, right_img (H, W, 3) BGR or RGB uint8.
        Outputs: disparity (H, W) float32 in pixels.
        """
        if not self.is_available or self.model is None:
            raise RuntimeError(
                f"IGEV-Stereo inference unavailable: {self.status}. "
                "Per PRD §6, use SGBMStereoEngine as declared baseline."
            )

        # Convert to RGB float tensors (B, C, H, W)
        if len(left_img.shape) == 2:
            left_rgb = np.stack([left_img] * 3, axis=-1).copy()
            right_rgb = np.stack([right_img] * 3, axis=-1).copy()
        else:
            left_rgb = np.ascontiguousarray(left_img[..., ::-1])
            right_rgb = np.ascontiguousarray(right_img[..., ::-1])

        h, w, _ = left_rgb.shape

        # Pad to multiple of 32 for encoder downsamplings
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        t_left = torch.from_numpy(left_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        t_right = torch.from_numpy(right_rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        if pad_h > 0 or pad_w > 0:
            t_left = F.pad(t_left, (0, pad_w, 0, pad_h), mode="replicate")
            t_right = F.pad(t_right, (0, pad_w, 0, pad_h), mode="replicate")

        with torch.no_grad():
            disp_pred = self.model(t_left, t_right, iters=12, test_mode=True)
            if isinstance(disp_pred, (list, tuple)):
                disp_pred = disp_pred[-1]

            # Unpad back to original shape
            if pad_h > 0 or pad_w > 0:
                disp_pred = disp_pred[:, :, :h, :w]

            disp_np = disp_pred.squeeze().cpu().numpy().astype(np.float32)

        # Invalidate negative / zero disparities
        disp_np[disp_np < 0.1] = 0.0

        return disp_np, None

    def get_backend_info(self) -> Dict:
        return {
            "backend": self.name,
            "status": self.status,
            "checkpoint": self.checkpoint_path,
            "is_available": self.is_available,
            "device": str(self.device),
        }
