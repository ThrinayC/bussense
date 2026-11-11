"""
bussense_infrastructure/models/water_model.py
=============================================
Real inference adapter for the Waterlogging Segmentation Model.
Exposes load(), predict(), benchmark(), metadata().
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch

from .architectures import LightweightRoadSegNet

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "infrastructure" / "water" / "best.pt"


class WaterModelAdapter:
    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.model: Optional[LightweightRoadSegNet] = None
        self._metadata: Dict[str, Any] = {}
        self.load()

    def load(self, weights_path: Optional[str] = None):
        if weights_path:
            self.weights_path = Path(weights_path)

        self.model = LightweightRoadSegNet(in_channels=3, num_classes=1).to(self.device)
        if self.weights_path.exists():
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            
            with open(self.weights_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()

            self._metadata = {
                "name": "LightweightRoadSegNet_Water",
                "task": "waterlogging_segmentation",
                "device": "CPU (Fallback)",
                "weights_path": str(self.weights_path),
                "sha256": h,
                "input_resolution": [256, 256],
                "parameters": sum(p.numel() for p in self.model.parameters())
            }
        else:
            self._metadata = {
                "name": "LightweightRoadSegNet_Water",
                "status": "UNINITIALIZED",
                "weights_path": str(self.weights_path)
            }
        self.model.eval()

    def predict(self, image_rgb: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """
        image_rgb: np.ndarray [H, W, 3] in RGB uint8
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        orig_h, orig_w = image_rgb.shape[:2]
        t0 = time.perf_counter()

        pil_img = Image.fromarray(image_rgb).resize((256, 256))
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask_256 = (probs > threshold).astype(np.uint8)
        mask_pil = Image.fromarray(binary_mask_256).resize((orig_w, orig_h), Image.NEAREST)
        final_mask = np.array(mask_pil)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        water_fraction = float(np.mean(final_mask))
        confidence = float(np.mean(probs[probs > threshold])) if np.any(probs > threshold) else 0.0

        return {
            "mask": final_mask,
            "confidence": round(confidence, 4),
            "water_pixel_fraction": round(water_fraction, 4),
            "latency_ms": round(latency_ms, 2)
        }

    def benchmark(self, num_runs: int = 30) -> Dict[str, Any]:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        latencies = []
        for _ in range(num_runs):
            res = self.predict(dummy)
            latencies.append(res["latency_ms"])

        return {
            "num_runs": num_runs,
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "fps": round(1000.0 / max(float(np.mean(latencies)), 1e-4), 2),
            "device": "CPU (Fallback)"
        }

    def metadata(self) -> Dict[str, Any]:
        return self._metadata
