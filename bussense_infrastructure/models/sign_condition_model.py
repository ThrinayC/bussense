"""
bussense_infrastructure/models/sign_condition_model.py
======================================================
Real inference adapter for the Sign Condition Classifier.
Exposes load(), predict(), benchmark(), metadata().
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch

from .architectures import SignConditionNet

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "infrastructure" / "sign_condition" / "best.pt"


class SignConditionModelAdapter:
    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("cpu")
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.model: Optional[SignConditionNet] = None
        self._metadata: Dict[str, Any] = {}
        self.conditions = SignConditionNet.CONDITIONS
        self.load()

    def load(self, weights_path: Optional[str] = None):
        if weights_path:
            self.weights_path = Path(weights_path)

        self.model = SignConditionNet(in_channels=3, num_classes=len(self.conditions)).to(self.device)
        if self.weights_path.exists():
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            with open(self.weights_path, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()

            self._metadata = {
                "name": "SignConditionNet",
                "task": "sign_condition_classification",
                "conditions": self.conditions,
                "device": "CPU (Fallback)",
                "weights_path": str(self.weights_path),
                "sha256": h,
                "input_resolution": [64, 64],
                "parameters": sum(p.numel() for p in self.model.parameters())
            }
        else:
            self._metadata = {
                "name": "SignConditionNet",
                "status": "UNINITIALIZED",
                "weights_path": str(self.weights_path)
            }
        self.model.eval()

    def predict(self, sign_crop_rgb: np.ndarray) -> Dict[str, Any]:
        """
        sign_crop_rgb: np.ndarray [H, W, 3] in RGB uint8
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        t0 = time.perf_counter()

        pil_img = Image.fromarray(sign_crop_rgb).resize((64, 64))
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        best_idx = int(np.argmax(probs))
        condition = self.conditions[best_idx]
        confidence = float(probs[best_idx])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "condition": condition,
            "confidence": round(confidence, 4),
            "probabilities": {cond: round(float(probs[i]), 4) for i, cond in enumerate(self.conditions)},
            "latency_ms": round(latency_ms, 2)
        }

    def benchmark(self, num_runs: int = 30) -> Dict[str, Any]:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
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
