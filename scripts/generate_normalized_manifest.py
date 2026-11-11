"""
scripts/generate_normalized_manifest.py
=======================================
Generates `data/normalized_manifest.json` for BusSense Phase 2.
Indexes all authentic dataset samples with:
- image_path, annotation_path
- split (train, val, test)
- dimensions [H, W, C]
- task and modality
- class annotations / metadata
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "data"

def get_img_dims(p: Path):
    try:
        with Image.open(p) as img:
            return [img.height, img.width, 3 if img.mode == "RGB" else 1]
    except Exception:
        return [0, 0, 0]

def build_fred_manifest():
    manifest = {"dataset": "fred", "task": "water_segmentation", "splits": {"train": [], "val": [], "test": []}}
    base_dir = DATASETS_DIR / "fred" / "raw"
    if not base_dir.exists():
        return manifest
    
    for split in ["train", "val", "test"]:
        img_dir = base_dir / "images" / split
        msk_dir = base_dir / "masks" / split
        if img_dir.exists():
            for img_p in sorted(img_dir.glob("*.png")):
                msk_p = msk_dir / img_p.name
                dims = get_img_dims(img_p)
                manifest["splits"][split].append({
                    "image_path": str(img_p.relative_to(BASE_DIR)).replace("\\", "/"),
                    "mask_path": str(msk_p.relative_to(BASE_DIR)).replace("\\", "/") if msk_p.exists() else None,
                    "dimensions": dims,
                    "class": "water"
                })
    return manifest

def build_ceymo_manifest():
    manifest = {"dataset": "ceymo", "task": "zebra_segmentation_detection", "splits": {"train": [], "val": [], "test": []}}
    base_dir = DATASETS_DIR / "ceymo" / "raw"
    if not base_dir.exists():
        return manifest

    split_map = {"train": "train", "val": "val", "test": "test"}
    for split_key, ceymo_split in split_map.items():
        img_dir = base_dir / "images" / ceymo_split
        lbl_dir = base_dir / "labels" / ceymo_split
        if img_dir.exists():
            for img_p in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                has_zebra = False
                labels = []
                if lbl_p.exists():
                    try:
                        with open(lbl_p, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    cls_id = int(parts[0])
                                    labels.append(cls_id)
                                    if cls_id == 5: # CeyMo pedestrian crossing
                                        has_zebra = True
                    except Exception:
                        pass
                dims = get_img_dims(img_p)
                manifest["splits"][split_key].append({
                    "image_path": str(img_p.relative_to(BASE_DIR)).replace("\\", "/"),
                    "label_path": str(lbl_p.relative_to(BASE_DIR)).replace("\\", "/") if lbl_p.exists() else None,
                    "dimensions": dims,
                    "labels": labels,
                    "has_zebra_crossing": has_zebra
                })
    return manifest

def build_mtsvd_manifest():
    manifest = {"dataset": "mtsvd", "task": "traffic_sign_detection", "splits": {"train": [], "val": [], "test": []}}
    base_dir = DATASETS_DIR / "mtsvd" / "raw"
    if not base_dir.exists():
        return manifest

    split_map = {"train": "train", "val": "val", "test": "test"}
    for split_key, sdir in split_map.items():
        split_path = base_dir / sdir
        if split_path.exists():
            for img_p in sorted(list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))):
                dims = get_img_dims(img_p)
                manifest["splits"][split_key].append({
                    "image_path": str(img_p.relative_to(BASE_DIR)).replace("\\", "/"),
                    "dimensions": dims,
                    "class": "traffic_sign"
                })
    return manifest

def build_idd_manifest():
    manifest = {"dataset": "idd", "task": "indian_domain_sanity", "splits": {"test": []}}
    base_dir = DATASETS_DIR / "idd" / "raw" / "images"
    if base_dir.exists():
        for img_p in sorted(list(base_dir.glob("*.jpg")) + list(base_dir.glob("*.png"))):
            dims = get_img_dims(img_p)
            manifest["splits"]["test"].append({
                "image_path": str(img_p.relative_to(BASE_DIR)).replace("\\", "/"),
                "dimensions": dims,
                "domain": "indian_road_corridor"
            })
    return manifest

def generate_manifest():
    print("=" * 65)
    print("BUSSENSE PHASE 2: GENERATING NORMALIZED DATASET MANIFEST")
    print("=" * 65)

    all_manifest = {
        "version": "2.0.0",
        "description": "BusSense Phase 2 Normalized Dataset Manifest for Real Perception Models",
        "datasets": {
            "fred": build_fred_manifest(),
            "ceymo": build_ceymo_manifest(),
            "mtsvd": build_mtsvd_manifest(),
            "idd": build_idd_manifest()
        },
        "summary": {}
    }

    for dname, dinfo in all_manifest["datasets"].items():
        total = sum(len(samples) for samples in dinfo["splits"].values())
        all_manifest["summary"][dname] = {
            "task": dinfo["task"],
            "total_samples": total,
            "split_counts": {s: len(samples) for s, samples in dinfo["splits"].items()}
        }
        print(f"[{dname.upper()}] Total: {total} | Splits: {all_manifest['summary'][dname]['split_counts']}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "normalized_manifest.json"
    with open(out_file, "w") as f:
        json.dump(all_manifest, f, indent=2)

    print(f"\nSaved normalized manifest to: {out_file}")
    print("=" * 65)
    return all_manifest

if __name__ == "__main__":
    generate_manifest()
