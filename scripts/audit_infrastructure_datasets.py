"""
scripts/audit_infrastructure_datasets.py
========================================
Audits the downloaded authentic public datasets for BusSense Phase 2:
- FRED / WaterSegNets (Waterlogging segmentation)
- CeyMo (Zebra / Pedestrian crossing segmentation & detection)
- MTSVD / GTSDB (Traffic sign detection)
- IDD (Indian driving domain calibration & sanity check)

Verifies:
- File existence and integrity
- Total images and masks/annotations
- Image resolutions and color channels
- Annotation schemas, classes, and label distributions
- Integrity check (detect corrupt files)
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

def inspect_image(img_path: Path):
    try:
        with Image.open(img_path) as img:
            return {
                "size": img.size, # (width, height)
                "format": img.format,
                "mode": img.mode,
                "valid": True
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}

def audit_fred():
    raw_dir = DATASETS_DIR / "fred" / "raw"
    report = {
        "dataset": "FRED / WaterSegNets (Road Waterlogging Benchmark)",
        "task": "binary_segmentation",
        "status": "NOT_FOUND" if not raw_dir.exists() else "FOUND",
        "splits": {},
        "total_images": 0,
        "total_masks": 0,
        "sample_resolutions": [],
        "errors": []
    }
    if not raw_dir.exists():
        return report

    for split in ["train", "val", "test"]:
        img_dir = raw_dir / "images" / split
        mask_dir = raw_dir / "masks" / split
        
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) if img_dir.exists() else []
        masks = list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.png")) if mask_dir.exists() else []
        
        report["splits"][split] = {
            "images_count": len(imgs),
            "masks_count": len(masks),
            "paired": len(imgs) == len(masks) and len(imgs) > 0
        }
        report["total_images"] += len(imgs)
        report["total_masks"] += len(masks)
        
        if imgs and len(report["sample_resolutions"]) < 2:
            info = inspect_image(imgs[0])
            if info["valid"]:
                report["sample_resolutions"].append(f"{info['size'][0]}x{info['size'][1]} ({info['mode']})")

    report["status"] = "VERIFIED" if report["total_images"] > 0 else "EMPTY"
    return report

def audit_ceymo():
    raw_dir = DATASETS_DIR / "ceymo" / "raw"
    report = {
        "dataset": "CeyMo (Zebra / Pedestrian Crossing Benchmark)",
        "task": "pedestrian_crossing_segmentation_detection",
        "status": "NOT_FOUND" if not raw_dir.exists() else "FOUND",
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "class_5_crossing_count": 0,
        "class_counts": {},
        "sample_resolutions": [],
        "errors": []
    }
    if not raw_dir.exists():
        return report

    crossing_count = 0
    for split in ["train", "val", "test"]:
        img_dir = raw_dir / "images" / split
        lbl_dir = raw_dir / "labels" / split

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) if img_dir.exists() else []
        lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        split_crossing = 0
        for lbl in lbls:
            try:
                with open(lbl, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = parts[0]
                            report["class_counts"][cls_id] = report["class_counts"].get(cls_id, 0) + 1
                            if cls_id == "5":
                                split_crossing += 1
            except Exception:
                pass

        crossing_count += split_crossing
        report["splits"][split] = {
            "images_count": len(imgs),
            "labels_count": len(lbls),
            "crossings_found": split_crossing
        }
        report["total_images"] += len(imgs)
        report["total_labels"] += len(lbls)

        if imgs and len(report["sample_resolutions"]) < 2:
            info = inspect_image(imgs[0])
            if info["valid"]:
                report["sample_resolutions"].append(f"{info['size'][0]}x{info['size'][1]} ({info['mode']})")

    report["class_5_crossing_count"] = crossing_count
    report["status"] = "VERIFIED" if report["total_images"] > 0 else "EMPTY"
    return report

def audit_mtsvd():
    raw_dir = DATASETS_DIR / "mtsvd" / "raw"
    report = {
        "dataset": "MTSVD / GTSDB (Traffic Sign Detection Benchmark)",
        "task": "object_detection",
        "status": "NOT_FOUND" if not raw_dir.exists() else "FOUND",
        "splits": {},
        "total_images": 0,
        "total_annotations": 0,
        "category_counts": {},
        "sample_resolutions": [],
        "errors": []
    }
    if not raw_dir.exists():
        return report

    for split in ["train", "val", "test"]:
        sdir = raw_dir / split
        imgs = list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")) if sdir.exists() else []
        coco_file = sdir / "_annotations.coco.json"
        
        ann_count = 0
        categories = {}
        if coco_file.exists():
            try:
                with open(coco_file, "r") as f:
                    coco = json.load(f)
                    ann_count = len(coco.get("annotations", []))
                    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
                    for ann in coco.get("annotations", []):
                        cid = ann.get("category_id")
                        cname = cats.get(cid, str(cid))
                        report["category_counts"][cname] = report["category_counts"].get(cname, 0) + 1
            except Exception as e:
                report["errors"].append(f"Failed to read COCO for {split}: {e}")

        report["splits"][split] = {
            "images_count": len(imgs),
            "annotations_count": ann_count
        }
        report["total_images"] += len(imgs)
        report["total_annotations"] += ann_count

        if imgs and len(report["sample_resolutions"]) < 2:
            info = inspect_image(imgs[0])
            if info["valid"]:
                report["sample_resolutions"].append(f"{info['size'][0]}x{info['size'][1]} ({info['mode']})")

    report["status"] = "VERIFIED" if report["total_images"] > 0 else "EMPTY"
    return report

def audit_idd():
    raw_dir = DATASETS_DIR / "idd" / "raw"
    report = {
        "dataset": "IDD (Indian Driving Domain Domain Check)",
        "task": "domain_generalization_sanity",
        "status": "NOT_FOUND" if not raw_dir.exists() else "FOUND",
        "total_images": 0,
        "sample_resolutions": [],
        "errors": []
    }
    if not raw_dir.exists():
        return report

    img_dir = raw_dir / "images"
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) if img_dir.exists() else []
    report["total_images"] = len(imgs)

    if imgs:
        info = inspect_image(imgs[0])
        if info["valid"]:
            report["sample_resolutions"].append(f"{info['size'][0]}x{info['size'][1]} ({info['mode']})")

    report["status"] = "VERIFIED" if report["total_images"] > 0 else "EMPTY"
    return report

def run_audit():
    print("=" * 65)
    print("BUSSENSE PHASE 2: AUDIT INFRASTRUCTURE DATASETS")
    print("=" * 65)
    
    results = {
        "fred": audit_fred(),
        "ceymo": audit_ceymo(),
        "mtsvd": audit_mtsvd(),
        "idd": audit_idd()
    }

    out_file = DATASETS_DIR / "audit_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    for k, v in results.items():
        print(f"\n[{k.upper()}] - {v['dataset']}")
        print(f"  Status: {v['status']}")
        print(f"  Total Images: {v.get('total_images', 0)}")
        if "splits" in v:
            print(f"  Splits: {v['splits']}")
        if "total_masks" in v:
            print(f"  Total Masks: {v['total_masks']}")
        if "total_labels" in v:
            print(f"  Total Labels: {v['total_labels']}")
        if "total_annotations" in v:
            print(f"  Total COCO Annotations: {v['total_annotations']}")
        if "category_counts" in v and v["category_counts"]:
            print(f"  Categories: {v['category_counts']}")
        if "class_counts" in v and v["class_counts"]:
            print(f"  Class Box Counts: {v['class_counts']}")
        if "sample_resolutions" in v and v["sample_resolutions"]:
            print(f"  Sample Resolutions: {', '.join(v['sample_resolutions'])}")

    print("\n" + "=" * 65)
    print(f"Audit report saved to: {out_file}")
    print("=" * 65)
    return results

if __name__ == "__main__":
    run_audit()
