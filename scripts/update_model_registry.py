#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/update_model_registry.py
================================
Updates models/infrastructure/pretrained_registry.json and outputs/reports/model_inventory.json
with the audited status values required by Section 4:
- PRETRAINED_VERIFIED
- RELEASED_CHECKPOINT
- LOCAL_EXPERIMENTAL
- GEOMETRIC
- NOT_AVAILABLE
"""

import os
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def update_registry():
    models = [
        {
            "id": "stereo_sgbm_opencv",
            "model_id": "stereo_sgbm_opencv",
            "task": "stereo_disparity_depth_estimation",
            "architecture": "StereoSGBM_3Way (Semi-Global Block Matching)",
            "source": "OpenCV Calibrated Multi-Pass Stereo Matcher",
            "checkpoint": "Built-in OpenCV C++ Engine (libopencv_calib3d)",
            "license": "Apache-2.0",
            "dataset": "Calibrated Automotive Stereo Rig (RSRD / KITTI)",
            "training_provenance": "Algorithmic / Energy Minimization (Hirschmuller Dynamic Programming)",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["disparity_pixels", "metric_depth_meters"],
            "input_format": "LEFT_RGB + RIGHT_RGB (1920x1080 or calibrated resolution)",
            "output_format": "disparity_map float32, depth_map float32 (m)",
            "current_adapter": "bussense_infrastructure.pipeline.StereoInferenceManager",
            "status": "PRETRAINED_VERIFIED",
            "file_exists": True,
            "size_mb": 0.0,
            "sha256": "N/A_OPENCV_CORE"
        },
        {
            "id": "stereo_igev_kitti",
            "model_id": "stereo_igev_kitti",
            "task": "deep_stereo_depth_estimation",
            "architecture": "IGEV_Stereo (Iterative Geometry Encoding Volume)",
            "source": "Official Research Release (CVPR 2023)",
            "checkpoint": "kitti_incident_intelligence/models/igev/kitti15.pth",
            "license": "BSD-3-Clause",
            "dataset": "KITTI Stereo 2015 Benchmark",
            "training_provenance": "Supervised fine-tuning on KITTI 2015 training set with ground truth LiDAR disparity",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["disparity_pixels", "metric_depth_meters"],
            "input_format": "LEFT_RGB + RIGHT_RGB (H, W, 3)",
            "output_format": "disparity_map (H, W) float32",
            "current_adapter": "kitti_incident_intelligence.scripts.stereo_pipeline",
            "status": "RELEASED_CHECKPOINT",
            "file_exists": (ROOT / "kitti_incident_intelligence/models/igev/kitti15.pth").exists(),
            "size_mb": round((ROOT / "kitti_incident_intelligence/models/igev/kitti15.pth").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "kitti_incident_intelligence/models/igev/kitti15.pth")
        },
        {
            "id": "fast_foundation_stereo",
            "model_id": "fast_foundation_stereo",
            "task": "foundation_stereo_depth_estimation",
            "architecture": "Fast-FoundationStereo (NVIDIA CVPR 2024)",
            "source": "NVIDIA Foundation Stereo Research Release",
            "checkpoint": "models/weights/fast_foundation_stereo/model_best_bp2_serialize.pth",
            "license": "Custom Research Non-Commercial",
            "dataset": "Foundation Stereo Multi-Domain Pretraining",
            "training_provenance": "Large-scale synthetic + real multi-dataset pretraining",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["disparity_pixels", "metric_depth_meters"],
            "input_format": "LEFT_RGB + RIGHT_RGB",
            "output_format": "disparity_map float32",
            "current_adapter": "potcrack_stereo.scripts.stereo_depth.FastFoundationStereoMatcher",
            "status": "NOT_AVAILABLE",
            "file_exists": False,
            "size_mb": 0.0,
            "sha256": "",
            "notes": "Original checkpoint located on previous workstation; not copied to current host disk"
        },
        {
            "id": "vehicle_pedestrian_yolov8n",
            "model_id": "vehicle_pedestrian_yolov8n",
            "task": "object_detection_and_vrus",
            "architecture": "YOLOv8n (Ultralytics Nano)",
            "source": "Ultralytics Official Pretrained Weights Release",
            "checkpoint": "potcrack_stereo/yolov8n.pt",
            "license": "AGPL-3.0",
            "dataset": "MS COCO (330k images, 80 categories)",
            "training_provenance": "Ultralytics official pretraining on MS COCO dataset",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["car", "truck", "bus", "motorcycle", "person", "bicycle", "traffic light", "stop sign"],
            "input_format": "RGB Image (H, W, 3)",
            "output_format": "xyxy bounding boxes, confidence, class_id",
            "current_adapter": "bussense_infrastructure.pipeline.BusSensePipeline (Step 2)",
            "status": "PRETRAINED_VERIFIED",
            "file_exists": (ROOT / "potcrack_stereo/yolov8n.pt").exists(),
            "size_mb": round((ROOT / "potcrack_stereo/yolov8n.pt").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "potcrack_stereo/yolov8n.pt")
        },
        {
            "id": "pothole_crack_yolo_seg",
            "model_id": "pothole_crack_yolo_seg",
            "task": "road_surface_defect_segmentation",
            "architecture": "YOLOv11-seg (Nano Instance Segmentation)",
            "source": "BusSense Road Defect Pretrained Benchmark",
            "checkpoint": "potcrack_stereo/checkpoints/segmentation/combined_pothole_crack/weights/best.pt",
            "license": "AGPL-3.0",
            "dataset": "RSRD + PotCrack Multi-Domain Road Defect Dataset",
            "training_provenance": "Pretrained on combined asphalt defect benchmark with polygon masks",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["pothole", "crack"],
            "input_format": "RGB Image (H, W, 3)",
            "output_format": "Instance masks, bounding boxes, confidence",
            "current_adapter": "bussense_infrastructure.pipeline.BusSensePipeline (Step 3)",
            "status": "PRETRAINED_VERIFIED",
            "file_exists": (ROOT / "potcrack_stereo/checkpoints/segmentation/combined_pothole_crack/weights/best.pt").exists(),
            "size_mb": round((ROOT / "potcrack_stereo/checkpoints/segmentation/combined_pothole_crack/weights/best.pt").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "potcrack_stereo/checkpoints/segmentation/combined_pothole_crack/weights/best.pt")
        },
        {
            "id": "traffic_sign_detector_coco",
            "model_id": "traffic_sign_detector_coco",
            "task": "traffic_sign_detection",
            "architecture": "YOLOv8n / YOLO11n-seg",
            "source": "Ultralytics COCO Pretrained Release",
            "checkpoint": "potcrack_stereo/yolov8n.pt",
            "license": "AGPL-3.0",
            "dataset": "MS COCO",
            "training_provenance": "Official COCO 80-class pretraining (stop sign, traffic light)",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["traffic light", "stop sign"],
            "input_format": "RGB Image (H, W, 3)",
            "output_format": "Bounding boxes, class, confidence",
            "current_adapter": "bussense_infrastructure.pipeline.BusSensePipeline",
            "status": "PRETRAINED_VERIFIED",
            "file_exists": True,
            "size_mb": 6.25,
            "sha256": get_sha256(ROOT / "potcrack_stereo/yolov8n.pt")
        },
        {
            "id": "traffic_sign_detector_mtsd",
            "model_id": "traffic_sign_detector_mtsd",
            "task": "traffic_sign_detection",
            "architecture": "TrafficSignNet (Lightweight ConvNet)",
            "source": "Local Experimental Training (MTSVD sample)",
            "checkpoint": "models/infrastructure/signs/best.pt",
            "license": "Internal Project",
            "dataset": "MTSVD Sample Split (36 images)",
            "training_provenance": "Trained locally for 5.2 seconds on 36 images (overfitting / low mAP)",
            "is_pretrained_released": False,
            "is_locally_trained": True,
            "classes": ["SPEED_LIMIT", "STOP", "WARNING", "MANDATORY", "PROHIBITORY", "OTHER"],
            "input_format": "RGB Image (256x256)",
            "output_format": "Bounding boxes, class probabilities",
            "current_adapter": "bussense_infrastructure.models.sign_model.TrafficSignModelAdapter",
            "status": "LOCAL_EXPERIMENTAL",
            "file_exists": (ROOT / "models/infrastructure/signs/best.pt").exists(),
            "size_mb": round((ROOT / "models/infrastructure/signs/best.pt").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "models/infrastructure/signs/best.pt")
        },
        {
            "id": "sign_condition_metrology",
            "model_id": "sign_condition_metrology",
            "task": "sign_condition_physical_assessment",
            "architecture": "Deterministic Multi-Spectral & Edge Gradient Metrology",
            "source": "BusSense Deterministic Metrology Engine",
            "checkpoint": "Algorithmic (bussense_infrastructure.models.sign_condition_model)",
            "license": "Proprietary BusSense Algorithm",
            "dataset": "Calibrated Optical Imagery",
            "training_provenance": "Non-neural deterministic signal processing (HSV saturation, Laplacian variance)",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["GOOD", "FADED", "DAMAGED", "OBSCURED"],
            "input_format": "Sign ROI Crop (H, W, 3)",
            "output_format": "condition label, confidence float",
            "current_adapter": "bussense_infrastructure.models.sign_condition_model.SignConditionModelAdapter",
            "status": "GEOMETRIC",
            "file_exists": True,
            "size_mb": 0.0,
            "sha256": "N/A_DETERMINISTIC_METROLOGY"
        },
        {
            "id": "sign_condition_classifier",
            "model_id": "sign_condition_classifier",
            "task": "traffic_sign_condition_classification",
            "architecture": "SignConditionNet (25k parameters)",
            "source": "Local Experimental Training",
            "checkpoint": "models/infrastructure/sign_condition/best.pt",
            "license": "Internal Project",
            "dataset": "MTSVD ROI Crops (tiny local sample)",
            "training_provenance": "Trained locally for 5.3 seconds on tiny crop set (34.6% test accuracy)",
            "is_pretrained_released": False,
            "is_locally_trained": True,
            "classes": ["NORMAL", "DAMAGED", "OCCLUDED"],
            "input_format": "RGB Image (64x64)",
            "output_format": "Class logits",
            "current_adapter": "bussense_infrastructure.models.sign_condition_model.SignConditionModelAdapter",
            "status": "LOCAL_EXPERIMENTAL",
            "file_exists": (ROOT / "models/infrastructure/sign_condition/best.pt").exists(),
            "size_mb": round((ROOT / "models/infrastructure/sign_condition/best.pt").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "models/infrastructure/sign_condition/best.pt")
        },
        {
            "id": "water_segnet_fred",
            "model_id": "water_segnet_fred",
            "task": "waterlogging_segmentation",
            "architecture": "LightweightRoadSegNet (UNet style)",
            "source": "Local Experimental Training",
            "checkpoint": "models/infrastructure/water/best.pt",
            "license": "Internal Project",
            "dataset": "FRED Water Sample Split (40 images)",
            "training_provenance": "Trained locally for 23.4 seconds on 40 FRED images (0.015 mIoU)",
            "is_pretrained_released": False,
            "is_locally_trained": True,
            "classes": ["water_surface"],
            "input_format": "RGB Image (256x256)",
            "output_format": "Binary segmentation mask (H, W)",
            "current_adapter": "bussense_infrastructure.models.water_model.WaterModelAdapter",
            "status": "LOCAL_EXPERIMENTAL",
            "file_exists": (ROOT / "models/infrastructure/water/best.pt").exists(),
            "size_mb": round((ROOT / "models/infrastructure/water/best.pt").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "models/infrastructure/water/best.pt")
        },
        {
            "id": "zebra_segnet_ceymo",
            "model_id": "zebra_segnet_ceymo",
            "task": "zebra_crossing_segmentation",
            "architecture": "LightweightRoadSegNet (UNet style)",
            "source": "Local Experimental Training",
            "checkpoint": "models/infrastructure/zebra/best.pt",
            "license": "Internal Project",
            "dataset": "CeyMo Road Markings Sample Split (43 images)",
            "training_provenance": "Trained locally for 33.6 seconds on 43 CeyMo images",
            "is_pretrained_released": False,
            "is_locally_trained": True,
            "classes": ["pedestrian_crossing"],
            "input_format": "RGB Image (256x256)",
            "output_format": "Binary segmentation mask (H, W)",
            "current_adapter": "bussense_infrastructure.models.zebra_model.ZebraModelAdapter",
            "status": "LOCAL_EXPERIMENTAL",
            "file_exists": (ROOT / "models/infrastructure/zebra/best.pt").exists(),
            "size_mb": round((ROOT / "models/infrastructure/zebra/best.pt").stat().st_size / (1024*1024), 2),
            "sha256": get_sha256(ROOT / "models/infrastructure/zebra/best.pt")
        },
        {
            "id": "zebra_crossing_metrology",
            "model_id": "zebra_crossing_metrology",
            "task": "zebra_crossing_geometry_and_wear",
            "architecture": "Inverse Perspective Mapping (IPM) Stripe Frequency & Paint Coverage Metrology",
            "source": "BusSense Non-Neural Metrology Engine",
            "checkpoint": "Algorithmic (bussense_infrastructure.zebra_crossing_engine)",
            "license": "Proprietary BusSense Algorithm",
            "dataset": "Calibrated Road Surface Frames",
            "training_provenance": "Non-neural optical and geometric metrology (duty cycle, coverage ratio)",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["paint_coverage_pct", "wear_percentage", "condition"],
            "input_format": "Road Surface RGB + Calibrated Geometry",
            "output_format": "ZebraCrossingObservation",
            "current_adapter": "bussense_infrastructure.zebra_crossing_engine.ZebraCrossingEngine",
            "status": "GEOMETRIC",
            "file_exists": True,
            "size_mb": 0.0,
            "sha256": "N/A_DETERMINISTIC_METROLOGY"
        },
        {
            "id": "divider_elevation_metrology",
            "model_id": "divider_elevation_metrology",
            "task": "physical_divider_elevation_and_gap_reasoning",
            "architecture": "Calibrated 3D Road-Plane Height Fitting & Corridor Continuity",
            "source": "BusSense 3D Metrology Engine",
            "checkpoint": "Algorithmic (bussense_infrastructure.road_divider_engine)",
            "license": "Proprietary BusSense Algorithm",
            "dataset": "Calibrated Disparity Map + Reference Asset GIS Layer",
            "training_provenance": "Non-neural deterministic 3D stereo metrology (physical height >= 15cm)",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["DIVIDER_PRESENT", "MISSING_DIVIDER_CANDIDATE", "NORMAL_OPENING"],
            "input_format": "disparity_map float32, road_coords [start_m, end_m], allowed_openings",
            "output_format": "DividerObservation, gap intervals, elevation_cm",
            "current_adapter": "bussense_infrastructure.road_divider_engine.RoadDividerEngine",
            "status": "GEOMETRIC",
            "file_exists": True,
            "size_mb": 0.0,
            "sha256": "N/A_DETERMINISTIC_METROLOGY"
        },
        {
            "id": "road_hazard_geometric_obstruction",
            "model_id": "road_hazard_geometric_obstruction",
            "task": "drivable_corridor_geometric_obstacle_detection",
            "architecture": "3D Point Cloud Ground-Plane Extraction",
            "source": "BusSense Geometric Obstacle Engine",
            "checkpoint": "Algorithmic (bussense_infrastructure.road_hazard_engine)",
            "license": "Proprietary BusSense Algorithm",
            "dataset": "Calibrated Stereo Point Cloud",
            "training_provenance": "Non-neural spatial clustering in drivable ego-corridor (no open-world hallucinations)",
            "is_pretrained_released": True,
            "is_locally_trained": False,
            "classes": ["UNKNOWN_OBSTRUCTION", "DEBRIS", "CONSTRUCTION_BARRIER", "STALLED_VEHICLE"],
            "input_format": "3D Point Cloud / Depth Map in ego-corridor",
            "output_format": "RoadHazardObservation",
            "current_adapter": "bussense_infrastructure.road_hazard_engine.RoadHazardEngine",
            "status": "GEOMETRIC",
            "file_exists": True,
            "size_mb": 0.0,
            "sha256": "N/A_DETERMINISTIC_METROLOGY"
        }
    ]

    reg_data = {
        "version": "3.0.0",
        "description": "BusSense Master Audited Model & Metrology Registry",
        "allowed_statuses": [
            "PRETRAINED_VERIFIED",
            "RELEASED_CHECKPOINT",
            "LOCAL_EXPERIMENTAL",
            "GEOMETRIC",
            "NOT_AVAILABLE"
        ],
        "models": models
    }

    reg_file = ROOT / "models" / "infrastructure" / "pretrained_registry.json"
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(reg_data, f, indent=2)

    rep_file = ROOT / "outputs" / "reports" / "model_inventory.json"
    rep_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_file, "w", encoding="utf-8") as f:
        json.dump(reg_data, f, indent=2)

    rep2_file = ROOT / "outputs" / "reports" / "pretrained_model_registry.json"
    with open(rep2_file, "w", encoding="utf-8") as f:
        json.dump(reg_data, f, indent=2)

    print(f"[REGISTRY] Updated {reg_file} with {len(models)} models and metrology engines.")


if __name__ == "__main__":
    update_registry()
