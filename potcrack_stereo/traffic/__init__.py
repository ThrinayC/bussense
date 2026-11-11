"""
BusSense Vehicular Stereo Perception & Traffic Sensing Module.
Additive package for stereo vehicle detection, tracking, 3D ground-contact localization,
and road-plane traffic density estimation.
"""

from .detector import TrafficDetector, VEHICLE_CLASSES, TARGET_CLASSES
from .tracker import TrafficTracker
from .spatial_analyzer import TrafficSpatialAnalyzer
from .density_engine import TrafficDensityEngine

__all__ = [
    "TrafficDetector",
    "TrafficTracker",
    "TrafficSpatialAnalyzer",
    "TrafficDensityEngine",
    "VEHICLE_CLASSES",
    "TARGET_CLASSES",
]
