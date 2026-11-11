# BusSense Mega Master Dataset Audit

| Source | Modality | Resolution | FPS | Baseline | Scenario Family | Runtime Valid | GT Available | Demo Selected |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KITTI Raw Baseline | `real_stereo` | 1242x375 | 10.0 | 0.533 m | VRU Safety Corridor, Cyclist Disamb... | ✓ | ✓ | ✓ |
| KITTI Raw Traffic | `real_stereo` | 1242x375 | 10.0 | 0.533 m | Multi-Vehicle Traffic Stream, Virtu... | ✓ | ✓ | ✓ |
| EvTTC IEEE RAL 2025 | `real_stereo` | 1920x1200 | 10.0 | 0.450 m | Pre-Collision Emergency Braking, Cl... | ✓ | ✓ | ✓ |
| EvTTC IEEE RAL 2025 | `real_stereo` | 1920x1200 | 10.0 | 0.450 m | High-Relative-Speed Emergency Braki... | ✓ | ✓ | ✓ |
| EvTTC IEEE RAL 2025 | `real_stereo` | 1920x1200 | 10.0 | 0.450 m | Sudden Pedestrian Crossing Conflict... | ✓ | ✓ | ✓ |
| Extreme Driving Conditions | `real_stereo` | 1280x720 | 10.0 | 0.120 m | Complex Multi-Agent Urban Stream & ... | ✓ | ✓ | ✓ |
| Extreme Driving Conditions | `real_stereo` | 1280x720 | 10.0 | 0.120 m | Aggressive Lateral Cut-In & Evasive... | ✓ | ✓ | ✓ |
| Native Stereo Pothole Metrology | `real_stereo` | 960x540 | 10.0 | 0.200 m | Pothole Depth (cm), Shoelace Area (... | ✓ | ✓ | ✓ |
| Native Stereo Pothole Metrology | `real_stereo` | 1240x608 | 10.0 | 0.200 m | Calibrated Road Plane Metrology & D... | ✓ | ✓ | ✓ |
| Native Road Surface Crack Networks | `real_stereo` | 1920x1080 | 10.0 | 0.200 m | Crack Network Skeletonization, Leng... | ✓ | ✓ | ✓ |
| Public Wikimedia / Police Video | `monocular` | 1280x720 | 10.0 | None | Urban Pedestrian Hit-and-Run Candid... | ✓ | ✗ | ✓ |
| Public Wikimedia / Police Video | `monocular` | 2560x1440 | 10.0 | None | High-Speed Collision Candidate & Vi... | ✓ | ✗ | ✓ |
| Public Wikimedia / Police Video | `monocular` | 1920x1080 | 10.0 | None | Suspect Departure & ANPR Plate Harv... | ✓ | ✗ | ✓ |


## Ethical & Modality Separation Rules
- **Absolute Stereo Rule**: All standard road perception, vehicle tracking, VRU protection, and traffic flow strictly require synchronized `LEFT RGB + RIGHT RGB`.
- **Monocular Hit-and-Run Exception**: Authenticated real stereo public crash videos are unavailable. The hit-and-run module operates in isolated `monocular` test mode (`metric_depth_valid = false`, `source_modality = 'monocular'`). No metric depth or stereo range is fabricated.
