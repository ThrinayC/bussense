# BusSense — Vehicle Stereo Pothole 3D Measurement MVP

BusSense is an autonomous, forward-facing vehicle stereo inspection system that detects road potholes and performs calibrated 3D geometric metrology (depth in cm, footprint area in m², length and width in meters) using **ONLY LEFT RGB + RIGHT RGB** imagery.

---

## 1. Actual Deployment Requirements

- **Final Input**: `LEFT RGB + RIGHT RGB` from a vehicle-mounted forward-facing stereo camera.
- **No Monocular Depth**: Monocular depth hallucination is strictly prohibited.
- **No GT Depth at Inference**: Ground-truth depth is used strictly during evaluation/cross-validation, never during final inference.
- **Strict Inference Path**:
  ```
  LEFT RGB + RIGHT RGB
          ↓
  Stereo Disparity & Depth (Fast-FoundationStereo / SGBM)
          ↓
  LEFT RGB + Predicted Depth
          ↓
  Pothole Segmentation (Fine-tuned YOLO11-seg)
          ↓
  3D Pothole ROI & Ring Extraction
          ↓
  Local Road-Surface Plane Estimation (RANSAC)
          ↓
  Pothole Depth (p95 cm) + Footprint Area (Ray Intersection + Shoelace in m²)
          ↓
  Temporal Tracking & Smoothed Metrology
          ↓
  Annotated Full-Screen Inspection Video (outputs/demo/pothole_measurement_demo.mp4)
  ```

---

## 2. Metrology Principles & Math

### A. Road Plane Surface Fitting
For each detected pothole:
1. Extract a surrounding road ring using morphological dilation: $\text{Ring} = \text{Dilate}(\text{Mask}) \setminus \text{Mask}$.
2. Project valid depth pixels in the ring into 3D camera coordinates:
   $$X = \frac{(u - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y) \cdot Z}{f_y}, \quad Z = Z$$
3. Fit the local road plane using RANSAC:
   $$aX + bY + cZ + d = 0, \quad \text{with } \|\mathbf{n}\|_2 = 1, \quad \mathbf{n} = [a, b, c]^T$$

### B. Pothole Depth Metrology
- Depression distance below the fitted local road plane:
  $$\delta_i = -(\mathbf{n} \cdot \mathbf{P}_i + d)$$
- **Primary Depth Metric**: **95th percentile depression ($d_{95}$ in cm)** to eliminate outlier spikes.
- Secondary metrics: median depression ($d_{50}$) and filtered max ($d_{\max}$).

### C. Footprint Area Metrology (Ray-Plane Intersection + Shoelace)
- Rather than orthogonal projection of 3D noisy points, each pothole contour pixel $(u_k, v_k)$ is unprojected into a calibrated camera ray:
  $$\mathbf{r}_k = \left[\frac{u_k - c_x}{f_x}, \frac{v_k - c_y}{f_y}, 1.0\right]^T$$
- Intersect ray with the fitted road plane $\mathbf{n} \cdot \mathbf{P} + d = 0$:
  $$t_k = -\frac{d}{\mathbf{n} \cdot \mathbf{r}_k}, \quad \mathbf{P}_k = t_k \mathbf{r}_k$$
- Establish an orthonormal 2D basis $(\mathbf{u}_{\text{plane}}, \mathbf{v}_{\text{plane}})$ on the road plane.
- Project 3D points to local 2D planar coordinates:
  $$x_k = \mathbf{P}_k \cdot \mathbf{u}_{\text{plane}}, \quad y_k = \mathbf{P}_k \cdot \mathbf{v}_{\text{plane}}$$
- Compute physical footprint area via the Shoelace polygon formula:
  $$A = \frac{1}{2} \left| \sum_{k=0}^{N-1} (x_k y_{k+1} - x_{k+1} y_k) \right| \quad (\text{in } \text{m}^2)$$
- Length and width are derived from the minimum rotated bounding rectangle on the plane.

---

## Performance Benchmark Results on Tested Hardware

- **Tested Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) / Intel Core i7 / Windows 11
- **Input Image Resolution**: $960 \times 540$ (RSRD stereo stream)

### Stereo Model Benchmark Comparison (Multi-Sequence RSRD Sample, 30 Frames)

| Metric | Fast-FoundationStereo | OpenCV SGBM |
| :--- | :--- | :--- |
| **Disparity Mean EPE** | **0.297 px** | 0.255 px |
| **Disparity Median EPE** | 0.290 px | **0.241 px** |
| **Bad-Pixel Rate (>3px)** | 0.31% | **0.14%** |
| **Depth MAE** | 8.53 cm | **7.69 cm** |
| **Depth RMSE** | **0.2284 m** | 0.2464 m |
| **Depth AbsRel** | 0.0131 | **0.0122** |
| **Stereo Latency (ms)** | 1439 ms | **69.0 ms** |
| **Stereo Speed (FPS)** | 0.7 FPS | **14.5 FPS** |

*Both models are fully pluggable via a common interface.*

### End-to-End Pipeline Latency Breakdown

| Stage | Fast-FoundationStereo Mode | SGBM Mode |
| :--- | :--- | :--- |
| **1. Stereo Disparity / Depth** | 1399.4 ms | 69.0 ms |
| **2. YOLO11-seg Pothole Segmentation** | 49.9 ms | 49.9 ms |
| **3. Local Road Plane & 3D Metrology** | 5.1 ms | 5.1 ms |
| **4. Temporal Tracking & Smoothing** | 0.03 ms | 0.03 ms |
| **5. HUD & Telemetry Card Rendering** | 58.7 ms | 58.7 ms |
| **Total End-to-End Latency** | **1513.1 ms** | **182.7 ms** |
| **Effective Pipeline Speed** | **0.66 FPS** | **5.47 FPS** |

---

## 3. Dataset & Ground-Truth Validation

### A. G-7 Pavement Damages Session-Based Split (No Leakage)
Pothole color RGB: `(42, 125, 209)` (BGR: `209, 125, 42`).
Continuous recording sessions are strictly isolated:
- **Train Split**: 227 frames (48 pothole frames across 6 sessions + 7 negative road sessions)
- **Validation Split**: 50 frames (11 pothole frames across 2 sessions + 5 negative road sessions)
- **Test Split**: 149 frames (11 pothole frames across 2 sessions + 5 negative road sessions)

### B. RSRD Irregular Unevenness Audit
Audited 50 surface patches:
- Irregular Road Unevenness / Waviness / Rutting: **35 patches**
- Flat Road Surface: **15 patches**
- Verified Localized Potholes: **0 patches**
- *Conclusion*: RSRD irregular road unevenness is shallow (depth < 3 cm) and broad; it must NOT be assumed to be a pothole without patch verification.

### C. 3D Geometry Cross-Check (Stereo Predicted vs Ground-Truth Depth)
On 15 verified stereo road frames:
- **Mean Depth Absolute Difference**: **0.26 cm**
- **Median Depth Absolute Difference**: **0.30 cm**
- **Mean Area Absolute Difference**: **0.0846 m²**

---

## 4. Execution Commands

To reproduce all results or run on custom stereo data:

```powershell
# Activate environment
..\bussense_venv\Scripts\Activate.ps1

# 1. Run Stereo Model Benchmark
python scripts/evaluate_stereo.py --model fast_foundation --num_frames 30
python scripts/evaluate_stereo.py --model sgbm --num_frames 30

# 2. Run G-7 Dataset Preparation & Leak-Free Session Split
python scripts/prepare_g7.py

# 3. Train / Evaluate YOLO11-seg Pothole Segmenter
python scripts/train_pothole_seg.py

# 4. Run RSRD Unevenness Audit
python scripts/audit_rsrd_unevenness.py

# 5. Run 3D Geometry Cross-Check
python scripts/evaluate_geometry.py

# 6. Run End-to-End Pipeline & Generate Demo Video
python scripts/make_video.py
```

Outputs:
- Full-Screen Inspection Video: `outputs/demo/pothole_measurement_demo.mp4`
- 2x2 Quad Inspection Video: `outputs/demo/pothole_measurement_debug.mp4`
- Performance Profile: `outputs/demo/performance.json`

---

## 5. Vehicle Stereo Tracking & Traffic Sensing

BusSense includes an integrated, additive vehicular perception module that executes concurrently with 3D pothole metrology strictly from **LEFT RGB + RIGHT RGB** imagery.

### A. Modular Architecture & Execution Pipeline
```
               LEFT RGB + RIGHT RGB
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
 Stereo Disparity & Depth           LEFT RGB
 (SGBM / Fast-FoundationStereo)        │
         │              ┌──────────────┴──────────────┐
         │              ▼                             ▼
         │      Traffic Detector             Pothole Segmentation
         │      (YOLO: car, truck, bus,       (Fine-tuned YOLO11-seg)
         │       motorcycle)                          │
         │              │                             ▼
         │              ▼                    Local Road Plane Fit
         │          ByteTrack                (RANSAC: aX+bY+cZ+d=0)
         │         (Vehicle #01, #02)                 │
         │              │                             ▼
         ├──────────────┼─────────────────────────────┤
         ▼              ▼                             ▼
  3D Ground-Contact  Pairwise Euclidean &     3D Pothole Metrology
  Ray-Plane Position Same-Lane Gap [HEURISTIC] (p95 Depth cm, Area m²)
  (X, Y, Z meters)   (D_ij, |ΔZ| meters)              │
         │              │                             │
         └──────────────┼─────────────────────────────┘
                        ▼
         Observable Road-Plane Polygon Area & Density
         - 4-vertex 3D planar polygon on road surface
         - Density = N_veh / A_polygon (veh/m², veh/km²)
         - Guardrail: 'DENSITY: UNRELIABLE' if plane/polygon invalid
                        │
                        ▼
              Combined BusSense HUD
```

### B. Vehicle Class Handling
- **Supported Detector Classes**: `car`, `truck`, `bus`, `motorcycle` (standard MS-COCO).
- **Note on Vans**: MS-COCO does NOT have a dedicated `van` class; in standard YOLO models, vans are naturally detected under `car` or `truck`.
- **KITTI Class Mapping**: For tracking evaluation against the KITTI benchmark, KITTI ground-truth `Van` is mapped to generic vehicle alongside `Car` (`car` -> `["Car", "Van"]`, `truck` -> `["Truck"]`). Pedestrians and cyclists are strictly excluded from vehicular metrics.

### C. 3D Vehicle Ground-Contact Localization (Ray-Plane Intersection)
Rather than taking a single center pixel or noisy 2D bounding box average:
1. **Ground-Contact Image Point**: $(u_{\text{bc}}, v_{\text{bc}}) = \left(\frac{x_1 + x_2}{2}, y_2\right)$.
2. **Calibrated Camera Ray**: $\mathbf{r} = \left[\frac{u_{\text{bc}} - c_x}{f_x}, \frac{v_{\text{bc}} - c_y}{f_y}, 1.0\right]^T$.
3. **Intersection with Fitted Local Road Plane** ($aX + bY + cZ + d = 0$):
   $$t_{\text{plane}} = -\frac{d}{a r_x + b r_y + c r_z}, \quad \mathbf{P}_{\text{ground}} = t_{\text{plane}} \mathbf{r}$$
4. **Disparity Sanity/Validity Check**: Disparity is sampled in a local ROI near $(u_{\text{bc}}, v_{\text{bc}})$ with IQR outlier rejection to obtain $Z_{\text{stereo}} = \frac{f \cdot B}{d_{\text{median}}}$. The ray-plane depth is verified against $Z_{\text{stereo}}$ ($|t_{\text{plane}} - Z_{\text{stereo}}| \le \max(4.0, 0.35 Z_{\text{stereo}})$).
5. **Guardrail**: If disparity is invalid or $< 5$ inliers, the vehicle distance is flagged `DISTANCE: UNRELIABLE`.

### D. Inter-Vehicle & Ego Proximity Metrics
- **Pairwise 3D Euclidean Distance**: $D_{ij} = \|\mathbf{P}_i - \mathbf{P}_j\|_2 = \sqrt{(X_i - X_j)^2 + (Y_i - Y_j)^2 + (Z_i - Z_j)^2}$.
- **SAME-LANE HEURISTIC**: When $|\Delta X_{ij}| < 1.8\text{ m}$, the vehicles are estimated to occupy the same traffic lane, and the longitudinal gap $|\Delta Z_{ij}|$ is reported. This is explicitly documented and labeled as an initial heuristic approximation rather than a universally fixed lane boundary.
- **Ego-Vehicle Proximity**: Nearest vehicle ID, class, 3D distance, and forward longitudinal gap are continuously tracked.

### E. Explicit Observable Road-Plane Density Formulation
- **Bounded 3D Polygon on Road Plane**: A 4-vertex planar polygon is constructed from camera FOV ray intersections at $Z_{\min} = 2.0\text{ m}$ and $Z_{\max} = 35.0\text{ m}$, clamped to the visible lateral road lane envelope ($\pm 6.0\text{ m}$):
  $$V_1 = (X_{\text{near,left}}, Y, Z_{\min}), \quad V_2 = (X_{\text{near,right}}, Y, Z_{\min}), \quad V_3 = (X_{\text{far,right}}, Y, Z_{\max}), \quad V_4 = (X_{\text{far,left}}, Y, Z_{\max})$$
- **Physical Area**: $A_{\text{polygon}} = \frac{1}{2} \|\sum_{k=1}^4 (V_k \times V_{k+1})\|$ in $\text{m}^2$.
- **Physical Density**: $\rho = \frac{N_{\text{vehicles\_in\_roi}}}{A_{\text{polygon}}}$, reported in $\text{veh}/\text{m}^2$ and $\text{veh}/\text{km}^2$.
- **Guardrail**: If road plane is missing, unstable ($|b| < 0.20$), inlier ratio $< 40\%$, or $A_{\text{polygon}} < 10\text{ m}^2$, the system reports `DENSITY: UNRELIABLE` without fabricating values.

### F. Quantitative KITTI Benchmark Validation Results
Evaluated on KITTI sequence `2011_09_26_drive_0005` (50 synchronized stereo pairs) against transformed 3D tracklet ground truth ($P_{\text{cam2}} = R_{\text{rect\_02}} (R_{\text{v2c}} P_{\text{velo}} + T_{\text{v2c}})$):
- **Precision**: 66.2%
- **Recall**: 45.3%
- **F1 Score**: 0.538
- **ID Switches**: 3
- **Tracking Continuity**: 93.0%
- **3D Distance MAE**: 3.15 m
- **3D Distance RMSE**: 3.16 m
- **Longitudinal Depth Z MAE**: 2.95 m
*(Structured report saved to `outputs/evaluation/kitti_traffic_eval.json`)*

### G. Final Real-World Stereo Demo Source
- **Selected Source**: Forward-Facing Urban Stereo Dashcam Sequence (`main_street_sbs.mp4`)
- **Duration**: 23.0 seconds continuous (460 frames @ 20 FPS)
- **Stereo Format**: Genuine side-by-side (SBS) stereo ($1280 \times 360$ total, $640 \times 360$ per eye)
- **Content**: Oncoming and leading vehicles (buses, cars) alongside real pavement defects/potholes.
- **Outputs**:
  - Full-Screen Inspection Demo: `outputs/demo/bussense_vehicle_pothole_demo.mp4`
  - 2x2 Synchronized Diagnostic Video: `outputs/demo/bussense_vehicle_pothole_debug.mp4`
  - Candidate Evaluation Contact Sheet: `outputs/demo/vehicle_pothole_candidate_contact_sheet.png`
