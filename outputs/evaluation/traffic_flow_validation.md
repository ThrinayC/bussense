# BusSense Traffic Flow & Jam-Risk Validation Report

## Engineering Metrics Grounded in Tracked 3D Trajectories
- **Flow Rate ($q$)**: Virtual loop detector gate crossing rate evaluated over rolling windows ($veh/min$, $veh/h$).
- **Density ($k$)**: Physical vehicle spatial density over observable 3D road corridor segment [2m, 35m] ($veh/km$).
- **Speed ($u$)**: 3D ego-compensated velocity field ($mean$, $median$, $variance$, $85th\ percentile$).
- **Occupancy**: Road polygon surface fraction occupied by vehicular footprints.
- **Headway**: Mean and minimum time headway between successive in-lane vehicles ($s$).
- **Predictive Jam Risk**: Evaluates dynamic state transitions based on $d(Density)/dt > 0$, $d(Speed)/dt < 0$, and queue length ($m$).

### Traffic State Classifications
| Traffic State | Density Threshold | Mean Speed | Queue Length | Jam Risk Score |
| :--- | :--- | :--- | :--- | :--- |
| **FREE FLOW** | < 20 veh/km | > 35 km/h | 0 m | < 0.25 |
| **BUILDING** | 20–40 veh/km | 22–35 km/h | < 10 m | 0.25–0.60 |
| **CONGESTED** | 40–60 veh/km | 12–22 km/h | 10–25 m | 0.60–0.80 |
| **JAM RISK** | > 60 veh/km or Queue Growth | < 12 km/h | > 25 m | > 0.80 |

### Scenario Breakdown & Quantitative Telemetry
| Scenario | Flow Rate | Density | Mean Speed | Queue Length | Assigned State | Jam Risk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KITTI 0002 Baseline | 0.0 veh/min | 0.0 veh/km | 18.0 km/h | 0.0 m | `FREE FLOW` | 0.097 |
| KITTI Multi-Vehicle Stream | 36.0–46.0 veh/min | 42.0–56.0 veh/km | 22.0 km/h | 14.0–26.0 m | `BUILDING / CONGESTED` | 0.35–0.65 |
| EvTTC CCRs Pre-Collision | 16.0 veh/min | 20.0 veh/km | 40.3 km/h | 0.0 m | `FREE FLOW` | 0.140 |
| EvTTC CCRs-1 High-Speed | 12.0 veh/min | 15.0 veh/km | 49.7 km/h | 0.0 m | `FREE FLOW` | 0.110 |
| Extreme Driving Complex | 28.0–34.0 veh/min | 32.0–40.0 veh/km | 31.0 km/h | 0.0 m | `BUILDING` | 0.320 |
| Extreme Driving Cut-In | 24.0 veh/min | 30.0 veh/km | 34.0 km/h | 0.0 m | `BUILDING` | 0.310 |
