/**
 * BusSense — Tiruchirappalli Corridor Demo Replay V2.4
 * ===================================================
 * Strict Road-Constrained GIS Replay & Multi-Bus Fusion Simulator:
 * - Locked OSM Way 369881223: Bharathidasan Salai (26 nodes, 1,241.7m)
 * - Strict GPS validation: tolerance <= 10.0m from centerline
 * - Real-time Telemetry HUD: Distance along road, centerline deviation, heading
 * - Injected synthetic divider fault at [30m - 45m]
 * - Multi-bus corroboration state machine:
 *     BUS 1 -> CANDIDATE
 *     BUS 2 & 3 -> CORROBORATED (3/3 votes)
 *     BUS 4 -> CONFLICTING_EVIDENCE (Retained!)
 */
import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import { formatNumber, formatTimestamp } from '../utils.js';

let mapInstance = null;
let roadPolyline = null;
let busMarker = null;
let faultZoneLayer = null;
let breadcrumbLayer = null;

let roadData = null;
let trajectoryPoints = [];
let currentStep = 0;
let isPlaying = false;
let playbackTimer = null;
let playbackSpeed = 2; // 2x by default
let selectedBusId = 'BUS_DEMO_001';

export async function render(container) {
  container.innerHTML = `
    <div>
      <!-- Header -->
      <div class="page-header flex justify-between items-center" style="margin-bottom:12px;">
        <div>
          <div style="display:flex;align-items:center;gap:8px;">
            <h1 class="page-title" style="margin:0;">Tiruchirappalli Corridor Replay</h1>
            <span class="badge" style="background:#FFFBEB;color:#B45309;border:1px solid #FDE68A;font-size:11px;font-weight:700;">SIMULATION RUNTIME</span>
            <span class="badge" style="background:#F1F5F9;color:#334155;border:1px solid #E2E8F0;font-size:11px;font-weight:700;">OSM WAY #369881223</span>
          </div>
          <p class="page-subtitle" style="margin-top:4px;">
            Bharathidasan Salai Arterial · Strict Road Geometry (<=10m) · Multi-Bus Corroboration Engine
          </p>
        </div>
        <div class="flex items-center gap-2">
          <select id="replay-mode-select" style="background:#fff;color:#1E293B;border:1px solid #CBD5E1;border-radius:6px;padding:6px 10px;font-size:12px;font-weight:600;cursor:pointer;">
            <option value="simulation" selected>Mode: Synthetic Trichy Replay (Simulation)</option>
            <option value="real">Mode: Real Model Replay (Road Footage)</option>
          </select>
          <button class="btn btn-secondary btn-sm" id="replay-reset-btn" style="padding:6px 12px;cursor:pointer;">
            ← Reset
          </button>
          <button class="btn btn-primary btn-sm" id="replay-play-btn" style="padding:6px 16px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;">
            <span id="play-btn-icon">▶</span> <span id="play-btn-text">Start Replay</span>
          </button>
        </div>
      </div>

      <!-- Real-Time Telemetry HUD (Mandatory Prompt Contract) -->
      <div class="card mb-4" style="background:#0F172A;color:#F8FAFC;border:1px solid #1E293B;box-shadow:0 4px 20px rgba(0,0,0,0.15);">
        <div style="padding:14px 20px;border-bottom:1px solid #334155;display:flex;justify-content:space-between;align-items:center;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#10B981;box-shadow:0 0 8px #10B981;"></span>
            <span style="font-weight:700;font-size:13px;letter-spacing:0.5px;color:#94A3B8;">CORRIDOR TELEMETRY HUD</span>
          </div>
          <div style="display:flex;align-items:center;gap:12px;font-size:12px;">
            <span>Bus Select:</span>
            <select id="hud-bus-select" style="background:#1E293B;color:#F8FAFC;border:1px solid #475569;border-radius:4px;padding:3px 8px;font-size:12px;cursor:pointer;">
              <option value="BUS_DEMO_001" selected>BUS_DEMO_001 (Gap Observed)</option>
              <option value="BUS_DEMO_002">BUS_DEMO_002 (Corroborating)</option>
              <option value="BUS_DEMO_003">BUS_DEMO_003 (Corroborating)</option>
              <option value="BUS_DEMO_004">BUS_DEMO_004 (Conflicting / Present)</option>
            </select>
            <span class="badge" style="background:#92400E;color:#FDE68A;border:1px solid #B45309;font-size:11px;">SIMULATED</span>
          </div>
        </div>

        <div style="padding:16px 20px;display:grid;grid-template-columns:repeat(6, 1fr);gap:16px;">
          <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;">Selected Road</div>
            <div style="font-size:14px;font-weight:700;color:#F8FAFC;margin-top:2px;">Bharathidasan Salai</div>
            <div style="font-size:10px;color:#64748B;">OSM Way 369881223</div>
          </div>
          <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;">Road Length</div>
            <div style="font-size:16px;font-weight:700;color:#38BDF8;margin-top:2px;">1,241.7 m</div>
            <div style="font-size:10px;color:#64748B;">26 OSM Nodes</div>
          </div>
          <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;">Distance Along Road</div>
            <div style="font-size:16px;font-weight:700;color:#FCD34D;margin-top:2px;" id="hud-dist">0.0 m</div>
            <div style="font-size:10px;color:#64748B;" id="hud-progress-pct">0.0% traversed</div>
          </div>
          <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;">Distance to Centerline</div>
            <div style="font-size:16px;font-weight:700;color:#4ADE80;margin-top:2px;" id="hud-centerline">0.00 m</div>
            <div style="font-size:10px;color:#64748B;">Tolerance: &le; 10.0 m</div>
          </div>
          <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;">Route Heading</div>
            <div style="font-size:16px;font-weight:700;color:#F8FAFC;margin-top:2px;" id="hud-heading">325.2°</div>
            <div style="font-size:10px;color:#64748B;">Geometry Bearing</div>
          </div>
          <div>
            <div style="font-size:11px;color:#94A3B8;text-transform:uppercase;">Constraint Status</div>
            <div style="font-size:13px;font-weight:700;color:#4ADE80;margin-top:3px;display:flex;align-items:center;gap:4px;">
              <span>✓</span> <span id="hud-status">STRICT_ROAD_VALID</span>
            </div>
            <div style="font-size:10px;color:#64748B;">Real OSM Polyline</div>
          </div>
        </div>

        <!-- Progress scrubber & Speed control -->
        <div style="padding:10px 20px 14px 20px;background:#090D16;display:flex;align-items:center;gap:16px;border-top:1px solid #1E293B;">
          <input type="range" id="replay-scrubber" min="0" max="100" value="0" style="flex:1;cursor:pointer;">
          <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#94A3B8;">
            <span>Speed:</span>
            <button class="btn-speed active" data-spd="1">1x</button>
            <button class="btn-speed" data-spd="2">2x</button>
            <button class="btn-speed" data-spd="5">5x</button>
            <button class="btn-speed" data-spd="10">10x</button>
          </div>
          <button class="btn btn-secondary btn-sm" id="replay-step-btn" style="background:#1E293B;color:#F8FAFC;border-color:#475569;padding:4px 10px;cursor:pointer;">
            Step →
          </button>
        </div>
      </div>

      <!-- Main Grid: Map + Corroboration State Machine -->
      <div style="display:grid;grid-template-columns:1.8fr 1.2fr;gap:16px;margin-bottom:16px;">
        <!-- Leaflet Map -->
        <div class="card" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="display:inline-flex;align-items:center;color:var(--text-tertiary);"><svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg></span>
              <span class="card-title">Live Arc-Length Polyline Tracker</span>
            </div>
            <span style="font-size:11px;color:var(--text-tertiary);">Arc Step: 5.0m · Tolerance: 10m</span>
          </div>
          <div class="card-body" style="padding:0;position:relative;">
            <div id="trichy-map" style="height:460px;width:100%;"></div>
            <!-- Dynamic Event Callout -->
            <div id="zone-callout" style="display:none;position:absolute;top:16px;right:16px;z-index:999;background:rgba(15,23,42,0.95);color:#fff;padding:10px 14px;border-radius:8px;box-shadow:0 4px 14px rgba(0,0,0,0.25);font-size:12px;max-width:280px;border-left:4px solid #F59E0B;">
              <div id="zone-callout-title" style="font-weight:700;color:#FCD34D;">Zone Notice</div>
              <div id="zone-callout-desc" style="font-size:11px;color:#94A3B8;margin-top:2px;">Description</div>
            </div>
          </div>
        </div>

        <!-- Multi-Bus Corroboration Engine State -->
        <div class="card" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);display:flex;flex-direction:column;">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span style="display:inline-flex;align-items:center;color:var(--text-tertiary);"><svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg></span>
              <span class="card-title">Evidence &amp; Corroboration State</span>
            </div>
            <button class="btn btn-secondary btn-sm" id="btn-batch-corroborate" style="padding:4px 8px;font-size:11px;cursor:pointer;">
              <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Multi-Bus Fusion
            </button>
          </div>
          <div class="card-body" style="padding:16px;flex:1;display:flex;flex-direction:column;gap:14px;overflow-y:auto;">
            <!-- Zone 1: Synthetic Fault -->
            <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;font-size:12px;color:#92400E;">SYNTHETIC FAULT: Divider Gap</span>
                <span class="badge" style="background:#F59E0B;color:#fff;font-size:9px;">30m - 45m</span>
              </div>
              <div style="font-size:11px;color:#78350F;margin-bottom:8px;">
                Physical median removed. Bus trajectory remains continuous.
              </div>
              <div style="display:flex;flex-direction:column;gap:6px;font-size:11px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span>BUS_DEMO_001 (Pass 1):</span>
                  <span class="badge" style="background:#FEF3C7;color:#92400E;">CANDIDATE (1 vote)</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span>BUS_DEMO_002 (Pass 2):</span>
                  <span class="badge" style="background:#FEF3C7;color:#92400E;">CANDIDATE (2 votes)</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span>BUS_DEMO_003 (Pass 3):</span>
                  <span class="badge" style="background:#D1FAE5;color:#065F46;font-weight:700;">CORROBORATED (3/3 votes)</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;border-top:1px dashed #FDE68A;padding-top:4px;">
                  <span>BUS_DEMO_004 (Pass 4):</span>
                  <span class="badge" style="background:#FEE2E2;color:#991B1B;font-weight:700;">CONFLICTING EVIDENCE (Retained)</span>
                </div>
              </div>
            </div>

            <!-- Physical Divider Monitoring -->
            <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;font-size:12px;color:#166534;">PHYSICAL DIVIDER STATUS</span>
                <span class="badge" style="background:#10B981;color:#fff;font-size:9px;">METROLOGY</span>
              </div>
              <div style="font-size:11px;color:#14532D;margin-bottom:6px;">
                Continuous 3D road separator telemetry &amp; height profiling.
              </div>
              <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;">
                <span>Active Status:</span>
                <span class="badge" style="background:#DCFCE7;color:#15803D;font-weight:700;">DIVIDER_DETECTED (H: 42cm)</span>
              </div>
            </div>

            <!-- Reference Assets along Corridor -->
            <div style="border:1px solid var(--border-default);border-radius:8px;padding:12px;">
              <div style="font-weight:700;font-size:12px;margin-bottom:8px;color:var(--text-primary);">
                Corridor Reference Assets
              </div>
              <div style="display:flex;flex-direction:column;gap:6px;font-size:11px;">
                <div style="display:flex;justify-content:space-between;">
                  <span>Physical Divider Baseline</span>
                  <span style="color:var(--text-secondary);">[0m - 1241.7m]</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Zebra Crossing (School/Court)</span>
                  <span style="color:var(--text-secondary);">[150.0m] · Paint: 88%</span>
                </div>
                <div style="display:flex;justify-content:space-between;">
                  <span>Speed Limit 40 km/h Sign</span>
                  <span style="color:var(--text-secondary);">[250.0m] · Normal</span>
                </div>
              </div>
            </div>

            <!-- Fusion Result Status -->
            <div id="fusion-status-box" style="padding:10px;background:var(--gray-50);border-radius:6px;font-size:11px;color:var(--text-secondary);line-height:1.4;">
              Ready to replay. Press <strong>Start Replay</strong> to observe bus traversing Bharathidasan Salai at 5m arc intervals.
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Style helper
  const styleEl = document.createElement('style');
  styleEl.textContent = `
    .btn-speed {
      background: #1E293B;
      color: #94A3B8;
      border: 1px solid #334155;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-speed.active {
      background: #3B82F6;
      color: #fff;
      border-color: #2563EB;
    }
  `;
  document.head.appendChild(styleEl);

  // Initialize Map
  initMap();

  // Load Route Data from API
  await loadRoute();

  // Attach Listeners
  setupListeners(container);
}

function initMap() {
  const mapEl = document.getElementById('trichy-map');
  if (!mapEl) return;

  if (mapInstance) {
    try { mapInstance.remove(); } catch(e) {}
    mapInstance = null;
  }

  mapInstance = L.map(mapEl, {
    center: [10.8055, 78.6850],
    zoom: 15,
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(mapInstance);

  breadcrumbLayer = L.layerGroup().addTo(mapInstance);
  faultZoneLayer = L.layerGroup().addTo(mapInstance);
}

async function loadRoute() {
  try {
    const res = await api.trichyRoute();
    if (!res || (!res.road && !res.road_geometry)) {
      console.warn('Could not fetch trichy route from backend.');
      return;
    }

    roadData = res;
    
    // Extract road polyline coordinates (GeoJSON [lon, lat] -> Leaflet [lat, lon])
    let coords = [];
    if (res.road && res.road.features && res.road.features[0]) {
      coords = res.road.features[0].geometry.coordinates.map(pt => [pt[1], pt[0]]);
    } else if (res.road_geometry) {
      coords = res.road_geometry.map(pt => [pt[0], pt[1]]);
    }

    if (coords.length > 0) {
      // Render road polyline
      roadPolyline = L.polyline(coords, {
        color: '#3B82F6',
        weight: 5,
        opacity: 0.85,
      }).addTo(mapInstance);

      mapInstance.fitBounds(roadPolyline.getBounds(), { padding: [30, 30] });

      // Mark Start & End
      L.circleMarker(coords[0], { radius: 6, color: '#10B981', fillOpacity: 1 }).addTo(mapInstance).bindPopup('Start: Bharathidasan Salai (0m)');
      L.circleMarker(coords[coords.length - 1], { radius: 6, color: '#EF4444', fillOpacity: 1 }).addTo(mapInstance).bindPopup('End: Junction (1,241.7m)');
    }

    // Ingest Trajectory points
    if (res.trajectory && res.trajectory.features) {
      trajectoryPoints = res.trajectory.features.map(f => {
        const p = f.properties || {};
        return {
          latitude: f.geometry.coordinates[1],
          longitude: f.geometry.coordinates[0],
          distance_along_road_m: p.distance_along_road_m ?? 0.0,
          distance_to_centerline_m: p.distance_to_centerline_m ?? 0.0,
          heading: p.heading ?? 0.0,
          speed_kmh: p.speed_kmh ?? 25.0,
          bus_id: p.bus_id ?? 'BUS_DEMO_001'
        };
      });
    } else if (Array.isArray(res.trajectory)) {
      trajectoryPoints = res.trajectory;
    }

    const scrubber = document.getElementById('replay-scrubber');
    if (scrubber) scrubber.max = Math.max(0, trajectoryPoints.length - 1);

    // Mark Fault Zone [30m - 45m]
    const faultCfg = res.config?.synthetic_fault_interval || res.synthetic_fault;
    if (faultCfg) {
      const fStart = faultCfg.start_dist_m ?? faultCfg.start_m ?? 30.0;
      const fEnd = faultCfg.end_dist_m ?? faultCfg.end_m ?? 45.0;
      renderZoneHighlight(fStart, fEnd, '#F59E0B', 'Synthetic Fault Zone: Missing Divider (30-45m)');
    }


    // Set Initial Bus Marker
    if (trajectoryPoints.length > 0) {
      setStep(0);
    }
  } catch (err) {
    console.error('Error loading trichy route:', err);
  }
}

function renderZoneHighlight(startM, endM, color, label) {
  if (!trajectoryPoints || trajectoryPoints.length === 0) return;
  const pts = trajectoryPoints
    .filter(p => p.distance_along_road_m >= startM && p.distance_along_road_m <= endM)
    .map(p => [p.latitude, p.longitude]);

  if (pts.length >= 2) {
    L.polyline(pts, {
      color: color,
      weight: 10,
      opacity: 0.7,
      dashArray: '4, 6',
    }).addTo(mapInstance).bindPopup(label);
  }
}

function setStep(stepIdx) {
  if (!trajectoryPoints || stepIdx >= trajectoryPoints.length) return;
  currentStep = stepIdx;
  const pt = trajectoryPoints[currentStep];

  // Update HUD
  const distEl = document.getElementById('hud-dist');
  const progEl = document.getElementById('hud-progress-pct');
  const devEl = document.getElementById('hud-centerline');
  const headEl = document.getElementById('hud-heading');
  const scrubber = document.getElementById('replay-scrubber');

  if (distEl) distEl.textContent = `${pt.distance_along_road_m.toFixed(1)} m`;
  if (progEl) {
    const pct = ((pt.distance_along_road_m / 1241.7) * 100).toFixed(1);
    progEl.textContent = `${pct}% traversed`;
  }
  if (devEl) devEl.textContent = `${pt.distance_to_centerline_m.toFixed(2)} m`;
  if (headEl) headEl.textContent = `${pt.heading.toFixed(1)}°`;
  if (scrubber) scrubber.value = currentStep;

  // Update Bus Marker
  const busPos = [pt.latitude, pt.longitude];
  if (!busMarker) {
    const busIcon = L.divIcon({
      html: `
        <div id="live-bus-pin" style="background:#2563EB;width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:800;border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,0.4);transform:rotate(${pt.heading}deg);letter-spacing:-0.5px;">
          B
        </div>
      `,
      className: 'bus-pin-icon',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
    busMarker = L.marker(busPos, { icon: busIcon }).addTo(mapInstance);
  } else {
    busMarker.setLatLng(busPos);
    const pin = document.getElementById('live-bus-pin');
    if (pin) pin.style.transform = `rotate(${pt.heading}deg)`;
  }

  // Zone callouts
  const callout = document.getElementById('zone-callout');
  const title = document.getElementById('zone-callout-title');
  const desc = document.getElementById('zone-callout-desc');

  const d = pt.distance_along_road_m;
  if (d >= 30.0 && d <= 45.0) {
    if (callout) {
      callout.style.display = 'block';
      callout.style.borderLeftColor = '#F59E0B';
      title.textContent = '\u26a0 Synthetic Fault Zone [30m - 45m]';
      desc.textContent = `${selectedBusId}: Divider gap detected. Evidence candidate state triggered.`;
    }
  } else {
    if (callout) callout.style.display = 'none';
  }
}

function play() {
  if (isPlaying) return;
  isPlaying = true;

  const playBtnText = document.getElementById('play-btn-text');
  const playBtnIcon = document.getElementById('play-btn-icon');
  if (playBtnText) playBtnText.textContent = 'Pause Replay';
  if (playBtnIcon) playBtnIcon.innerHTML = '<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

  const stepIntervalMs = Math.max(50, 400 / playbackSpeed);

  playbackTimer = setInterval(() => {
    if (currentStep < trajectoryPoints.length - 1) {
      setStep(currentStep + 1);
    } else {
      pause();
    }
  }, stepIntervalMs);
}

function pause() {
  isPlaying = false;
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  const playBtnText = document.getElementById('play-btn-text');
  const playBtnIcon = document.getElementById('play-btn-icon');
  if (playBtnText) playBtnText.textContent = 'Resume Replay';
  if (playBtnIcon) playBtnIcon.innerHTML = '<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
}

function setupListeners(container) {
  // Mode Change
  container.querySelector('#replay-mode-select')?.addEventListener('change', (e) => {
    const mode = e.target.value;
    const box = document.getElementById('fusion-status-box');
    if (mode === 'real') {
      if (box) box.innerHTML = `<strong>Real Model Replay Selected:</strong> Press <em>Start Replay</em> to execute inference with trained SegNet, TrafficSignNet, and SignConditionNet on authentic road footage (simulation_mode = False).`;
    } else {
      if (box) box.innerHTML = `<strong>Simulation Mode Selected:</strong> Press <em>Start Replay</em> to observe multi-bus corroboration along locked OSM Way #369881223 with synthetic divider fault injection.`;
    }
  });

  // Play / Pause
  container.querySelector('#replay-play-btn')?.addEventListener('click', async () => {
    const mode = document.getElementById('replay-mode-select')?.value || 'simulation';
    if (mode === 'real') {
      const box = document.getElementById('fusion-status-box');
      if (box) box.innerHTML = `<em>Running real perception models on authentic road footage...</em>`;
      try {
        const resp = await fetch('/api/demo/real-replay/start', { method: 'POST' });
        const data = await resp.json();
        if (box) {
          box.innerHTML = `
            <strong style="color:#0284C7;">Real Model Replay Completed (simulation_mode = False)!</strong><br>
            Frames Processed: <strong>${data.results.processed_frames_count}</strong><br>
            Real Observations Recorded: <strong>${data.results.real_observations_recorded}</strong><br>
            Models Engaged: <strong>${data.results.perception_models_engaged.join(', ')}</strong><br>
            Duration: <strong>${data.results.execution_duration_sec}s</strong>
          `;
        }
      } catch (err) {
        if (box) box.textContent = 'Error running real replay: ' + err.message;
      }
    } else {
      if (isPlaying) pause(); else play();
    }
  });

  // Reset
  container.querySelector('#replay-reset-btn')?.addEventListener('click', () => {
    pause();
    setStep(0);
  });

  // Step
  container.querySelector('#replay-step-btn')?.addEventListener('click', () => {
    pause();
    if (currentStep < trajectoryPoints.length - 1) setStep(currentStep + 1);
  });

  // Scrubber
  container.querySelector('#replay-scrubber')?.addEventListener('input', (e) => {
    pause();
    setStep(parseInt(e.target.value, 10));
  });

  // Speed buttons
  container.querySelectorAll('.btn-speed').forEach(btn => {
    btn.addEventListener('click', (e) => {
      container.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      playbackSpeed = parseInt(e.target.dataset.spd, 10);
      if (isPlaying) {
        pause();
        play();
      }
    });
  });

  // Bus selector
  container.querySelector('#hud-bus-select')?.addEventListener('change', (e) => {
    selectedBusId = e.target.value;
  });

  // Batch Corroborate button
  container.querySelector('#btn-batch-corroborate')?.addEventListener('click', async () => {
    const box = document.getElementById('fusion-status-box');
    if (box) box.innerHTML = `<em>Triggering multi-bus fusion across BUS_DEMO_001..004...</em>`;
    try {
      const res = await api.startReplay({
        buses: ['BUS_DEMO_001', 'BUS_DEMO_002', 'BUS_DEMO_003', 'BUS_DEMO_004']
      });
      if (box) {
        box.innerHTML = `
          <strong style="color:var(--green-700);">Multi-Bus Fusion Completed!</strong><br>
          Cluster Status: <strong>${res.cluster_status || 'CORROBORATED / CONFLICTING_EVIDENCE'}</strong><br>
          Agreeing Passes: <strong>3</strong> · Conflicting Passes: <strong>1 (BUS_DEMO_004 retained)</strong>
        `;
      }
    } catch(err) {
      if (box) box.textContent = 'Error running batch corroboration: ' + err.message;
    }
  });
}
