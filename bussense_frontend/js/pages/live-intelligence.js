/**
 * BusSense — Live Intelligence Dashboard (10 Perception Domains)
 * ==============================================================
 * Comprehensive, structured telemetry surfacing all 10 intelligence domains:
 * 1. [ROAD HEALTH]
 * 2. [TRAFFIC & SAFETY]
 * 3. [WATERLOGGING]
 * 4. [ROAD MARKINGS]
 * 5. [TRAFFIC SIGNS]
 * 6. [ROAD DIVIDERS]
 * 7. [ROAD HAZARDS]
 * 8. [STEREO 3D]
 * 9. [GIS / LOCATION]
 * 10. [MULTI-BUS EVIDENCE]
 */
import api from '../api.js';

let pollTimer = null;
let isPolling = true;

export async function render(container) {
  container.innerHTML = `
    <div>
      <!-- Header -->
      <div class="page-header flex justify-between items-center" style="margin-bottom:16px;">
        <div>
          <div style="display:flex;align-items:center;gap:8px;">
            <h1 class="page-title" style="margin:0;">Live Edge Intelligence</h1>
            <span class="badge" style="background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;font-size:11px;font-weight:700;">10 DOMAINS ACTIVE</span>
            <span class="badge" style="background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;font-size:11px;font-weight:700;">REAL-TIME</span>
          </div>
          <p class="page-subtitle" style="margin-top:4px;">
            Stereo 3D Metrology · Multi-Bus Evidence Corroboration · Road Health &amp; Autonomous Safety Engine
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn btn-secondary btn-sm" id="intel-poll-toggle" style="padding:6px 12px;cursor:pointer;">
            <span id="poll-toggle-icon"><svg width="8" height="8" viewBox="0 0 8 8" fill="#10B981"><circle cx="4" cy="4" r="4"/></svg></span> Auto-Refresh (3s)
          </button>
          <button class="btn btn-primary btn-sm" id="intel-refresh-btn" style="padding:6px 14px;cursor:pointer;">
            ↻ Fetch Live State
          </button>
        </div>
      </div>

      <!-- 10 Domains Grid -->
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));gap:16px;" id="domains-grid">
        <!-- Loading skeletons -->
        <div class="card" style="padding:24px;text-align:center;color:var(--text-secondary);grid-column:1/-1;">
          Loading live intelligence telemetry...
        </div>
      </div>
    </div>
  `;

  // Bind controls
  const refreshBtn = container.querySelector('#intel-refresh-btn');
  const pollBtn = container.querySelector('#intel-poll-toggle');

  if (refreshBtn) refreshBtn.addEventListener('click', loadIntelligence);
  if (pollBtn) {
    pollBtn.addEventListener('click', () => {
      isPolling = !isPolling;
      const icon = document.getElementById('poll-toggle-icon');
      if (icon) icon.innerHTML = isPolling ? '<svg width="8" height="8" viewBox="0 0 8 8" fill="#10B981"><circle cx="4" cy="4" r="4"/></svg>' : '<svg width="8" height="8" viewBox="0 0 8 8" fill="#CBD5E1"><circle cx="4" cy="4" r="4"/></svg>';
    });
  }

  await loadIntelligence();

  // Polling loop
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (isPolling) loadIntelligence();
  }, 3000);
}

export function destroy() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function loadIntelligence() {
  try {
    const data = await api.liveIntelligence();
    if (!data || !data.domains) return;
    renderDomains(data.domains, data.timestamp);
  } catch (err) {
    console.error('Error fetching live intelligence:', err);
  }
}

function renderDomains(d, ts) {
  const grid = document.getElementById('domains-grid');
  if (!grid) return;

  const rh = d.road_health || {};
  const tsaf = d.traffic_safety || {};
  const wl = d.waterlogging || {};
  const rm = d.road_markings || {};
  const sgn = d.traffic_signs || {};
  const rd = d.road_dividers || {};
  const rhz = d.road_hazards || {};
  const s3d = d.stereo_3d || {};
  const gis = d.gis_location || {};
  const mb = d.multibus_evidence || {};

  grid.innerHTML = `
    <!-- 1. ROAD HEALTH -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="5" y1="8" x2="19" y2="8"/><line x1="5" y1="14" x2="19" y2="14"/><line x1="5" y1="20" x2="11" y2="20"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">1. ROAD HEALTH</strong>
        </div>
        <span class="badge" style="background:${rh.health_score_pct >= 80 ? '#10B981' : '#F59E0B'};color:#fff;font-size:10px;font-weight:700;">
          ${rh.health_score_pct}% HEALTH (${rh.road_condition})
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Pothole Detections</div>
          <div style="font-size:16px;font-weight:700;color:#DC2626;margin-top:2px;">${rh.pothole_count}</div>
          <div style="font-size:10px;color:#94A3B8;">Max Depth: <strong>${rh.max_pothole_depth_cm} cm</strong> · Area: ${rh.total_pothole_area_m2} m²</div>
        </div>
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Crack Detections</div>
          <div style="font-size:16px;font-weight:700;color:#D97706;margin-top:2px;">${rh.crack_count}</div>
          <div style="font-size:10px;color:#94A3B8;">Total Length: <strong>${rh.total_crack_length_m} m</strong></div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Hazard State: <strong style="color:#1E293B;">${rh.hazard_state}</strong></span>
        <a href="#road-health" style="color:#3B82F6;text-decoration:none;font-weight:600;">View Defects →</a>
      </div>
    </div>

    <!-- 2. TRAFFIC & SAFETY -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="5" y="2" width="14" height="20" rx="3"/><circle cx="12" cy="7" r="2" fill="currentColor" stroke="none"/><circle cx="12" cy="13" r="2"/><circle cx="12" cy="19" r="2"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">2. TRAFFIC &amp; SAFETY</strong>
        </div>
        <span class="badge" style="background:${tsaf.safety_state === 'SAFE' ? '#10B981' : '#EF4444'};color:#fff;font-size:10px;font-weight:700;">
          ${tsaf.safety_state}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Tracked Vehicles</div>
          <div style="font-size:16px;font-weight:700;color:#2563EB;margin-top:2px;">${tsaf.tracked_vehicles}</div>
          <div style="font-size:10px;color:#94A3B8;">Lead Dist: <strong>${tsaf.lead_distance_m} m</strong> · Speed: ${tsaf.vehicle_speed_kmh} km/h</div>
        </div>
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Flow &amp; Congestion</div>
          <div style="font-size:16px;font-weight:700;color:#0D9488;margin-top:2px;">${tsaf.traffic_flow_vpm} <span style="font-size:11px;font-weight:400;">v/m</span></div>
          <div style="font-size:10px;color:#94A3B8;">Density: ${tsaf.traffic_density_vpk} v/km · Headway: ${tsaf.headway_gap_s}s</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>VRU Detections: <strong>${tsaf.vru_detections}</strong> (Congestion: <strong>${tsaf.congestion_risk}</strong>)</span>
        <a href="#traffic" style="color:#3B82F6;text-decoration:none;font-weight:600;">Traffic Intel →</a>
      </div>
    </div>

    <!-- 3. WATERLOGGING -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">3. WATERLOGGING</strong>
        </div>
        <span class="badge" style="background:#0284C7;color:#fff;font-size:10px;font-weight:700;">
          ${wl.waterlogging_state}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F0F9FF;padding:8px 10px;border-radius:6px;border:1px solid #BAE6FD;">
          <div style="color:#0369A1;font-size:10px;text-transform:uppercase;">Road Water Coverage</div>
          <div style="font-size:16px;font-weight:700;color:#0284C7;margin-top:2px;">${wl.water_coverage_pct}%</div>
          <div style="font-size:10px;color:#0369A1;">Persistence: <strong>${wl.persistence_frames} frames</strong></div>
        </div>
        <div style="background:#F0F9FF;padding:8px 10px;border-radius:6px;border:1px solid #BAE6FD;">
          <div style="color:#0369A1;font-size:10px;text-transform:uppercase;">Weather / Flood State</div>
          <div style="font-size:14px;font-weight:700;color:#075985;margin-top:3px;">${wl.severe_weather_state}</div>
          <div style="font-size:10px;color:#0369A1;">Overlay: <strong>${wl.overlay_type}</strong></div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Transparent Mask: <strong style="color:#10B981;">Active</strong> (Road Surface Only)</span>
        <a href="#infrastructure" style="color:#3B82F6;text-decoration:none;font-weight:600;">Metrology →</a>
      </div>
    </div>

    <!-- 4. ROAD MARKINGS -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">4. ROAD MARKINGS &amp; ZEBRA</strong>
        </div>
        <span class="badge" style="background:#10B981;color:#fff;font-size:10px;font-weight:700;">
          ${rm.lane_condition}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Lane Health &amp; Paint</div>
          <div style="font-size:16px;font-weight:700;color:#059669;margin-top:2px;">${rm.coverage_pct}%</div>
          <div style="font-size:10px;color:#94A3B8;">Continuity: <strong>${rm.continuity_pct}%</strong> · Faded: ${rm.faded_status}</div>
        </div>
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Zebra Crossing Wear</div>
          <div style="font-size:16px;font-weight:700;color:#0D9488;margin-top:2px;">${rm.zebra_paint_pct}% <span style="font-size:10px;font-weight:400;color:#64748B;">paint</span></div>
          <div style="font-size:10px;color:#94A3B8;">Wear: <strong>${rm.zebra_wear_pct}%</strong> (${rm.zebra_detected ? 'Detected' : 'None'})</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Bus Lane / Corridor: <strong style="color:#1E293B;">${rm.bus_lane_class}</strong></span>
        <span class="badge" style="background:#F1F5F9;color:#475569;border:1px solid #E2E8F0;font-size:9px;">CALIBRATED WEAR</span>
      </div>
    </div>

    <!-- 5. TRAFFIC SIGNS -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="2"/><line x1="12" y1="16" x2="12" y2="21"/><line x1="8" y1="21" x2="16" y2="21"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">5. TRAFFIC SIGNS &amp; OCR</strong>
        </div>
        <span class="badge" style="background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;font-size:10px;font-weight:700;">
          ${sgn.condition}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Detected Class</div>
          <div style="font-size:14px;font-weight:700;color:#1E293B;margin-top:2px;">${sgn.detected_class}</div>
          <div style="font-size:10px;color:#94A3B8;">Confidence: <strong>${(sgn.confidence * 100).toFixed(0)}%</strong></div>
        </div>
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">OCR Text / Reading</div>
          <div style="font-size:16px;font-weight:700;color:#1E293B;margin-top:2px;">${sgn.ocr_result}</div>
          <div style="font-size:10px;color:#94A3B8;">Missing State: <strong>${sgn.missing_sign_state}</strong></div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Multi-Frame Corroboration: <strong style="color:#10B981;">Verified</strong></span>
        <span class="badge" style="background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;font-size:9px;">TrafficSignNet</span>
      </div>
    </div>

    <!-- 6. ROAD DIVIDERS -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">6. ROAD DIVIDERS &amp; GAPS</strong>
        </div>
        <span class="badge" style="background:${rd.missing_state === 'DIVIDER_GAP' ? '#F59E0B' : '#10B981'};color:#fff;font-size:10px;font-weight:700;">
          ${rd.missing_state}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#FFFBEB;padding:8px 10px;border-radius:6px;border:1px solid #FDE68A;">
          <div style="color:#92400E;font-size:10px;text-transform:uppercase;">Physical Continuity</div>
          <div style="font-size:16px;font-weight:700;color:#B45309;margin-top:2px;">${rd.divider_present ? 'PRESENT' : 'ABSENT'}</div>
          <div style="font-size:10px;color:#92400E;">Gaps: <strong>${rd.physical_gaps_count}</strong> · Length: <strong>${rd.physical_gap_length_m} m</strong></div>
        </div>
        <div style="background:#FFFBEB;padding:8px 10px;border-radius:6px;border:1px solid #FDE68A;">
          <div style="color:#92400E;font-size:10px;text-transform:uppercase;">3D Height Profile</div>
          <div style="font-size:16px;font-weight:700;color:#B45309;margin-top:2px;">${rd.median_height_cm} cm</div>
          <div style="font-size:10px;color:#92400E;">IQR: <strong>${rd.height_iqr_cm} cm</strong> · Condition: ${rd.divider_condition}</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Road Separator: <strong style="color:#1E293B;">Dual-Carriageway Median</strong></span>
        <a href="#trichy-replay" style="color:#3B82F6;text-decoration:none;font-weight:600;">View Replay →</a>
      </div>
    </div>

    <!-- 7. ROAD HAZARDS -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">7. ROAD HAZARDS</strong>
        </div>
        <span class="badge" style="background:${rhz.hazard_severity === 'HIGH' ? '#DC2626' : '#F59E0B'};color:#fff;font-size:10px;font-weight:700;">
          ${rhz.hazard_severity} SEVERITY
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#FEF2F2;padding:8px 10px;border-radius:6px;border:1px solid #FECACA;">
          <div style="color:#991B1B;font-size:10px;text-transform:uppercase;">Active Hazards</div>
          <div style="font-size:16px;font-weight:700;color:#DC2626;margin-top:2px;">${rhz.active_hazard_count}</div>
          <div style="font-size:10px;color:#991B1B;">Potholes: <strong>${rhz.pothole_hazards}</strong> · Cracks: <strong>${rhz.crack_hazards}</strong></div>
        </div>
        <div style="background:#FEF2F2;padding:8px 10px;border-radius:6px;border:1px solid #FECACA;">
          <div style="color:#991B1B;font-size:10px;text-transform:uppercase;">Water &amp; Obstacles</div>
          <div style="font-size:14px;font-weight:700;color:#991B1B;margin-top:3px;">${rhz.water_hazard}</div>
          <div style="font-size:10px;color:#991B1B;">Debris / Unknown Obstruction Support</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Supported: <strong>${(rhz.supported_classes || []).slice(0, 3).join(', ')}...</strong></span>
        <span class="badge" style="background:#DC2626;color:#fff;font-size:9px;">MULTI-SIGNAL</span>
      </div>
    </div>

    <!-- 8. STEREO 3D -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 3H3v18l3-3 3 3 3-3 3 3 3-3 3 3V3z"/><line x1="3" y1="9" x2="7" y2="9"/><line x1="3" y1="14" x2="5" y2="14"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">8. STEREO 3D &amp; ROAD PLANE</strong>
        </div>
        <span class="badge" style="background:#10B981;color:#fff;font-size:10px;font-weight:700;">
          ${s3d.stereo_status}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Disparity Availability</div>
          <div style="font-size:16px;font-weight:700;color:#2563EB;margin-top:2px;">${s3d.disparity_availability_pct}%</div>
          <div style="font-size:10px;color:#94A3B8;">Dense surface disparity active</div>
        </div>
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Road-Plane Validity</div>
          <div style="font-size:16px;font-weight:700;color:#059669;margin-top:2px;">${s3d.road_plane_valid ? 'VALID' : 'ESTIMATED'}</div>
          <div style="font-size:10px;color:#94A3B8;">Lead Dist: <strong>${s3d.lead_distance_m}m</strong> · Pothole Depth: <strong>${s3d.pothole_depth_cm}cm</strong></div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Normal Vector: <code style="font-size:10px;">${JSON.stringify(s3d.plane_normal || [])}</code></span>
        <span class="badge" style="background:#3B82F6;color:#fff;font-size:9px;">CALIBRATED BASELINE</span>
      </div>
    </div>

    <!-- 9. GIS / LOCATION -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">9. GIS &amp; LOCATION</strong>
        </div>
        <span class="badge" style="background:#F1F5F9;color:#334155;border:1px solid #E2E8F0;font-size:10px;font-weight:700;">
          OSM WAY #${gis.osm_way_id}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Road &amp; Segment</div>
          <div style="font-size:14px;font-weight:700;color:#0F172A;margin-top:2px;">${gis.osm_road_name}</div>
          <div style="font-size:10px;color:#64748B;">Segment: <strong>${gis.road_segment}</strong></div>
        </div>
        <div style="background:#F8FAFC;padding:8px 10px;border-radius:6px;border:1px solid #EDF2F7;">
          <div style="color:#64748B;font-size:10px;text-transform:uppercase;">Chainage Station</div>
          <div style="font-size:16px;font-weight:700;color:#D97706;margin-top:2px;">${gis.chainage_m} m</div>
          <div style="font-size:10px;color:#64748B;">${gis.chainage_progress_pct}% traversed of ${gis.total_length_m}m</div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>GPS: <strong>${gis.gps_latitude}°, ${gis.gps_longitude}°</strong> (Dev: <strong>${gis.centerline_deviation_m}m</strong>)</span>
        <span style="color:#059669;font-weight:600;">Bearing: ${gis.heading_deg}°</span>
      </div>
    </div>

    <!-- 10. MULTI-BUS EVIDENCE -->
    <div class="card" style="background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:16px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;border-bottom:1px solid #F1F5F9;padding-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg></span>
          <strong style="font-size:13px;color:#1E293B;">10. MULTI-BUS EVIDENCE</strong>
        </div>
        <span class="badge" style="background:${mb.evidence_state === 'CORROBORATED' ? '#10B981' : '#F59E0B'};color:#fff;font-size:10px;font-weight:700;">
          ${mb.evidence_state}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
        <div style="background:#F0FDF4;padding:8px 10px;border-radius:6px;border:1px solid #BBF7D0;">
          <div style="color:#166534;font-size:10px;text-transform:uppercase;">Corroborating Fleet</div>
          <div style="font-size:16px;font-weight:700;color:#15803D;margin-top:2px;">${mb.corroboration_votes} / ${mb.observing_buses_count} buses</div>
          <div style="font-size:10px;color:#166534;">Persistence: <strong>${mb.temporal_persistence_frames} frames</strong> (3 days)</div>
        </div>
        <div style="background:#F0FDF4;padding:8px 10px;border-radius:6px;border:1px solid #BBF7D0;">
          <div style="color:#166534;font-size:10px;text-transform:uppercase;">Fusion Confidence</div>
          <div style="font-size:16px;font-weight:700;color:#15803D;margin-top:2px;">${(mb.fusion_confidence * 100).toFixed(0)}%</div>
          <div style="font-size:10px;color:#166534;">Conflicts: <strong>${mb.conflicts_count} (${mb.retained_conflicting_bus})</strong></div>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#64748B;">
        <span>Multi-Pass Corroboration Engine: <strong style="color:#10B981;">Active</strong></span>
        <a href="#trichy-replay" style="color:#3B82F6;text-decoration:none;font-weight:600;">Inspect Fusion →</a>
      </div>
    </div>
  `;
}
