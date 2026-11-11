/**
 * BusSense — Road Infrastructure & Waterlogging Dashboard V2.4
 * ==========================================================
 * Metrology & Verification for:
 * - Waterlogging extent metrology (road UNION water ego-corridor)
 * - Traffic signs & condition (Normal, Damaged, Fallen, Occluded)
 * - Zebra crossings & paint wear metrology (Visible, Faded, Partial)
 * - Road dividers & physical median gaps (local road coordinate metrology)
 * - Road hazards & 3D geometric obstructions
 * - Evidence state machine & Multi-bus fusion status
 */
import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import {
  timeAgo,
  severityBadge,
  confidenceBar,
  formatNumber,
  formatTimestamp,
} from '../utils.js';

let mapInstance = null;
let markersLayer = null;
let allEvents = [];
let allAssets = [];
let currentFilter = 'ALL';
let pollTimer = null;

export async function render(container) {
  container.innerHTML = `
    <div>
      <!-- Header -->
      <div class="page-header flex justify-between items-center" style="margin-bottom:16px;">
        <div>
          <div style="display:flex;align-items:center;gap:8px;">
            <h1 class="page-title" style="margin:0;">Infrastructure &amp; Waterlogging Metrology</h1>
            <span class="badge" style="background:#0284C7;color:#fff;font-size:11px;font-weight:700;">METROLOGY ENGINE</span>
          </div>
          <p class="page-subtitle" style="margin-top:4px;">
            Calibrated Stereo Road-Plane · Multi-Bus Corroboration · Evidence State Machine
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button class="btn btn-secondary btn-sm" id="infra-refresh-btn" style="padding:6px 12px;cursor:pointer;">
            ↻ Refresh Data
          </button>
          <a href="#trichy-replay" class="btn btn-primary btn-sm" style="padding:6px 14px;text-decoration:none;display:inline-flex;align-items:center;gap:6px;">
            <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg></span> Run Trichy Replay
          </a>
        </div>
      </div>

      <!-- KPI Grid -->
      <div class="kpi-grid mb-4" id="infra-kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Waterlogging Incidents</div>
          <div class="kpi-value" id="kpi-water-count">—</div>
          <div class="kpi-sub" id="kpi-water-sub">Ego-corridor extent &gt; 15%</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Traffic Signs Monitored</div>
          <div class="kpi-value" id="kpi-signs-count">—</div>
          <div class="kpi-sub" id="kpi-signs-sub">Normal / Damaged / Fallen</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Zebra Crossings</div>
          <div class="kpi-value" id="kpi-zebra-count">—</div>
          <div class="kpi-sub" id="kpi-zebra-sub">Paint wear metrology</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Dividers &amp; Gaps</div>
          <div class="kpi-value" id="kpi-divider-count">—</div>
          <div class="kpi-sub" id="kpi-divider-sub">Physical median continuity</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Active Hazards</div>
          <div class="kpi-value" id="kpi-hazard-count">—</div>
          <div class="kpi-sub" id="kpi-hazard-sub">Debris &amp; Unknown 3D</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">🔄 Corroborated Claims</div>
          <div class="kpi-value" id="kpi-corroborated-count">—</div>
          <div class="kpi-sub" id="kpi-corroborated-sub">Multi-bus verified</div>
        </div>
      </div>

      <!-- Map & Filters Section -->
      <div class="card mb-4" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);">
        <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;border-bottom:1px solid rgba(241,245,249,0.8);">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:16px;">📍</span>
            <span class="card-title">Infrastructure Spatial Map &amp; Corroboration Layers</span>
          </div>
          <!-- Filter Tabs -->
          <div style="display:flex;gap:4px;background:var(--gray-100);padding:3px;border-radius:6px;" id="infra-filter-tabs">
            <button class="infra-tab active" data-filter="ALL">All</button>
            <button class="infra-tab" data-filter="WATERLOGGING">Waterlogging</button>
            <button class="infra-tab" data-filter="SIGNS">Signs</button>
            <button class="infra-tab" data-filter="ZEBRA">Zebra</button>
            <button class="infra-tab" data-filter="DIVIDERS">Dividers</button>
            <button class="infra-tab" data-filter="HAZARDS">Hazards</button>
            <button class="infra-tab" data-filter="SIMULATED">Simulated Only</button>
          </div>
        </div>
        <div class="card-body" style="padding:0;position:relative;">
          <div id="infra-map" style="height:380px;width:100%;border-radius:0 0 var(--radius-md) var(--radius-md);"></div>
          <!-- Legend Bar -->
          <div style="position:absolute;bottom:12px;left:12px;z-index:999;background:rgba(255,255,255,0.92);backdrop-filter:blur(4px);padding:6px 12px;border-radius:20px;box-shadow:0 2px 8px rgba(0,0,0,0.1);font-size:11px;display:flex;gap:12px;align-items:center;">
            <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:#0284C7;display:inline-block;"></span> Waterlogging</span>
            <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:#3B82F6;display:inline-block;"></span> Traffic Sign</span>
            <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:#0D9488;display:inline-block;"></span> Zebra Crossing</span>
            <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:#F97316;display:inline-block;"></span> Road Divider</span>
            <span style="display:flex;align-items:center;gap:4px;"><span style="width:10px;height:10px;border-radius:50%;background:#DC2626;display:inline-block;"></span> Road Hazard</span>
            <span style="display:flex;align-items:center;gap:4px;border-left:1px solid #ccc;padding-left:8px;"><span class="badge" style="background:#FFFBEB;color:#B45309;border:1px solid #FDE68A;font-size:9px;padding:1px 4px;">SIM</span> Synthetic</span>
          </div>
        </div>
      </div>

      <!-- Events & Candidate Asset Table -->
      <div class="card" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);">
        <div class="card-header" style="display:flex;justify-content:space-between;align-items:center;padding:14px 20px;border-bottom:1px solid rgba(241,245,249,0.8);">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:16px;">📋</span>
            <span class="card-title">Infrastructure Defect &amp; Observation Registry</span>
          </div>
          <span style="font-size:12px;color:var(--text-tertiary);" id="table-record-count">Loading…</span>
        </div>
        <div class="table-container">
          <table class="data-table" id="infra-events-table">
            <thead>
              <tr>
                <th>Event ID</th>
                <th>Type</th>
                <th>Corridor / Road</th>
                <th>Severity</th>
                <th>Metrology / Condition</th>
                <th>Evidence State</th>
                <th>Simulated / Real</th>
                <th>Timestamp</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody id="infra-table-body">
              <tr><td colspan="9" style="text-align:center;padding:32px;color:var(--text-tertiary);">Loading infrastructure metrology data…</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  // Style tabs inline
  const styleEl = document.createElement('style');
  styleEl.textContent = `
    .infra-tab {
      background: transparent;
      border: none;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 500;
      color: var(--text-secondary);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .infra-tab.active {
      background: #fff;
      color: var(--text-primary);
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
  `;
  document.head.appendChild(styleEl);

  // Initialize Leaflet Map
  initMap();

  // Load data
  await loadInfrastructureData();

  // Tab click listeners
  container.querySelectorAll('.infra-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      container.querySelectorAll('.infra-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      currentFilter = e.target.dataset.filter;
      applyFilter();
    });
  });

  // Refresh button
  container.querySelector('#infra-refresh-btn')?.addEventListener('click', loadInfrastructureData);
}

function initMap() {
  const mapEl = document.getElementById('infra-map');
  if (!mapEl) return;

  if (mapInstance) {
    try { mapInstance.remove(); } catch(e) {}
    mapInstance = null;
  }

  // Centered on Bharathidasan Salai, Tiruchirappalli
  mapInstance = L.map(mapEl, {
    center: [10.8055, 78.6850],
    zoom: 15,
    zoomControl: true,
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(mapInstance);

  markersLayer = L.layerGroup().addTo(mapInstance);
}

async function loadInfrastructureData() {
  try {
    const [eventsRes, assetsRes] = await Promise.all([
      api.infrastructureEvents({ limit: 100 }),
      api.infrastructureAssets({})
    ]);

    allEvents = Array.isArray(eventsRes) ? eventsRes : [];
    allAssets = Array.isArray(assetsRes) ? assetsRes : [];

    updateKPIs();
    applyFilter();
  } catch (err) {
    console.error('Failed to load infrastructure data:', err);
  }
}

function updateKPIs() {
  let waterCount = 0;
  let signsCount = 0;
  let zebraCount = 0;
  let dividerCount = 0;
  let hazardCount = 0;
  let corroboratedCount = 0;

  for (const ev of allEvents) {
    const t = ev.event_type || '';
    const st = ev.status || '';
    if (t.includes('WATERLOGGING')) waterCount++;
    if (t.includes('SIGN')) signsCount++;
    if (t.includes('ZEBRA')) zebraCount++;
    if (t.includes('DIVIDER')) dividerCount++;
    if (t.includes('HAZARD') || t.includes('OBSTRUCTION') || t.includes('BLOCKAGE')) hazardCount++;
    if (st === 'CORROBORATED' || st === 'VERIFIED') corroboratedCount++;
  }

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl('kpi-water-count', waterCount);
  setEl('kpi-signs-count', signsCount);
  setEl('kpi-zebra-count', zebraCount);
  setEl('kpi-divider-count', dividerCount);
  setEl('kpi-hazard-count', hazardCount);
  setEl('kpi-corroborated-count', corroboratedCount);
}

function applyFilter() {
  let filtered = allEvents;

  if (currentFilter === 'WATERLOGGING') {
    filtered = allEvents.filter(e => (e.event_type || '').includes('WATERLOGGING'));
  } else if (currentFilter === 'SIGNS') {
    filtered = allEvents.filter(e => (e.event_type || '').includes('SIGN'));
  } else if (currentFilter === 'ZEBRA') {
    filtered = allEvents.filter(e => (e.event_type || '').includes('ZEBRA'));
  } else if (currentFilter === 'DIVIDERS') {
    filtered = allEvents.filter(e => (e.event_type || '').includes('DIVIDER'));
  } else if (currentFilter === 'HAZARDS') {
    filtered = allEvents.filter(e => (e.event_type || '').includes('HAZARD') || (e.event_type || '').includes('OBSTRUCTION'));
  } else if (currentFilter === 'SIMULATED') {
    filtered = allEvents.filter(e => e.is_simulated || e.details?.simulation_mode);
  }

  renderTable(filtered);
  renderMapMarkers(filtered);
}

function renderTable(events) {
  const tbody = document.getElementById('infra-table-body');
  const countEl = document.getElementById('table-record-count');
  if (!tbody) return;

  if (countEl) countEl.textContent = `${events.length} records`;

  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:32px;color:var(--text-tertiary);">No infrastructure events match the selected filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = events.map(ev => {
    const isSim = ev.is_simulated || ev.details?.simulation_mode;
    const simBadge = isSim
      ? `<span class="badge" style="background:#FFFBEB;color:#B45309;border:1px solid #FDE68A;font-size:10px;font-weight:700;padding:2px 6px;">SIMULATED</span>`
      : `<span class="badge" style="background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;font-size:10px;font-weight:700;padding:2px 6px;">VERIFIED</span>`;

    // Metrology summary
    let metrology = '—';
    if (ev.event_type === 'WATERLOGGING' && ev.details?.water_coverage_ratio !== undefined) {
      metrology = `Water: <strong>${(ev.details.water_coverage_ratio * 100).toFixed(1)}%</strong> road coverage`;
    } else if (ev.event_type?.includes('SIGN')) {
      metrology = `Condition: <strong>${ev.details?.sign_condition || 'NORMAL'}</strong> (${ev.details?.sign_class || 'Sign'})`;
    } else if (ev.event_type?.includes('ZEBRA')) {
      metrology = `Paint: <strong>${ev.details?.paint_coverage_ratio ? (ev.details.paint_coverage_ratio * 100).toFixed(0) + '%' : '—'}</strong> (${ev.details?.condition || 'OBSERVED'})`;
    } else if (ev.event_type?.includes('DIVIDER')) {
      if (ev.details?.gap_length_m) {
        metrology = `Gap: <strong>${ev.details.gap_length_m.toFixed(1)}m</strong> (${ev.details.gap_start_m?.toFixed(0)}m - ${ev.details.gap_end_m?.toFixed(0)}m)`;
      } else {
        metrology = `Divider Status: ${ev.details?.condition || 'NORMAL'}`;
      }
    } else if (ev.details?.hazard_type) {
      metrology = `Hazard: <strong>${ev.details.hazard_type}</strong>`;
    }

    // Status state
    let statusStyle = 'background:var(--gray-100);color:var(--gray-600);';
    if (ev.status === 'CORROBORATED' || ev.status === 'VERIFIED') statusStyle = 'background:var(--green-100);color:var(--green-700);font-weight:600;';
    if (ev.status === 'CANDIDATE') statusStyle = 'background:var(--amber-100);color:var(--amber-800);';
    if (ev.status === 'CONFLICTING_EVIDENCE') statusStyle = 'background:var(--red-100);color:var(--red-700);font-weight:700;';

    const statusBadge = `<span class="badge-severity" style="${statusStyle}">${(ev.status || 'OBSERVED').replace(/_/g, ' ')}</span>`;

    return `
      <tr>
        <td style="font-family:monospace;font-size:12px;font-weight:600;color:var(--text-primary);">${ev.event_id || '—'}</td>
        <td><strong>${(ev.event_type || '—').replace(/_/g, ' ')}</strong></td>
        <td>${ev.corridor_id || 'Bharathidasan Salai'}</td>
        <td>${severityBadge(ev.severity || 'MEDIUM')}</td>
        <td style="font-size:12px;">${metrology}</td>
        <td>${statusBadge}</td>
        <td>${simBadge}</td>
        <td style="font-size:11px;color:var(--text-secondary);">${timeAgo(ev.created_at || ev.timestamp)}</td>
        <td>
          <button class="btn btn-secondary btn-sm btn-inspect-ev" data-id="${ev.event_id}" style="padding:4px 8px;font-size:11px;cursor:pointer;">
            Inspect
          </button>
        </td>
      </tr>
    `;
  }).join('');

  tbody.querySelectorAll('.btn-inspect-ev').forEach(btn => {
    btn.addEventListener('click', () => {
      const eid = btn.dataset.id;
      openEventEvidence(eid);
    });
  });
}

function renderMapMarkers(events) {
  if (!markersLayer || !mapInstance) return;
  markersLayer.clearLayers();

  const colorMap = {
    WATERLOGGING: '#0284C7',
    SIGN:         '#3B82F6',
    ZEBRA:        '#0D9488',
    DIVIDER:      '#F97316',
    HAZARD:       '#DC2626',
    OBSTRUCTION:  '#DC2626',
  };

  events.forEach(ev => {
    const lat = ev.latitude || ev.location?.latitude;
    const lon = ev.longitude || ev.location?.longitude;
    if (!lat || !lon) return;

    let markerColor = '#64748B';
    for (const [key, col] of Object.entries(colorMap)) {
      if ((ev.event_type || '').includes(key)) {
        markerColor = col;
        break;
      }
    }

    const isSim = ev.is_simulated || ev.details?.simulation_mode;

    const iconHtml = `
      <div style="background:${markerColor};width:14px;height:14px;border-radius:50%;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,0.3);position:relative;">
        ${isSim ? '<span style="position:absolute;top:-4px;right:-4px;width:6px;height:6px;background:#F59E0B;border-radius:50%;border:1px solid #fff;"></span>' : ''}
      </div>
    `;

    const customIcon = L.divIcon({
      html: iconHtml,
      className: 'infra-marker-icon',
      iconSize: [14, 14],
      iconAnchor: [7, 7]
    });

    const marker = L.marker([lat, lon], { icon: customIcon }).addTo(markersLayer);

    const popupHtml = `
      <div style="font-family:var(--font-family);min-width:180px;">
        <div style="font-weight:700;font-size:12px;margin-bottom:4px;color:var(--text-primary);">${(ev.event_type || '').replace(/_/g, ' ')}</div>
        <div style="font-size:11px;color:var(--text-secondary);margin-bottom:4px;">ID: <code>${ev.event_id}</code></div>
        <div style="font-size:11px;margin-bottom:6px;">Status: <strong>${ev.status || 'OBSERVED'}</strong></div>
        ${isSim ? '<span class="badge" style="background:#FFFBEB;color:#B45309;border:1px solid #FDE68A;font-size:9px;">SIMULATED</span>' : '<span class="badge" style="background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;font-size:9px;">VERIFIED</span>'}
        <button style="margin-top:8px;width:100%;background:var(--blue-500);color:#fff;border:none;border-radius:4px;padding:4px;font-size:11px;cursor:pointer;" onclick="window.openEventModal('${ev.event_id}')">
          View Evidence
        </button>
      </div>
    `;

    marker.bindPopup(popupHtml);
  });
}
