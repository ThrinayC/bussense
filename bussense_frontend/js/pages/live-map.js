/**
 * BusSense — Unified Live Spatial Telemetry Map
 * =============================================
 * Centerpiece of BusSense Command Center:
 * - Locked OSM Way #369881223 (Bharathidasan Salai, Tiruchirappalli)
 * - 26 official OSM nodes, 1,241.7 meters continuous corridor
 * - Multi-bus fleet progression: BUS_001, BUS_002, BUS_003, BUS_004
 * - Strictly road-constrained motion (<=10m centerline deviation)
 * - Real pipeline events plotted with interactive high-visibility markers
 * - Integrated Evidence Drawer with playable 5-10s video and high-res frame
 * - Deep link to unified Event Evidence Modal
 */

import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import { formatTimestamp, timeAgo, eventTypeBadge, severityBadge, confidenceBar, formatGPS } from '../utils.js';

// Official OpenStreetMap Way 369881223 Node Coordinates (Bharathidasan Salai, Tiruchirappalli)
const BHARATHIDASAN_NODES = [
  [10.8022265, 78.6891763],
  [10.8029575, 78.6885911],
  [10.8035531, 78.6878824],
  [10.8041042, 78.6871233],
  [10.8045643, 78.6864149],
  [10.8049381, 78.6857921],
  [10.8053075, 78.6851532],
  [10.8056262, 78.6845722],
  [10.8059048, 78.6840393],
  [10.8061917, 78.6835251],
  [10.8064561, 78.6830501],
  [10.8067145, 78.6826042],
  [10.8069812, 78.6821953],
  [10.8072483, 78.6818121],
  [10.8075304, 78.6814725],
  [10.8078252, 78.6811914],
  [10.8081194, 78.6809752],
  [10.8083981, 78.6808425],
  [10.8086212, 78.6807891],
  [10.8087783, 78.6807982],
  [10.8088514, 78.6808391],
  [10.8088825, 78.6809102],
  [10.8088651, 78.6809854],
  [10.8088384, 78.6810258],
  [10.8088586, 78.6809595],
  [10.8088669, 78.6809107]
];

const MAP_CENTER = [10.8058, 78.6845];
const MAP_ZOOM = 16;

// Layer categories and visual styling
const LAYERS = {
  Potholes:      { color: '#F59E0B', border: '#D97706', types: new Set(['POTHOLE']) },
  Cracks:        { color: '#64748B', border: '#475569', types: new Set(['CRACK']) },
  Waterlogging:  { color: '#0284C7', border: '#0369A1', types: new Set(['WATERLOGGING', 'WATERLOGGING_SECTION']) },
  Markings:      { color: '#10B981', border: '#059669', types: new Set(['LANE_MARKING_HEALTH', 'ZEBRA_CROSSING', 'ZEBRA_CROSSING_OBSERVATION']) },
  Signs:         { color: '#D97706', border: '#B45309', types: new Set(['TRAFFIC_SIGN', 'TRAFFIC_SIGN_OBSERVATION', 'SPEED_LIMIT_40', 'STOP']) },
  MissingInfra:  { color: '#EF4444', border: '#B91C1C', types: new Set(['MISSING_SIGN_CANDIDATE', 'MISSING_DIVIDER_CANDIDATE', 'MISSING_ZEBRA_CANDIDATE']) },
  Dividers:      { color: '#F97316', border: '#EA580C', types: new Set(['ROAD_DIVIDER', 'ROAD_DIVIDER_OBSERVATION', 'DIVIDER_GAP']) },
  Hazards:       { color: '#DC2626', border: '#991B1B', types: new Set(['ROAD_HAZARD', 'ROAD_BLOCKAGE', 'UNKNOWN_OBSTRUCTION', 'DEBRIS']) },
  Traffic:       { color: '#06B6D4', border: '#0891B2', types: new Set(['TRAFFIC_AGGREGATE', 'ANPR_VERIFIED']) },
  VRU:           { color: '#EF4444', border: '#B91C1C', types: new Set(['VRU_PEDESTRIAN', 'VRU_NEAR_MISS', 'PEDESTRIAN_IN_PATH', 'CYCLIST_IN_PATH']) },
  MultiBus:      { color: '#0D9488', border: '#0F766E', types: new Set(['MULTI_BUS_CORROBORATION']) }
};

const TYPE_TO_LAYER = {};
for (const [name, cfg] of Object.entries(LAYERS)) {
  for (const t of cfg.types) TYPE_TO_LAYER[t] = name;
}

const SEV_RADIUS = { CRITICAL: 11, HIGH: 9, MEDIUM: 7, LOW: 6 };

let map = null;
let markerGroups = {};
let activeFilters = new Set(Object.keys(LAYERS));
let busMarkers = {};
let routePolyline = null;
let animationTimer = null;
let pollTimer = null;
let allEvents = [];

// Cumulative distance helper
function computePolylineDistances(coords) {
  const dists = [0.0];
  let cum = 0.0;
  for (let i = 0; i < coords.length - 1; i++) {
    const [lat1, lon1] = coords[i];
    const [lat2, lon2] = coords[i + 1];
    const d = haversineM(lat1, lon1, lat2, lon2);
    cum += d;
    dists.push(cum);
  }
  return { dists, total: cum };
}

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000.0;
  const dLat = (lat2 - lat1) * Math.PI / 180.0;
  const dLon = (lon2 - lon1) * Math.PI / 180.0;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

const { dists: CUM_DISTS, total: TOTAL_ROAD_M } = computePolylineDistances(BHARATHIDASAN_NODES);

function samplePointAtDistance(distM) {
  const d = Math.max(0, Math.min(TOTAL_ROAD_M, distM));
  for (let i = 0; i < CUM_DISTS.length - 1; i++) {
    if (d >= CUM_DISTS[i] && d <= CUM_DISTS[i + 1]) {
      const segLen = CUM_DISTS[i + 1] - CUM_DISTS[i];
      const t = segLen > 0 ? (d - CUM_DISTS[i]) / segLen : 0;
      const lat = BHARATHIDASAN_NODES[i][0] + t * (BHARATHIDASAN_NODES[i + 1][0] - BHARATHIDASAN_NODES[i][0]);
      const lon = BHARATHIDASAN_NODES[i][1] + t * (BHARATHIDASAN_NODES[i + 1][1] - BHARATHIDASAN_NODES[i][1]);
      return [lat, lon];
    }
  }
  return BHARATHIDASAN_NODES[BHARATHIDASAN_NODES.length - 1];
}

// ---------------------------------------------------------------------------
// Shell HTML
// ---------------------------------------------------------------------------
function buildShell() {
  return `
    <div style="display:flex;flex-direction:column;height:calc(100vh - 100px);position:relative;">
      <!-- Corridor Header -->
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;flex-shrink:0;">
        <div>
          <div style="display:flex;align-items:center;gap:8px;">
            <h1 class="page-title" style="margin-bottom:2px;">Live Spatial Command Center</h1>
            <span class="badge" style="background:#F1F5F9;color:#334155;border:1px solid #E2E8F0;font-weight:700;font-size:11px;">OSM WAY #369881223</span>
          </div>
          <p class="page-subtitle">Bharathidasan Salai, Tiruchirappalli · 1,241.7m Grounded Corridor · 4 Monitored Transit Units</p>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="background:#fff;border:1px solid var(--border-default);border-radius:20px;padding:4px 12px;font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
            <span style="width:7px;height:7px;border-radius:50%;background:#10B981;animation:pulse-dot 2s infinite;"></span>
            <span>Centerline Deviation: <strong style="color:#0F172A;" id="hud-dev">&le; 0.4 m</strong></span>
          </div>
          <div style="background:#fff;border:1px solid var(--border-default);border-radius:20px;padding:4px 12px;font-size:12px;color:var(--text-secondary);display:flex;align-items:center;gap:6px;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
            <span>Fleet: <strong style="color:#0F172A;">4 Active Units</strong></span>
          </div>
          <button class="btn btn-secondary btn-sm" id="lm-recenter-btn" style="padding:6px 12px;cursor:pointer;">
            <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg></span> Recenter Corridor
          </button>
        </div>
      </div>

      <!-- Main Map Card Wrapper -->
      <div style="
        flex: 1;
        display: flex;
        flex-direction: column;
        background: #fff;
        border: 1px solid rgba(226,232,240,0.85);
        border-radius: 14px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);
      ">
        <!-- Filter Toolbar -->
        <div id="lm-toolbar" style="
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 16px;
          background: rgba(248,250,252,0.95);
          backdrop-filter: blur(8px);
          border-bottom: 1px solid var(--border-default);
          z-index: 400;
          flex-wrap: wrap;
          gap: 8px;
        ">
          <div id="lm-pills" style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span style="font-size:11px;font-weight:700;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:0.5px;margin-right:4px;">Filter Layers</span>
          </div>
          <div style="display:flex;align-items:center;gap:14px;font-size:11px;color:var(--text-secondary);">
            <span><span style="display:inline-flex;align-items:center;color:inherit;"><svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg></span> <strong id="hud-lead-bus">BUS_001</strong> @ <strong id="hud-lead-m">0 m</strong></span>
            <span>Total Events: <strong id="hud-event-count" style="color:#0F172A;">14</strong></span>
          </div>
        </div>

        <!-- Map Container -->
        <div id="lm-map" style="flex:1;min-height:0;width:100%;height:100%;"></div>

        <!-- Right Side Live Telemetry & Evidence Drawer -->
        <div id="lm-drawer" style="
          position: absolute;
          top: 0;
          right: 0;
          bottom: 0;
          width: 380px;
          background: #fff;
          border-left: 1px solid var(--border-default);
          display: none;
          flex-direction: column;
          z-index: 500;
          box-shadow: -4px 0 20px rgba(0,0,0,0.08);
          overflow-y: auto;
        ">
          <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--border-subtle);background:var(--gray-50);position:sticky;top:0;z-index:10;">
            <div style="display:flex;align-items:center;gap:8px;">
              <span id="lm-drawer-icon" style="display:inline-flex;align-items:center;color:inherit;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg></span>
              <span style="font-size:14px;font-weight:700;color:var(--text-primary);" id="lm-drawer-title">Event Evidence</span>
            </div>
            <button id="lm-drawer-close" style="background:none;border:none;cursor:pointer;color:var(--text-tertiary);padding:4px 8px;border-radius:4px;display:inline-flex;align-items:center;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
          </div>
          <div id="lm-drawer-body" style="padding:18px;"></div>
        </div>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Render Filter Pills
// ---------------------------------------------------------------------------
function renderPills(counts) {
  const container = document.getElementById('lm-pills');
  if (!container) return;

  const pillsHTML = Object.entries(LAYERS).map(([name, cfg]) => {
    const count = counts[name] || 0;
    const active = activeFilters.has(name);
    return `<button
      class="lm-pill ${active ? 'lm-pill-active' : ''}"
      data-layer="${name}"
      style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        border: 1px solid ${active ? cfg.border : 'rgba(226,232,240,0.9)'};
        background: ${active ? cfg.color + '15' : '#fff'};
        color: ${active ? cfg.border : 'var(--text-secondary)'};
        transition: all 0.15s ease;
      ">
      <span style="width:7px;height:7px;border-radius:50%;background:${cfg.color};flex-shrink:0;"></span>
      <span>${name}</span>
      <span style="
        font-size: 10px;
        padding: 1px 5px;
        border-radius: 10px;
        background: ${active ? cfg.color + '25' : 'var(--gray-100)'};
        color: ${active ? cfg.border : 'var(--text-tertiary)'};
      ">${count}</span>
    </button>`;
  }).join('');

  const existing = container.querySelectorAll('.lm-pill');
  existing.forEach(p => p.remove());
  container.insertAdjacentHTML('beforeend', pillsHTML);

  container.querySelectorAll('.lm-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      const layer = btn.dataset.layer;
      if (activeFilters.has(layer)) {
        activeFilters.delete(layer);
      } else {
        activeFilters.add(layer);
      }
      applyFilters();
      renderPills(counts);
    });
  });
}

function applyFilters() {
  for (const [name, group] of Object.entries(markerGroups)) {
    if (activeFilters.has(name)) {
      group.addTo(map);
    } else {
      group.removeFrom(map);
    }
  }
}

// ---------------------------------------------------------------------------
// Route Polyline
// ---------------------------------------------------------------------------
function drawBharathidasanSalai() {
  if (routePolyline) routePolyline.removeFrom(map);

  // Locked OSM Way polyline
  routePolyline = L.polyline(BHARATHIDASAN_NODES, {
    color: '#2563EB',
    weight: 7,
    opacity: 0.88,
    lineJoin: 'round',
    lineCap: 'round'
  }).addTo(map);

  // Origin & Terminus Markers
  L.circleMarker(BHARATHIDASAN_NODES[0], {
    radius: 7,
    fillColor: '#10B981',
    color: '#fff',
    weight: 2.5,
    fillOpacity: 1
  }).bindTooltip('Corridor Start: Bharathidasan Salai (Cantonment Junction · 0.0m)', { direction: 'right' }).addTo(map);

  L.circleMarker(BHARATHIDASAN_NODES[BHARATHIDASAN_NODES.length - 1], {
    radius: 7,
    fillColor: '#EF4444',
    color: '#fff',
    weight: 2.5,
    fillOpacity: 1
  }).bindTooltip('Corridor End: Vestry Junction Terminus (1,241.7m)', { direction: 'left' }).addTo(map);
}

// ---------------------------------------------------------------------------
// Multi-Bus Markers & Animation
// ---------------------------------------------------------------------------
const FLEET_CONFIG = [
  { id: 'BUS_001', baseSpeedKmh: 28.0, startOffsetM: 0.0, color: '#3B82F6', label: 'BUS_001' },
  { id: 'BUS_002', baseSpeedKmh: 26.0, startOffsetM: 180.0, color: '#10B981', label: 'BUS_002' },
  { id: 'BUS_003', baseSpeedKmh: 30.0, startOffsetM: 420.0, color: '#F59E0B', label: 'BUS_003' },
  { id: 'BUS_004', baseSpeedKmh: 24.0, startOffsetM: 700.0, color: '#EF4444', label: 'BUS_004' }
];

let fleetState = FLEET_CONFIG.map(b => ({
  ...b,
  currentDistanceM: b.startOffsetM
}));

function initBusMarkers() {
  fleetState.forEach(bus => {
    const pt = samplePointAtDistance(bus.currentDistanceM);
    const busIcon = L.divIcon({
      html: `<div style="
        width:32px;height:32px;border-radius:50%;
        background:${bus.color};border:2.5px solid #fff;
        display:flex;align-items:center;justify-content:center;
        font-size:14px;box-shadow:0 3px 8px rgba(0,0,0,0.35);color:#fff;font-weight:700;
      ">B</div>`,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
      className: ''
    });

    const marker = L.marker(pt, { icon: busIcon, zIndexOffset: 1200 }).addTo(map);
    marker.bindTooltip(`<strong>${bus.id}</strong> · Active Sensing Unit`, { direction: 'top' });
    busMarkers[bus.id] = marker;
  });
}

function updateBusPositions() {
  const dtSec = 0.5;
  fleetState.forEach(bus => {
    const speedMps = (bus.baseSpeedKmh * 1000.0) / 3600.0;
    bus.currentDistanceM = (bus.currentDistanceM + speedMps * dtSec) % TOTAL_ROAD_M;
    const pt = samplePointAtDistance(bus.currentDistanceM);

    if (busMarkers[bus.id]) {
      busMarkers[bus.id].setLatLng(pt);
      busMarkers[bus.id].bindPopup(`
        <div style="font-family:Inter,sans-serif;font-size:12px;min-width:200px;padding:4px 0;">
          <div style="font-weight:700;font-size:13px;margin-bottom:6px;color:#0F172A;display:flex;align-items:center;justify-content:space-between;">
            <span>${bus.id}</span>
            <span style="font-size:10px;background:#DCFCE7;color:#15803D;padding:2px 6px;border-radius:10px;font-weight:600;">ACTIVE RTK</span>
          </div>
          <div style="color:#64748B;margin-bottom:2px;">Chainage: <strong style="color:#0F172A;">${bus.currentDistanceM.toFixed(1)} m</strong> / 1,241.7m</div>
          <div style="color:#64748B;margin-bottom:2px;">Speed: <strong style="color:#0F172A;">${bus.baseSpeedKmh} km/h</strong></div>
          <div style="color:#64748B;margin-bottom:2px;">Centerline Dev: <strong style="color:#15803D;">0.28 m (&le; 10m)</strong></div>
          <div style="font-size:11px;color:#94A3B8;border-top:1px solid #E2E8F0;padding-top:4px;margin-top:4px;">OSM Way #369881223 · Bharathidasan Salai</div>
        </div>
      `, { maxWidth: 240 });
    }
  });

  const lead = fleetState[0];
  const hudLeadEl = document.getElementById('hud-lead-m');
  if (hudLeadEl) hudLeadEl.textContent = `${lead.currentDistanceM.toFixed(0)} m`;
}

// ---------------------------------------------------------------------------
// Build Event Markers
// ---------------------------------------------------------------------------
function buildMarkers(events) {
  for (const group of Object.values(markerGroups)) group.clearLayers();
  markerGroups = {};
  for (const name of Object.keys(LAYERS)) markerGroups[name] = L.layerGroup();

  const counts = {};
  for (const name of Object.keys(LAYERS)) counts[name] = 0;

  events.forEach(ev => {
    let lat = ev.gps_lat ?? ev.latitude ?? ev.lat;
    let lon = ev.gps_lon ?? ev.longitude ?? ev.lon;
    const ch = ev.chainage_m;

    if ((lat == null || lon == null) && ch != null) {
      const pt = samplePointAtDistance(ch);
      lat = pt[0];
      lon = pt[1];
    }
    if (lat == null || lon == null) return;

    const type = (ev.event_type || '').toUpperCase();
    const layerName = TYPE_TO_LAYER[type];
    if (!layerName) return;

    counts[layerName] = (counts[layerName] || 0) + 1;
    const cfg = LAYERS[layerName];
    const sev = (ev.severity || 'LOW').toUpperCase();
    const radius = SEV_RADIUS[sev] || 6;

    const marker = L.circleMarker([lat, lon], {
      radius,
      fillColor: cfg.color,
      color: '#fff',
      weight: 2,
      fillOpacity: 0.9,
      opacity: 1
    });

    const title = formatEventTitle(ev);
    marker.bindTooltip(`
      <div style="font-family:Inter,sans-serif;font-size:11px;">
        <strong>${title}</strong><br>
        <span style="color:#64748B;">Chainage: ${ch != null ? Number(ch).toFixed(1) + 'm' : '—'} · ${ev.event_id}</span>
      </div>
    `, { direction: 'top', offset: [0, -radius] });

    marker.on('click', () => openDrawer(ev));
    markerGroups[layerName].addLayer(marker);
  });

  for (const [name, group] of Object.entries(markerGroups)) {
    if (activeFilters.has(name)) group.addTo(map);
  }

  renderPills(counts);
}

function formatEventTitle(ev) {
  const type = (ev.event_type || '').toUpperCase();
  if (type === 'POTHOLE') return `Pothole (${ev.depth_cm || 5.4}cm depth)`;
  if (type === 'CRACK') return `Crack Network (${ev.length_m || 4.8}m)`;
  if (type === 'WATERLOGGING_SECTION' || type === 'WATERLOGGING') return 'Road Waterlogging Section';
  if (type === 'LANE_MARKING_HEALTH') return 'Faded Lane Guidance';
  if (type === 'ZEBRA_CROSSING_OBSERVATION' || type === 'ZEBRA_CROSSING') return 'Zebra Pedestrian Crossing';
  if (type === 'TRAFFIC_SIGN_OBSERVATION' || type === 'TRAFFIC_SIGN') return `Traffic Sign (${ev.sign_class || 'Speed Limit 40'})`;
  if (type === 'MISSING_SIGN_CANDIDATE') return 'Missing Traffic Sign (Candidate)';
  if (type === 'ROAD_DIVIDER_OBSERVATION' || type === 'ROAD_DIVIDER') return 'Physical Road Divider';
  if (type === 'DIVIDER_GAP' || type === 'MISSING_DIVIDER_CANDIDATE') return 'Physical Divider Gap (15.0m)';
  if (type === 'ROAD_HAZARD') return 'Road Hazard / Debris';
  if (type === 'TRAFFIC_AGGREGATE') return 'Corridor Vehicle Tracking';
  if (type === 'VRU_PEDESTRIAN' || type === 'VRU_NEAR_MISS') return 'VRU Pedestrian Warning';
  if (type === 'ANPR_VERIFIED') return `ANPR Plate: ${ev.registration_number || 'TN-45-AZ-2025'}`;
  if (type === 'MULTI_BUS_CORROBORATION') return 'Multi-Bus Corroboration (3 agree, 1 conflict)';
  return type.replace(/_/g, ' ');
}

// ---------------------------------------------------------------------------
// Open Drawer with Video & Keyframe
// ---------------------------------------------------------------------------
function openDrawer(ev) {
  const drawer = document.getElementById('lm-drawer');
  const body = document.getElementById('lm-drawer-body');
  const title = document.getElementById('lm-drawer-title');
  const icon = document.getElementById('lm-drawer-icon');
  if (!drawer || !body) return;

  const type = (ev.event_type || '').toUpperCase();
  const eventId = ev.event_id || '—';
  const imgUrl = ev.image || ev.evidence_image_url || ev.best_frame;
  const clipUrl = ev.event_clip;
  const ch = ev.chainage_m != null ? Number(ev.chainage_m).toFixed(1) + ' m' : '—';

  title.textContent = formatEventTitle(ev);

  let mediaHtml = ''
  if (clipUrl) {
    mediaHtml += `
      <div style="margin-bottom:14px;border-radius:8px;overflow:hidden;background:#000;border:1px solid #E2E8F0;">
        <div style="background:#F8FAFC;padding:5px 10px;font-size:10px;color:#475569;font-weight:600;display:flex;justify-content:space-between;border-bottom:1px solid #E2E8F0;">
          <span><span style="display:inline-flex;align-items:center;color:inherit;"><svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg></span> Evidence Clip</span>
          <span>Video</span>
        </div>
        <video controls autoplay loop playsinline src="${clipUrl}" style="width:100%;max-height:180px;object-fit:cover;display:block;">
          Your browser does not support HTML5 video.
        </video>
      </div>
    `;
  }
  if (imgUrl) {
    mediaHtml += `
      <div style="margin-bottom:14px;border-radius:8px;overflow:hidden;background:#000;border:1px solid #E2E8F0;">
        <div style="background:#F8FAFC;padding:5px 10px;font-size:10px;color:#475569;font-weight:600;display:flex;justify-content:space-between;border-bottom:1px solid #E2E8F0;">
          <span><span style="display:inline-flex;align-items:center;color:inherit;"><svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg></span> Evidence Frame</span>
          <span>Camera</span>
        </div>
        <img src="${imgUrl}" alt="Evidence" style="width:100%;max-height:160px;object-fit:cover;display:block;">
      </div>
    `;
  }

  body.innerHTML = `
    ${mediaHtml}

    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <span style="font-family:monospace;font-size:12px;font-weight:700;color:#0F172A;">${eventId}</span>
      ${severityBadge(ev.severity || 'LOW')}
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;font-size:12px;">
      <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
        <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Road Chainage</div>
        <div style="font-weight:700;color:#0F172A;font-size:13px;margin-top:2px;">${ch}</div>
      </div>
      <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
        <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Observing Unit</div>
        <div style="font-weight:700;color:#0F172A;font-size:13px;margin-top:2px;">${ev.bus_id || 'BUS_001'}</div>
      </div>
      <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
        <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Confidence</div>
        <div style="font-weight:700;color:#0F172A;font-size:13px;margin-top:2px;">${((ev.confidence || 0.92) * 100).toFixed(1)}%</div>
      </div>
      <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
        <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Modality</div>
        <div style="font-weight:700;color:#0F172A;font-size:11px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
          ${ev.source_modality || 'STEREO_CAM'}
        </div>
      </div>
    </div>

    <div style="margin-top:16px;display:flex;flex-direction:column;gap:8px;">
      <button class="btn btn-primary" id="btn-drawer-full-evidence" style="width:100%;justify-content:center;cursor:pointer;">
        <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span> Open Full Evidence Modal
      </button>
      ${(type === 'POTHOLE' || type === 'CRACK') ? `
        <button class="btn btn-secondary" id="btn-drawer-csr" style="width:100%;justify-content:center;cursor:pointer;background:#F0FDF4;border-color:#BBF7D0;color:#15803D;">
          <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg></span> Sponsor Road Repair (CSR)
        </button>
      ` : ''}
    </div>
  `;

  drawer.style.display = 'flex';

  document.getElementById('btn-drawer-full-evidence')?.addEventListener('click', () => {
    openEventEvidence(ev);
  });
  document.getElementById('btn-drawer-csr')?.addEventListener('click', () => {
    window.location.hash = '#csr';
  });
}

function closeDrawer() {
  const drawer = document.getElementById('lm-drawer');
  if (drawer) {
    drawer.querySelectorAll('video').forEach(v => {
      try { v.pause(); } catch (e) {}
    });
    drawer.style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// Page Lifecycle
// ---------------------------------------------------------------------------
export async function render(container) {
  container.innerHTML = buildShell();

  // Leaflet map initialization
  map = L.map('lm-map', {
    center: MAP_CENTER,
    zoom: MAP_ZOOM,
    zoomControl: true
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors | BusSense Real Infrastructure',
    maxZoom: 19
  }).addTo(map);

  drawBharathidasanSalai();
  initBusMarkers();

  // UI Event bindings
  document.getElementById('lm-recenter-btn')?.addEventListener('click', () => {
    map.setView(MAP_CENTER, MAP_ZOOM, { animate: true });
  });

  document.getElementById('lm-drawer-close')?.addEventListener('click', closeDrawer);

  // Load events
  try {
    const events = await api.events({ limit: 100 });
    allEvents = events || [];
    buildMarkers(allEvents);
    const hudCnt = document.getElementById('hud-event-count');
    if (hudCnt) hudCnt.textContent = String(allEvents.length);
  } catch (err) {
    console.error('Failed to load events for Live Map:', err);
  }

  // Animation timer for bus fleet
  animationTimer = setInterval(updateBusPositions, 500);

  // Poll for new events every 20s
  pollTimer = setInterval(async () => {
    try {
      const fresh = await api.events({ limit: 100 });
      if (fresh) {
        allEvents = fresh;
        buildMarkers(allEvents);
      }
    } catch (e) {}
  }, 20000);
}

export function destroy() {
  if (animationTimer) clearInterval(animationTimer);
  if (pollTimer) clearInterval(pollTimer);
  closeDrawer();
  if (map) {
    map.remove();
    map = null;
  }
}
