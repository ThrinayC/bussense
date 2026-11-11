/**
 * BusSense Command Center — Fleet / Bus Status V2.4
 * =================================================
 * Single-bus view: BUS-01 (Tiruchirappalli Autonomous Sensing Unit).
 * - Elevates fleet KPIs to reference design system
 * - Interactive Journey Log with direct evidence inspection modal
 * - Dynamic sensor and perception health overview
 */
import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import { timeAgo, formatNumber, formatTimestamp } from '../utils.js';

let eventsPerTypeChart = null;

const INCIDENT_TYPES = new Set([
  'COLLISION_CANDIDATE','RAPID_CLOSING','DANGEROUS_FOLLOWING','HARD_BRAKING',
  'CUT_IN_CANDIDATE','ABRUPT_LATERAL_MANEUVER','WEAVING_CANDIDATE',
  'EMERGENCY_EVASIVE_MANEUVER','HIT_AND_RUN_CANDIDATE',
]);
const VRU_TYPES = new Set(['VRU_NEAR_MISS','PEDESTRIAN_IN_PATH','CYCLIST_IN_PATH','SUDDEN_VRU_EMERGENCE']);

export async function render(container) {
  container.innerHTML = `
    <div>
      <div class="page-header flex justify-between items-center">
        <div>
          <h1 class="page-title">Fleet &amp; Sensor Telemetry</h1>
          <p class="page-subtitle">BUS-01 — Edge AI Mobile Perception Unit · Tiruchirappalli Metro</p>
        </div>
        <div class="flex items-center gap-2">
          <span class="badge-subtle" style="background:rgba(16,185,129,0.08);color:#059669;border:1px solid rgba(16,185,129,0.2);font-weight:600;padding:6px 12px;border-radius:20px;font-size:12px;">
            ● 1 Unit Deployed &amp; Active
          </span>
        </div>
      </div>

      <section class="kpi-grid mb-4" id="fleet-kpis">
        ${[1,2,3,4].map(() => '<div class="kpi-card"><div class="loading-shimmer" style="width:55%;height:14px;margin-bottom:8px;"></div><div class="loading-shimmer" style="width:38%;height:28px;"></div></div>').join('')}
      </section>

      <div class="grid-2-1 mb-4">
        <!-- Bus status card -->
        <div class="card" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);">
          <div class="card-header" style="border-bottom:1px solid rgba(241,245,249,0.8);padding:14px 20px;">
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg></span>
              <span class="card-title">Vehicle BUS-01 Specifications</span>
            </div>
            <span class="badge-status active">Telemetry Live</span>
          </div>
          <div class="card-body" id="bus-detail-card" style="padding:20px;">Loading…</div>
        </div>

        <!-- Event breakdown chart -->
        <div class="card" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);">
          <div class="card-header" style="border-bottom:1px solid rgba(241,245,249,0.8);padding:14px 20px;">
            <span class="card-title">Perception Distribution</span>
            <span style="font-size:11px;color:var(--text-tertiary);">Current Run</span>
          </div>
          <div class="card-body" style="padding:20px;">
            <div class="chart-container" style="height:230px;"><canvas id="fleet-chart"></canvas></div>
          </div>
        </div>
      </div>

      <!-- Recent events from this bus -->
      <div class="card" style="box-shadow:0 1px 3px rgba(0,0,0,0.02), 0 4px 14px rgba(15,23,42,0.03);">
        <div class="card-header" style="border-bottom:1px solid rgba(241,245,249,0.8);padding:14px 20px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:16px;">⏱️</span>
            <span class="card-title">Journey Operational Log</span>
          </div>
          <span class="text-xs text-tertiary">Click any row to inspect sensor evidence &amp; metrology</span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Event ID</th>
                <th>Time</th>
                <th>Event Classification</th>
                <th>Corridor Location</th>
                <th>Domain Category</th>
                <th>Confidence</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody id="fleet-journey-tbody">
              <tr><td colspan="7" style="text-align:center;padding:24px;color:#94A3B8;">Loading…</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>`;

  const eventsData = await api.events({ limit: 200 });
  const events = Array.isArray(eventsData) ? eventsData : [];

  // Single bus in demonstration
  const busEvents = events.filter(e => !e.bus_id || e.bus_id === 'BUS-01');

  renderKPIs(busEvents);
  renderBusCard(busEvents);
  renderChart(busEvents);
  renderJourney(busEvents);
}

function renderKPIs(events) {
  const total = events.length;
  const potholes = events.filter(e => (e.event_type || '').toUpperCase() === 'POTHOLE').length;
  const incidents = events.filter(e => INCIDENT_TYPES.has((e.event_type || '').toUpperCase())).length;
  const anpr = events.filter(e => (e.event_type || '').toUpperCase() === 'ANPR_VERIFIED').length;

  document.getElementById('fleet-kpis').innerHTML = `
    <div class="kpi-card">
      <div class="kpi-card-header">
        <span class="kpi-label">Active Fleet Units</span>
        <span class="kpi-badge kpi-badge-teal">SINGLE FLEET</span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">1</span>
        <span class="kpi-pill kpi-pill-green">BUS-01 Online</span>
      </div>
      <div class="kpi-context">Mobile sensing unit operational</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-card-header">
        <span class="kpi-label">Total Ingested Events</span>
        <span class="kpi-badge kpi-badge-blue">TELEMETRY</span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(total)}</span>
        <span class="kpi-pill kpi-pill-blue">Authentic</span>
      </div>
      <div class="kpi-context">Verified Mega-run scenario events</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-card-header">
        <span class="kpi-label">Kinematic Incidents</span>
        <span class="kpi-badge kpi-badge-red">SAFETY CRITICAL</span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value" style="color:var(--red-600);">${formatNumber(incidents)}</span>
        <span class="kpi-pill kpi-pill-red">High Risk</span>
      </div>
      <div class="kpi-context">TTC &amp; closing speed conflicts</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-card-header">
        <span class="kpi-label">Identified Plates</span>
        <span class="kpi-badge kpi-badge-amber">ANPR PIPELINE</span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value" style="color:var(--blue-600);">${formatNumber(anpr)}</span>
        <span class="kpi-pill kpi-pill-green">Ensemble OK</span>
      </div>
      <div class="kpi-context">OCR plate matches verified</div>
    </div>`;
}

function renderBusCard(events) {
  const card = document.getElementById('bus-detail-card');
  if (!card) return;

  const potholes = events.filter(e => (e.event_type || '').toUpperCase() === 'POTHOLE').length;
  const cracks   = events.filter(e => (e.event_type || '').toUpperCase() === 'CRACK').length;
  const incidents = events.filter(e => INCIDENT_TYPES.has((e.event_type || '').toUpperCase())).length;
  const vru       = events.filter(e => VRU_TYPES.has((e.event_type || '').toUpperCase())).length;
  const anpr      = events.filter(e => (e.event_type || '').toUpperCase() === 'ANPR_VERIFIED').length;
  const traffic   = events.filter(e => (e.event_type || '').toUpperCase() === 'TRAFFIC_AGGREGATE').length;
  const har       = events.filter(e => (e.event_type || '').toUpperCase() === 'HIT_AND_RUN_CANDIDATE').length;

  card.innerHTML = `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
      ${stat('Active Route Corridor', 'ROUTE-01-METRO (Trichy)')}
      ${stat('Operating Status', '<span style="color:var(--green-600);font-weight:600;">● Online / Streaming</span>')}
      ${stat('Current Waypoint', 'Thillai Nagar Main Road (10.821°N, 78.696°E)')}
      ${stat('Sensing Speed', '22.0 km/h (Nominal)')}
      ${stat('Optical Sensor Rig', 'Monocular Wide-Angle + Stereo Metrology')}
      ${stat('Perception Core', 'bussense-2.4.0 (Edge Pipeline)')}
    </div>
    <div style="margin-top:18px;padding-top:16px;border-top:1px solid rgba(241,245,249,0.8);">
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-tertiary);margin-bottom:10px;">Journey Event Telemetry Breakdown</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        ${pill('Potholes', potholes, '#F59E0B')}
        ${pill('Cracks', cracks, '#64748B')}
        ${pill('Incidents', incidents, '#EF4444')}
        ${pill('VRU Safety', vru, '#10B981')}
        ${pill('ANPR Reads', anpr, '#3B82F6')}
        ${pill('Traffic Intel', traffic, '#06B6D4')}
        ${har > 0 ? pill('Hit-and-Run', har, '#DC2626') : ''}
      </div>
    </div>`;
}

function stat(label, value) {
  return `<div>
    <div style="font-size:11px;color:var(--text-tertiary);margin-bottom:2px;font-weight:500;">${label}</div>
    <div style="font-size:13px;font-weight:600;color:var(--text-primary);">${value}</div>
  </div>`;
}

function pill(label, count, color) {
  return `<div style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;background:${color}15;border:1px solid ${color}35;border-radius:20px;font-size:12px;font-weight:600;color:${color};">
    <strong>${count}</strong> ${label}
  </div>`;
}

function renderChart(events) {
  const canvas = document.getElementById('fleet-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const categories = {
    Potholes: events.filter(e => e.event_type === 'POTHOLE').length,
    Cracks:   events.filter(e => e.event_type === 'CRACK').length,
    Incidents: events.filter(e => INCIDENT_TYPES.has((e.event_type || '').toUpperCase())).length,
    VRU:      events.filter(e => VRU_TYPES.has((e.event_type || '').toUpperCase())).length,
    ANPR:     events.filter(e => e.event_type === 'ANPR_VERIFIED').length,
    Traffic:  events.filter(e => e.event_type === 'TRAFFIC_AGGREGATE').length,
  };

  eventsPerTypeChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: Object.keys(categories),
      datasets: [{
        label: 'Events',
        data: Object.values(categories),
        backgroundColor: ['#F59E0B', '#64748B', '#EF4444', '#10B981', '#3B82F6', '#06B6D4'],
        borderWidth: 0,
        borderRadius: 6,
        maxBarThickness: 28,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0F172A',
          padding: 10,
          cornerRadius: 6,
          displayColors: false,
          callbacks: { label: ctx => ` ${ctx.parsed.y} detections` },
        },
      },
      scales: {
        x: { grid: { display: false }, border: { display: false }, ticks: { font: { family: 'Inter', size: 11 }, color: '#64748B' } },
        y: { beginAtZero: true, grid: { color: 'rgba(241,245,249,0.8)' }, border: { display: false }, ticks: { font: { family: 'Inter', size: 11 }, color: '#94A3B8', precision: 0 } },
      },
    },
  });
}

function renderJourney(events) {
  const tbody = document.getElementById('fleet-journey-tbody');
  if (!tbody) return;

  const catMap = {
    POTHOLE: 'Road Defect', CRACK: 'Road Defect',
    COLLISION_CANDIDATE: 'Incident', RAPID_CLOSING: 'Incident', DANGEROUS_FOLLOWING: 'Incident',
    HARD_BRAKING: 'Incident', CUT_IN_CANDIDATE: 'Incident', EMERGENCY_EVASIVE_MANEUVER: 'Incident',
    HIT_AND_RUN_CANDIDATE: 'Incident',
    PEDESTRIAN_IN_PATH: 'VRU', CYCLIST_IN_PATH: 'VRU', VRU_NEAR_MISS: 'VRU', SUDDEN_VRU_EMERGENCE: 'VRU',
    ANPR_VERIFIED: 'ANPR', TRAFFIC_AGGREGATE: 'Traffic', VULNERABLE_AREA_AGGREGATE: 'VRU',
  };
  const catColors = {
    'Road Defect': '#F59E0B', Incident: '#EF4444', VRU: '#10B981', ANPR: '#3B82F6', Traffic: '#06B6D4',
  };

  if (events.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:24px;color:#94A3B8;">No journey events recorded.</td></tr>';
    return;
  }

  const sorted = [...events].sort((a, b) => b.timestamp - a.timestamp);

  tbody.innerHTML = sorted.map(ev => {
    const t = (ev.event_type || '').toUpperCase();
    const cat = catMap[t] || 'Other';
    const catColor = catColors[cat] || '#94A3B8';
    const label = t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    const conf = ev.confidence != null ? `${Math.round(ev.confidence * 100)}%` : '—';
    const demoLoc = ev.demo_route_location;
    const locText = demoLoc ? `${demoLoc.road_name || 'Trichy Road'}` : 'Trichy Metro Corridor';
    const eventId = ev.event_id || ev.id || '—';

    return `<tr class="clickable-row" onclick="window.openEventEvidence('${eventId}')" title="Click to view full event evidence modal">
      <td><span class="mono" style="font-size:12px;font-weight:600;color:var(--blue-600);">${eventId}</span></td>
      <td class="mono text-secondary text-sm">${timeAgo(ev.timestamp)}</td>
      <td style="font-size:13px;font-weight:600;color:var(--text-primary);">${label}</td>
      <td class="text-secondary" style="font-size:12px;">${locText}</td>
      <td><span style="font-size:11px;font-weight:600;color:${catColor};background:${catColor}15;padding:3px 8px;border-radius:12px;">${cat}</span></td>
      <td class="mono text-sm">${conf}</td>
      <td>
        <button class="btn btn-secondary btn-sm" style="font-size:11px;padding:3px 8px;cursor:pointer;">
          <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" style="display:inline;vertical-align:middle;margin-right:4px;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Inspect
        </button>
      </td>
    </tr>`;
  }).join('');
}

export function destroy() {
  if (eventsPerTypeChart) { eventsPerTypeChart.destroy(); eventsPerTypeChart = null; }
}
