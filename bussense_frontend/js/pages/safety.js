/**
 * BusSense — Safety & Incidents Page
 * Shows collision candidates, hard braking, evasive maneuvers, etc.
 */
import api from '../api.js';
import {
  timeAgo,
  severityBadge,
  severityColor,
  eventTypeBadge,
  confidenceBar,
  formatNumber,
} from '../utils.js';

// Incident event types we care about
const INCIDENT_TYPES = new Set([
  'COLLISION_CANDIDATE',
  'RAPID_CLOSING',
  'DANGEROUS_FOLLOWING',
  'HARD_BRAKING',
  'CUT_IN_CANDIDATE',
  'ABRUPT_LATERAL_MANEUVER',
  'WEAVING_CANDIDATE',
  'EMERGENCY_EVASIVE_MANEUVER',
]);

let refreshTimer = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function truncateId(id) {
  if (!id) return '—';
  const s = String(id);
  return s.length > 10 ? s.slice(0, 10) + '…' : s;
}

function ttcCell(val) {
  if (val == null) return '<span class="text-tertiary">—</span>';
  const n = Number(val);
  const cls = n < 1.0 ? 'ttc-danger' : '';
  return `<span class="${cls}">${n.toFixed(2)} s</span>`;
}

function rangeCell(val) {
  if (val == null) return '<span class="text-tertiary">—</span>';
  return `${Number(val).toFixed(1)} m`;
}

// ---------------------------------------------------------------------------
// Compute KPIs
// ---------------------------------------------------------------------------

function computeKpis(incidents) {
  const total = incidents.length;
  const critical = incidents.filter(
    (e) => (e.severity || '').toUpperCase() === 'CRITICAL'
  ).length;

  const ttcValues = incidents
    .map((e) => e.details?.ttc_s ?? e.ttc_s)
    .filter((v) => v != null && !isNaN(v));
  const avgTtc =
    ttcValues.length > 0
      ? ttcValues.reduce((a, b) => a + Number(b), 0) / ttcValues.length
      : null;

  const activeTracks = new Set(
    incidents
      .map((e) => e.details?.track_id ?? e.track_id)
      .filter((v) => v != null)
  ).size;

  return { total, critical, avgTtc, activeTracks };
}

// ---------------------------------------------------------------------------
// Severity breakdown counts
// ---------------------------------------------------------------------------

const SEVERITY_ORDER = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

function severityCounts(incidents) {
  const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  for (const e of incidents) {
    const s = (e.severity || 'LOW').toUpperCase();
    if (s in counts) counts[s]++;
    else counts.LOW++;
  }
  return counts;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

export async function render(container) {
  container.innerHTML = `
    <div class="page-content">
      <div class="page-header">
        <h2 class="page-title">Safety &amp; Incidents</h2>
        <p class="page-subtitle">Collision candidates, hard braking, evasive maneuvers and other safety events</p>
      </div>

      <!-- KPI Row -->
      <div class="kpi-grid mb-4" id="safety-kpis">
        ${kpiSkeleton()}
      </div>

      <!-- Severity Breakdown -->
      <div class="card mb-4" id="severity-breakdown">
        <div class="card-header">
          <span class="card-title">Severity Breakdown</span>
        </div>
        <div class="card-body">
          <div class="loading-shimmer" style="width:100%;height:28px;border-radius:6px;"></div>
        </div>
      </div>

      <!-- Incidents Table -->
      <div class="card" id="incidents-table-card">
        <div class="card-header">
          <span class="card-title">Incident Events</span>
          <span class="text-xs text-secondary" id="incidents-count"></span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table" id="incidents-table">
              <tr>
                <th style="width:110px;">Event ID</th>
                <th>Type</th>
                <th style="width:90px;">Severity</th>
                <th>Trichy Corridor Location</th>
                <th style="width:90px;">Range</th>
                <th style="width:85px;">TTC</th>
                <th style="width:105px;">Time</th>
                <th style="width:105px;">Confidence</th>
                <th style="width:95px;text-align:right;">Evidence</th>
              </tr>
            <tbody id="incidents-tbody">
              <tr><td colspan="10" style="text-align:center;padding:32px;">
                <div class="loading-shimmer" style="width:200px;height:16px;margin:0 auto;border-radius:4px;"></div>
              </td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <style>
      .ttc-danger {
        color: var(--red-500);
        font-weight: 600;
      }
      .severity-bar-track {
        display: flex;
        width: 100%;
        height: 28px;
        border-radius: 6px;
        overflow: hidden;
        background: var(--gray-100);
      }
      .severity-bar-segment {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 600;
        color: #fff;
        white-space: nowrap;
        min-width: 0;
        transition: width 0.3s ease;
      }
      .severity-bar-segment.empty { display: none; }
      .severity-legend {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        flex-wrap: wrap;
      }
      .severity-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--text-secondary);
      }
      .severity-legend-swatch {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        flex-shrink: 0;
      }
    </style>
  `;

  await loadData();

  // Auto-refresh every 30s
  refreshTimer = setInterval(loadData, 30000);
}

async function loadData() {
  const data = await api.events({ limit: 200 });
  if (!data) return;

  const events = Array.isArray(data) ? data : data.events || data.items || [];
  const incidents = events.filter((e) => INCIDENT_TYPES.has(e.event_type));

  renderKpis(incidents);
  renderSeverityBar(incidents);
  renderTable(incidents);
}

// ---------------------------------------------------------------------------
// KPI cards
// ---------------------------------------------------------------------------

function kpiSkeleton() {
  return Array(4)
    .fill(
      `<div class="kpi-card">
        <div class="loading-shimmer" style="width:80px;height:12px;margin-bottom:8px;border-radius:3px;"></div>
        <div class="loading-shimmer" style="width:60px;height:28px;border-radius:4px;"></div>
      </div>`
    )
    .join('');
}

function renderKpis(incidents) {
  const { total, critical, avgTtc, activeTracks } = computeKpis(incidents);

  document.getElementById('safety-kpis').innerHTML = `
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Total Incidents</span>
        <span class="kpi-icon-badge" style="background:#EFF6FF;color:#2563EB;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(total)}</span>
        <span class="kpi-pill positive">Active Sensing</span>
      </div>
      <div class="kpi-context">Bus BUS-01 kinematic corridor events</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Critical Alerts</span>
        <span class="kpi-icon-badge" style="background:#FEF2F2;color:#DC2626;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value" style="color:${critical > 0 ? '#DC2626' : 'var(--text-primary)'}">${formatNumber(critical)}</span>
        <span class="kpi-pill negative">High Priority</span>
      </div>
      <div class="kpi-context">${total > 0 ? ((critical / total) * 100).toFixed(0) + '% of detected incidents' : 'All clear'}</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Mean Time-to-Collision</span>
        <span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;">⏱️</span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${avgTtc != null ? avgTtc.toFixed(2) + '<span style="font-size:14px;font-weight:500;"> s</span>' : '—'}</span>
        <span class="kpi-pill warning">Kinematic TTC</span>
      </div>
      <div class="kpi-context">Stereo + monocular trajectory model</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Active Conflict Tracks</span>
        <span class="kpi-icon-badge" style="background:#F0FDF4;color:#16A34A;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(activeTracks)}</span>
        <span class="kpi-pill positive">Tracked Targets</span>
      </div>
      <div class="kpi-context">Unique vehicle &amp; agent tracks</div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Severity breakdown bar
// ---------------------------------------------------------------------------

function renderSeverityBar(incidents) {
  const counts = severityCounts(incidents);
  const total = incidents.length || 1; // avoid division by zero

  const segments = SEVERITY_ORDER.map((sev) => {
    const count = counts[sev];
    const pct = (count / total) * 100;
    const color = severityColor(sev);
    const empty = count === 0 ? 'empty' : '';
    const label = pct >= 8 ? `${sev} ${count}` : count > 0 ? `${count}` : '';
    return `<div class="severity-bar-segment ${empty}" style="width:${pct}%;background:${color};" title="${sev}: ${count}">${label}</div>`;
  }).join('');

  const legend = SEVERITY_ORDER.map((sev) => {
    return `<div class="severity-legend-item">
      <div class="severity-legend-swatch" style="background:${severityColor(sev)};"></div>
      ${sev} <strong>${counts[sev]}</strong>
    </div>`;
  }).join('');

  const el = document.getElementById('severity-breakdown');
  if (!el) return;

  el.querySelector('.card-body').innerHTML = `
    <div class="severity-bar-track">${segments}</div>
    <div class="severity-legend">${legend}</div>
  `;
}

// ---------------------------------------------------------------------------
// Incidents table
// ---------------------------------------------------------------------------

function renderTable(incidents) {
  const countEl = document.getElementById('incidents-count');
  if (countEl) countEl.textContent = `${incidents.length} incidents`;

  const tbody = document.getElementById('incidents-tbody');
  if (!tbody) return;

  if (incidents.length === 0) {
    tbody.innerHTML = `
      <tr><td colspan="9">
        <div class="empty-state">
          <div class="empty-state-icon"><svg width="32" height="32" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div>
          <h3>No incidents</h3>
          <p>No safety incidents found in the current data window.</p>
        </div>
      </td></tr>`;
    return;
  }

  // Sort by timestamp descending (newest first)
  const sorted = [...incidents].sort((a, b) => {
    const tA = a.timestamp || a.created_at || 0;
    const tB = b.timestamp || b.created_at || 0;
    return tB - tA;
  });

  tbody.innerHTML = sorted
    .map((e) => {
      const d = e.details || {};
      const demoLoc = e.demo_route_location;
      const locationText = demoLoc
        ? `${demoLoc.road_name || 'Trichy Road'} (${demoLoc.landmark || 'Corridor'})`
        : 'Tiruchirappalli Corridor';
      const rangeMVal = d.range_m ?? e.range_m;
      const ttcVal = d.ttc_s ?? e.ttc_s;
      const conf = d.confidence ?? e.confidence;
      const ts = e.timestamp || e.created_at;

      const hasMedia = Boolean(e.event_clip || e.image);
      const mediaBadge = hasMedia
        ? `<span class="badge-status active" style="font-size:10px;padding:2px 6px;"><svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" style="display:inline;vertical-align:middle;margin-right:3px;"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>${e.event_clip ? 'Clip' : 'Frame'}</span>`
        : `<span class="text-tertiary">—</span>`;

      return `<tr class="clickable-row" data-event-id="${e.event_id}" onclick="window.openEventEvidence('${e.event_id}')" title="Click to inspect incident evidence" style="cursor:pointer;">
        <td><span class="mono" style="font-weight:600;color:var(--blue-600);">${e.event_id || '—'}</span></td>
        <td>${eventTypeBadge(e.event_type)}</td>
        <td>${severityBadge(e.severity)}</td>
        <td style="font-size:12px;color:#1E293B;font-weight:500;">${locationText}</td>
        <td class="mono">${rangeCell(rangeMVal)}</td>
        <td class="mono">${ttcCell(ttcVal)}</td>
        <td class="text-secondary" style="font-size:11.5px;" title="${ts ? new Date(ts * 1000).toLocaleString() : ''}">${timeAgo(ts)}</td>
        <td>${confidenceBar(conf)}</td>
        <td style="text-align:right;">${mediaBadge}</td>
      </tr>`;
    })
    .join('');

  // Attach global helper to open modal
  window.showIncidentEvidence = (eventId) => {
    const ev = incidents.find(x => x.event_id === eventId);
    window.openEventEvidence(eventId, { cachedEvent: ev });
  };
}

// ---------------------------------------------------------------------------
// Destroy
// ---------------------------------------------------------------------------

export function destroy() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}
