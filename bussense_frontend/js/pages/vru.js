/**
 * BusSense — VRU Safety Page
 * Shows VRU events (pedestrian / cyclist / near-miss) and vulnerable area segments.
 */
import api from '../api.js';
import {
  timeAgo,
  severityBadge,
  eventTypeBadge,
  confidenceBar,
  formatNumber,
} from '../utils.js';

// VRU event types we care about
const VRU_TYPES = new Set([
  'PEDESTRIAN_IN_PATH',
  'CYCLIST_IN_PATH',
  'VRU_NEAR_MISS',
  'SUDDEN_VRU_EMERGENCE',
]);

// Derive a human-readable VRU type from the event_type field
function vruTypeLabel(eventType) {
  switch (eventType) {
    case 'PEDESTRIAN_IN_PATH': return 'Pedestrian';
    case 'CYCLIST_IN_PATH':    return 'Cyclist';
    case 'VRU_NEAR_MISS':      return 'Near Miss';
    case 'SUDDEN_VRU_EMERGENCE': return 'Sudden VRU';
    default: return eventType || '—';
  }
}

// Color a vulnerability index value
function vulnIndexHtml(val) {
  if (val == null) return '—';
  const n = Number(val);
  let color = 'var(--green-600)';
  if (n > 60) color = 'var(--red-500)';
  else if (n > 30) color = 'var(--amber-600)';
  return `<span style="font-weight:600;color:${color}">${n.toFixed(1)}</span>`;
}

// Truncate an ID for display
function truncId(id) {
  if (!id) return '—';
  const s = String(id);
  return s.length > 10 ? s.slice(0, 10) + '…' : s;
}

// Format a numeric field with a unit suffix; return '—' for missing values
function numWithUnit(val, unit) {
  if (val == null || isNaN(val)) return '—';
  return `${Number(val).toFixed(1)} ${unit}`;
}

/* ── State ─────────────────────────────────────────────── */
let _interval = null;

/* ── Render ────────────────────────────────────────────── */
export async function render(container) {
  // Page shell — skeleton while loading
  container.innerHTML = `
    <div class="page-content">
      <div class="page-header">
        <h2 class="page-title">VRU Safety</h2>
        <p class="page-subtitle">Vulnerable Road User events, near-misses, and high-risk segments</p>
      </div>

      <!-- KPIs -->
      <div class="kpi-grid mb-4" id="vru-kpis">
        ${kpiSkeleton(4)}
      </div>

      <!-- VRU Events table -->
      <div class="card mb-4" id="vru-events-card">
        <div class="card-header">
          <span class="card-title">VRU Events</span>
        </div>
        <div class="card-body" style="padding:0;">
          <div style="padding:var(--space-12);text-align:center;color:var(--text-tertiary)">Loading…</div>
        </div>
      </div>

      <!-- Vulnerable Segments table -->
      <div class="card" id="vru-segments-card">
        <div class="card-header">
          <span class="card-title">Vulnerable Segments</span>
        </div>
        <div class="card-body" style="padding:0;">
          <div style="padding:var(--space-12);text-align:center;color:var(--text-tertiary)">Loading…</div>
        </div>
      </div>
    </div>`;

  await loadData();
}

/* ── Destroy ───────────────────────────────────────────── */
export function destroy() {
  if (_interval) {
    clearInterval(_interval);
    _interval = null;
  }
}

/* ── Data fetching ─────────────────────────────────────── */
async function loadData() {
  const [eventsRaw, vruAreasRaw] = await Promise.all([
    api.events({ limit: 200 }),
    api.vruAreas(),
  ]);

  // Normalise — API may return array directly or { items: [] }
  const allEvents = Array.isArray(eventsRaw)
    ? eventsRaw
    : (eventsRaw?.items ?? eventsRaw?.events ?? []);

  const vruEvents = allEvents.filter(e => VRU_TYPES.has(e.event_type));

  const segments = Array.isArray(vruAreasRaw)
    ? vruAreasRaw
    : (vruAreasRaw?.segments ?? vruAreasRaw?.items ?? vruAreasRaw?.areas ?? []);

  renderKpis(vruEvents, segments);
  renderEventsTable(vruEvents);
  renderSegmentsTable(segments);
}

/* ── KPIs ──────────────────────────────────────────────── */
function renderKpis(events, segments) {
  const total = events.length;
  const nearMisses = events.filter(e => e.event_type === 'VRU_NEAR_MISS').length;
  const totalSegments = segments.length;

  const indices = segments
    .map(s => s.vulnerability_index)
    .filter(v => v != null && !isNaN(v));
  const avgIndex = indices.length
    ? (indices.reduce((a, b) => a + Number(b), 0) / indices.length)
    : null;

  const kpiContainer = document.getElementById('vru-kpis');
  if (!kpiContainer) return;

  kpiContainer.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">VRU Events</span>
        <span class="kpi-icon-badge" style="background:#EFF6FF;color:#2563EB;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M13 4a1 1 0 1 1-2 0 1 1 0 0 1 2 0"/><path d="M10 8h4l1 4-3 1-2-2-3.5 1z"/><path d="M9 16l1-4 2 2 1 4M15 16l-1-4"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(total)}</span>
        <span class="kpi-pill positive">Stereo Tracked</span>
      </div>
      <div class="kpi-context">Pedestrians &amp; cyclists in transit path</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Near Miss Conflicts</span>
        <span class="kpi-icon-badge" style="background:#FEF2F2;color:#DC2626;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value" style="color:${nearMisses > 0 ? '#DC2626' : 'var(--text-primary)'}">${formatNumber(nearMisses)}</span>
        <span class="kpi-pill negative">High Risk</span>
      </div>
      <div class="kpi-context">TTC &lt; 1.5s conflict trajectories</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Vulnerable Corridors</span>
        <span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="18" r="3"/><path d="M6 18L9.5 6H12"/><polyline points="12 6 17 6"/><path d="M17 6l-1.5 12"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(totalSegments)}</span>
        <span class="kpi-pill warning">Spatial Zones</span>
      </div>
      <div class="kpi-context">Shastri Rd &amp; Salai Rd aggregate zones</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Avg Vulnerability Index</span>
        <span class="kpi-icon-badge" style="background:#F0FDF4;color:#16A34A;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${avgIndex != null ? avgIndex.toFixed(1) : '—'}</span>
        <span class="kpi-pill neutral">Safety Metric</span>
      </div>
      <div class="kpi-context">Crossing density &amp; speed index</div>
    </div>
  `;
}

function kpiSkeleton(count) {
  return Array.from({ length: count }, () => `
    <div class="kpi-card">
      <div class="loading-shimmer" style="width:80px;height:14px;margin-bottom:8px;border-radius:4px;"></div>
      <div class="loading-shimmer" style="width:60px;height:28px;border-radius:4px;"></div>
    </div>`).join('');
}

/* ── VRU Events Table ──────────────────────────────────── */
function renderEventsTable(events) {
  const card = document.getElementById('vru-events-card');
  if (!card) return;

  const body = card.querySelector('.card-body');

  if (!events.length) {
    body.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg></div>
        <h3>No VRU events</h3>
        <p>No vulnerable road user events found in the current dataset.</p>
      </div>`;
    return;
  }

  body.innerHTML = `
    <div style="overflow-x:auto;">
      <table class="data-table">
        <thead>
          <tr>
            <th style="width:110px;">Event ID</th>
            <th>Type</th>
            <th style="width:90px;">VRU Type</th>
            <th style="width:90px;">Severity</th>
            <th>Trichy Corridor Location</th>
            <th style="width:85px;">Range</th>
            <th style="width:85px;">TTC</th>
            <th style="width:105px;">Time</th>
            <th style="width:105px;">Confidence</th>
            <th style="width:95px;text-align:right;">Evidence</th>
          </tr>
        </thead>
        <tbody>
          ${events.map(e => {
            const demoLoc = e.demo_route_location;
            const locationText = demoLoc
              ? `${demoLoc.road_name || 'Trichy Road'} (${demoLoc.landmark || 'Corridor'})`
              : 'Tiruchirappalli Corridor';
            const hasMedia = Boolean(e.event_clip || e.image);
            const mediaBtn = hasMedia
              ? `<span class="badge-status active" style="font-size:10px;padding:2px 6px;"><svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg> ${e.event_clip ? 'Clip' : 'Frame'}</span>`
              : '<span class="text-tertiary">—</span>';

            return `
            <tr class="clickable-row" data-event-id="${e.event_id}" onclick="window.openEventEvidence('${e.event_id}')" title="Click to view VRU safety evidence" style="cursor:pointer;">
              <td><span class="mono" style="font-weight:600;color:var(--blue-600);">${e.event_id || '—'}</span></td>
              <td>${eventTypeBadge(e.event_type)}</td>
              <td><strong>${vruTypeLabel(e.event_type)}</strong></td>
              <td>${severityBadge(e.severity)}</td>
              <td style="font-size:12px;color:#1E293B;font-weight:500;">${locationText}</td>
              <td class="mono">${numWithUnit(e.range_m, 'm')}</td>
              <td class="mono">${numWithUnit(e.ttc_s, 's')}</td>
              <td class="text-secondary" style="font-size:11.5px;" title="${e.timestamp || ''}">${timeAgo(e.timestamp)}</td>
              <td>${confidenceBar(e.confidence)}</td>
              <td style="text-align:right;">${mediaBtn}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;

  window.showVruEvidence = (eventId) => {
    const ev = events.find(x => x.event_id === eventId);
    window.openEventEvidence(eventId, { cachedEvent: ev });
  };
}

/* ── Vulnerable Segments Table ─────────────────────────── */
function renderSegmentsTable(segments) {
  const card = document.getElementById('vru-segments-card');
  if (!card) return;

  const body = card.querySelector('.card-body');

  if (!segments.length) {
    body.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg></div>
        <h3>No segments</h3>
        <p>No vulnerable area segments found.</p>
      </div>`;
    return;
  }

  body.innerHTML = `
    <div style="overflow-x:auto;">
      <table class="data-table">
        <thead>
          <tr>
            <th>Segment ID</th>
            <th>Pedestrians</th>
            <th>Cyclists</th>
            <th>Crossings</th>
            <th>Conflicts</th>
            <th>Near Misses</th>
            <th>Vulnerability Index</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
          ${segments.map(s => `
            <tr>
              <td class="mono" title="${s.segment_id || ''}">${truncId(s.segment_id)}</td>
              <td>${s.pedestrians ?? s.pedestrian_count ?? '—'}</td>
              <td>${s.cyclists ?? s.cyclist_count ?? '—'}</td>
              <td>${s.crossings ?? s.crossing_count ?? '—'}</td>
              <td>${s.conflicts ?? s.conflict_count ?? '—'}</td>
              <td>${s.near_misses ?? s.near_miss_count ?? '—'}</td>
              <td>${vulnIndexHtml(s.vulnerability_index)}</td>
              <td>${severityBadge(s.severity)}</td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
}
