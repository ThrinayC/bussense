/**
 * BusSense Command Center — Overview / Dashboard V2 (Reference Quality)
 * Single-bus (BUS-01), Tiruchirappalli 12-Waypoint Metro Corridor.
 */
import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import {
  timeAgo, eventTypeBadge, severityBadge, severityColor,
  confidenceBar, formatNumber, formatTimestamp,
} from '../utils.js';

let timelineChart = null;
let severityChart = null;

export async function render(container) {
  container.innerHTML = `
    <div>
      <!-- Page Header with Corridor Telemetry -->
      <div class="page-header" style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:20px;">
        <div>
          <h1 class="page-title" style="font-size:22px;font-weight:700;letter-spacing:-0.02em;color:var(--text-primary);">Operational Overview</h1>
          <p class="page-subtitle" style="font-size:13px;color:var(--text-secondary);margin-top:2px;">
            Mobile Sensing Telemetry · Bus <strong style="color:var(--text-primary);">BUS-01</strong> · Tiruchirappalli Metro Corridor
          </p>
        </div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
          <div style="display:inline-flex;align-items:center;gap:6px;background:#FFFFFF;border:1px solid rgba(226,232,240,0.85);padding:6px 12px;border-radius:20px;font-size:12px;font-weight:500;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
            <span style="width:7px;height:7px;border-radius:50%;background:#10B981;display:inline-block;animation:pulse-dot 2s infinite;"></span>
            <span>Route: <strong>Central Bus Stand ➔ Srirangam</strong></span>
          </div>
          <button class="btn btn-secondary btn-sm" id="btn-refresh-overview" style="font-size:12px;padding:6px 12px;border-radius:8px;">
            ↻ Refresh
          </button>
        </div>
      </div>

      <!-- KPI Grid: Reference Quality Layout -->
      <section class="kpi-grid mb-4" id="kpi-grid">
        ${[1,2,3,4].map(() => `
          <div class="kpi-card">
            <div class="loading-shimmer" style="width:60%;height:14px;margin-bottom:8px;"></div>
            <div class="loading-shimmer" style="width:40%;height:28px;"></div>
          </div>
        `).join('')}
      </section>

      <!-- Visualizations Grid: 2-1 Layout -->
      <section class="grid-2-1 mb-4" style="gap:20px;">
        <!-- Left: Event Timeline & Category Breakdown -->
        <div class="card">
          <div class="card-header">
            <div style="display:flex;align-items:center;gap:8px;">
              <span class="card-title">Perception Event Dynamics · 24h Timeline</span>
            </div>
            <span class="badge-status active" style="font-size:11px;">Live</span>
          </div>
          <div class="card-body">
            <div class="chart-container" style="height:220px;">
              <canvas id="timeline-chart"></canvas>
            </div>

            <!-- Segmented Category Breakdown (Reference Style) -->
            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #F1F5F9;">
              <div style="font-size:11.5px;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">
                Corridor Event Composition
              </div>
              <div class="segmented-bar" id="category-segmented-bar">
                <div class="segmented-segment loading-shimmer" style="width:100%;background:var(--gray-100);"></div>
              </div>
              <div class="segmented-legend" id="category-legend"></div>
            </div>
          </div>
        </div>

        <!-- Right: Severity & Operational Status -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Severity &amp; Fleet Status</span>
            <span class="badge-status active">Live</span>
          </div>
          <div class="card-body">
            <div class="chart-container" style="height:190px;">
              <canvas id="severity-chart"></canvas>
            </div>

            <!-- Operational Summary Chips -->
            <div style="margin-top:16px;display:flex;flex-direction:column;gap:8px;">
              <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#F8FAFC;border-radius:8px;font-size:12px;border:1px solid #F1F5F9;">
                <span style="color:var(--text-secondary);">Sensing Unit:</span>
                <span class="mono" style="font-weight:700;color:var(--text-primary);">BUS-01 (TN-45-N-2025)</span>
              </div>
              <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#F8FAFC;border-radius:8px;font-size:12px;border:1px solid #F1F5F9;">
                <span style="color:var(--text-secondary);">Traffic Corridor State:</span>
                <span style="font-weight:700;color:#059669;">Free Flow (22 km/h)</span>
              </div>
              <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#F0FDF4;border-radius:8px;font-size:12px;border:1px solid #BBF7D0;">
                <span style="color:#166534;font-weight:500;">CSR Sponsorship Pool:</span>
                <span style="font-weight:700;color:#15803D;">₹90,00,000 Uncommitted</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Recent Events Feed -->
      <section class="card">
        <div class="card-header">
          <div>
            <span class="card-title">Recent Real-Time Events</span>
            <span style="font-size:12px;color:var(--text-tertiary);margin-left:8px;" id="events-count"></span>
          </div>
          <span style="font-size:11px;color:var(--text-secondary);">
            Click any row to open full evidence modal
          </span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th style="width:110px;">Time</th>
                <th style="width:120px;">Event ID</th>
                <th>Type</th>
                <th style="width:90px;">Severity</th>
                <th>Trichy Corridor Location</th>
                <th style="width:110px;">Confidence</th>
                <th style="width:90px;text-align:right;">Evidence</th>
              </tr>
            </thead>
            <tbody id="events-tbody">
              <tr>
                <td colspan="7" class="text-secondary" style="text-align:center;padding:28px;">
                  <div class="loading-shimmer" style="width:180px;height:14px;margin:0 auto;"></div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  `;

  // Attach refresh action
  document.getElementById('btn-refresh-overview')?.addEventListener('click', () => render(container));

  // Load backend data
  const [healthData, eventsData] = await Promise.all([
    api.health(),
    api.events({ limit: 200 }),
  ]);

  const events = Array.isArray(eventsData) ? eventsData : [];

  renderKPIs(healthData, events);
  renderTimelineChart(events);
  renderSeverityChart(events);
  renderCategoryBreakdown(events);
  renderEventsTable(events.slice(0, 10));
}

function renderKPIs(health, events) {
  const totalEvents = events.length;
  const criticalCount = events.filter(e => (e.severity || '').toUpperCase() === 'CRITICAL').length;
  const roadDefects = events.filter(e => {
    const t = (e.event_type || '').toUpperCase();
    return t === 'POTHOLE' || t === 'CRACK';
  }).length;
  const incidentCount = events.filter(e => {
    const t = (e.event_type || '').toUpperCase();
    return ['COLLISION_CANDIDATE','RAPID_CLOSING','HARD_BRAKING','CUT_IN_CANDIDATE',
      'EMERGENCY_EVASIVE_MANEUVER','HIT_AND_RUN_CANDIDATE'].includes(t);
  }).length;

  document.getElementById('kpi-grid').innerHTML = `
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Total Events</span>
        <span class="kpi-icon-badge" style="background:#EFF6FF;color:#2563EB;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(totalEvents)}</span>
        <span class="kpi-pill positive">● Live Stream</span>
      </div>
      <div class="kpi-context">Across 13 authentic Mega scenarios</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Critical Alerts</span>
        <span class="kpi-icon-badge" style="background:#FEF2F2;color:#DC2626;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value" style="${criticalCount > 0 ? 'color:#DC2626;' : ''}">${formatNumber(criticalCount)}</span>
        <span class="kpi-pill negative">Immediate Action</span>
      </div>
      <div class="kpi-context">Hit-and-Run, near miss &amp; deep cavity</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Road Defects</span>
        <span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 12a4 4 0 0 0 8 0"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(roadDefects)}</span>
        <span class="kpi-pill warning">₹52,800 Est.</span>
      </div>
      <div class="kpi-context">6 Potholes · 3 Surface Crack Networks</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Kinematic Incidents</span>
        <span class="kpi-icon-badge" style="background:#F0FDF4;color:#16A34A;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(incidentCount)}</span>
        <span class="kpi-pill positive">TTC Tracked</span>
      </div>
      <div class="kpi-context">Evasive maneuvers &amp; cut-in candidates</div>
    </div>
  `;
}

function renderTimelineChart(events) {
  const canvas = document.getElementById('timeline-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const now = Date.now();
  const cutoff = now - 24 * 60 * 60 * 1000;
  const buckets = new Array(24).fill(0);
  const labels = [];

  for (let i = 0; i < 24; i++) {
    const h = new Date(cutoff + i * 3600000);
    labels.push(h.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false }));
  }

  events.forEach(ev => {
    const tsMs = typeof ev.timestamp === 'number' ? ev.timestamp * 1000 : new Date(ev.timestamp).getTime();
    if (tsMs >= cutoff && tsMs <= now) {
      buckets[Math.min(Math.floor((tsMs - cutoff) / 3600000), 23)]++;
    }
  });

  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, 'rgba(37, 99, 235, 0.25)');
  gradient.addColorStop(1, 'rgba(37, 99, 235, 0.00)');

  timelineChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Detected Events',
        data: buckets,
        borderColor: '#2563EB',
        borderWidth: 2,
        backgroundColor: gradient,
        fill: true,
        tension: 0.35,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: '#2563EB',
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#FFFFFF',
          borderColor: 'rgba(226,232,240,0.9)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          titleFont: { family: 'Inter', size: 12, weight: '600' },
          titleColor: '#1E293B',
          bodyFont: { family: 'Inter', size: 12 },
          bodyColor: '#475569',
          displayColors: false,
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.y} perception events`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { family: 'Inter', size: 10 }, color: '#94A3B8', maxTicksLimit: 8 },
        },
        y: {
          beginAtZero: true,
          grid: { color: '#F1F5F9' },
          ticks: { font: { family: 'Inter', size: 10 }, color: '#94A3B8', precision: 0 },
        },
      },
    },
  });
}

function renderCategoryBreakdown(events) {
  const counts = {
    incidents: 0,
    defects: 0,
    vru: 0,
    anpr: 0,
    traffic: 0
  };

  events.forEach(e => {
    const t = (e.event_type || '').toUpperCase();
    if (['COLLISION_CANDIDATE','RAPID_CLOSING','HARD_BRAKING','CUT_IN_CANDIDATE','EMERGENCY_EVASIVE_MANEUVER','HIT_AND_RUN_CANDIDATE','WEAVING_CANDIDATE','ABRUPT_LATERAL_MANEUVER','DANGEROUS_FOLLOWING'].includes(t)) {
      counts.incidents++;
    } else if (t === 'POTHOLE' || t === 'CRACK') {
      counts.defects++;
    } else if (['PEDESTRIAN_IN_PATH','CYCLIST_IN_PATH','VRU_NEAR_MISS','SUDDEN_VRU_EMERGENCE','VULNERABLE_AREA_AGGREGATE'].includes(t)) {
      counts.vru++;
    } else if (t === 'ANPR_VERIFIED') {
      counts.anpr++;
    } else if (t === 'TRAFFIC_AGGREGATE') {
      counts.traffic++;
    }
  });

  const total = events.length || 1;
  const pInc = Math.round((counts.incidents / total) * 100);
  const pDef = Math.round((counts.defects / total) * 100);
  const pVru = Math.round((counts.vru / total) * 100);
  const pAnpr = Math.round((counts.anpr / total) * 100);
  const pTraf = Math.max(0, 100 - (pInc + pDef + pVru + pAnpr));

  const bar = document.getElementById('category-segmented-bar');
  if (bar) {
    bar.innerHTML = `
      <div class="segmented-segment" style="width:${pInc}%;background:#EF4444;" title="Incidents: ${counts.incidents}"></div>
      <div class="segmented-segment" style="width:${pDef}%;background:#F59E0B;" title="Road Defects: ${counts.defects}"></div>
      <div class="segmented-segment" style="width:${pVru}%;background:#10B981;" title="VRU Safety: ${counts.vru}"></div>
      <div class="segmented-segment" style="width:${pAnpr}%;background:#3B82F6;" title="ANPR: ${counts.anpr}"></div>
      <div class="segmented-segment" style="width:${pTraf}%;background:#14B8A6;" title="Traffic: ${counts.traffic}"></div>
    `;
  }

  const legend = document.getElementById('category-legend');
  if (legend) {
    legend.innerHTML = `
      <div class="segmented-item" style="border-left-color:#EF4444;">
        <div class="segmented-item-val">${counts.incidents}</div>
        <div class="segmented-item-lbl">Incidents (${pInc}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#F59E0B;">
        <div class="segmented-item-val">${counts.defects}</div>
        <div class="segmented-item-lbl">Road Defects (${pDef}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#10B981;">
        <div class="segmented-item-val">${counts.vru}</div>
        <div class="segmented-item-lbl">VRU Safety (${pVru}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#3B82F6;">
        <div class="segmented-item-val">${counts.anpr}</div>
        <div class="segmented-item-lbl">ANPR (${pAnpr}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#14B8A6;">
        <div class="segmented-item-val">${counts.traffic}</div>
        <div class="segmented-item-lbl">Traffic Intel (${pTraf}%)</div>
      </div>
    `;
  }
}

function renderSeverityChart(events) {
  const canvas = document.getElementById('severity-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  events.forEach(e => {
    const s = (e.severity || 'LOW').toUpperCase();
    if (s in counts) counts[s]++;
    else counts.LOW++;
  });

  const sevs = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
  const colors = { CRITICAL: '#EF4444', HIGH: '#F59E0B', MEDIUM: '#3B82F6', LOW: '#94A3B8' };

  severityChart = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: sevs.map(s => s.charAt(0) + s.slice(1).toLowerCase()),
      datasets: [{
        data: sevs.map(s => counts[s]),
        backgroundColor: sevs.map(s => colors[s]),
        borderWidth: 2,
        borderColor: '#FFFFFF',
        hoverOffset: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            font: { family: 'Inter', size: 11 },
            color: '#64748B',
            padding: 12,
            usePointStyle: true,
            pointStyleWidth: 7,
          },
        },
        tooltip: {
          backgroundColor: '#FFFFFF',
          borderColor: 'rgba(226,232,240,0.9)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8,
          titleColor: '#1E293B',
          bodyColor: '#475569',
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed} events`,
          },
        },
      },
    },
  });
}

function renderEventsTable(events) {
  const tbody = document.getElementById('events-tbody');
  const countEl = document.getElementById('events-count');
  if (!tbody) return;
  if (countEl) countEl.textContent = `(${events.length} latest)`;

  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-secondary" style="text-align:center;padding:24px;">No events recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = events.map(ev => {
    const demoLoc = ev.demo_route_location;
    const locText = demoLoc ? `${demoLoc.road_name || 'Trichy Road'} (${demoLoc.landmark || 'Corridor'})` : 'Tiruchirappalli Metro Corridor';
    const hasMedia = Boolean(ev.event_clip || ev.image || ev.evidence_image_url);
    const mediaBadge = hasMedia
      ? `<span style="display:inline-flex;align-items:center;gap:3px;font-size:10px;font-weight:600;color:#0369A1;background:#F0F9FF;border:1px solid #BAE6FD;padding:2px 7px;border-radius:10px;">📷 Media</span>`
      : `<span style="color:#94A3B8;font-size:11px;">Telemetry only</span>`;

    return `
      <tr class="clickable-row" data-event-id="${ev.event_id}" style="cursor:pointer;">
        <td class="mono" style="font-size:11.5px;color:var(--text-secondary);" title="${formatTimestamp(ev.timestamp)}">${timeAgo(ev.timestamp)}</td>
        <td><span class="mono" style="font-weight:600;color:var(--blue-600);display:inline-block;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:bottom;" title="${ev.event_id || ''}">${ev.event_id || '—'}</span></td>
        <td>${eventTypeBadge(ev.event_type)}</td>
        <td>${severityBadge(ev.severity)}</td>
        <td style="font-size:12px;color:#1E293B;font-weight:500;">${locText}</td>
        <td>${confidenceBar(ev.confidence)}</td>
        <td style="text-align:right;">${mediaBadge}</td>
      </tr>
    `;
  }).join('');

  // Wire interactive click on every single row to open evidence modal
  tbody.querySelectorAll('.clickable-row').forEach(row => {
    row.addEventListener('click', () => {
      const eventId = row.dataset.eventId;
      const ev = events.find(x => x.event_id === eventId);
      openEventEvidence(eventId, { cachedEvent: ev });
    });
  });
}

export function destroy() {
  if (timelineChart) { timelineChart.destroy(); timelineChart = null; }
  if (severityChart) { severityChart.destroy(); severityChart = null; }
}
