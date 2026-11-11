/**
 * BusSense — Traffic Intelligence Page (Reference Quality)
 * Real-time corridor traffic dynamics from mobile sensing bus BUS-01.
 */
import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import {
  timeAgo,
  trafficStateBadge,
  formatNumber,
  pct,
  formatTimestamp,
} from '../utils.js';

let chartInstance = null;
let refreshTimer = null;

/* ── State color map ─────────────────────────────────── */
const STATE_META = {
  FREE_FLOW:  { label: 'Free Flow',  color: '#22C55E',  bg: '#F0FDF4', css: 'free_flow'  },
  BUILDING:   { label: 'Building',   color: '#F59E0B',  bg: '#FFFBEB', css: 'building'   },
  CONGESTED:  { label: 'Congested',  color: '#F97316',  bg: '#FFF7ED', css: 'congested'  },
  JAM_RISK:   { label: 'Jam Risk',   color: '#EF4444',  bg: '#FEF2F2', css: 'jam_risk'   },
};

/* ── Helpers ──────────────────────────────────────────── */
function mean(arr, key) {
  if (!arr.length) return 0;
  const sum = arr.reduce((s, r) => s + (Number(r[key]) || 0), 0);
  return sum / arr.length;
}

function countByState(records) {
  const counts = { FREE_FLOW: 0, BUILDING: 0, CONGESTED: 0, JAM_RISK: 0 };
  for (const r of records) {
    const s = (r.traffic_state || '').toUpperCase();
    if (s in counts) counts[s]++;
  }
  return counts;
}

/* ── Jam Risk color helper ────────────────────────────── */
function jamRiskCell(value) {
  if (value == null) return '—';
  const display = pct(value);
  const color = value > 0.5 ? 'var(--red-500)' : 'inherit';
  return `<span style="color:${color};font-weight:${value > 0.5 ? 700 : 500}">${display}</span>`;
}

/* ── Build page HTML ──────────────────────────────────── */
function buildHTML() {
  return `
    <div class="page-content">
      <div class="page-header" style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:20px;">
        <div>
          <h2 class="page-title">Traffic Intelligence</h2>
          <p class="page-subtitle">Corridor dynamics &amp; virtual gate flow rates · Bus BUS-01 · Tiruchirappalli Metro</p>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="display:inline-flex;align-items:center;gap:6px;background:#FFFFFF;border:1px solid rgba(226,232,240,0.85);padding:6px 12px;border-radius:20px;font-size:12px;font-weight:500;">
            <span style="width:7px;height:7px;border-radius:50%;background:#10B981;display:inline-block;animation:pulse-dot 2s infinite;"></span>
            <span>Virtual Gate Sampling: <strong>30s Rolling</strong></span>
          </div>
        </div>
      </div>

      <!-- Elevated KPI Row (Reference Quality: 4 Cards) -->
      <div class="kpi-grid mb-4" id="traffic-kpis">
        ${[1,2,3,4].map(() => `
          <div class="kpi-card">
            <div class="loading-shimmer" style="width:60%;height:14px;margin-bottom:8px;"></div>
            <div class="loading-shimmer" style="width:40%;height:28px;"></div>
          </div>
        `).join('')}
      </div>

      <!-- Main content: table + state distribution -->
      <div class="grid-2-1" style="gap:20px;">
        <!-- Records Table -->
        <div class="card">
          <div class="card-header">
            <div>
              <span class="card-title">Corridor Traffic Records</span>
              <span class="text-xs text-tertiary" id="traffic-count" style="margin-left:8px;"></span>
            </div>
            <span style="font-size:11px;color:var(--text-secondary);">Click row for camera evidence</span>
          </div>
          <div style="overflow-x:auto;">
            <table class="data-table" id="traffic-table">
              <thead>
                <tr>
                  <th style="width:110px;">Event ID</th>
                  <th>Corridor Location</th>
                  <th style="width:100px;">State</th>
                  <th>Flow Rate</th>
                  <th>Density</th>
                  <th>Mean Speed</th>
                  <th>Jam Risk</th>
                  <th style="width:80px;text-align:right;">Evidence</th>
                </tr>
              </thead>
              <tbody id="traffic-tbody">
                <tr><td colspan="8" style="text-align:center;padding:28px;color:#94A3B8;">Loading traffic telemetry…</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Right: Traffic State Breakdown & Segmented Gauge -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Corridor State Distribution</span>
            <span class="badge-status active">Live</span>
          </div>
          <div class="card-body">
            <div class="chart-container" style="height:210px;">
              <canvas id="jam-risk-chart"></canvas>
            </div>

            <!-- Segmented State Bar (Reference Style) -->
            <div style="margin-top:16px;padding-top:14px;border-top:1px solid #F1F5F9;">
              <div style="font-size:11px;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">
                State Proportions Across Windows
              </div>
              <div class="segmented-bar" id="traffic-segmented-bar"></div>
              <div class="segmented-legend" id="traffic-segmented-legend"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

/* ── Render KPIs ──────────────────────────────────────── */
function renderKPIs(records) {
  const avgFlow = mean(records, 'flow_rate_vpm');
  const avgSpeed = mean(records, 'mean_speed_kmh');
  const avgDensity = mean(records, 'density_veh_per_km');
  const jamCount = records.filter(r => {
    const s = (r.traffic_state || '').toUpperCase();
    return s === 'JAM_RISK' || s === 'CONGESTED';
  }).length;

  const container = document.getElementById('traffic-kpis');
  if (!container) return;

  container.innerHTML = `
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Corridor Flow Rate</span>
        <span class="kpi-icon-badge" style="background:#EFF6FF;color:#2563EB;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><rect x="1" y="3" width="15" height="13" rx="2"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${avgFlow.toFixed(1)} <span style="font-size:14px;font-weight:500;color:var(--text-secondary);">vpm</span></span>
        <span class="kpi-pill positive">Virtual Gate</span>
      </div>
      <div class="kpi-context">Mean volume along Metro route</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Mean Stream Speed</span>
        <span class="kpi-icon-badge" style="background:#F0FDF4;color:#16A34A;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${avgSpeed.toFixed(1)} <span style="font-size:14px;font-weight:500;color:var(--text-secondary);">km/h</span></span>
        <span class="kpi-pill positive">Optimal Flow</span>
      </div>
      <div class="kpi-context">Bus BUS-01 stream average</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Spatial Density</span>
        <span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${avgDensity.toFixed(1)} <span style="font-size:14px;font-weight:500;color:var(--text-secondary);">veh/km</span></span>
        <span class="kpi-pill neutral">Corridor k</span>
      </div>
      <div class="kpi-context">Within normal transit bounds</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Congestion / Jam Risk</span>
        <span class="kpi-icon-badge" style="background:${jamCount > 0 ? '#FEF2F2' : '#EFF6FF'};color:${jamCount > 0 ? '#DC2626' : '#2563EB'};"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value" style="${jamCount > 0 ? 'color:#DC2626;' : ''}">${formatNumber(jamCount)}</span>
        <span class="kpi-pill ${jamCount > 0 ? 'negative' : 'positive'}">${jamCount > 0 ? 'Alert Active' : 'All Clear'}</span>
      </div>
      <div class="kpi-context">Physical camera keyframe attached</div>
    </div>
  `;
}

/* ── Render Table ─────────────────────────────────────── */
function renderTable(records) {
  const tbody = document.getElementById('traffic-tbody');
  const countEl = document.getElementById('traffic-count');
  if (!tbody) return;

  if (countEl) countEl.textContent = `(${records.length} records)`;

  if (!records.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-secondary" style="text-align:center;padding:28px;">No traffic records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map(r => {
    const eventId = r.event_id || r.id || '—';
    const demoLoc = r.demo_route_location;
    const locationText = demoLoc ? `${demoLoc.road_name || 'Trichy Road'} (${demoLoc.landmark || ''})` : 'Tiruchirappalli Corridor';
    const flowText = r.flow_rate_vpm != null ? `${r.flow_rate_vpm.toFixed(1)} vpm` : '—';
    const densityText = r.density_veh_per_km != null ? `${r.density_veh_per_km.toFixed(1)} veh/km` : '—';
    const speedText = r.mean_speed_kmh != null ? `${r.mean_speed_kmh.toFixed(1)} km/h` : '—';
    const hasMedia = Boolean(r.image || r.evidence_image_url);

    const mediaBadge = hasMedia
      ? `<span class="badge-status active" style="font-size:10px;padding:2px 6px;"><svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg> Frame</span>`
      : `<span style="color:#94A3B8;font-size:11px;">Telemetry</span>`;

    return `
      <tr class="clickable-row" data-event-id="${eventId}" style="cursor:pointer;">
        <td><span class="mono" style="font-weight:600;color:var(--blue-600);">${eventId}</span></td>
        <td style="font-size:12px;color:#1E293B;font-weight:500;">${locationText}</td>
        <td>${trafficStateBadge(r.traffic_state)}</td>
        <td class="mono">${flowText}</td>
        <td class="mono">${densityText}</td>
        <td class="mono">${speedText}</td>
        <td>${jamRiskCell(r.jam_risk)}</td>
        <td style="text-align:right;">${mediaBadge}</td>
      </tr>
    `;
  }).join('');

  // Wire row click to open unified evidence modal
  tbody.querySelectorAll('.clickable-row').forEach(row => {
    row.addEventListener('click', () => {
      const id = row.dataset.eventId;
      const rec = records.find(x => (x.event_id || x.id) === id);
      openEventEvidence(id, { cachedEvent: rec });
    });
  });
}

/* ── Render Chart & Segmented Breakdown ─────────────────────────────────── */
function renderChart(records) {
  const canvas = document.getElementById('jam-risk-chart');
  if (!canvas || !window.Chart) return;

  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }

  const counts = countByState(records);
  const labels = Object.values(STATE_META).map(m => m.label);
  const data = [counts.FREE_FLOW, counts.BUILDING, counts.CONGESTED, counts.JAM_RISK];
  const colors = ['#22C55E', '#F59E0B', '#F97316', '#EF4444'];
  const total = records.length || 1;

  chartInstance = new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderColor: '#fff',
        borderWidth: 2,
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
            padding: 12,
            usePointStyle: true,
            pointStyleWidth: 7,
            font: { family: "'Inter', sans-serif", size: 11 },
            color: '#64748B',
          },
        },
        tooltip: {
          backgroundColor: '#0F172A',
          padding: 10,
          cornerRadius: 8,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed} windows (${Math.round((ctx.parsed / total) * 100)}%)`,
          },
        },
      },
    },
  });

  // Segmented Breakdown Bar
  const pFF = Math.round((counts.FREE_FLOW / total) * 100);
  const pBld = Math.round((counts.BUILDING / total) * 100);
  const pCng = Math.round((counts.CONGESTED / total) * 100);
  const pJam = Math.max(0, 100 - (pFF + pBld + pCng));

  const bar = document.getElementById('traffic-segmented-bar');
  if (bar) {
    bar.innerHTML = `
      <div class="segmented-segment" style="width:${pFF}%;background:#22C55E;" title="Free Flow: ${counts.FREE_FLOW}"></div>
      <div class="segmented-segment" style="width:${pBld}%;background:#F59E0B;" title="Building: ${counts.BUILDING}"></div>
      <div class="segmented-segment" style="width:${pCng}%;background:#F97316;" title="Congested: ${counts.CONGESTED}"></div>
      <div class="segmented-segment" style="width:${pJam}%;background:#EF4444;" title="Jam Risk: ${counts.JAM_RISK}"></div>
    `;
  }

  const legend = document.getElementById('traffic-segmented-legend');
  if (legend) {
    legend.innerHTML = `
      <div class="segmented-item" style="border-left-color:#22C55E;">
        <div class="segmented-item-val">${counts.FREE_FLOW}</div>
        <div class="segmented-item-lbl">Free Flow (${pFF}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#F59E0B;">
        <div class="segmented-item-val">${counts.BUILDING}</div>
        <div class="segmented-item-lbl">Building (${pBld}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#F97316;">
        <div class="segmented-item-val">${counts.CONGESTED}</div>
        <div class="segmented-item-lbl">Congested (${pCng}%)</div>
      </div>
      <div class="segmented-item" style="border-left-color:#EF4444;">
        <div class="segmented-item-val">${counts.JAM_RISK}</div>
        <div class="segmented-item-lbl">Jam Risk (${pJam}%)</div>
      </div>
    `;
  }
}

/* ── Load data ────────────────────────────────────────── */
async function loadData() {
  const records = await api.traffic({ limit: 100 });
  const list = Array.isArray(records) ? records : (records?.items || []);

  renderKPIs(list);
  renderTable(list);
  renderChart(list);
}

/* ── Mount / Export ───────────────────────────────────── */
export async function render(container) {
  container.innerHTML = buildHTML();
  await loadData();
  refreshTimer = setInterval(loadData, 30000);
}

export function destroy() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
}
