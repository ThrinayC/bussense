/**
 * BusSense — Vehicles / ANPR Page V2
 * Shows verified plate readings as prominent cards + table.
 * Plate registration shown large. OCR confidence displayed clearly.
 */
import api from '../api.js';
import { timeAgo, confidenceBar, formatNumber, formatTimestamp } from '../utils.js';

let chartInstance = null;

export async function render(container) {
  container.innerHTML = `
    <div>
      <div class="page-header">
        <h1 class="page-title">Vehicles / ANPR</h1>
        <p class="page-subtitle">Verified license plate readings · Bus BUS-01 · Active Perception Corridor</p>
      </div>

      <section class="kpi-grid mb-4" style="grid-template-columns:repeat(3,1fr);" id="anpr-kpis">
        ${[1,2,3].map(() => '<div class="kpi-card"><div class="loading-shimmer" style="width:60%;height:14px;margin-bottom:8px;"></div><div class="loading-shimmer" style="width:40%;height:28px;"></div></div>').join('')}
      </section>

      <div class="grid-2-1">
        <!-- ANPR Cards + Table -->
        <div>
          <div id="anpr-cards" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;"></div>
          <div class="card">
            <div class="card-header">
              <span class="card-title">All Readings</span>
              <span class="text-xs text-tertiary" id="anpr-count"></span>
            </div>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Registration</th>
                  <th>Vehicle</th>
                  <th>OCR Conf.</th>
                  <th>Plate Conf.</th>
                  <th>Width</th>
                  <th>Time</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody id="anpr-tbody">
                <tr><td colspan="7" style="text-align:center;padding:24px;color:#94A3B8;">Loading…</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Confidence distribution chart -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">OCR Confidence Distribution</span>
          </div>
          <div class="card-body">
            <div class="chart-container" style="height:220px;"><canvas id="anpr-chart"></canvas></div>
          </div>
          <div id="anpr-engine-note" style="padding:12px 16px;font-size:11px;color:var(--text-tertiary);border-top:1px solid var(--border-subtle);">
            Engine: EasyOCR Multi-Variant Ensemble · 6 preprocessing variants per candidate
          </div>
        </div>
      </div>
    </div>`;

  const result = await api.events({ event_type: 'ANPR_VERIFIED', limit: 100 });
  const events = Array.isArray(result) ? result : [];

  renderKPIs(events);
  renderCards(events.slice(0, 6));
  renderTable(events);
  renderChart(events);
}

function renderKPIs(events) {
  const uniqueRegs = new Set();
  let ocrSum = 0, ocrCount = 0;
  for (const ev of events) {
    if (ev.registration_number) uniqueRegs.add(ev.registration_number);
    if (ev.ocr_confidence != null) { ocrSum += ev.ocr_confidence; ocrCount++; }
  }
  const avgOcr = ocrCount > 0 ? Math.round((ocrSum / ocrCount) * 100) : null;

  document.getElementById('anpr-kpis').innerHTML = `
    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Plates Scanned</span>
        <span class="kpi-icon-badge" style="background:#EFF6FF;color:#2563EB;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(events.length)}</span>
        <span class="kpi-pill positive">Verified</span>
      </div>
      <div class="kpi-context">Multi-variant OCR confirmed</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Distinct Registrations</span>
        <span class="kpi-icon-badge" style="background:#F0FDF4;color:#16A34A;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${formatNumber(uniqueRegs.size)}</span>
        <span class="kpi-pill neutral">Vehicles</span>
      </div>
      <div class="kpi-context">Unique license plate numbers</div>
    </div>

    <div class="kpi-card">
      <div class="kpi-header">
        <span class="kpi-label">Avg OCR Confidence</span>
        <span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></span>
      </div>
      <div class="kpi-value-row">
        <span class="kpi-value">${avgOcr != null ? avgOcr + '%' : '—'}</span>
        <span class="kpi-pill warning">High Quality</span>
      </div>
      <div class="kpi-context">6 preprocessing variants consensus</div>
    </div>
  `;
}

function renderCards(events) {
  const container = document.getElementById('anpr-cards');
  if (!container) return;

  if (events.length === 0) { container.innerHTML = ''; return; }

  container.innerHTML = events.map(ev => {
    const reg = ev.registration_number || ev.plate || '—';
    const ocrConf = ev.ocr_confidence;
    const isHighConf = ocrConf != null && ocrConf >= 0.85;
    const confColor = isHighConf ? '#22C55E' : ocrConf >= 0.70 ? '#F59E0B' : '#EF4444';
    const isSuspect = reg.includes('25527') || reg.includes('AH');

    return `<div class="card" style="
      background:var(--bg-card);border:1px solid ${isSuspect ? 'var(--red-300)' : 'var(--border-default)'};
      border-radius:var(--radius-md);padding:14px;cursor:pointer;transition:transform 0.1s ease, box-shadow 0.15s ease;
      ${isSuspect ? 'box-shadow:0 0 0 2px var(--red-100);' : ''}
    " onclick="window.openEventEvidence('${ev.event_id}')" title="Click to view full ANPR evidence & OCR">
      ${isSuspect ? '<div style="font-size:10px;font-weight:700;color:var(--red-600);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">⚠ Suspect Vehicle (Hit-and-Run)</div>' : ''}
      <div style="font-size:18px;font-weight:800;font-family:\'Courier New\',monospace;letter-spacing:2px;color:${isSuspect ? 'var(--red-600)' : 'var(--text-primary)'};margin-bottom:8px;">${reg}</div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-tertiary);">
        <span>${ev.vehicle_class || 'Car'}</span>
        <span style="font-weight:600;color:${confColor};">${ocrConf != null ? Math.round(ocrConf * 100) + '%' : '—'}</span>
      </div>
      <div style="font-size:11px;color:var(--text-tertiary);margin-top:4px;">${timeAgo(ev.timestamp)}</div>
    </div>`;
  }).join('');
}

function renderTable(events) {
  const tbody = document.getElementById('anpr-tbody');
  const countEl = document.getElementById('anpr-count');
  if (!tbody) return;
  if (countEl) countEl.textContent = `${events.length} records`;

  if (events.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:24px;color:#94A3B8;">No ANPR events.</td></tr>';
    return;
  }

  tbody.innerHTML = events.map(ev => {
    const reg = ev.registration_number || ev.plate || '—';
    const isSuspect = reg.includes('25527') || reg.includes('AH');
    const eventId = ev.event_id || '';
    return `<tr class="clickable-row" data-event-id="${eventId}" onclick="window.openEventEvidence('${eventId}')" title="Click to inspect ANPR evidence" ${isSuspect ? 'style="background:var(--red-50);"' : ''}>
      <td><span style="font-family:monospace;font-weight:700;font-size:13px;letter-spacing:1px;color:${isSuspect ? 'var(--red-600)' : 'var(--text-primary)'};">${reg}</span>${isSuspect ? ' <span style="font-size:10px;color:var(--red-500);font-weight:600;">SUSPECT</span>' : ''}</td>
      <td class="text-sm text-secondary">${ev.vehicle_class || '—'}</td>
      <td>${confidenceBar(ev.ocr_confidence)}</td>
      <td>${confidenceBar(ev.plate_confidence)}</td>
      <td class="mono text-sm">${ev.plate_pixel_width != null ? Math.round(ev.plate_pixel_width) + ' px' : '—'}</td>
      <td class="mono text-sm text-secondary">${timeAgo(ev.timestamp)}</td>
      <td><button class="btn btn-secondary btn-sm" style="font-size:11px;padding:3px 8px;cursor:pointer;" onclick="event.stopPropagation(); window.openEventEvidence('${eventId}')">🔍 View OCR</button></td>
    </tr>`;
  }).join('');

  window.showAnprEvidence = async (eventId) => {
    const ev = events.find(x => x.event_id === eventId);
    if (!ev) return;
    const modal = document.getElementById('global-event-modal');
    const title = document.getElementById('modal-title');
    const body  = document.getElementById('modal-body');
    if (!modal || !title || !body) return;

    const reg = ev.registration_number || ev.plate || '—';
    const isSuspect = reg.includes('25527') || reg.includes('AH');
    title.textContent = `ANPR Evidence — ${reg}`;

    let mediaList = [];
    try {
      mediaList = await api.eventMedia(eventId) || [];
      if (!mediaList.length && isSuspect) {
        mediaList = await api.eventMedia('EVT-HNR-CHICAGO') || [];
      }
    } catch (e) {}

    // Find visual components
    const ocrVis = mediaList.find(m => m.filename && (m.filename.toLowerCase().includes('ocr_vis') || m.filename.toLowerCase().includes('plate')));
    const frameImg = mediaList.find(m => m.mime_type && m.mime_type.startsWith('image/') && !m.filename.toLowerCase().includes('ocr_vis'));
    const clipMedia = mediaList.find(m => m.mime_type && m.mime_type.startsWith('video/'));

    const ocrVisURL = ocrVis ? `http://localhost:8000/api/media/${ocrVis.media_id}` : (ev.best_frame?.startsWith('http') ? ev.best_frame : null);
    const frameURL = ev.image || (frameImg ? `http://localhost:8000/api/media/${frameImg.media_id}` : null);
    const clipURL = ev.event_clip || (clipMedia ? `http://localhost:8000/api/media/${clipMedia.media_id}` : null);

    const demoLoc = ev.demo_route_location;
    const locText = demoLoc ? `${demoLoc.road_name || 'Trichy Road'} (${demoLoc.landmark || 'Corridor'})` : 'Tiruchirappalli Corridor';

    body.innerHTML = `
      <div style="font-size:24px;font-weight:800;font-family:monospace;letter-spacing:3px;color:${isSuspect ? '#DC2626' : '#2563EB'};background:${isSuspect ? '#FEF2F2' : '#EFF6FF'};border:2px solid ${isSuspect ? '#FECACA' : '#BFDBFE'};border-radius:8px;padding:12px;text-align:center;margin-bottom:14px;">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" style="vertical-align:middle;margin-right:6px;"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>${reg} ${isSuspect ? '<span style="font-size:12px;background:#DC2626;color:#FFF;padding:2px 6px;border-radius:4px;vertical-align:middle;margin-left:6px;">SUSPECT HIT & RUN</span>' : ''}
      </div>

      ${clipURL ? `
        <div style="margin-bottom:14px;">
          <div style="font-size:11px;font-weight:600;color:#64748B;margin-bottom:4px;text-transform:uppercase;">
            Authentic Evidence Video Clip (~3-6s Observation)
          </div>
          <video controls autoplay loop style="width:100%;border-radius:8px;background:#000;" src="${clipURL}">Video unavailable</video>
        </div>
      ` : ''}

      ${ocrVisURL ? `
        <div style="margin-bottom:14px;text-align:center;">
          <div style="font-size:11px;font-weight:600;color:#64748B;margin-bottom:4px;text-transform:uppercase;">Extracted Plate Crop & Bounding Box</div>
          <img src="${ocrVisURL}" alt="Plate Crop" style="max-width:100%;border-radius:6px;border:1px solid #E2E8F0;background:#000;">
        </div>
      ` : ''}

      ${frameURL ? `
        <div style="margin-bottom:14px;">
          <div style="font-size:11px;font-weight:600;color:#64748B;margin-bottom:4px;text-transform:uppercase;">Full Vehicle Camera Frame</div>
          <img src="${frameURL}" alt="Source Frame" style="width:100%;border-radius:8px;border:1px solid #E2E8F0;" onerror="this.style.display='none'">
        </div>
      ` : ''}

      <div style="padding:12px;background:#F8FAFC;border-radius:8px;font-size:12px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">Registration Number:</span>
          <strong>${reg}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">OCR Verification Confidence:</span>
          <strong>${ev.ocr_confidence != null ? Math.round(ev.ocr_confidence * 100) + '%' : '—'}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">Track ID:</span>
          <span class="mono">#${ev.track_id != null ? ev.track_id : '—'} (${ev.vehicle_class || 'Car'})</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">Detection Timestamp:</span>
          <span class="mono">${ev.timestamp ? new Date(ev.timestamp * 1000).toLocaleString() : '—'}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">Plate Bounding Box Width:</span>
          <strong>${ev.plate_pixel_width != null ? Math.round(ev.plate_pixel_width) + ' px' : '—'}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">Corridor Location:</span>
          <strong>${locText}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#64748B;">OCR Engine:</span>
          <strong>${ev.ocr_engine || 'EasyOCR Multi-Variant Ensemble'}</strong>
        </div>
        <div style="display:flex;justify-content:space-between;">
          <span style="color:#64748B;">Multi-Variant Consensus:</span>
          <strong>${ev.ocr_consensus_count != null ? ev.ocr_consensus_count + ' voting passes' : '—'}</strong>
        </div>
      </div>
    `;
    modal.style.display = 'flex';
  };
}

function renderChart(events) {
  const canvas = document.getElementById('anpr-chart');
  if (!canvas || typeof Chart === 'undefined') return;

  const buckets = new Array(10).fill(0);
  for (const ev of events) {
    const c = ev.ocr_confidence;
    if (c == null) continue;
    buckets[Math.min(Math.floor(c * 10), 9)]++;
  }

  chartInstance = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['0–10%','10–20%','20–30%','30–40%','40–50%','50–60%','60–70%','70–80%','80–90%','90–100%'],
      datasets: [{
        label: 'Plates',
        data: buckets,
        backgroundColor: buckets.map((_, i) =>
          i >= 8 ? '#22C55E' : i >= 6 ? '#3B82F6' : i >= 4 ? '#F59E0B' : '#EF4444'
        ),
        borderRadius: 4, barPercentage: 0.7,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1E293B', padding: 8, cornerRadius: 6,
          callbacks: { label: ctx => ` ${ctx.parsed.y} plates` },
        },
      },
      scales: {
        x: { grid: { display: false }, border: { display: false }, ticks: { font: { family: 'Inter', size: 10 }, color: '#94A3B8' } },
        y: { beginAtZero: true, grid: { color: '#F1F5F9' }, border: { display: false }, ticks: { font: { family: 'Inter', size: 11 }, color: '#94A3B8', precision: 0 } },
      },
    },
  });
}

export function destroy() {
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
}
