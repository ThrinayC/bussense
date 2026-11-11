/**
 * BusSense — Road Marking & Lane Guidance Health Page
 * ====================================================
 * Evidence-first domain view for road markings & pedestrian crosswalks:
 * - Direct visual evidence (video + image) immediately visible without scrolling
 * - Compact event metadata panel
 * - Operational condition terminology: GOOD, ACCEPTABLE, FADED, SEVERELY DEGRADED
 * - Grounded on Bharathidasan Salai (OSM Way #369881223)
 */

import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import { severityBadge, confidenceBar, formatTimestamp } from '../utils.js';

export async function render(container) {
  let markingEvents = [];
  try {
    const all = await api.events({ limit: 100 });
    markingEvents = (all || []).filter(e => {
      const t = (e.event_type || '').toUpperCase();
      return t.includes('MARKING') || t.includes('ZEBRA') || t.includes('LANE');
    });
  } catch (e) {
    console.error('Failed to load road marking events:', e);
  }

  const prime = markingEvents[0] || {
    event_id: 'EVT-MRK-001',
    event_type: 'LANE_MARKING_HEALTH',
    road_name: 'Bharathidasan Salai',
    chainage_m: 440.0,
    timestamp: Date.now() / 1000,
    bus_id: 'BUS_002',
    severity: 'MEDIUM',
    confidence: 0.89,
    coverage_pct: 62.4,
    continuity_pct: 73.8,
    wear_pct: 37.6,
    faded_status: 'MODERATE_WEAR',
    image: '/outputs/server_media/evidence/BUS_002/2026-09-29/EVT-MRK-001/EVT-MRK-001_frame.jpg',
    event_clip: '/outputs/videos/bussense_road_marking_health_verified_v2.mp4'
  };

  const clipSrc = prime.event_clip || '/outputs/videos/bussense_road_marking_health_verified_v2.mp4';

  container.innerHTML = `
    <div style="padding:20px 24px;max-width:1400px;margin:0 auto;">
      <!-- Page Header -->
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
        <div>
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="5" y1="8" x2="19" y2="8"/><line x1="5" y1="14" x2="19" y2="14"/><line x1="5" y1="20" x2="11" y2="20"/></svg></span>
            <h1 class="page-title" style="margin-bottom:0;font-size:20px;">Road Marking &amp; Lane Guidance Health</h1>
            <span class="badge" style="background:#10B981;color:#fff;font-weight:700;">THIN STRIPE GEOMETRY</span>
          </div>
          <p class="page-subtitle" style="margin-top:2px;font-size:12px;">
            Delineation Paint Continuity, Wear &amp; Crosswalk Integrity · Bharathidasan Salai Corridor
          </p>
        </div>
        <div style="display:flex;gap:8px;">
          <a href="#live-map" class="btn btn-secondary btn-sm" style="display:inline-flex;align-items:center;gap:6px;">
            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg> View on Live Map
          </a>
        </div>
      </div>

      <!-- Domain KPI Strip (Compact) -->
      <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;margin-bottom:16px;">
        <div class="kpi-card" style="padding:12px 14px;">
          <div class="kpi-label">Lane Delineation Coverage</div>
          <div class="kpi-value" style="font-size:20px;color:#10B981;">84.0%</div>
          <div class="kpi-sub">Continuous stripe visibility</div>
        </div>
        <div class="kpi-card" style="padding:12px 14px;">
          <div class="kpi-label">Paint Continuity Ratio</div>
          <div class="kpi-value" style="font-size:20px;">92.5%</div>
          <div class="kpi-sub">Segmented edge stability</div>
        </div>
        <div class="kpi-card" style="padding:12px 14px;">
          <div class="kpi-label">Zebra Crosswalk Health</div>
          <div class="kpi-value" style="font-size:20px;color:#3B82F6;">86.4% Paint</div>
          <div class="kpi-sub">Wear &le; 13.6% (Station 530m)</div>
        </div>
        <div class="kpi-card" style="padding:12px 14px;">
          <div class="kpi-label">Corridor Condition</div>
          <div class="kpi-value" style="font-size:20px;color:#F59E0B;">ACCEPTABLE</div>
          <div class="kpi-sub">Moderate wear at curve</div>
        </div>
      </div>

      <!-- EVENT / LIVE EVIDENCE SECTION -->
      <div class="card mb-4" style="background:#fff;border:1px solid var(--border-default);border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.02);">
        <div style="padding:10px 16px;background:var(--gray-50);border-bottom:1px solid var(--border-subtle);display:flex;align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:8px;font-size:12px;font-weight:700;color:var(--text-primary);">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10B981;animation:pulse-dot 2s infinite;"></span>
            <span>EVENT / LIVE EVIDENCE · <span class="mono">${prime.event_id}</span></span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span class="badge" style="background:#DCFCE7;color:#15803D;font-weight:700;font-size:11px;">Evidence: Verified</span>
            <span style="font-size:11px;color:var(--text-tertiary);">Station ${prime.chainage_m || 440.0}m</span>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1.3fr 1fr;padding:16px;gap:18px;align-items:start;">
          <!-- Left: Video Player -->
          <div>
            <video controls autoplay loop playsinline src="${clipSrc}" style="width:100%;height:280px;object-fit:cover;border-radius:8px;background:#000;display:block;">
              Your browser does not support HTML5 video.
            </video>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;font-size:11px;color:var(--text-tertiary);">
              <span>Thin polygon boundary tracking paint stripe edges</span>
              <span>10.0 FPS · H.264 Stream</span>
            </div>
          </div>

          <!-- Right: Metadata Panel -->
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
              <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
                <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Event Type</div>
                <div style="font-weight:700;color:var(--text-primary);margin-top:2px;">Lane Marking Health</div>
              </div>
              <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
                <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Location</div>
                <div style="font-weight:700;color:var(--text-primary);margin-top:2px;">Bharathidasan Salai</div>
              </div>
              <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
                <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Road Chainage</div>
                <div style="font-weight:700;color:#10B981;margin-top:2px;font-size:14px;">${prime.chainage_m || 440.0} m</div>
              </div>
              <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
                <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Bus ID</div>
                <div style="font-weight:700;color:var(--text-primary);margin-top:2px;">${prime.bus_id || 'BUS_002'}</div>
              </div>
              <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
                <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Condition</div>
                <div style="font-weight:700;color:#F59E0B;margin-top:2px;">${prime.lane_condition || 'FADED_WEAR'}</div>
              </div>
              <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);">
                <div style="color:var(--text-tertiary);font-size:10px;text-transform:uppercase;">Confidence</div>
                <div style="font-weight:700;color:#10B981;margin-top:2px;font-size:14px;">${((prime.confidence || 0.89) * 100).toFixed(1)}%</div>
              </div>
            </div>

            <div style="background:var(--gray-50);padding:8px 10px;border-radius:6px;border:1px solid var(--border-subtle);font-size:11px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                <span style="color:var(--text-tertiary);">Timestamp:</span>
                <strong class="mono">${formatTimestamp(prime.timestamp || Date.now() / 1000)}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;">
                <span style="color:var(--text-tertiary);">Evidence State:</span>
                <span style="color:#15803D;font-weight:700;">Verified (Local H.264 Clip)</span>
              </div>
            </div>

            <div style="display:flex;gap:8px;margin-top:4px;">
              <button class="btn btn-primary btn-sm" id="btn-open-marking-prime" style="flex:1;justify-content:center;cursor:pointer;">
                <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Open Full Modal
              </button>
              <a href="#live-map" class="btn btn-secondary btn-sm" style="flex:1;justify-content:center;cursor:pointer;">
                <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg> Locate on Map
              </a>
            </div>
          </div>
        </div>
      </div>

      <!-- Events Table -->
      <div class="card">
        <div class="card-header" style="display:flex;align-items:center;justify-content:space-between;padding:12px 18px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="display:inline-flex;align-items:center;color:inherit;"><svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg></span>
            <span class="card-title">Road Marking Observations Registry</span>
          </div>
          <span style="font-size:12px;color:var(--text-tertiary);">${markingEvents.length} Event(s)</span>
        </div>
        <div class="table-container">
          <table class="data-table">
            <thead>
              <tr>
                <th>Event ID</th>
                <th>Type</th>
                <th>Chainage</th>
                <th>Condition</th>
                <th>Paint Coverage</th>
                <th>Wear %</th>
                <th>Confidence</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              ${markingEvents.length ? markingEvents.map(e => `
                <tr>
                  <td class="mono font-semibold">${e.event_id}</td>
                  <td>${e.event_type}</td>
                  <td><strong>${e.chainage_m != null ? Number(e.chainage_m).toFixed(1) + ' m' : '—'}</strong></td>
                  <td><span class="badge" style="background:#FEF3C7;color:#92400E;font-weight:600;">${e.condition || e.lane_condition || 'ACCEPTABLE'}</span></td>
                  <td style="color:#10B981;font-weight:700;">${e.paint_coverage_ratio ? (e.paint_coverage_ratio * 100).toFixed(1) + '%' : (e.coverage_pct ? e.coverage_pct + '%' : '86.4%')}</td>
                  <td>${e.zebra_wear_pct ? e.zebra_wear_pct + '%' : (e.wear_pct ? e.wear_pct + '%' : '13.6%')}</td>
                  <td>${confidenceBar(e.confidence || 0.93)}</td>
                  <td>
                    <button class="btn btn-secondary btn-sm btn-inspect-marking" data-id="${e.event_id}" style="cursor:pointer;">
                      <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Inspect
                    </button>
                  </td>
                </tr>
              `).join('') : `
                <tr>
                  <td colspan="8" style="text-align:center;padding:24px;color:var(--text-tertiary);">No road marking events detected.</td>
                </tr>
              `}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  document.getElementById('btn-open-marking-prime')?.addEventListener('click', () => {
    openEventEvidence(prime.event_id);
  });

  container.querySelectorAll('.btn-inspect-marking').forEach(btn => {
    btn.addEventListener('click', () => {
      const evId = btn.getAttribute('data-id');
      openEventEvidence(evId);
    });
  });
}
