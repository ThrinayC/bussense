/**
 * BusSense — Road Health Page
 * Pothole & crack defect monitoring with KPIs, filterable table, Leaflet map,
 * and integrated CSR / Road Repair Funding module.
 */
import api from '../api.js';
import { openEventEvidence } from '../event-modal.js';
import { calculateRepairCost, formatINR, matchCsrPartners } from '../csr-cost-model.js';
import {
  timeAgo,
  severityBadge,
  severityColor,
  eventTypeBadge,
  confidenceBar,
  formatNumber,
  formatTimestamp,
} from '../utils.js';

let mapInstance = null;
let markersLayer = null;
let markersByEventId = {};
let allDefects = [];
let csrPartners = [];
let csrRequests = [];
let selectedCsrPotholeId = null;
let selectedCsrCompanyId = null;
let activeTab = 'all';

// ---------------------------------------------------------------------------
// Road condition badge
// ---------------------------------------------------------------------------
function roadConditionBadge(condition) {
  if (!condition) return '—';
  const c = condition.toLowerCase();
  const styles = {
    good:      'background:var(--green-100);color:var(--green-600)',
    fair:      'background:var(--amber-100);color:var(--amber-600)',
    poor:      'background:#FEE2E2;color:#B91C1C',
    very_poor: 'background:var(--red-100);color:var(--red-600)',
  };
  const style = styles[c] || 'background:var(--gray-100);color:var(--gray-600)';
  const label = condition.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  return `<span class="badge-severity" style="${style}">${label}</span>`;
}

// ---------------------------------------------------------------------------
// CSR Request status badge
// ---------------------------------------------------------------------------
function csrStatusBadge(status) {
  if (!status) return '<span style="color:var(--text-tertiary);font-size:11px;">Not Requested</span>';
  const s = status.toUpperCase();
  const classMap = {
    'REQUESTED': 'requested',
    'UNDER REVIEW': 'under-review',
    'APPROVED': 'approved',
    'REJECTED': 'rejected',
    'FUNDED': 'funded',
    'COMPLETED': 'completed'
  };
  const cls = classMap[s] || 'requested';
  return `<span class="badge-csr-status ${cls}">${s}</span>`;
}

// ---------------------------------------------------------------------------
// Compute KPIs from defect list
// ---------------------------------------------------------------------------
function computeKpis(defects) {
  let potholes = 0;
  let cracks = 0;
  let depthSum = 0;
  let depthCount = 0;
  let worstCondition = 'Good';
  const conditionRank = { good: 0, fair: 1, poor: 2, very_poor: 3 };
  let worstRank = -1;

  for (const d of defects) {
    const type = (d.event_type || d.defect_type || '').toUpperCase();
    if (type === 'POTHOLE') {
      potholes++;
      if (d.depth_cm != null) { depthSum += d.depth_cm; depthCount++; }
    } else if (type === 'CRACK') {
      cracks++;
    }
    const rc = (d.road_condition || '').toLowerCase();
    if (conditionRank[rc] != null && conditionRank[rc] > worstRank) {
      worstRank = conditionRank[rc];
      worstCondition = rc.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
  }

  return {
    potholes,
    cracks,
    avgDepth: depthCount ? (depthSum / depthCount).toFixed(1) : '—',
    worstCondition,
  };
}

// ---------------------------------------------------------------------------
// Compute CSR specific KPIs
// ---------------------------------------------------------------------------
function computeCsrKpis() {
  const totalPool = csrPartners.reduce((acc, p) => acc + (Number(p.available_budget_inr) || 0), 0);
  
  // Calculate backlog for all potholes
  let backlogCost = 0;
  for (const d of allDefects) {
    const est = calculateRepairCost(d);
    if (est) backlogCost += est.total_cost;
  }

  const activeReqsCount = csrRequests.length;
  const totalRequested = csrRequests.reduce((acc, r) => acc + (Number(r.requested_amount_inr) || 0), 0);

  return {
    totalPool,
    backlogCost,
    activeReqsCount,
    totalRequested
  };
}

// ---------------------------------------------------------------------------
// Render KPI row
// ---------------------------------------------------------------------------
function renderKpis(container) {
  if (!container) return;

  if (activeTab === 'csr') {
    const csrKpis = computeCsrKpis();
    container.innerHTML = `
      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#ECFDF5;color:#059669;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg></span> Available CSR Pool</span>
          <span class="kpi-badge kpi-badge-teal">CORPORATE POOL</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#059669;font-size:24px;">${formatINR(csrKpis.totalPool)}</span>
          <span class="kpi-pill kpi-pill-green">Verified</span>
        </div>
        <div class="kpi-context">${csrPartners.length} active corporate sponsors</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="8" x2="19" y2="8"/><line x1="5" y1="14" x2="19" y2="14"/><line x1="5" y1="20" x2="11" y2="20"/></svg></span> Repair Backlog</span>
          <span class="kpi-badge kpi-badge-amber">ESTIMATED</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#D97706;font-size:24px;">${formatINR(csrKpis.backlogCost)}</span>
          <span class="kpi-pill kpi-pill-amber">IRC Backlog</span>
        </div>
        <div class="kpi-context">Engineering cost across all detected defects</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#EFF6FF;color:#2563EB;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg></span> Active CSR Requests</span>
          <span class="kpi-badge kpi-badge-blue">WORKFLOW</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value">${csrKpis.activeReqsCount}</span>
          <span class="kpi-pill kpi-pill-blue">Submitted</span>
        </div>
        <div class="kpi-context">Corporate sponsorship proposals</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#ECFDF5;color:#059669;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg></span> Committed Funding</span>
          <span class="kpi-badge kpi-badge-teal">ALLOCATED</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#059669;font-size:24px;">${formatINR(csrKpis.totalRequested)}</span>
          <span class="kpi-pill kpi-pill-green">Allocated</span>
        </div>
        <div class="kpi-context">Dedicated to rapid road patching</div>
      </div>
    `;
  } else {
    const kpis = computeKpis(allDefects);
    container.innerHTML = `
      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#FFFBEB;color:#D97706;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12a4 4 0 0 0 8 0"/></svg></span> Total Potholes</span>
          <span class="kpi-badge kpi-badge-amber">DEFECT METROLOGY</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value">${formatNumber(kpis.potholes)}</span>
          <span class="kpi-pill kpi-pill-amber">Stereo Depth</span>
        </div>
        <div class="kpi-context">Volumetric &amp; depth profiling</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#F1F5F9;color:#475569;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></span> Total Cracks</span>
          <span class="kpi-badge kpi-badge-blue">SURFACE INTEG.</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value">${formatNumber(kpis.cracks)}</span>
          <span class="kpi-pill kpi-pill-blue">Active</span>
        </div>
        <div class="kpi-context">Longitudinal &amp; alligator fissures</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#ECFDF5;color:#059669;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="8" x2="19" y2="8"/><line x1="5" y1="14" x2="19" y2="14"/><line x1="5" y1="20" x2="11" y2="20"/></svg></span> Mean Pothole Depth</span>
          <span class="kpi-badge kpi-badge-teal">SEVERITY</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value">${kpis.avgDepth}<span style="font-size:14px;font-weight:500;color:var(--text-secondary);margin-left:2px">cm</span></span>
          <span class="kpi-pill ${Number(kpis.avgDepth) > 5 ? 'kpi-pill-red' : 'kpi-pill-amber'}">Threshold Met</span>
        </div>
        <div class="kpi-context">Photogrammetric laser &amp; stereo profiling</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-card-header">
          <span class="kpi-label"><span class="kpi-icon-badge" style="background:#FEF2F2;color:#DC2626;"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></span> Worst Condition</span>
          <span class="kpi-badge kpi-badge-red">CONDITION</span>
        </div>
        <div class="kpi-value-row">
          <span class="kpi-value" style="font-size:22px;color:var(--red-600);">${kpis.worstCondition}</span>
          <span class="kpi-pill kpi-pill-red">Intervention</span>
        </div>
        <div class="kpi-context">Corridor segment requires patch intervention</div>
      </div>
    `;
  }
}

// ---------------------------------------------------------------------------
// Filter defects by active tab
// ---------------------------------------------------------------------------
function filterDefects(defects, tab) {
  if (tab === 'all') return defects;
  if (tab === 'csr') return defects;
  return defects.filter(d => {
    const type = (d.event_type || d.defect_type || '').toUpperCase();
    return type === tab.toUpperCase();
  });
}

// ---------------------------------------------------------------------------
// Render defects table body
// ---------------------------------------------------------------------------
function renderTableBody(tbody, defects) {
  if (!defects.length) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;padding:var(--space-8);color:var(--text-tertiary)">No defects found</td></tr>`;
    return;
  }

  tbody.innerHTML = defects.map(d => {
    const type = (d.event_type || d.defect_type || '').toUpperCase();
    const isPothole = type === 'POTHOLE';
    const depthOrLength = isPothole
      ? (d.depth_cm != null ? `${Number(d.depth_cm).toFixed(1)} cm` : '—')
      : (d.length_m != null ? `${Number(d.length_m).toFixed(2)} m` : '—');
    const areaOrBranches = isPothole
      ? (d.area_m2 != null ? `${Number(d.area_m2).toFixed(3)} m²` : '—')
      : (d.branches != null ? `${d.branches} br` : '—');
    const eventId = d.event_id || d.id || '—';
    const truncatedId = eventId.length > 12 ? eventId.slice(0, 12) + '…' : eventId;
    const hasMedia = Boolean(d.event_clip || d.image);
    
    // Check CSR request status
    const req = csrRequests.find(r => r.pothole_event_id === eventId);
    const est = calculateRepairCost(d);

    const mediaBadge = hasMedia
      ? `<div style="display:flex;align-items:center;gap:4px;">
           <button class="btn btn-secondary btn-sm" style="font-size:11px;padding:3px 7px;cursor:pointer;" onclick="event.stopPropagation(); window.openEventEvidence('${eventId}')"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> ${d.event_clip ? 'Clip' : 'Frame'}</button>
           <button class="btn btn-primary btn-sm" style="font-size:11px;padding:3px 7px;cursor:pointer;background:#059669;border-color:#059669;" onclick="event.stopPropagation(); window.openCsrRequestFlow('${eventId}')" title="Request CSR sponsorship for this defect"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg> CSR</button>
         </div>`
      : `<span class="text-tertiary">—</span>`;

    return `<tr class="clickable-row" data-event-id="${eventId}" onclick="window.openEventEvidence('${eventId}')" title="Click to inspect defect evidence & metrology">
      <td><span class="mono truncate" style="max-width:105px;display:inline-block;font-weight:600;color:var(--blue-600);" title="${eventId}">${truncatedId}</span></td>
      <td>${eventTypeBadge(type)}</td>
      <td>${severityBadge(d.severity)}</td>
      <td><strong style="color:var(--text-primary)">${depthOrLength}</strong></td>
      <td>${areaOrBranches}</td>
      <td><strong style="color:#059669;font-family:Inter,sans-serif;font-size:12px;">${est ? formatINR(est.total_cost) : '—'}</strong></td>
      <td>${csrStatusBadge(req?.status)}</td>
      <td>${roadConditionBadge(d.road_condition)}</td>
      <td class="text-secondary">${timeAgo(d.timestamp)}</td>
      <td>${mediaBadge}</td>
    </tr>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Initialize or update Leaflet map markers
// ---------------------------------------------------------------------------
function updateMapMarkers(defects) {
  if (!mapInstance || !markersLayer) return;
  markersLayer.clearLayers();
  markersByEventId = {};

  const validLatLngs = [];

  for (const d of defects) {
    const lat = d.demo_route_location?.lat ?? d.gps_lat ?? d.latitude ?? d.lat;
    const lon = d.demo_route_location?.lon ?? d.gps_lon ?? d.longitude ?? d.lon ?? d.lng;
    if (lat == null || lon == null) continue;

    validLatLngs.push([lat, lon]);

    const type = (d.event_type || d.defect_type || '').toUpperCase();
    const isPothole = type === 'POTHOLE';
    const color = isPothole ? '#F59E0B' : '#64748B';
    const eventId = d.event_id || d.id;

    // Check CSR request status
    const req = csrRequests.find(r => r.pothole_event_id === eventId);
    const est = calculateRepairCost(d);

    // Size proportional to severity
    const sevSizes = { CRITICAL: 12, HIGH: 10, MEDIUM: 8, LOW: 6 };
    const radius = sevSizes[(d.severity || '').toUpperCase()] || 7;

    const marker = L.circleMarker([lat, lon], {
      radius,
      color: req ? '#059669' : color,
      fillColor: req ? '#10B981' : color,
      fillOpacity: 0.75,
      weight: req ? 2.5 : 1.5,
    });

    if (eventId) {
      markersByEventId[eventId] = marker;
    }

    // Direct click opens the unified event evidence modal
    marker.on('click', () => {
      window.openEventEvidence(eventId, { cachedEvent: d });
    });

    // Tooltip for instant hover preview
    const locName = d.demo_route_location?.landmark || d.demo_route_location?.road_name || 'Trichy Corridor';
    const metricText = isPothole
      ? `Depth: ${d.depth_cm != null ? d.depth_cm + ' cm' : '—'} · Area: ${d.area_m2 != null ? d.area_m2 + ' m²' : '—'}`
      : `Length: ${d.length_m != null ? d.length_m + ' m' : '—'}`;
    const csrTag = req
      ? `<span style="display:inline-block;margin-top:2px;font-weight:700;color:#059669;"><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg> CSR: ${req.status} (${req.company_name?.split(' ')[0] || 'Sponsor'})</span>`
      : `<span style="display:inline-block;margin-top:2px;color:#64748B;">CSR: Not Requested · Est: <strong>${formatINR(est?.total_cost)}</strong></span>`;

    marker.bindTooltip(`
      <div style="font-family:Inter,sans-serif;font-size:12px;padding:2px 4px;line-height:1.4;">
        <strong style="color:#0F172A">${isPothole ? '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;"><circle cx="12" cy="12" r="9"/><path d="M8 12a4 4 0 0 0 8 0"/></svg> Pothole Metrology' : '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;"><polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Crack Network'}</strong> · <span style="font-weight:700;color:${color}">${d.severity || 'DEFECT'}</span><br>
        <span style="color:#475569">${metricText} · ${locName}</span><br>
        ${csrTag}<br>
        <span style="font-size:10px;font-weight:600;color:#2563EB;">Click to open full evidence modal</span>
      </div>
    `, { direction: 'top', offset: [0, -6] });

    markersLayer.addLayer(marker);
  }

  // Robust bounds fitting handling 0, 1, or multiple markers without crashing
  if (validLatLngs.length === 1) {
    mapInstance.setView(validLatLngs[0], 16);
  } else if (validLatLngs.length > 1) {
    try {
      const bounds = L.latLngBounds(validLatLngs).pad(0.15);
      if (bounds && bounds.isValid && bounds.isValid()) {
        mapInstance.fitBounds(bounds, { maxZoom: 16 });
      }
    } catch (e) {
      console.warn('fitBounds error:', e);
    }
  }
}

// ---------------------------------------------------------------------------
// Render CSR / Road Repair Funding Sub-Page
// ---------------------------------------------------------------------------
function renderCsrModule(container) {
  if (!container) return;

  // Selected defect
  const currentDefect = allDefects.find(d => (d.event_id || d.id) === selectedCsrPotholeId)
    || allDefects.find(d => (d.event_type || '').toUpperCase() === 'POTHOLE')
    || allDefects[0];

  if (currentDefect) {
    selectedCsrPotholeId = currentDefect.event_id || currentDefect.id;
  }

  // Cost calculation for selected defect
  const est = calculateRepairCost(currentDefect) || {
    total_cost: 5850,
    material_cost: 1016,
    labour_cost: 1598,
    equipment_cost: 1500,
    traffic_management: 1200,
    contingency: 531,
    severity: 'HIGH',
    disclaimer: 'Engineering estimate based on BusSense stereo metrology.'
  };

  // Partners matching
  const matchedPartners = matchCsrPartners(currentDefect, csrPartners);
  
  if (!selectedCsrCompanyId && matchedPartners.length) {
    selectedCsrCompanyId = matchedPartners[0].company_id;
  }
  const selectedPartner = csrPartners.find(p => p.company_id === selectedCsrCompanyId) || matchedPartners[0];

  // Build Pothole Options for Dropdown
  const potholeOptions = allDefects
    .filter(d => (d.event_type || '').toUpperCase() === 'POTHOLE' || (d.event_type || '').toUpperCase() === 'CRACK')
    .map(d => {
      const id = d.event_id || d.id;
      const road = d.demo_route_location?.road_name || d.demo_route_location?.landmark || 'Trichy Corridor';
      const isPot = (d.event_type || '').toUpperCase() === 'POTHOLE';
      const meta = isPot ? `${d.depth_cm}cm depth · ${d.area_m2}m²` : `${d.length_m}m crack`;
      const isSel = id === selectedCsrPotholeId ? 'selected' : '';
      return `<option value="${id}" ${isSel}>${id} — ${road} (${d.severity} · ${meta})</option>`;
    }).join('');

  container.innerHTML = `
    <!-- Top Wizard Card Grid -->
    <div class="grid-2 mb-4" style="gap:20px;">

      <!-- LEFT: Selected Defect & Engineering Cost Estimate -->
      <div class="card" style="border-top:3px solid #059669;">
        <div class="card-header" style="flex-direction:column;align-items:stretch;gap:10px;">
          <div class="flex items-center justify-between">
            <span class="card-title" style="display:flex;align-items:center;gap:6px;">
              <span style="display:inline-flex;align-items:center;"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12a4 4 0 0 0 8 0"/></svg></span> 1. Road Defect Metrology &amp; Cost
            </span>
            <span class="badge-demo">VERIFIED METROLOGY</span>
          </div>

          <div>
            <label style="font-size:11px;font-weight:600;color:var(--text-secondary);margin-bottom:4px;display:block;">
              Select Pothole / Defect to Sponsor:
            </label>
            <select id="csr-pothole-picker" class="form-input" style="width:100%;font-size:12px;padding:7px 10px;border-radius:6px;border:1px solid var(--border-default);background:var(--bg-main);font-weight:500;">
              ${potholeOptions}
            </select>
          </div>
        </div>

        <div class="card-body">
          <!-- Real Full Camera Frame -->
          ${currentDefect?.image ? `
            <div style="margin-bottom:14px;position:relative;background:#0F172A;border-radius:8px;overflow:hidden;border:1px solid var(--border-default);">
              <div style="position:absolute;top:8px;left:8px;background:rgba(15,23,42,0.85);color:#38BDF8;font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;letter-spacing:0.5px;backdrop-filter:blur(4px);">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg> FULL CAMERA FRAME (NATIVE STEREO METROLOGY)
              </div>
              <img src="${currentDefect.image}" alt="Pothole Full Frame" style="width:100%;max-height:220px;object-fit:cover;display:block;" onerror="this.style.display='none'">
            </div>
          ` : ''}

          <!-- Defect Details Grid -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
            <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:8px 12px;">
              <div style="font-size:10px;color:#92400E;font-weight:600;text-transform:uppercase;">Measured Depth</div>
              <div style="font-size:18px;font-weight:800;color:#B45309;">${currentDefect?.depth_cm != null ? currentDefect.depth_cm + ' cm' : '—'}</div>
              <div style="font-size:10px;color:#B45309;">Stereo Metrology</div>
            </div>
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:8px 12px;">
              <div style="font-size:10px;color:#475569;font-weight:600;text-transform:uppercase;">Surface Area</div>
              <div style="font-size:18px;font-weight:800;color:#1E293B;">${currentDefect?.area_m2 != null ? currentDefect.area_m2 + ' m²' : '—'}</div>
              <div style="font-size:10px;color:#64748B;">Cavity Footprint</div>
            </div>
          </div>

          <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;margin-bottom:14px;font-size:12px;">
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span style="color:#64748B;">Corridor Location:</span>
              <strong style="color:#1E293B;">${currentDefect?.demo_route_location?.road_name || 'Trichy Road'} (${currentDefect?.demo_route_location?.landmark || 'Active Route'})</strong>
            </div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span style="color:#64748B;">Sensing Fleet Unit:</span>
              <span class="mono" style="font-weight:600;">${currentDefect?.bus_id || 'BUS-01'}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span style="color:#64748B;">Defect Severity:</span>
              <span>${severityBadge(currentDefect?.severity)}</span>
            </div>
          </div>

          <!-- ESTIMATED REPAIR COST CARD -->
          <div style="background:#F0FDF4;border:1.5px solid #86EFAC;border-radius:10px;padding:14px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
              <div style="font-size:11px;font-weight:700;color:#166534;text-transform:uppercase;letter-spacing:0.5px;">
                💰 ESTIMATED REPAIR COST
              </div>
              <span style="font-size:10px;background:#DCFCE7;color:#15803D;padding:2px 8px;border-radius:12px;font-weight:600;">
                ENGINEERING MODEL
              </span>
            </div>

            <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:8px;">
              <div style="font-size:28px;font-weight:900;color:#15803D;font-family:Inter,sans-serif;letter-spacing:-0.5px;">
                ${formatINR(est.total_cost)}
              </div>
              <button type="button" id="btn-toggle-csr-breakdown" style="background:none;border:none;color:#15803D;font-size:11px;font-weight:600;cursor:pointer;text-decoration:underline;">
                Cost Breakdown ▼
              </button>
            </div>

            <!-- Expandable Civil Breakdown -->
            <div id="csr-cost-breakdown-details" style="display:none;padding-top:10px;border-top:1px dashed #86EFAC;font-size:11px;color:#166534;">
              <div style="display:flex;justify-content:space-between;padding:3px 0;">
                <span>Polymer Cold-Mix Asphalt &amp; Tack Coat:</span>
                <strong class="mono">${formatINR(est.material_cost)}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:3px 0;">
                <span>Compaction Crew &amp; Surface Prep Labour:</span>
                <strong class="mono">${formatINR(est.labour_cost)}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:3px 0;">
                <span>Heavy Equipment Allowance (Compactor/Roller):</span>
                <strong class="mono">${formatINR(est.equipment_cost)}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:3px 0;">
                <span>Transit Corridor Traffic Channeling:</span>
                <strong class="mono">${formatINR(est.traffic_management)}</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:3px 0;">
                <span>Contingency &amp; Site QA (10%):</span>
                <strong class="mono">${formatINR(est.contingency)}</strong>
              </div>
            </div>

            <div style="font-size:10px;color:#166534;opacity:0.85;margin-top:6px;line-height:1.3;">
              <em>Estimate based on BusSense measured area/depth + configurable repair assumptions. Not an official municipal quotation.</em>
            </div>
          </div>
        </div>
      </div>

      <!-- RIGHT: Select CSR Partner & Request Form -->
      <div class="card" style="border-top:3px solid #3B82F6;">
        <div class="card-header">
          <span class="card-title" style="display:flex;align-items:center;gap:6px;">
            <span>🤝</span> 2. Choose CSR Partner &amp; Submit Request
          </span>
          <span style="font-size:11px;color:var(--text-secondary);">${matchedPartners.length} Available Sponsors</span>
        </div>

        <div class="card-body">
          <label style="font-size:11px;font-weight:600;color:var(--text-secondary);margin-bottom:8px;display:block;">
            Select Corporate Sponsor (Filtered for Tiruchirappalli Corridor):
          </label>

          <!-- Partner Cards Grid -->
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
            ${matchedPartners.map(p => {
              const isSelected = p.company_id === selectedCsrCompanyId;
              const availPct = Math.round((p.available_budget_inr / p.csr_budget_inr) * 100);
              return `
                <div class="csr-partner-card ${isSelected ? 'selected' : ''}" data-company-id="${p.company_id}">
                  <div>
                    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:6px;margin-bottom:6px;">
                      <div style="display:flex;align-items:center;gap:6px;">
                        <span style="font-size:20px;">${p.logo || '🏢'}</span>
                        <div style="font-size:12px;font-weight:700;color:#1E293B;line-height:1.2;">${p.company_name}</div>
                      </div>
                      ${isSelected ? '<span style="font-size:10px;background:#059669;color:#FFF;padding:2px 6px;border-radius:10px;font-weight:700;">ACTIVE</span>' : ''}
                    </div>

                    <div style="margin-bottom:6px;">
                      <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-secondary);">
                        <span>Available Allocation:</span>
                        <strong style="color:#059669;">${formatINR(p.available_budget_inr)}</strong>
                      </div>
                      <div class="csr-budget-bar">
                        <div class="csr-budget-progress" style="width:${availPct}%;"></div>
                      </div>
                      <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--text-tertiary);">
                        <span>Total CSR: ${formatINR(p.csr_budget_inr)}</span>
                        <span>${availPct}% Uncommitted</span>
                      </div>
                    </div>

                    <!-- Matching Tags -->
                    <div style="margin-bottom:8px;display:flex;flex-direction:column;gap:3px;">
                      ${(p.match_reasons || []).slice(0, 2).map(r => `
                        <span style="font-size:9.5px;color:#065F46;background:#ECFDF5;padding:1px 5px;border-radius:3px;display:inline-block;">✓ ${r}</span>
                      `).join('')}
                    </div>
                  </div>

                  <div style="padding-top:8px;border-top:1px solid var(--border-subtle);display:flex;align-items:center;justify-content:space-between;font-size:10px;color:var(--text-secondary);">
                    <span>${p.contact_name?.split(' ')[0] || 'Liaison'}</span>
                    <button class="btn ${isSelected ? 'btn-primary' : 'btn-secondary'} btn-sm" style="font-size:10px;padding:3px 8px;pointer-events:none;">
                      ${isSelected ? 'Selected ✓' : 'Select'}
                    </button>
                  </div>
                </div>
              `;
            }).join('')}
          </div>

          <!-- Sponsorship Request Form -->
          <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:14px;">
            <div style="font-size:12px;font-weight:700;color:#1E293B;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;">
              <span>Sponsorship Request Details</span>
              <span class="badge-demo">DEMO WORKFLOW</span>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">
              <div>
                <label style="font-size:11px;color:var(--text-secondary);display:block;margin-bottom:3px;">Selected Partner:</label>
                <div style="font-size:12px;font-weight:600;color:#1E293B;" id="csr-form-partner-name">${selectedPartner?.company_name || '—'}</div>
              </div>
              <div>
                <label style="font-size:11px;color:var(--text-secondary);display:block;margin-bottom:3px;">Target Defect:</label>
                <div style="font-size:12px;font-weight:600;color:#1E293B;">${currentDefect?.event_id || '—'} (${currentDefect?.demo_route_location?.road_name || 'Corridor'})</div>
              </div>
            </div>

            <div style="margin-bottom:10px;">
              <label style="font-size:11px;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:3px;">
                Requested Support Amount (INR):
              </label>
              <div style="display:flex;align-items:center;gap:8px;">
                <input type="number" id="csr-requested-amount-input" class="form-input" style="font-size:14px;font-weight:700;color:#15803D;padding:7px 10px;border-radius:6px;border:1px solid var(--border-default);background:#FFF;width:160px;" value="${est.total_cost}">
                <span style="font-size:11px;color:var(--text-secondary);">Defaulted to calculated engineering estimate</span>
              </div>
              <div id="csr-budget-warning" style="display:none;font-size:11px;color:#DC2626;margin-top:4px;font-weight:600;">
                ⚠️ Warning: Requested amount exceeds this partner's uncommitted allocation!
              </div>
            </div>

            <div style="margin-bottom:12px;">
              <label style="font-size:11px;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:3px;">
                Operator Priority Notes &amp; Civic Justification:
              </label>
              <textarea id="csr-operator-notes" rows="2" style="width:100%;font-size:12px;padding:8px 10px;border-radius:6px;border:1px solid var(--border-default);background:#FFF;font-family:Inter,sans-serif;" placeholder="e.g. Critical arterial roadway repair request — heavy transit corridor with frequent commuter buses."></textarea>
            </div>

            <div style="display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:10px;color:var(--text-tertiary);">Status will initialize as <strong>REQUESTED</strong></span>
              <button class="btn btn-primary" id="btn-submit-csr-request" style="background:#059669;border-color:#059669;padding:8px 16px;font-size:12px;font-weight:700;display:inline-flex;align-items:center;gap:6px;cursor:pointer;">
                <span>🚀</span> SEND CSR REQUEST
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom: CSR Request History Table -->
    <div class="card">
      <div class="card-header" style="flex-direction:row;align-items:center;justify-content:space-between;">
        <div>
          <h2 class="card-title">Recent CSR Sponsorship Requests</h2>
          <p style="font-size:12px;color:var(--text-secondary);margin-top:2px;">
            Auditable corporate sponsorship requests tied to verified BusSense road defects
          </p>
        </div>
        <span class="text-xs text-tertiary" id="csr-history-count">${csrRequests.length} request${csrRequests.length !== 1 ? 's' : ''}</span>
      </div>

      <div style="overflow-x:auto;">
        <table class="data-table" id="csr-history-table">
          <thead>
            <tr>
              <th>Request ID</th>
              <th>Sponsor Company</th>
              <th>Corridor Road</th>
              <th>Defect ID</th>
              <th>Severity</th>
              <th>Estimated Cost</th>
              <th>Requested Amount</th>
              <th>Status</th>
              <th>Date</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="csr-history-tbody">
            ${renderCsrHistoryRows(csrRequests)}
          </tbody>
        </table>
      </div>
    </div>
  `;

  // Wire Defect Dropdown
  const picker = document.getElementById('csr-pothole-picker');
  picker?.addEventListener('change', (e) => {
    selectedCsrPotholeId = e.target.value;
    renderCsrModule(container);
  });

  // Wire Cost Breakdown Toggle
  document.getElementById('btn-toggle-csr-breakdown')?.addEventListener('click', (e) => {
    const el = document.getElementById('csr-cost-breakdown-details');
    if (el) {
      const isHidden = el.style.display === 'none';
      el.style.display = isHidden ? 'block' : 'none';
      e.target.textContent = isHidden ? 'Hide Breakdown ▲' : 'Cost Breakdown ▼';
    }
  });

  // Wire Partner Card Click Selection
  container.querySelectorAll('.csr-partner-card').forEach(card => {
    card.addEventListener('click', () => {
      selectedCsrCompanyId = card.dataset.companyId;
      renderCsrModule(container);
    });
  });

  // Wire Budget Validation Warning
  const amountInput = document.getElementById('csr-requested-amount-input');
  const warningEl = document.getElementById('csr-budget-warning');
  amountInput?.addEventListener('input', () => {
    const val = Number(amountInput.value) || 0;
    const avail = Number(selectedPartner?.available_budget_inr) || 0;
    if (warningEl) warningEl.style.display = val > avail ? 'block' : 'none';
  });

  // Wire CSR Submission Button
  document.getElementById('btn-submit-csr-request')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-submit-csr-request');
    const amountVal = Number(amountInput?.value) || est.total_cost;
    const notesVal = document.getElementById('csr-operator-notes')?.value || 'Standard bus corridor repair request.';

    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Submitting...';

    const payload = {
      company_id: selectedCsrCompanyId || 'TVS_MOBILITY_CSR',
      pothole_event_id: selectedCsrPotholeId,
      bus_id: currentDefect?.bus_id || 'BUS-01',
      gps_lat: currentDefect?.demo_route_location?.lat ?? currentDefect?.gps_lat,
      gps_lon: currentDefect?.demo_route_location?.lon ?? currentDefect?.gps_lon,
      depth_cm: currentDefect?.depth_cm,
      area_m2: currentDefect?.area_m2,
      severity: currentDefect?.severity || 'MEDIUM',
      road_name: currentDefect?.demo_route_location?.road_name || 'Tiruchirappalli Corridor',
      estimated_repair_cost_inr: est.total_cost,
      requested_amount_inr: amountVal,
      evidence_image: currentDefect?.image || '',
      notes: notesVal
    };

    const res = await api.createCsrRequest(payload);
    if (res && res.request_id) {
      // Refresh requests and partners
      const [newRequests, newPartners] = await Promise.all([
        api.csrRequests(),
        api.csrPartners()
      ]);
      csrRequests = newRequests || [];
      csrPartners = newPartners || [];

      // Update KPIs & re-render CSR module
      renderKpis(document.getElementById('rh-kpis'));
      renderCsrModule(container);

      // Show toast or highlight
      setTimeout(() => {
        const row = document.querySelector(`tr[data-request-id="${res.request_id}"]`);
        if (row) {
          row.classList.add('highlight-row');
          row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          setTimeout(() => row.classList.remove('highlight-row'), 3500);
        }
      }, 100);
    } else {
      alert('Failed to submit CSR request. Please check server logs.');
      btn.disabled = false;
      btn.innerHTML = '<span>🚀</span> SEND CSR REQUEST';
    }
  });

  // Wire interactive table actions (advance status & view details)
  attachHistoryTableActions(container);
}

// ---------------------------------------------------------------------------
// Render history table rows
// ---------------------------------------------------------------------------
function renderCsrHistoryRows(requests) {
  if (!requests || !requests.length) {
    return `<tr><td colspan="10" style="text-align:center;padding:24px;color:var(--text-tertiary);">No CSR requests recorded yet.</td></tr>`;
  }

  return requests.map(r => {
    const isCompleted = r.status === 'COMPLETED' || r.status === 'FUNDED';
    const nextStatus = getNextStatus(r.status);

    return `
      <tr class="clickable-row" data-request-id="${r.request_id}">
        <td><span class="mono" style="font-weight:700;color:var(--blue-600);">${r.request_id}</span></td>
        <td>
          <div style="display:flex;align-items:center;gap:6px;">
            <span>${r.company_logo || '🏢'}</span>
            <span style="font-weight:600;font-size:12px;">${r.company_name || r.company_id}</span>
          </div>
        </td>
        <td style="font-size:12px;color:#1E293B;">${r.road_name || 'Trichy Route'}</td>
        <td>
          <button class="btn btn-secondary btn-sm" style="font-size:11px;padding:2px 7px;cursor:pointer;" onclick="event.stopPropagation(); window.openEventEvidence('${r.pothole_event_id}')">
            🕳️ ${r.pothole_event_id}
          </button>
        </td>
        <td>${severityBadge(r.severity)}</td>
        <td class="mono">${formatINR(r.estimated_repair_cost_inr)}</td>
        <td class="mono" style="font-weight:700;color:#059669;">${formatINR(r.requested_amount_inr)}</td>
        <td>${csrStatusBadge(r.status)}</td>
        <td class="text-secondary" style="font-size:11px;">${timeAgo(r.timestamp)}</td>
        <td>
          <div style="display:flex;align-items:center;gap:4px;">
            ${nextStatus ? `
              <button class="btn btn-secondary btn-sm btn-advance-status" data-request-id="${r.request_id}" data-next-status="${nextStatus}" style="font-size:10px;padding:3px 7px;cursor:pointer;" title="Demo: Advance workflow status to ${nextStatus}">
                ▶ ${nextStatus}
              </button>
            ` : '<span style="font-size:11px;color:#059669;font-weight:700;">✓ COMPLETE</span>'}
            <button class="btn btn-secondary btn-sm btn-view-csr" data-request-id="${r.request_id}" style="font-size:10px;padding:3px 6px;cursor:pointer;" title="View sponsorship dossier">
              👁️
            </button>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

function getNextStatus(curr) {
  const lifecycle = ['REQUESTED', 'UNDER REVIEW', 'APPROVED', 'FUNDED', 'COMPLETED'];
  const idx = lifecycle.indexOf(curr.toUpperCase());
  if (idx >= 0 && idx < lifecycle.length - 1) {
    return lifecycle[idx + 1];
  }
  return null;
}

function attachHistoryTableActions(container) {
  // Advance Status button
  container.querySelectorAll('.btn-advance-status').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const reqId = btn.dataset.requestId;
      const nextStat = btn.dataset.nextStatus;
      btn.disabled = true;
      btn.textContent = 'Updating...';

      const res = await api.updateCsrStatus(reqId, nextStat);
      if (res) {
        const [newRequests, newPartners] = await Promise.all([
          api.csrRequests(),
          api.csrPartners()
        ]);
        csrRequests = newRequests || [];
        csrPartners = newPartners || [];
        renderKpis(document.getElementById('rh-kpis'));
        renderCsrModule(container);
      }
    });
  });

  // View CSR Details button or Row click
  container.querySelectorAll('.btn-view-csr, #csr-history-tbody tr').forEach(el => {
    el.addEventListener('click', (e) => {
      const reqId = el.dataset.requestId;
      if (!reqId) return;
      openCsrDetailModal(reqId);
    });
  });
}

function openCsrDetailModal(requestId) {
  const req = csrRequests.find(r => r.request_id === requestId);
  if (!req) return;

  const modal = document.getElementById('global-event-modal');
  const title = document.getElementById('modal-title');
  const body = document.getElementById('modal-body');
  if (!modal || !title || !body) return;

  title.innerHTML = `<span style="display:inline-flex;align-items:center;gap:6px;">
    <span>🤝</span> CSR Sponsorship Dossier — <span class="mono">${req.request_id}</span>
  </span>`;

  body.innerHTML = `
    <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:12px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between;">
      <div>
        <div style="font-size:11px;font-weight:700;color:#166534;text-transform:uppercase;">Current Workflow Status</div>
        <div style="margin-top:2px;">${csrStatusBadge(req.status)}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:11px;color:#166534;">Requested Sponsorship</div>
        <div style="font-size:22px;font-weight:800;color:#15803D;">${formatINR(req.requested_amount_inr)}</div>
      </div>
    </div>

    ${req.evidence_image ? `
      <div style="margin-bottom:14px;background:#000;border-radius:8px;overflow:hidden;border:1px solid #E2E8F0;">
        <div style="padding:6px 10px;background:#F8FAFC;color:#475569;font-size:10px;font-weight:600;border-bottom:1px solid #E2E8F0;">
          📷 Pothole Evidence Frame
        </div>
        <img src="${req.evidence_image}" alt="Evidence Frame" style="width:100%;max-height:220px;object-fit:cover;display:block;">
      </div>
    ` : ''}

    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 14px;margin-bottom:14px;font-size:12px;">
      <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Sponsor Partner:</span>
        <strong style="color:#1E293B;">${req.company_logo || '🏢'} ${req.company_name || req.company_id}</strong>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Target Road Defect:</span>
        <button class="btn btn-secondary btn-sm" style="font-size:11px;padding:1px 6px;" onclick="window.openEventEvidence('${req.pothole_event_id}')">
          🕳️ ${req.pothole_event_id}
        </button>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Corridor Location:</span>
        <span>${req.road_name || 'Tiruchirappalli Corridor'}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Measured Geometry:</span>
        <span>${req.depth_cm ? req.depth_cm + ' cm depth' : ''} ${req.area_m2 ? '· ' + req.area_m2 + ' m² area' : ''} (${req.severity})</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Engineering Estimate:</span>
        <strong class="mono">${formatINR(req.estimated_repair_cost_inr)}</strong>
      </div>
      <div style="display:flex;justify-content:space-between;padding:4px 0;">
        <span style="color:#64748B;">Submission Timestamp:</span>
        <span>${formatTimestamp(req.timestamp)}</span>
      </div>
    </div>

    ${req.notes ? `
      <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:10px 12px;margin-bottom:14px;font-size:12px;">
        <div style="font-size:10px;font-weight:700;color:#92400E;text-transform:uppercase;margin-bottom:2px;">Operator Justification / Notes</div>
        <div style="color:#78350F;">${req.notes}</div>
      </div>
    ` : ''}

    <div style="display:flex;justify-content:flex-end;">
      <button class="btn btn-secondary" onclick="document.getElementById('global-event-modal').style.display='none'">Close</button>
    </div>
  `;
  modal.style.display = 'flex';
}

// ---------------------------------------------------------------------------
// Main render
// ---------------------------------------------------------------------------
export async function render(container) {
  if (location.hash === '#csr' || location.hash.includes('csr')) {
    activeTab = 'csr';
  } else {
    activeTab = 'all';
  }
  allDefects = [];
  csrPartners = [];
  csrRequests = [];

  container.innerHTML = `
    <div class="page-content">
      <div class="page-header">
        <h1 class="page-title">${activeTab === 'csr' ? 'CSR / Road Repair Funding' : 'Road Health'}</h1>
        <p class="page-subtitle">${activeTab === 'csr' ? 'Corporate road repair sponsorship & defect matching — Bus BUS-01 · Tiruchirappalli Corridor' : 'Pothole & crack defect monitoring — Bus BUS-01 · Active Perception Corridor'}</p>
      </div>

      <div class="kpi-grid mb-4" id="rh-kpis">
        ${Array(4).fill('<div class="kpi-card"><div class="loading-shimmer" style="width:80px;height:14px;margin-bottom:8px"></div><div class="loading-shimmer" style="width:60px;height:28px"></div></div>').join('')}
      </div>

      <!-- Tab Strip -->
      <div class="tabs mb-4" id="rh-tabs" style="border-bottom:1px solid var(--border-default);background:var(--bg-card);border-radius:var(--radius-md) var(--radius-md) 0 0;padding:4px 16px 0;">
        <div class="tab ${activeTab === 'all' ? 'active' : ''}" data-tab="all">All Defects</div>
        <div class="tab ${activeTab === 'pothole' ? 'active' : ''}" data-tab="pothole">Potholes</div>
        <div class="tab ${activeTab === 'crack' ? 'active' : ''}" data-tab="crack">Cracks</div>
        <div class="tab ${activeTab === 'csr' ? 'active' : ''}" data-tab="csr" style="font-weight:700;color:#059669;display:inline-flex;align-items:center;gap:6px;">
          <span>🤝</span> CSR / ROAD REPAIR FUNDING
        </div>
      </div>

      <!-- VIEW 1: Standard Defects Table + Map Grid -->
      <div id="rh-defects-view" class="grid-2-1">
        <div class="card">
          <div class="card-header" style="flex-direction:row;align-items:center;justify-content:space-between;">
            <h2 class="card-title">Defects</h2>
            <span class="text-xs text-tertiary" id="rh-count"></span>
          </div>
          <div style="overflow-x:auto">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Event ID</th>
                  <th>Type</th>
                  <th>Severity</th>
                  <th>Depth / Length</th>
                  <th>Area / Branches</th>
                  <th>Est. Repair</th>
                  <th>CSR Status</th>
                  <th>Condition</th>
                  <th>Time</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody id="rh-tbody">
                <tr><td colspan="10" style="text-align:center;padding:var(--space-8)">
                  <div class="loading-shimmer" style="width:200px;height:14px;margin:0 auto"></div>
                </td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Defect Map</h2>
            <div class="flex items-center gap-4" style="gap:12px">
              <span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--text-secondary)">
                <span style="width:8px;height:8px;border-radius:50%;background:#F59E0B;display:inline-block"></span> Potholes
              </span>
              <span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--text-secondary)">
                <span style="width:8px;height:8px;border-radius:50%;background:#64748B;display:inline-block"></span> Cracks
              </span>
              <span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--text-secondary)">
                <span style="width:8px;height:8px;border-radius:50%;background:#10B981;border:1px solid #059669;display:inline-block"></span> CSR Sponsored
              </span>
            </div>
          </div>
          <div class="map-container" id="rh-map" style="height:540px;border:none;border-radius:0"></div>
        </div>
      </div>

      <!-- VIEW 2: Integrated CSR Funding Module -->
      <div id="rh-csr-view" style="display:none;"></div>
    </div>
  `;

  // --- Initialize Leaflet map ---
  const mapEl = document.getElementById('rh-map');
  mapInstance = L.map(mapEl, { zoomControl: true }).setView([10.8150, 78.6918], 15);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 19,
  }).addTo(mapInstance);
  markersLayer = L.featureGroup().addTo(mapInstance);

  setTimeout(() => mapInstance.invalidateSize(), 200);

  // --- Tab handling ---
  const tabsEl = document.getElementById('rh-tabs');
  if (window.pendingCsrPotholeId) {
    selectedCsrPotholeId = window.pendingCsrPotholeId;
    window.pendingCsrPotholeId = null;
    activeTab = 'csr';
  } else if (window.location.hash === '#csr') {
    activeTab = 'csr';
  }

  if (tabsEl) {
    tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === activeTab));
  }

  tabsEl.addEventListener('click', (e) => {
    const tab = e.target.closest('.tab');
    if (!tab) return;
    activeTab = tab.dataset.tab;
    tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));

    if (activeTab === 'csr') {
      history.replaceState(null, '', '#csr');
      document.querySelectorAll('.sidebar-link').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-route') === '#csr');
      });
    } else {
      history.replaceState(null, '', '#road-health');
      document.querySelectorAll('.sidebar-link').forEach(el => {
        el.classList.toggle('active', el.getAttribute('data-route') === '#road-health');
      });
    }
    applyFilter();
  });

  // --- Fetch data in parallel ---
  const [data, partnersData, requestsData] = await Promise.all([
    api.roadEvents({ limit: 100 }),
    api.csrPartners(),
    api.csrRequests()
  ]);

  allDefects = Array.isArray(data) ? data : (data?.items || data?.results || []);
  csrPartners = Array.isArray(partnersData) ? partnersData : [];
  csrRequests = Array.isArray(requestsData) ? requestsData : [];

  if (!selectedCsrPotholeId && allDefects.length) {
    const defaultPot = allDefects.find(d => (d.event_type || '').toUpperCase() === 'POTHOLE');
    selectedCsrPotholeId = defaultPot ? (defaultPot.event_id || defaultPot.id) : (allDefects[0].event_id || allDefects[0].id);
  }

  // --- Render initial state ---
  applyFilter();
}

// ---------------------------------------------------------------------------
// Apply current tab filter to table & map or switch to CSR view
// ---------------------------------------------------------------------------
function applyFilter() {
  const defectsView = document.getElementById('rh-defects-view');
  const csrView = document.getElementById('rh-csr-view');
  const countEl = document.getElementById('rh-count');
  const kpisContainer = document.getElementById('rh-kpis');

  renderKpis(kpisContainer);

  if (activeTab === 'csr') {
    if (defectsView) defectsView.style.display = 'none';
    if (csrView) {
      csrView.style.display = 'block';
      renderCsrModule(csrView);
    }
  } else {
    if (csrView) csrView.style.display = 'none';
    if (defectsView) {
      defectsView.style.display = 'grid';
      setTimeout(() => mapInstance?.invalidateSize(), 100);
    }

    const filtered = filterDefects(allDefects, activeTab);
    const tbody = document.getElementById('rh-tbody');
    if (tbody) renderTableBody(tbody, filtered);
    if (countEl) countEl.textContent = `${filtered.length} defect${filtered.length !== 1 ? 's' : ''}`;
    updateMapMarkers(filtered);
  }
}

// ---------------------------------------------------------------------------
// Road defect evidence modal delegation
// ---------------------------------------------------------------------------
window.showRoadEvidence = (eventId) => {
  const ev = allDefects.find(x => (x.event_id || x.id) === eventId);
  openEventEvidence(eventId, { cachedEvent: ev });
};

// ---------------------------------------------------------------------------
// Trigger CSR Sponsorship flow from any evidence modal or table row
// ---------------------------------------------------------------------------
window.openCsrRequestFlow = (eventId, ev) => {
  if (window.location.hash !== '#road-health' && window.location.hash !== '#csr') {
    window.pendingCsrPotholeId = eventId;
    window.location.hash = '#csr';
    return;
  }

  activeTab = 'csr';
  selectedCsrPotholeId = eventId;
  const tabsEl = document.getElementById('rh-tabs');
  if (tabsEl) {
    tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'csr'));
  }
  history.replaceState(null, '', '#csr');
  document.querySelectorAll('.sidebar-link').forEach(el => {
    el.classList.toggle('active', el.getAttribute('data-route') === '#csr');
  });
  applyFilter();
  document.getElementById('rh-csr-view')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

// ---------------------------------------------------------------------------
// Locate defect on Road Health map from modal action
// ---------------------------------------------------------------------------
window.locateOnRoadMap = (eventId, lat, lon) => {
  if (!mapInstance) return;

  const ev = allDefects.find(x => (x.event_id || x.id) === eventId);
  if (ev) {
    const type = (ev.event_type || ev.defect_type || '').toLowerCase();
    if (activeTab !== 'all' && activeTab !== type) {
      activeTab = 'all';
      const tabsEl = document.getElementById('rh-tabs');
      if (tabsEl) {
        tabsEl.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'all'));
      }
      applyFilter();
    }
  }

  if (lat != null && lon != null) {
    mapInstance.setView([lat, lon], 17, { animate: true });
    const marker = markersByEventId[eventId];
    if (marker) {
      marker.openTooltip();
    }
  }

  const rows = document.querySelectorAll('#rh-tbody tr');
  rows.forEach(r => {
    if (r.dataset.eventId === eventId) {
      r.classList.add('highlight-row');
      r.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      setTimeout(() => r.classList.remove('highlight-row'), 3500);
    }
  });
};

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
export function destroy() {
  if (mapInstance) {
    mapInstance.remove();
    mapInstance = null;
  }
  markersLayer = null;
  markersByEventId = {};
  allDefects = [];
  csrPartners = [];
  csrRequests = [];
  activeTab = 'all';
}
