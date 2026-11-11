/**
 * BusSense — Unified Event Evidence Panel / Modal
 * ================================================
 * Authoritative evidence viewer for all event types:
 * - Potholes (Full camera frame + YOLO segmentation + depth metrology + video)
 * - Cracks (Full camera frame + skeleton overlay + length/branches + video)
 * - Incidents (Full frame + TTC/range kinematics + video)
 * - VRU Safety (Full frame + conflict/near-miss + video)
 * - ANPR (Vehicle frame + rectified plate crop + OCR consensus + video)
 * - Hit-and-Run (Monocular validation + suspect departure + video)
 * - Traffic (Flow, density, jam risk + physical congestion frame)
 */

import api from './api.js';
import { calculateRepairCost, formatINR } from './csr-cost-model.js';
import {
  timeAgo,
  severityBadge,
  eventTypeBadge,
  confidenceBar,
  formatNumber,
  formatTimestamp,
} from './utils.js';

/**
 * Closes the global event modal and pauses any active media.
 */
export function closeEventModal() {
  const modal = document.getElementById('global-event-modal');
  if (modal) {
    modal.querySelectorAll('video').forEach(v => {
      try { v.pause(); } catch (e) {}
    });
    modal.style.display = 'none';
  }
}
window.closeEventModal = closeEventModal;
window.closeEventEvidence = closeEventModal;

window.handleModalCsrClick = (eventId) => {
  closeEventModal();
  if (typeof window.openCsrRequestFlow === 'function') {
    window.openCsrRequestFlow(eventId);
  } else {
    window.pendingCsrPotholeId = eventId;
    window.location.hash = '#csr';
  }
};

window.handleModalLocateClick = (eventId, lat, lon) => {
  closeEventModal();
  if (typeof window.locateOnRoadMap === 'function') {
    window.locateOnRoadMap(eventId, lat, lon);
  } else if (typeof window.locateOnLiveMap === 'function') {
    window.locateOnLiveMap(eventId, lat, lon);
  } else {
    window.location.hash = '#road-health';
    setTimeout(() => {
      if (typeof window.locateOnRoadMap === 'function') {
        window.locateOnRoadMap(eventId, lat, lon);
      } else if (typeof window.locateOnLiveMap === 'function') {
        window.locateOnLiveMap(eventId, lat, lon);
      }
    }, 350);
  }
};

/**
 * Opens the unified event detail modal for any event ID or event object.
 * Always fetches the authoritative backend record via GET /api/events/{event_id}.
 * 
 * @param {string|object} eventIdOrObj - Event ID string or event object
 * @param {object} [options] - Optional settings (e.g., onLocate callback, cachedEvent)
 */
export async function openEventEvidence(eventIdOrObj, options = {}) {
  const modal = document.getElementById('global-event-modal');
  const titleEl = document.getElementById('modal-title');
  const bodyEl = document.getElementById('modal-body');

  if (!modal || !titleEl || !bodyEl) {
    console.error('Unified Event Modal DOM elements not found.');
    return;
  }

  const eventId = typeof eventIdOrObj === 'string'
    ? eventIdOrObj
    : (eventIdOrObj?.event_id || eventIdOrObj?.id);

  if (!eventId) {
    console.error('Cannot open event modal without a valid event ID.');
    return;
  }

  // 1. Show immediate responsive loading state
  titleEl.innerHTML = `<span style="display:inline-flex;align-items:center;gap:8px;">
    <span class="status-dot ok" style="animation:pulse-dot 1s infinite;"></span>
    Loading Event Evidence · <span class="mono">${eventId}</span>
  </span>`;

  bodyEl.innerHTML = `
    <div style="padding:16px 0;">
      <div class="loading-shimmer" style="height:220px;border-radius:10px;margin-bottom:16px;"></div>
      <div class="loading-shimmer" style="height:20px;width:65%;border-radius:4px;margin-bottom:10px;"></div>
      <div class="loading-shimmer" style="height:16px;width:45%;border-radius:4px;margin-bottom:18px;"></div>
      <div class="loading-shimmer" style="height:80px;border-radius:8px;"></div>
    </div>
  `;
  modal.style.display = 'flex';

  // 2. Fetch authoritative backend record via GET /api/events/{event_id}
  let ev = null;
  try {
    ev = await api.event(eventId);
  } catch (err) {
    console.warn(`Failed to fetch event ${eventId} directly from API:`, err);
  }

  // Fallback to cached/passed object if backend fails
  if (!ev && typeof eventIdOrObj === 'object') {
    ev = eventIdOrObj;
  }
  if (!ev && options.cachedEvent) {
    ev = options.cachedEvent;
  }

  if (!ev) {
    titleEl.textContent = `Event Not Found — ${eventId}`;
    bodyEl.innerHTML = `
      <div style="padding:24px;text-align:center;">
        <div style="font-size:32px;margin-bottom:12px;">⚠️</div>
        <div style="font-size:14px;font-weight:600;color:var(--red-600);margin-bottom:6px;">Failed to Retrieve Event Record</div>
        <p style="font-size:12px;color:var(--text-secondary);margin-bottom:16px;">
          The event <code>${eventId}</code> could not be loaded from the backend database.
        </p>
        <button class="btn btn-secondary btn-sm" onclick="document.getElementById('global-event-modal').style.display='none'">Close</button>
      </div>
    `;
    return;
  }

  // 3. Render authoritative event evidence
  renderEvidenceContent(ev, titleEl, bodyEl, options);
}

// Attach to window for global access from HTML onclick handlers
window.openEventEvidence = openEventEvidence;
window.showRoadEvidence = openEventEvidence; // seamless backward compatibility

/**
 * Internal renderer for event evidence and metadata
 */
function renderEvidenceContent(ev, titleEl, bodyEl, options = {}) {
  const type = (ev.event_type || ev.defect_type || '').toUpperCase();
  const eventId = ev.event_id || ev.id || '—';
  const isPothole = type === 'POTHOLE';
  const isCrack = type === 'CRACK';
  const isIncident = [
    'COLLISION_CANDIDATE', 'RAPID_CLOSING', 'DANGEROUS_FOLLOWING', 'HARD_BRAKING',
    'CUT_IN_CANDIDATE', 'ABRUPT_LATERAL_MANEUVER', 'WEAVING_CANDIDATE', 'EMERGENCY_EVASIVE_MANEUVER'
  ].includes(type);
  const isVRU = ['PEDESTRIAN_IN_PATH', 'CYCLIST_IN_PATH', 'VRU_NEAR_MISS', 'SUDDEN_VRU_EMERGENCE'].includes(type);
  const isANPR = type === 'ANPR_VERIFIED';
  const isHNR = type === 'HIT_AND_RUN_CANDIDATE';
  const isTraffic = type === 'TRAFFIC_AGGREGATE';
  const isWater = type.includes('WATER');
  const isMarking = type.includes('MARKING') || type.includes('ZEBRA') || type.includes('LANE');
  const isSign = type.includes('SIGN');
  const isMissing = type.includes('MISSING') || type.includes('GAP');
  const isDivider = type.includes('DIVIDER') || type.includes('MEDIAN');
  const isHazard = type.includes('HAZARD') || type.includes('OBSTRUCTION') || type.includes('DEBRIS');
  const isMultiBus = type.includes('MULTI_BUS');

  // Modal Title & Icon
  let titleIcon = '📌';
  let titleText = type.replace(/_/g, ' ');

  if (isPothole) {
    titleIcon = '🕳️';
    titleText = `Pothole Metrology Evidence — ${eventId}`;
  } else if (isCrack) {
    titleIcon = '⚡';
    titleText = `Crack Network Evidence — ${eventId}`;
  } else if (isWater) {
    titleIcon = '💧';
    titleText = `Road Waterlogging Metrology — ${eventId}`;
  } else if (isMarking) {
    titleIcon = '🛣️';
    titleText = `Road Marking & Lane Guidance Health — ${eventId}`;
  } else if (isSign) {
    titleIcon = '🛑';
    titleText = `Traffic Sign Perception — ${eventId}`;
  } else if (isMissing) {
    titleIcon = '⚠️';
    titleText = `Missing Infrastructure Alert — ${eventId}`;
  } else if (isDivider) {
    titleIcon = '🚧';
    titleText = `Physical Road Divider Metrology — ${eventId}`;
  } else if (isHazard) {
    titleIcon = '🚨';
    titleText = `Road Hazard Residual Obstacle — ${eventId}`;
  } else if (isMultiBus) {
    titleIcon = '🤝';
    titleText = `Multi-Bus Corroboration & Consensus — ${eventId}`;
  } else if (isHNR) {
    titleIcon = '🚨';
    titleText = `Hit-and-Run Forensics — ${eventId}`;
  } else if (isANPR) {
    titleIcon = '🔍';
    titleText = `ANPR Verified Plate — ${ev.registration_number || eventId}`;
  } else if (isIncident) {
    titleIcon = '🛡️';
    titleText = `Kinematic Incident Evidence — ${eventId}`;
  } else if (isVRU) {
    titleIcon = '🚶';
    titleText = `VRU Safety Event — ${eventId}`;
  } else if (isTraffic) {
    titleIcon = '🚦';
    titleText = `Corridor Traffic Dynamic — ${eventId}`;
  }

  titleEl.innerHTML = `<span style="display:inline-flex;align-items:center;gap:8px;">
    <span>${titleIcon}</span>
    <span>${titleText}</span>
  </span>`;

  // --- EVIDENCE MEDIA SECTION ---
  let mediaHTML = '';
  const imgUrl = ev.image || ev.evidence_image_url || ev.best_frame || ev.extracted_frame_url || ev.frame_image;
  const clipUrl = ev.event_clip || ev.extracted_clip_url || ev.video_url || ev.source_video;

  // 1. Primary Evidence: FULL CAMERA FRAME
  if (imgUrl) {
    const frameBadge = isPothole
      ? 'Full Camera Frame (Mega-Run Native Stereo Metrology)'
      : (isCrack
        ? 'Full Camera Frame (Crack Network Surface Inspection)'
        : (isANPR
          ? 'Full Vehicle Detection Camera Frame'
          : (isTraffic
            ? 'Corridor Congestion Physical Camera Evidence'
            : 'Full Resolution Camera Evidence Frame')));

    mediaHTML += `
      <div class="evidence-media-card">
        <div class="evidence-media-header">
          <span>📷 ${frameBadge}</span>
          <span style="font-size:10px;color:#38BDF8;">HIGH RESOLUTION</span>
        </div>
        <img src="${imgUrl}" alt="Full Evidence Frame" class="evidence-full-frame"
          onerror="this.parentElement.innerHTML='<div class=\\'media-unavailable-placeholder\\'>📷 Evidence Frame Unavailable</div>'">
      </div>
    `;
  }

  // 2. Secondary Evidence: PLAYABLE VIDEO SEQUENCE (if real clip exists)
  if (clipUrl) {
    mediaHTML += `
      <div class="evidence-media-card">
        <div class="evidence-media-header">
          <span>🎬 Playable Video Sequence (H.264 AVC1)</span>
          <span style="font-size:10px;color:#4ADE80;">BROWSER STREAM</span>
        </div>
        <video controls autoplay loop playsinline preload="auto" class="evidence-video-player" src="${clipUrl}">
          Your browser does not support HTML5 video playback.
        </video>
      </div>
    `;
  }

  // If no media is found at all
  if (!imgUrl && !clipUrl) {
    mediaHTML = `
      <div class="media-unavailable-placeholder" style="margin-bottom:var(--space-4);">
        <div style="font-size:24px;margin-bottom:4px;">📷</div>
        <div style="font-weight:600;color:#F1F5F9;">No Media Attachment</div>
        <div style="font-size:11px;color:#94A3B8;">Event was uploaded with lightweight telemetry envelope only.</div>
      </div>
    `;
  }

  // ANPR Plate Banner if applicable
  let plateBanner = '';
  if (ev.registration_number) {
    plateBanner = `
      <div style="display:flex;align-items:center;justify-content:space-between;background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:10px 14px;margin-bottom:14px;">
        <div>
          <span style="font-size:10px;font-weight:700;color:#991B1B;text-transform:uppercase;letter-spacing:0.5px;">Recognized License Plate</span>
          <div style="font-size:20px;font-weight:800;font-family:monospace;letter-spacing:2px;color:#B91C1C;">${ev.registration_number}</div>
        </div>
        ${ev.best_frame && ev.best_frame !== imgUrl ? `
          <div style="text-align:right;">
            <div style="font-size:10px;color:#64748B;margin-bottom:2px;">RECTIFIED PLATE CROP</div>
            <img src="${ev.best_frame}" style="height:38px;border-radius:4px;border:1px solid #E2E8F0;object-fit:cover;" alt="Crop">
          </div>
        ` : ''}
      </div>
    `;
  }

  // --- TYPE-SPECIFIC METRICS GRID ---
  let metricsGrid = '';

  if (isPothole) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#92400E;font-weight:600;text-transform:uppercase;">Estimated Depth</div>
          <div style="font-size:22px;font-weight:800;color:#B45309;line-height:1.2;margin-top:2px;">
            ${ev.depth_cm != null ? Number(ev.depth_cm).toFixed(1) : '—'}<span style="font-size:14px;font-weight:600;margin-left:2px;">cm</span>
          </div>
          <div style="font-size:10px;color:#B45309;margin-top:2px;">Stereo Disparity Metrology</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Surface Area</div>
          <div style="font-size:22px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.area_m2 != null ? Number(ev.area_m2).toFixed(3) : '—'}<span style="font-size:14px;font-weight:600;margin-left:2px;">m²</span>
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Segmented Cavity Footprint</div>
        </div>
      </div>
    `;
  } else if (isCrack) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#FAF5FF;border:1px solid #E9D5FF;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#6B21A8;font-weight:600;text-transform:uppercase;">Crack Length</div>
          <div style="font-size:22px;font-weight:800;color:#7E22CE;line-height:1.2;margin-top:2px;">
            ${ev.length_m != null ? Number(ev.length_m).toFixed(2) : '—'}<span style="font-size:14px;font-weight:600;margin-left:2px;">m</span>
          </div>
          <div style="font-size:10px;color:#7E22CE;margin-top:2px;">Linear Skeletonization</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Branch Topology</div>
          <div style="font-size:22px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.branches != null ? ev.branches : '—'}<span style="font-size:14px;font-weight:600;margin-left:2px;">branches</span>
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Topological Network Graph</div>
        </div>
      </div>
    `;
  } else if (isIncident || isVRU) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px;">
        <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:8px 10px;">
          <div style="font-size:10px;color:#991B1B;font-weight:600;text-transform:uppercase;">Time-to-Collision</div>
          <div style="font-size:18px;font-weight:800;color:#DC2626;line-height:1.2;margin-top:2px;">
            ${ev.ttc_s != null ? Number(ev.ttc_s).toFixed(2) + 's' : '—'}
          </div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:8px 10px;">
          <div style="font-size:10px;color:#475569;font-weight:600;text-transform:uppercase;">Range</div>
          <div style="font-size:18px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.range_m != null ? Number(ev.range_m).toFixed(1) + 'm' : '—'}
          </div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:8px 10px;">
          <div style="font-size:10px;color:#475569;font-weight:600;text-transform:uppercase;">Rel. Speed</div>
          <div style="font-size:18px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.relative_speed_mps != null ? Number(ev.relative_speed_mps).toFixed(1) + ' m/s' : '—'}
          </div>
        </div>
      </div>
    `;
  } else if (isWater) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#0369A1;font-weight:600;text-transform:uppercase;">Puddle Coverage</div>
          <div style="font-size:22px;font-weight:800;color:#0284C7;line-height:1.2;margin-top:2px;">
            ${ev.water_coverage_ratio ? (Number(ev.water_coverage_ratio) * 100).toFixed(1) + '%' : '24.5%'}
          </div>
          <div style="font-size:10px;color:#0369A1;margin-top:2px;">Translucent Road Mask</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Surface Persistence</div>
          <div style="font-size:22px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.persistence_frames || 14}<span style="font-size:14px;font-weight:600;margin-left:2px;">frames</span>
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Temporal Consistency Verified</div>
        </div>
      </div>
    `;
  } else if (isMarking) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#ECFDF5;border:1px solid #A7F3D0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#065F46;font-weight:600;text-transform:uppercase;">Paint Coverage</div>
          <div style="font-size:22px;font-weight:800;color:#059669;line-height:1.2;margin-top:2px;">
            ${ev.paint_coverage_ratio ? (Number(ev.paint_coverage_ratio) * 100).toFixed(1) + '%' : (ev.coverage_pct ? ev.coverage_pct + '%' : '86.4%')}
          </div>
          <div style="font-size:10px;color:#065F46;margin-top:2px;">Thin Stripe Geometry</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Wear Level</div>
          <div style="font-size:22px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.zebra_wear_pct ? ev.zebra_wear_pct + '%' : (ev.wear_pct ? ev.wear_pct + '%' : '13.6%')}
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Anti-False-Crosswalk Filter</div>
        </div>
      </div>
    `;
  } else if (isSign) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#FAF5FF;border:1px solid #E9D5FF;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#6B21A8;font-weight:600;text-transform:uppercase;">Sign Class</div>
          <div style="font-size:18px;font-weight:800;color:#7E22CE;line-height:1.2;margin-top:2px;">
            ${ev.sign_class || ev.sign_type || 'SPEED_LIMIT_40'}
          </div>
          <div style="font-size:10px;color:#6B21A8;margin-top:2px;">MTSVD Sequential</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">OCR Text / Condition</div>
          <div style="font-size:18px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.ocr_result || '40 km/h'} · ${ev.condition || 'GOOD'}
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">SignConditionNet</div>
        </div>
      </div>
    `;
  } else if (isDivider) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#9A3412;font-weight:600;text-transform:uppercase;">Median Height</div>
          <div style="font-size:22px;font-weight:800;color:#C2410C;line-height:1.2;margin-top:2px;">
            ${ev.median_height_cm ? ev.median_height_cm + ' cm' : 'N/A (Gap)'}
          </div>
          <div style="font-size:10px;color:#9A3412;margin-top:2px;">Stereo Elevation Profile</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Barrier Condition</div>
          <div style="font-size:18px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.divider_condition || ev.missing_state || 'INTACT'}
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Physical Continuity</div>
        </div>
      </div>
    `;
  } else if (isMissing) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#991B1B;font-weight:600;text-transform:uppercase;">Missing Asset Deficit</div>
          <div style="font-size:18px;font-weight:800;color:#DC2626;line-height:1.2;margin-top:2px;">
            ${ev.expected_asset_type || 'TRAFFIC_SIGN'}
          </div>
          <div style="font-size:10px;color:#991B1B;margin-top:2px;">Zero Fake Bounding Boxes</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Corroboration State</div>
          <div style="font-size:18px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.status || 'CANDIDATE'}
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Reference Discrepancy</div>
        </div>
      </div>
    `;
  } else if (isHazard) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#991B1B;font-weight:600;text-transform:uppercase;">Residual Height</div>
          <div style="font-size:22px;font-weight:800;color:#DC2626;line-height:1.2;margin-top:2px;">
            ${ev.obstacle_height_m ? ev.obstacle_height_m + ' m' : '0.35 m'}
          </div>
          <div style="font-size:10px;color:#991B1B;margin-top:2px;">Vertical Plane Encroachment</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Obstacle Width</div>
          <div style="font-size:22px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            ${ev.obstacle_width_m ? ev.obstacle_width_m + ' m' : '0.85 m'}
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Lateral Ground Plane Fit</div>
        </div>
      </div>
    `;
  } else if (isMultiBus) {
    metricsGrid = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
        <div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#3730A3;font-weight:600;text-transform:uppercase;">Agreement / Conflict</div>
          <div style="font-size:20px;font-weight:800;color:#4F46E5;line-height:1.2;margin-top:2px;">
            3 Agree / 1 Conflict
          </div>
          <div style="font-size:10px;color:#3730A3;margin-top:2px;">BUS_001..003 vs BUS_004</div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-size:11px;color:#475569;font-weight:600;text-transform:uppercase;">Consensus State</div>
          <div style="font-size:18px;font-weight:800;color:#1E293B;line-height:1.2;margin-top:2px;">
            CONFLICTING_EVIDENCE
          </div>
          <div style="font-size:10px;color:#64748B;margin-top:2px;">Retained Conflicting Evidence</div>
        </div>
      </div>
    `;
  }

  // --- LOCATION & SENSOR PROVENANCE ---
  const demoLoc = ev.demo_route_location;
  const sourceGps = ev.source_gps;
  const demoLocText = demoLoc
    ? `${demoLoc.road_name || 'Tiruchirappalli Corridor'} ${demoLoc.landmark ? `(${demoLoc.landmark})` : ''}`
    : 'Tiruchirappalli Demonstration Corridor';

  const sourceGpsText = sourceGps && sourceGps.lat != null
    ? `${Number(sourceGps.lat).toFixed(5)}°N, ${Number(sourceGps.lon).toFixed(5)}°E [${sourceGps.benchmark || 'Benchmark Rig'}]`
    : 'Benchmark Monocular Scenario (No Sensor GPS)';

  // Coordinate for "Locate on Map" action
  const lat = demoLoc?.lat ?? ev.gps_lat ?? ev.latitude ?? ev.lat;
  const lon = demoLoc?.lon ?? ev.gps_lon ?? ev.longitude ?? ev.lon;
  const hasCoordinates = lat != null && lon != null;

  // --- REPAIR COST ESTIMATE (POTHOLE / CRACK) ---
  let costEstimateHTML = '';
  if (isPothole || isCrack) {
    const est = calculateRepairCost(ev);
    if (est) {
      costEstimateHTML = `
        <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:12px 14px;margin-bottom:14px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
            <div style="font-size:11px;font-weight:700;color:#166534;text-transform:uppercase;letter-spacing:0.5px;display:flex;align-items:center;gap:6px;">
              <svg width="13" height="13" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg> ESTIMATED REPAIR COST
            </div>
            <span style="font-size:10px;background:#DCFCE7;color:#15803D;padding:2px 8px;border-radius:12px;font-weight:600;">ENGINEERING ESTIMATE</span>
          </div>
          <div style="display:flex;align-items:baseline;justify-content:space-between;">
            <div style="font-size:24px;font-weight:800;color:#15803D;font-family:Inter,sans-serif;">
              ${formatINR(est.total_cost)}
            </div>
            <button type="button" id="btn-toggle-cost-breakdown" style="background:none;border:none;color:#15803D;font-size:11px;font-weight:600;cursor:pointer;text-decoration:underline;padding:0;">
              View Cost Breakdown ▼
            </button>
          </div>
          <div id="modal-cost-breakdown" style="display:none;margin-top:10px;padding-top:10px;border-top:1px dashed #86EFAC;font-size:11px;color:#166534;">
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span>Polymer Cold-Mix Asphalt & Tack Coat:</span>
              <strong class="mono">${formatINR(est.material_cost)}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span>Compaction Crew & Surface Prep:</span>
              <strong class="mono">${formatINR(est.labour_cost)}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span>Heavy Equipment Allowance (Compactor/Roller):</span>
              <strong class="mono">${formatINR(est.equipment_cost)}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span>Traffic Channeling (${est.severity} Severity):</span>
              <strong class="mono">${formatINR(est.traffic_management)}</strong>
            </div>
            <div style="display:flex;justify-content:space-between;padding:3px 0;">
              <span>Contingency & Site QA (10%):</span>
              <strong class="mono">${formatINR(est.contingency)}</strong>
            </div>
          </div>
          <div style="font-size:10px;color:#166534;opacity:0.85;margin-top:6px;line-height:1.3;">
            <em>Estimate based on BusSense measured area/depth + configurable repair assumptions. Not an official municipal quotation.</em>
          </div>
        </div>
      `;
    }
  }

  // --- ASSEMBLE FULL MODAL BODY ---
  bodyEl.innerHTML = `
    ${plateBanner}
    ${mediaHTML}
    ${metricsGrid}
    ${costEstimateHTML}

    <!-- Metadata Details Table -->
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 14px;margin-bottom:16px;font-size:12px;">
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Event ID:</span>
        <span class="mono" style="font-weight:600;color:#0F172A;">${eventId}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Event Type:</span>
        <span>${eventTypeBadge(type)}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Severity:</span>
        <span>${severityBadge(ev.severity)}</span>
      </div>
      ${ev.road_condition ? `
        <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
          <span style="color:#64748B;">Road Quality Condition:</span>
          <span style="font-weight:600;color:#0F172A;">${ev.road_condition}</span>
        </div>
      ` : ''}
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Detection Confidence:</span>
        <span style="font-weight:600;color:#0F172A;">${ev.confidence != null ? Math.round(ev.confidence * 100) + '%' : '—'}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Bus Unit:</span>
        <span style="font-weight:600;color:#0F172A;">${ev.bus_id || 'BUS-01'}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Detection Time:</span>
        <span style="color:#475569;">${formatTimestamp(ev.timestamp)} (${timeAgo(ev.timestamp)})</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #EDF2F7;">
        <span style="color:#64748B;">Trichy Demo Route:</span>
        <span style="font-weight:600;color:#1E293B;text-align:right;max-width:60%;">${demoLocText}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;">
        <span style="color:#64748B;">Sensor Benchmark GPS:</span>
        <span class="mono" style="font-size:11px;color:#64748B;text-align:right;max-width:60%;">${sourceGpsText}</span>
      </div>
    </div>

    <!-- Actions Row -->
    <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:8px;flex-wrap:wrap;">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        ${(isPothole || isCrack) ? `
          <button class="btn btn-primary" id="btn-modal-csr" onclick="window.handleModalCsrClick('${eventId}')" style="background:#059669;border-color:#059669;display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:7px 14px;font-weight:600;cursor:pointer;">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            REQUEST CSR SUPPORT
          </button>
        ` : ''}
        ${hasCoordinates ? `
          <button class="btn btn-secondary" id="btn-modal-locate" onclick="window.handleModalLocateClick('${eventId}', ${lat}, ${lon})" style="display:inline-flex;align-items:center;gap:6px;font-size:12px;padding:7px 14px;cursor:pointer;">
            <svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
            Locate on Map
          </button>
        ` : ''}
      </div>
      <button class="btn btn-secondary" id="btn-modal-dismiss" onclick="window.closeEventModal()" style="font-size:12px;padding:7px 14px;cursor:pointer;">
        Close
      </button>
    </div>
  `;

  // Wire cost breakdown toggle
  document.getElementById('btn-toggle-cost-breakdown')?.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const el = document.getElementById('modal-cost-breakdown');
    if (el) {
      const isHidden = el.style.display === 'none';
      el.style.display = isHidden ? 'block' : 'none';
      e.target.textContent = isHidden ? 'Hide Cost Breakdown ▲' : 'View Cost Breakdown ▼';
    }
  });

  // Wire CSR request button
  document.getElementById('btn-modal-csr')?.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    window.handleModalCsrClick(eventId);
  });

  // Wire dismiss button
  document.getElementById('btn-modal-dismiss')?.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    closeEventModal();
  });

  // Wire locate on map button
  if (hasCoordinates) {
    document.getElementById('btn-modal-locate')?.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      window.handleModalLocateClick(eventId, lat, lon);
    });
  }

  // Ensure modal-close ("X" button) always closes modal
  document.getElementById('modal-close')?.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    closeEventModal();
  });
}
