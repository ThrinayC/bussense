/**
 * BusSense Notification Engine
 * ============================
 * Polls the backend every 20s for new high-priority events.
 * Manages a local notification store, bell badge, panel, and toast.
 */
import api from './api.js';
import { playCriticalAlarm } from './audio-alerts.js';
import { ICONS } from './icons.js';

const HIGH_PRIORITY_TYPES = new Set([
  'HIT_AND_RUN_CANDIDATE', 'COLLISION_CANDIDATE', 'EMERGENCY_EVASIVE_MANEUVER',
  'PEDESTRIAN_IN_PATH', 'CYCLIST_IN_PATH', 'VRU_NEAR_MISS',
]);
const HIGH_PRIORITY_SEVERITY = new Set(['CRITICAL', 'HIGH']);

let notifications = [];    // [{id, event, read, addedAt}]
let seenEventIds = new Set();
let pollInterval = null;
let onOpenEvent = null;    // callback when notification clicked

// ---------------------------------------------------------------------------
// Priority classification
// ---------------------------------------------------------------------------
function isHighPriority(ev) {
  const type = (ev.event_type || '').toUpperCase();
  const sev = (ev.severity || '').toUpperCase();
  if (HIGH_PRIORITY_TYPES.has(type)) return true;
  if (type === 'POTHOLE' && sev === 'CRITICAL') return true;
  if (type === 'CRACK' && sev === 'CRITICAL') return true;
  if (type === 'TRAFFIC_AGGREGATE' && (ev.traffic_state === 'JAM_RISK' || ev.traffic_state === 'CONGESTED')) return true;
  return false;
}

function getNotifIcon(ev) {
  const type = (ev.event_type || '').toUpperCase();
  const iconMap = {
    HIT_AND_RUN_CANDIDATE:       ICONS.hit_and_run,
    COLLISION_CANDIDATE:         ICONS.collision,
    EMERGENCY_EVASIVE_MANEUVER:  ICONS.evasive,
    PEDESTRIAN_IN_PATH:          ICONS.pedestrian,
    CYCLIST_IN_PATH:             ICONS.cyclist,
    VRU_NEAR_MISS:               ICONS.vru_near_miss,
    POTHOLE:                     ICONS.pothole,
    CRACK:                       ICONS.crack,
    TRAFFIC_AGGREGATE:           ICONS.traffic,
    RAPID_CLOSING:               ICONS.rapid_close,
  };
  return iconMap[type] || ICONS.warning;
}

function getNotifTitle(ev) {
  const labels = {
    HIT_AND_RUN_CANDIDATE: 'Hit-and-Run Candidate', COLLISION_CANDIDATE: 'Collision Candidate',
    EMERGENCY_EVASIVE_MANEUVER: 'Emergency Evasive Maneuver', PEDESTRIAN_IN_PATH: 'Pedestrian In Path',
    CYCLIST_IN_PATH: 'Cyclist In Path', VRU_NEAR_MISS: 'VRU Near Miss',
    POTHOLE: 'Severe Pothole Detected', CRACK: 'Critical Road Crack', TRAFFIC_AGGREGATE: 'Jam Risk Alert',
    RAPID_CLOSING: 'Rapid Closing',
  };
  return labels[(ev.event_type || '').toUpperCase()] || (ev.event_type || '').replace(/_/g, ' ');
}

function timeAgo(timestamp) {
  const diff = Date.now() / 1000 - timestamp;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

// ---------------------------------------------------------------------------
// Check for new events
// ---------------------------------------------------------------------------
async function checkForNewEvents() {
  const data = await api.events({ limit: 100 });
  const events = Array.isArray(data) ? data : [];

  let addedNew = false;
  for (const ev of events) {
    if (!ev.event_id || seenEventIds.has(ev.event_id)) continue;
    seenEventIds.add(ev.event_id);

    if (isHighPriority(ev)) {
      notifications.unshift({ id: ev.event_id, event: ev, read: false, addedAt: Date.now() });
      addedNew = true;
    }
  }

  if (addedNew) {
    updateBellBadge();
    showToast(notifications[0]);
  }
}

// ---------------------------------------------------------------------------
// Bell badge
// ---------------------------------------------------------------------------
function updateBellBadge() {
  const unread = notifications.filter(n => !n.read).length;
  const badge = document.getElementById('notif-badge');
  const bell = document.getElementById('notif-bell');
  if (badge) {
    badge.textContent = unread;
    badge.style.display = unread > 0 ? 'flex' : 'none';
  }
  if (bell) {
    bell.style.color = unread > 0 ? 'var(--red-500)' : 'var(--text-secondary)';
  }
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
function showToast(notif) {
  const existing = document.getElementById('bs-toast');
  if (existing) existing.remove();

  const ev = notif.event;
  const sev = (ev.severity || '').toUpperCase();
  if (sev === 'CRITICAL' || sev === 'HIGH') {
    playCriticalAlarm();
  }

  const toast = document.createElement('div');
  toast.id = 'bs-toast';
  toast.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:#fff;border:1px solid #E2E8F0;border-left:4px solid ${sev === 'CRITICAL' ? '#EF4444' : '#F59E0B'};
    border-radius:10px;padding:14px 18px;min-width:300px;max-width:360px;
    box-shadow:0 8px 24px rgba(15,23,42,0.12);
    display:flex;align-items:flex-start;gap:12px;
    animation:slideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    font-family:Inter,sans-serif;cursor:pointer;
  `;
  toast.innerHTML = `
    <span style="font-size:22px;flex-shrink:0;">${getNotifIcon(ev)}</span>
    <div style="flex:1;min-width:0;">
      <div style="font-size:11px;font-weight:700;letter-spacing:0.5px;color:${sev === 'CRITICAL' ? 'var(--red-600)' : 'var(--amber-600)'};margin-bottom:2px;">${sev} ALERT</div>
      <div style="font-size:13px;font-weight:700;color:#0F172A;margin-bottom:2px;">${getNotifTitle(ev)}</div>
      <div style="font-size:11px;color:#64748B;">Corridor: ${ev.demo_route_location?.road_name || 'Trichy'} · ${timeAgo(ev.timestamp)}</div>
    </div>
    <button onclick="this.parentElement.remove()" style="background:none;border:none;color:#94A3B8;cursor:pointer;font-size:16px;padding:0;flex-shrink:0;">✕</button>
    <div class="toast-progress"></div>
  `;
  toast.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;
    notif.read = true;
    updateBellBadge();
    toast.remove();
    if (onOpenEvent) onOpenEvent(ev);
  });
  document.body.appendChild(toast);

  let dismissTimer = setTimeout(() => { if (toast.parentElement) toast.remove(); }, 6000);
  toast.addEventListener('mouseenter', () => clearTimeout(dismissTimer));
  toast.addEventListener('mouseleave', () => {
    dismissTimer = setTimeout(() => { if (toast.parentElement) toast.remove(); }, 3000);
  });
}

// ---------------------------------------------------------------------------
// Notification panel
// ---------------------------------------------------------------------------
export function openPanel() {
  const existing = document.getElementById('notif-panel');
  if (existing) { existing.remove(); return; }

  const panel = document.createElement('div');
  panel.id = 'notif-panel';
  panel.style.cssText = `
    position:fixed;top:56px;right:8px;width:340px;max-height:480px;
    background:#fff;border:1px solid #E2E8F0;border-radius:12px;
    box-shadow:0 8px 24px rgba(0,0,0,0.12);z-index:9000;
    display:flex;flex-direction:column;overflow:hidden;font-family:Inter,sans-serif;
  `;
  panel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid #F1F5F9;">
      <span style="font-size:13px;font-weight:600;color:#1E293B;">Notifications</span>
      <button id="notif-mark-all" style="background:none;border:none;font-size:11px;color:#3B82F6;cursor:pointer;font-weight:500;">Mark all read</button>
    </div>
    <div id="notif-list" style="flex:1;overflow-y:auto;"></div>
    ${notifications.length === 0 ? '' : ''}
  `;

  renderPanelList(panel.querySelector('#notif-list'));

  panel.querySelector('#notif-mark-all').addEventListener('click', () => {
    notifications.forEach(n => n.read = true);
    updateBellBadge();
    renderPanelList(panel.querySelector('#notif-list'));
  });

  document.body.appendChild(panel);

  // Close on outside click
  setTimeout(() => {
    document.addEventListener('click', function handler(e) {
      if (!panel.contains(e.target) && e.target.id !== 'notif-bell') {
        panel.remove();
        document.removeEventListener('click', handler);
      }
    });
  }, 10);
}

function renderPanelList(container) {
  if (!container) return;
  if (notifications.length === 0) {
    container.innerHTML = `<div style="padding:32px 16px;text-align:center;color:#94A3B8;font-size:13px;">No alerts. All clear.</div>`;
    return;
  }

  container.innerHTML = notifications.map((notif, i) => {
    const ev = notif.event;
    const severityColors = { CRITICAL: '#EF4444', HIGH: '#F59E0B', MEDIUM: '#3B82F6', LOW: '#94A3B8' };
    const sev = (ev.severity || '').toUpperCase();
    return `<div data-notif-idx="${i}" style="
      display:flex;gap:12px;padding:12px 16px;border-bottom:1px solid #F8FAFC;
      background:${notif.read ? '#fff' : '#EFF6FF'};cursor:pointer;
      transition:background 0.1s;
    " class="notif-item">
      <span style="font-size:18px;flex-shrink:0;margin-top:2px;">${getNotifIcon(ev)}</span>
      <div style="flex:1;min-width:0;">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
          <span style="font-size:10px;font-weight:700;color:${severityColors[sev] || '#64748B'};text-transform:uppercase;">${sev}</span>
          ${!notif.read ? '<span style="width:6px;height:6px;background:#3B82F6;border-radius:50%;flex-shrink:0;"></span>' : ''}
        </div>
        <div style="font-size:12px;font-weight:600;color:#1E293B;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${getNotifTitle(ev)}</div>
        <div style="font-size:11px;color:#64748B;">Bus: ${ev.bus_id || '—'} · ${timeAgo(ev.timestamp)}</div>
      </div>
    </div>`;
  }).join('');

  container.querySelectorAll('.notif-item').forEach(item => {
    item.addEventListener('click', () => {
      const idx = parseInt(item.dataset.notifIdx);
      const notif = notifications[idx];
      if (!notif) return;
      notif.read = true;
      updateBellBadge();
      const panel = document.getElementById('notif-panel');
      if (panel) panel.remove();
      if (onOpenEvent) onOpenEvent(notif.event);
    });
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
export function initNotifications(opts = {}) {
  if (opts.onOpenEvent) onOpenEvent = opts.onOpenEvent;

  // Seed with existing high-priority events (mark as read — they're historical)
  api.events({ limit: 200 }).then(data => {
    const events = Array.isArray(data) ? data : [];
    for (const ev of events) {
      if (!ev.event_id || seenEventIds.has(ev.event_id)) continue;
      seenEventIds.add(ev.event_id);
      if (isHighPriority(ev)) {
        notifications.push({ id: ev.event_id, event: ev, read: true, addedAt: Date.now() });
      }
    }
    updateBellBadge();
  });

  // Poll for new events
  pollInterval = setInterval(checkForNewEvents, 20000);
}

export function destroyNotifications() {
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}
