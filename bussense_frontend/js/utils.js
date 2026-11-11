/**
 * BusSense Shared Utilities
 * Common formatting, time, severity, and DOM helpers.
 */
import { ICONS } from './icons.js';

// --- Time ---
export function timeAgo(timestamp) {
  if (!timestamp) return '—';
  const now = Date.now() / 1000;
  const ts = typeof timestamp === 'number' ? timestamp : new Date(timestamp).getTime() / 1000;
  const diff = now - ts;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function formatTimestamp(ts) {
  if (!ts) return '—';
  const d = typeof ts === 'number' ? new Date(ts * 1000) : new Date(ts);
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
  });
}

// --- Severity ---
export function severityBadge(severity) {
  const s = (severity || '').toLowerCase();
  return `<span class="badge-severity ${s}">${severity || 'N/A'}</span>`;
}

export function severityColor(severity) {
  const map = { CRITICAL: '#EF4444', HIGH: '#F59E0B', MEDIUM: '#3B82F6', LOW: '#94A3B8' };
  return map[(severity || '').toUpperCase()] || '#94A3B8';
}

// --- Event Types ---
const EVENT_TYPE_MAP = {
  POTHOLE:                   { label: 'Pothole',      css: 'pothole',     icon: ICONS.pothole },
  CRACK:                     { label: 'Crack',        css: 'crack',       icon: ICONS.crack },
  COLLISION_CANDIDATE:       { label: 'Collision',    css: 'incident',    icon: ICONS.collision },
  RAPID_CLOSING:             { label: 'Rapid Close',  css: 'incident',    icon: ICONS.rapid_close },
  DANGEROUS_FOLLOWING:       { label: 'Following',    css: 'incident',    icon: ICONS.following },
  HARD_BRAKING:              { label: 'Hard Brake',   css: 'incident',    icon: ICONS.hard_brake },
  CUT_IN_CANDIDATE:          { label: 'Cut-In',       css: 'incident',    icon: ICONS.cut_in },
  ABRUPT_LATERAL_MANEUVER:   { label: 'Lateral',      css: 'incident',    icon: ICONS.lateral },
  WEAVING_CANDIDATE:         { label: 'Weaving',      css: 'incident',    icon: ICONS.weaving },
  EMERGENCY_EVASIVE_MANEUVER:{ label: 'Evasive',      css: 'incident',    icon: ICONS.evasive },
  HIT_AND_RUN_CANDIDATE:     { label: 'Hit & Run',    css: 'hit-and-run', icon: ICONS.hit_and_run },
  PEDESTRIAN_IN_PATH:        { label: 'Pedestrian',   css: 'vru',         icon: ICONS.pedestrian },
  CYCLIST_IN_PATH:           { label: 'Cyclist',      css: 'vru',         icon: ICONS.cyclist },
  VRU_NEAR_MISS:             { label: 'VRU Near Miss',css: 'vru',         icon: ICONS.vru_near_miss },
  SUDDEN_VRU_EMERGENCE:      { label: 'Sudden VRU',   css: 'vru',         icon: ICONS.sudden_vru },
  ANPR_VERIFIED:             { label: 'ANPR',         css: 'anpr',        icon: ICONS.anpr },
  TRAFFIC_AGGREGATE:         { label: 'Traffic',      css: 'traffic',     icon: ICONS.traffic },
  VULNERABLE_AREA_AGGREGATE: { label: 'VRU Area',     css: 'vru',         icon: ICONS.vru_area },
};

export function eventTypeBadge(type) {
  const info = EVENT_TYPE_MAP[type] || { label: type, css: '', icon: ICONS.info };
  return `<span class="badge-event ${info.css}" style="display:inline-flex;align-items:center;gap:3px;">${info.icon}${info.label}</span>`;
}

export function eventTypeLabel(type) {
  return (EVENT_TYPE_MAP[type] || { label: type }).label;
}

export function eventTypeIcon(type) {
  return (EVENT_TYPE_MAP[type] || { icon: ICONS.info }).icon;
}

// --- Traffic ---
export function trafficStateBadge(state) {
  const s = (state || '').toLowerCase();
  const labels = {
    free_flow: 'Free Flow',
    building: 'Building',
    congested: 'Congested',
    jam_risk: 'Jam Risk',
  };
  return `<span class="badge-traffic ${s}">${labels[s] || state}</span>`;
}

// --- Numbers ---
export function formatNumber(n, decimals = 0) {
  if (n == null || isNaN(n)) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return Number(n).toFixed(decimals);
}

export function pct(value) {
  if (value == null) return '—';
  return (value * 100).toFixed(1) + '%';
}

// --- DOM ---
export function el(tag, attrs = {}, children = '') {
  const props = Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(' ');
  return `<${tag} ${props}>${children}</${tag}>`;
}

export function createEl(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

export function formatGPS(lat, lon) {
  if (lat == null || lon == null) return '<span style="color:var(--text-tertiary);font-style:italic;">Unavailable</span>';
  return `${Number(lat).toFixed(4)}°, ${Number(lon).toFixed(4)}°`;
}

// --- Confidence ---
export function confidenceBar(val) {
  if (val == null) return '—';
  const pctVal = Math.round(val * 100);
  const color = pctVal >= 80 ? 'var(--green-500)' : pctVal >= 60 ? 'var(--amber-500)' : 'var(--red-500)';
  return `<div style="display:flex;align-items:center;gap:6px;">
    <div style="width:48px;height:4px;background:var(--gray-200);border-radius:2px;overflow:hidden;">
      <div style="width:${pctVal}%;height:100%;background:${color};border-radius:2px;"></div>
    </div>
    <span class="text-xs text-secondary">${pctVal}%</span>
  </div>`;
}
