/**
 * BusSense SVG Icon Library
 * =========================
 * Provides tiny inline SVG strings for use in HTML template literals.
 * All icons are 14×14, stroke-based, currentColor — they inherit text color automatically.
 *
 * Usage:
 *   import { icon } from './icons.js';
 *   `<span class="badge-event">${icon('pothole')} Pothole</span>`
 */

// Base SVG wrapper — 14x14 inline, vertically centred
const svg = (path) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0;margin-right:2px;">${path}</svg>`;

// --- Icon definitions ---
export const ICONS = {
  // Road Health
  pothole:      svg('<circle cx="12" cy="12" r="9"/><path d="M8 12a4 4 0 0 0 8 0"/>'),
  crack:        svg('<polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'),
  waterlogging: svg('<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>'),
  road_marking: svg('<line x1="5" y1="8" x2="19" y2="8"/><line x1="5" y1="14" x2="19" y2="14"/><line x1="5" y1="20" x2="11" y2="20"/>'),

  // Traffic Incidents
  collision:    svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'),
  rapid_close:  svg('<polyline points="15 18 9 12 15 6"/><polyline points="7 18 1 12 7 6"/>'),
  following:    svg('<rect x="1" y="3" width="15" height="13" rx="2"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="18" r="2"/>'),
  hard_brake:   svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  cut_in:       svg('<path d="M9 18 Q15 12 9 6"/><polyline points="14 6 9 6 9 11"/>'),
  lateral:      svg('<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>'),
  weaving:      svg('<path d="M2 16 Q6.5 8 11 12 Q15.5 16 20 8" fill="none"/>'),
  evasive:      svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'),
  hit_and_run:  svg('<path d="M13 4a1 1 0 1 1-2 0 1 1 0 0 1 2 0"/><path d="M7 21l5-10 5 5"/><path d="M15 11l3 5"/>'),

  // VRU
  pedestrian:   svg('<circle cx="12" cy="4" r="2"/><path d="M10 8h4l1 4-3 1-2-2-3.5 1z"/><path d="M9 16l1-4 2 2 1 4M15 16l-1-4"/>'),
  cyclist:      svg('<circle cx="18" cy="18" r="3"/><circle cx="6" cy="18" r="3"/><path d="M6 18L9.5 6H12"/><polyline points="12 6 17 6"/><path d="M17 6l-1.5 12"/>'),
  vru_near_miss:svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  sudden_vru:   svg('<circle cx="12" cy="4" r="2"/><path d="M9 20l3-10 3 10"/><path d="M6 8l6 2 6-2"/>'),
  vru_area:     svg('<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>'),

  // Infrastructure
  anpr:         svg('<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>'),
  traffic:      svg('<rect x="5" y="2" width="14" height="20" rx="3"/><circle cx="12" cy="7" r="2" fill="currentColor" stroke="none"/><circle cx="12" cy="13" r="2"/><circle cx="12" cy="19" r="2"/>'),
  sign:         svg('<rect x="3" y="4" width="18" height="12" rx="2"/><line x1="12" y1="16" x2="12" y2="21"/><line x1="8" y1="21" x2="16" y2="21"/>'),
  divider:      svg('<line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>'),
  hazard:       svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  missing:      svg('<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>'),

  // General UI
  search:       svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'),
  pin:          svg('<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>'),
  warning:      svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
  info:         svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'),
  camera:       svg('<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>'),
  film:         svg('<rect x="2" y="2" width="20" height="20" rx="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/>'),
  download:     svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>'),
  bell:         svg('<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>'),
  volume:       svg('<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>'),
  mute:         svg('<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>'),
  close:        svg('<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>'),
  keyboard:     svg('<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M6 8h.01M10 8h.01M14 8h.01M18 8h.01M8 12h.01M12 12h.01M16 12h.01M7 16h10"/>'),
  money:        svg('<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>'),
  error_page:   svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'),
  not_found:    svg('<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/>'),
  company:      svg('<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>'),
  arrow_down:   svg('<polyline points="6 9 12 15 18 9"/>'),
  arrow_up:     svg('<polyline points="18 15 12 9 6 15"/>'),
  check:        svg('<polyline points="20 6 9 17 4 12"/>'),
};

/**
 * Get an SVG icon string by name.
 * @param {string} name - Icon name from ICONS map
 * @returns {string} Inline SVG HTML string, or empty string if not found
 */
export function icon(name) {
  return ICONS[name] || ICONS.info || '';
}

export default icon;
