/**
 * BusSense API Client
 * Typed fetch wrapper for all backend endpoints.
 */
const API_BASE = (typeof window !== 'undefined' && window.location && window.location.origin) 
  ? `${window.location.origin}/api` 
  : 'http://localhost:8000/api';

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const text = await res.text();
      console.error(`API ${res.status}: ${url}`, text);
      return null;
    }
    return await res.json();
  } catch (err) {
    console.error(`API fetch failed: ${url}`, err);
    return null;
  }
}

const api = {
  // Health
  health: () => apiFetch('/health'),

  // Events — list
  events: (params = {}) => {
    const q = new URLSearchParams();
    if (params.event_type) q.set('event_type', params.event_type);
    if (params.severity) q.set('severity', params.severity);
    if (params.bus_id) q.set('bus_id', params.bus_id);
    if (params.limit) q.set('limit', params.limit);
    if (params.offset) q.set('offset', params.offset);
    return apiFetch(`/events?${q}`);
  },

  // Single event
  event: (id) => apiFetch(`/events/${encodeURIComponent(id)}`),

  // Road events
  roadEvents: (params = {}) => {
    const q = new URLSearchParams();
    if (params.defect_type) q.set('defect_type', params.defect_type);
    if (params.road_condition) q.set('road_condition', params.road_condition);
    if (params.limit) q.set('limit', params.limit);
    return apiFetch(`/road-events?${q}`);
  },

  // Traffic
  traffic: (params = {}) => {
    const q = new URLSearchParams();
    if (params.bus_id) q.set('bus_id', params.bus_id);
    if (params.traffic_state) q.set('traffic_state', params.traffic_state);
    if (params.limit) q.set('limit', params.limit);
    return apiFetch(`/traffic?${q}`);
  },

  // VRU areas
  vruAreas: () => apiFetch('/vru-areas'),

  // Media for event
  eventMedia: (eventId) => apiFetch(`/media/event/${encodeURIComponent(eventId)}`),

  // CSR Sponsorship & Funding
  csrPartners: () => apiFetch('/csr/partners'),
  csrRequests: (status) => apiFetch('/csr/requests' + (status ? `?status=${encodeURIComponent(status)}` : '')),
  csrRequest: (id) => apiFetch(`/csr/requests/${encodeURIComponent(id)}`),
  createCsrRequest: (data) => apiFetch('/csr/requests', { method: 'POST', body: JSON.stringify(data) }),
  updateCsrStatus: (id, status) => apiFetch(`/csr/requests/${encodeURIComponent(id)}/status`, { method: 'PATCH', body: JSON.stringify({ status }) }),

  // Infrastructure & Road Hazard Extensions
  infrastructureEvents: (params = {}) => {
    const q = new URLSearchParams();
    if (params.event_type) q.set('event_type', params.event_type);
    if (params.status) q.set('status', params.status);
    if (params.corridor_id) q.set('corridor_id', params.corridor_id);
    if (params.limit) q.set('limit', params.limit);
    return apiFetch(`/infrastructure/events?${q}`);
  },
  infrastructureAssets: (params = {}) => {
    const q = new URLSearchParams();
    if (params.asset_type) q.set('asset_type', params.asset_type);
    if (params.corridor_id) q.set('corridor_id', params.corridor_id);
    return apiFetch(`/infrastructure/assets?${q}`);
  },
  trichyRoute: () => apiFetch('/demo/routes/trichy'),
  startReplay: (data = {}) => apiFetch('/demo/replay/start', { method: 'POST', body: JSON.stringify(data) }),
  replayStatus: () => apiFetch('/demo/replay/status'),
  buses: () => apiFetch('/buses'),
  liveIntelligence: () => apiFetch('/telemetry/live-intelligence'),
};

export default api;

