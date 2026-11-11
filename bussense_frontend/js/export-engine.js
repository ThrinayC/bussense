/**
 * BusSense Client-Side Telemetry & Defect Export Engine
 * Generates verified CSV & JSON corridor reports directly in the browser.
 */
import api from './api.js';

function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function escapeCsv(val) {
  if (val == null) return '';
  const str = String(val).replace(/"/g, '""');
  return `"${str}"`;
}

export async function exportCorridorEventsCsv() {
  const events = await api.events({ limit: 500 });
  if (!Array.isArray(events) || !events.length) {
    alert('No events available for export.');
    return;
  }

  const headers = [
    'Event ID', 'Type', 'Severity', 'Timestamp', 'Bus ID',
    'Road Name', 'Landmark', 'Latitude', 'Longitude', 'Confidence',
    'Depth (cm)', 'Area (m2)', 'Crack Length (m)', 'TTC (s)', 'Plate Number'
  ];

  const rows = events.map(ev => [
    escapeCsv(ev.event_id || ev.id),
    escapeCsv(ev.event_type || ev.defect_type),
    escapeCsv(ev.severity),
    escapeCsv(ev.timestamp ? new Date(typeof ev.timestamp === 'number' ? ev.timestamp * 1000 : ev.timestamp).toISOString() : ''),
    escapeCsv(ev.bus_id || 'BUS-01'),
    escapeCsv(ev.demo_route_location?.road_name || 'Tiruchirappalli Corridor'),
    escapeCsv(ev.demo_route_location?.landmark || ''),
    escapeCsv(ev.demo_route_location?.lat ?? ev.gps_lat),
    escapeCsv(ev.demo_route_location?.lon ?? ev.gps_lon),
    escapeCsv(ev.confidence != null ? Math.round(ev.confidence * 100) + '%' : ''),
    escapeCsv(ev.depth_cm),
    escapeCsv(ev.area_m2),
    escapeCsv(ev.length_m),
    escapeCsv(ev.ttc_s),
    escapeCsv(ev.registration_number || ev.details?.registration_number || '')
  ].join(','));

  const csv = [headers.join(','), ...rows].join('\r\n');
  const filename = `BusSense_Corridor_Telemetry_${new Date().toISOString().slice(0, 10)}.csv`;
  downloadBlob(csv, filename, 'text/csv;charset=utf-8;');
}

export async function exportCorridorEventsJson() {
  const events = await api.events({ limit: 500 });
  const json = JSON.stringify(events, null, 2);
  const filename = `BusSense_Corridor_Events_${new Date().toISOString().slice(0, 10)}.json`;
  downloadBlob(json, filename, 'application/json;charset=utf-8;');
}
